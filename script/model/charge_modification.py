from mlpotential.combine import *
import torch
from mlpotential.dataloader import DataIterator
from sklearn.linear_model import LinearRegression
import argparse
import numpy as np

from mlpotential.combine import ChargeModel, ChargeEnsembleModel, ChargeCompleteModel, ChargeCompleteEnsembleModel
import h5py
from tqdm import tqdm

def get_argument():
    parser = argparse.ArgumentParser('charge_model_modification', 
                                     description='Adjust the model mean and the data for training')
    parser.add_argument('-t', '--type', default='charge')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-d', '--data', default='data.hdf5')
    parser.add_argument('-o', '--output_model', default='output_model.pt')
    parser.add_argument('-u', '--output_data', default='data_substract.hdf5')
    parser.add_argument('-g', '--gpu', default='0')
    args = parser.parse_args()
    return args

def substract_energy(model_type, model_name, data_name, output, gpu='0'):
    if model_type == 'charge':
        model = ChargeModel()
    elif model_type == 'charge_ensemble':
        model = ChargeEnsembleModel()
    elif model_type == 'charge_complete':
        model = ChargeCompleteModel()
    elif model_type == 'charge_complete_ensemble':
        model = ChargeCompleteEnsembleModel()
    model.read(model_name)
    inp = h5py.File(data_name, 'r')
    if gpu =='0':
        device = torch.device('cpu')
    elif gpu == '1':
        device = torch.device('cuda')
    else:
        raise ValueError(f'Unvalid value for gpu argument of {gpu}')
    model = model.to(device)
    outp = h5py.File(output, 'w')
    n_train = len(inp['train'])
    n_test = len(inp['test'])
    outp.create_group('train')
    for i in tqdm(range(n_train), desc='Train'):
        outp.create_group(f'train/{i}')
        sub = inp[f'train/{i}']
        outp.copy(source=sub['atomic_numbers'], dest=outp[f'train/{i}'], name='atomic_numbers')
        outp.copy(source=sub['coordinates'], dest=outp[f'train/{i}'], name='coordinates')
        atomic_numbers = torch.tensor(np.array(sub['atomic_numbers']), dtype=torch.int64, device=device)
        positions = torch.tensor(np.array(sub['coordinates']), dtype=torch.float32, requires_grad=True, device=device)
        energies = model.batch_compute_energy(atomic_numbers, positions)
        forces = -torch.autograd.grad(energies, positions, torch.ones_like(energies))[0]
        e = np.array(sub['energies'], dtype=np.float64)
        f = np.array(sub['forces'], dtype=np.float32)
        f -= forces.detach().cpu().numpy()
        e -= energies.detach().cpu().numpy()
        outp.create_dataset(name=f'train/{i}/energies', data=e)
        outp.create_dataset(name=f'train/{i}/forces', data=f)
    outp.create_group('test')
    for i in tqdm(range(n_test), desc='Test'):
        outp.create_group(f'test/{i}')
        sub = inp[f'test/{i}']
        outp.copy(source=sub['atomic_numbers'], dest=outp[f'test/{i}'], name='atomic_numbers')
        outp.copy(source=sub['coordinates'], dest=outp[f'test/{i}'], name='coordinates')
        atomic_numbers = torch.tensor(np.array(sub['atomic_numbers']), dtype=torch.int64, device=device)
        positions = torch.tensor(np.array(sub['coordinates']), dtype=torch.float32, requires_grad=True, device=device)
        energies = model.batch_compute_energy(atomic_numbers, positions)
        forces = -torch.autograd.grad(energies, positions, torch.ones_like(energies))[0]
        e = np.array(sub['energies'], dtype=np.float64)
        f = np.array(sub['forces'], dtype=np.float32)
        f -= forces.detach().cpu().numpy()
        e -= energies.detach().cpu().numpy()
        outp.create_dataset(name=f'test/{i}/energies', data=e)
        outp.create_dataset(name=f'test/{i}/forces', data=f)
    inp.close()
    outp.close()

def change_mean(model_type, model_name, data_name, output, gpu='0'):
    if model_type == 'charge':
        model = ChargeModel()
        new_model = ChargeModel()
    elif model_type == 'charge_ensemble':
        model = ChargeEnsembleModel()
        new_model = ChargeEnsembleModel()
    elif model_type == 'charge_complete':
        model = ChargeCompleteModel()
        new_model = ChargeCompleteModel()
    elif model_type == 'charge_complete_ensemble':
        model = ChargeCompleteEnsembleModel()
        new_model = ChargeCompleteEnsembleModel()
    else:
        raise ValueError(f'Unsupported model type {model_type}')
    model.read(model_name)
    data = DataIterator(data_name, ['atomic_numbers', 'coordinates', 'energies'])
    data.mode = 'all'
    loader = data.dataloader(shuffle=True)
    e_lst = []
    count = {}
    for element in model.element_list:
        count[element] = []
    for batch_data in loader:
        atomic_numbers = batch_data['atomic_numbers'].detach().cpu().numpy()
        energies = batch_data['energies'].detach().cpu().numpy()
        e_lst += energies.tolist()
        for element in model.element_list:
            count[element] += (atomic_numbers == element).sum(axis=1).tolist()
    tmp = []
    for element in model.element_list:
        tmp.append(np.array(count[element]))
    x = np.stack(tmp, 1)
    energies_model = LinearRegression(fit_intercept=False)
    energies_model.fit(x, e_lst)
    for coef in energies_model.coef_:
        print(coef, end=' ')
    print()
    # Change the model here
    tmp = model.dump()
    if model_type == 'charge' or model_type == 'charge_complete':
        tmp['short_network']['params']['shifts'] = [coef.item() for coef in energies_model.coef_]
    elif model_type == 'charge_ensemble' or model_type == 'charge_complete_ensemble':
        n = len(tmp['short_ensemble'])
        for i in range(n):
            tmp['short_ensemble'][i]['params']['shifts'] = [coef.item() for coef in energies_model.coef_]
    new_model.load(tmp)
    new_model.write(output)

def main():
    args = get_argument()
    substract_energy(args.type, args.model, args.data, args.output_data, args.gpu)
    change_mean(args.type, args.model, args.output_data, args.output_model, args.gpu)

if __name__ == '__main__':
    main()