'''
Training script for the delta learning
'''

import argparse
import torch
import time
from mlpotential.delta import *
from mlpotential.dataloader import DataIterator
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_argument():
    parser = argparse.ArgumentParser('train_delta', 
                                     description='Training script for delta learning')
    parser.add_argument('-m', '--model', default='model.pt')
    parser.add_argument('-c', '--checkpoint', default='checkpoint.pt')
    parser.add_argument('-b', '--before', default='dftb')
    parser.add_argument('-a', '--after', default='pbe0')
    parser.add_argument('-t', '--type', default='short')
    parser.add_argument('-d', '--data', default='data.hdf5')
    parser.add_argument('-g', '--gpu', default='cpu')
    parser.add_argument('-e', '--epoch', default='100')
    parser.add_argument('-l', '--learningrate', default='0.001')
    parser.add_argument('-r', '--restart', default='0')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    try:
        device = torch.device(args.gpu)
    except:
        raise ValueError(f'Unvalid value for gpu argument of {args.gpu}')
    
    if args.type == 'short':
        model = DeltaNetwork()
        model.read(args.model)
    elif args.type == 'ensemble':
        model = DeltaEnsemble()
        model.read(args.model)
    else:
        raise ValueError(f'Unvalid value for model type of {args.type}')
    
    data = DataIterator(args.data, ['atomic_numbers', 'coordinates', args.before, args.after])
    loader = data.dataloader(shuffle=True)
    criterion = torch.nn.MSELoss()
    criterion_eval = torch.nn.MSELoss(reduction='sum') # To return RMSE, need to sum up all the MSE
    model = model.to(device=device)

    if args.restart == '0':
        train_time = []
        validation_time = []
        start_iteration = 0
        rmse_train = []
        rmse_test = []
        best_rmse = 100000.0
        best_model = None
        bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
        weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
        optimizer = torch.optim.AdamW([{'params': bias_params, 'weight_decay':0.0},
                                    {'params': weight_params, 'weight_decay':0.0001}
                                        ], lr=float(args.learningrate))
    elif args.restart == '1':
        save_dict = torch.load(args.checkpoint, map_location=device)
        train_time = save_dict['train_time']
        validation_time = save_dict['validation_time']
        start_iteration = save_dict['epoch_finished']
        rmse_train = save_dict['rmse_train']
        rmse_test = save_dict['rmse_test']
        best_rmse = save_dict['best_rmse']
        best_model = save_dict['best_model']
        model.load(save_dict['current_model'])
        model = model.to(device=device)
        bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
        weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
        optimizer = torch.optim.AdamW([{'params': bias_params, 'weight_decay':0.0},
                                    {'params': weight_params, 'weight_decay':0.0001}
                                        ], lr=float(args.learningrate)) # To ensure optimizer get the correct model
        optimizer.load_state_dict(save_dict['optimizer'])
    else:
        raise ValueError(f'Incorrect option for restart {args.restart}')
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=0)

    for epoch in range(start_iteration, int(args.epoch)):
        torch.cuda.synchronize()
        begin_time = time.time()
        data.mode = 'train'
        for batch_data in loader:
            optimizer.zero_grad()
            atomic_numbers = batch_data['atomic_numbers'].to(torch.int64).to(device)
            positions = batch_data['coordinates'].to(torch.float32).to(device)
            before = batch_data[args.before].to(torch.float64).to(device)
            after = batch_data[args.after].to(torch.float64).to(device)
            

if __name__ == '__main__':
    main()
