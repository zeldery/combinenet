'''
CALCULATE THE QUANTITY FOR INIT THE MODEL
GENERATE THE BATCH DATA FOR TRAINING
'''

from combinenet.dataloader import H5PyScanner
from sklearn.linear_model import LinearRegression
import numpy as np

# Summary the data using scan_individual function

def collective_quantity(file_name):
    element_list = [1,6,7,8]
    element_count = {k:[] for k in element_list}
    scanner = H5PyScanner(['atomic_numbers', 'energies'], 'atomic_numbers')
    energies_list = []
    for dat in scanner.scan_individual([file_name]): # Support list of files
        for element in element_list:
            element_count[element].append((np.array(dat['atomic_numbers']) == element).sum())
        energies_list.append(dat['energies'])
    tmp = []
    for element in element_list:
        tmp.append(np.array(element_count[element]))
    x = np.stack(tmp, 1)
    print(f'n_structure = {x.shape[0]} n_atoms = {x.sum()}')
    energies_model = LinearRegression(fit_intercept=False)
    energies_model.fit(x, energies_list)
    for coef in energies_model.coef_:
        print(coef, end=' ')
    print()

def atomic_quantity(file_name):
    element_list = [1,6,7,8]
    scanner = H5PyScanner(['atomic_numbers', 'M1', 'M2', 'M3', 'Veff'], 'atomic_numbers')
    m1_list = {k:[] for k in element_list}
    m2_list = {k:[] for k in element_list}
    m3_list = {k:[] for k in element_list}
    v_list = {k:[] for k in element_list}
    n_structure = 0
    n_atoms = 0
    for dat in scanner.scan_individual([file_name]):
        n_structure += 1
        n_atoms += len(dat['atomic_numbers'])
        for i, element in enumerate(dat['atomic_numbers']):
            m1_list[element].append(dat['M1'][i])
            m2_list[element].append(dat['M2'][i])
            m3_list[element].append(dat['M3'][i])
            v_list[element].append(dat['Veff'][i])
    print(f'n_structure = {n_structure} n_atoms = {n_atoms}')
    for element in element_list:
        print(element)
        m1 = np.array(m1_list[element])
        print(f'{m1.mean()} {m1.std()}')
        m2 = np.array(m2_list[element])
        print(f'{m2.mean()} {m2.std()}')
        m3 = np.array(m3_list[element])
        print(f'{m3.mean()} {m3.std()}')
        v = np.array(v_list[element])
        print(f'{v.mean()} {v.std()}')
    
def generate():
    scanner = H5PyScanner(['atomic_numbers', 'coordinates', 'energies', 'forces', 'q', 'M1', 'M2', 'M3', 'Veff'], 'atomic_numbers')
    scanner.generate_iterator(['element_hcno.hdf5'], 200, 'pbe0xdm-ani1x-hcno-200.h5', 0.2)
    scanner.generate_iterator(['element_hcno.hdf5'], 1000, 'pbe0xdm-ani1x-hcno-1000.h5', 0.2)


def main():
    collective_quantity('data_energy_charge_2.hdf5')
    atomic_quantity('data_xdm_2.hdf5')
    generate()

if __name__ == '__main__':
    main()