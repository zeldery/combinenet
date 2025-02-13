'''
Delta learning
'''

import torch
from torch import nn
from .net import IndexNetwork, NetworkEnsemble

class DeltaNetwork(nn.Module):
    '''
    Delta network
    delta = (output - output_mean) - (input - input_mean)
    output = input + (output_mean - input_mean) + delta
    '''
    def __init__(self):
        super().__init__()

    def init(self, dimensions, activations, shifts_inp, shifts_outp):
        self.shifts_inp = shifts_inp.copy()
        self.shifts_outp = shifts_outp.copy()
        self.networks = IndexNetwork()
        shifts = [outp - inp for inp, outp in zip(shifts_inp, shifts_outp)]
        alphas = [1.0 for _ in shifts_inp]
        self.networks.init(dimensions, activations, shifts, alphas, torch.float64, True)

    def compute(self, index, aev, x):
        return self.networks.compute(index, aev) + x

    def batch_compute(self, index, aev, x):
        return self.networks.batch_compute(index, aev) + x

    def load(self, data):
        params = data['params']
        self.shifts_inp = params['shifts_inp'].copy()
        self.shifts_outp = params['shifts_outp'].copy()
        self.networks = IndexNetwork()
        self.networks.load(data['networks'])

    def dump(self):
        params = {'shifts_inp': self.shifts_inp, 'shifts_outp': self.shifts_outp}
        return {'params': params, 'networks': self.networks.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)

class DeltaEnsemble(nn.Module):
    '''
    Similar to the DeltaNetwork, but using NetworkEnsemble instead of IndexNetwork
    '''
    def __init__(self):
        super().__init__()

    def set(self, network_list, shifts_inp, shifts_outp):
        '''
        set will change the shifts and alphas value of individual network
        '''
        self.shifts_inp = shifts_inp.copy()
        self.shifts_outp = shifts_outp.copy()
        lst = [network.dump() for network in network_list]
        shifts = [outp - inp for inp, outp in zip(shifts_inp, shifts_outp)]
        alphas = [1.0 for _ in shifts_inp]
        for i in range(len(lst)):
            lst[i]['params']['shifts'] = shifts.copy()
            lst[i]['params']['alphas'] = alphas.copy()
            lst[i]['params']['output_type'] = torch.float64
            lst[i]['params']['sum_up'] = True
        network_list_final = []
        for params in lst:
            new_network = IndexNetwork()
            new_network.load(params)
            network_list_final.append(new_network)
        self.ensemble = NetworkEnsemble()
        self.ensemble.set(network_list_final)

    def compute(self, index, aev, x):
        return self.ensemble.compute(index, aev) + x

    def batch_compute(self, index, aev, x):
        return self.ensemble.batch_compute(index, aev) + x
    
    def load(self, data):
        params = data['params']
        self.shifts_inp = params['shifts_inp'].copy()
        self.shifts_outp = params['shifts_outp'].copy()
        self.ensemble = NetworkEnsemble()
        self.ensemble.load(data['ensemble'])

    def dump(self):
        params = {'shifts_inp': self.shifts_inp, 'shifts_outp': self.shifts_outp}
        return {'params': params, 'ensemble': self.ensemble.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)