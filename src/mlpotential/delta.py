'''
Delta learning
'''

import numpy as np
import torch
from torch import nn
from net import IndexNetwork, IndexShift, NetworkEnsemble

class DeltaNetwork(nn.Module):
    '''
    delta = (y-shift_y)/alpha_y - (x-shift_x)/alpha_x
    y = shift_y + alpha_y * ((x-shift_x)/alpha_x + delta)
    y = shift_y + alpha_y * delta + alpha_y/alpha_x * (x-shift_x)
    y = (shift_y + alpha_y * delta) + ( - alpha_y/alpha_x * shift_x + alpha_y/alpha_x * x)
    '''
    def __init__(self):
        super().__init__()

    def init(self, dimensions, activations, shifts_inp, alphas_inp, shifts_outp, alphas_outp):
        self.shifts_inp = shifts_inp.copy()
        self.alphas_inp = alphas_inp.copy()
        self.shifts_outp = shifts_outp.copy()
        self.alphas_outp = alphas_outp.copy()
        self.networks = IndexNetwork()
        self.networks.init(dimensions, activations, shifts_outp, alphas_outp, torch.float64, False)
        scaler_alphas = [x/y for x,y in zip(alphas_outp, alphas_inp)]
        scaler_shifts = [-x*y for x,y in zip(scaler_alphas, shifts_inp)]
        self.scaler = IndexShift(scaler_shifts, scaler_alphas)

    def compute(self, index, aev, x):
        delta = self.networks.compute(index, aev)
        original = self.scaler.compute(index, x)
        return delta.sum() + original.sum()

    def batch_compute(self, index, aev, x):
        delta = self.networks.batch_compute(index, aev)
        original = self.scaler.batch_compute(index, x)
        return delta.sum(dim=1) + original.sum(dim=1)

    def load(self, data):
        params = data['params']
        self.shifts_inp = params['shifts_inp'].copy()
        self.alphas_inp = params['alphas_inp'].copy()
        self.shifts_outp = params['shifts_outp'].copy()
        self.alphas_outp = params['alphas_outp'].copy()
        self.networks = IndexNetwork()
        self.networks.load(data['networks'])
        scaler_alphas = [x/y for x,y in zip(self.alphas_outp, self.alphas_inp)]
        scaler_shifts = [-x*y for x,y in zip(scaler_alphas, self.shifts_inp)]
        self.scaler = IndexShift(scaler_shifts, scaler_alphas)

    def dump(self):
        params = {'shifts_inp': self.shifts_inp, 'alphas_inp': self.alphas_inp,
                  'shifts_outp': self.shifts_outp, 'alphas_outp': self.alphas_outp}
        return {'params': params, 'networks': self.networks.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)

class DeltaEnsemble(nn.Module):
    '''
    
    '''
    def __init__(self):
        super().__init__()

    def set(self, network_list, shifts_inp, alphas_inp, shifts_outp, alphas_outp):
        self.shifts_inp = shifts_inp.copy()
        self.alphas_inp = alphas_inp.copy()
        self.shifts_outp = shifts_outp.copy()
        self.alphas_outp = alphas_outp.copy()
        scaler_alphas = [x/y for x,y in zip(alphas_outp, alphas_inp)]
        scaler_shifts = [-x*y for x,y in zip(scaler_alphas, shifts_inp)]
        self.scaler = IndexShift(scaler_shifts, scaler_alphas)
        self.ensemble = NetworkEnsemble()
        self.ensemble.set(network_list)
    
    def compute(self, index, aev, x):
        delta = self.ensemble.compute(index, aev)
        original = self.scaler.compute(index, x)
        return delta.sum() + original.sum()

    def batch_compute(self, index, aev, x):
        delta = self.ensemble.batch_compute(index, aev)
        original = self.scaler.batch_compute(index, x)
        return delta.sum(dim=1) + original.sum(dim=1)
    
    def load(self, data):
        params = data['params']
        self.shifts_inp = params['shifts_inp'].copy()
        self.alphas_inp = params['alphas_inp'].copy()
        self.shifts_outp = params['shifts_outp'].copy()
        self.alphas_outp = params['alphas_outp'].copy()
        self.ensemble = NetworkEnsemble()
        self.ensemble.load(data['ensemble'])
        scaler_alphas = [x/y for x,y in zip(self.alphas_outp, self.alphas_inp)]
        scaler_shifts = [-x*y for x,y in zip(scaler_alphas, self.shifts_inp)]
        self.scaler = IndexShift(scaler_shifts, scaler_alphas)

    def dump(self):
        params = {'shifts_inp': self.shifts_inp, 'alphas_inp': self.alphas_inp,
                  'shifts_outp': self.shifts_outp, 'alphas_outp': self.alphas_outp}
        return {'params': params, 'ensemble': self.ensemble.dump()}

    def read(self, file_name):
        data = torch.load(file_name, map_location=torch.device('cpu'), weights_only=True)
        self.load(data)

    def write(self, file_name):
        data = self.dump()
        torch.save(data, file_name)