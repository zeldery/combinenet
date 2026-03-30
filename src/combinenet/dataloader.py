'''
Handle the hdf5 data type, create ready-to-use data for pytorch
'''

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.linear_model import LinearRegression

class H5PyScanner:
    '''
    Scanner for the list of h5py files based on the field name
    The H5 file can have as many layers as possible, as long as it has the named fields in the list
    The exception is used for atomic_numbers or species in the case of 1 common value for all data in that batch
    '''
    def __init__(self, field_list, exception=None):
        self.field_list = field_list.copy()
        self.exception = exception
    
    def scan_(self, sub):
        '''
        Internal use for scan the h5py directory
        '''
        lst_key = list(sub.keys())
        if len(lst_key) > 0:
            if isinstance(sub[lst_key[0]], h5py.Group):
                for key in lst_key:
                    yield from self.scan_(sub[key])
            else:
                for field in self.field_list:
                    if field not in lst_key:
                        raise KeyError(f'{field} not in dataset')
                tmp = {}
                for field in self.field_list:
                    tmp[field] = np.array(sub[field])
                yield tmp

    def scan(self, list_file_name):
        '''
        Real scan for list of file name here
        '''
        for name in list_file_name:
            file = h5py.File(name)
            yield from self.scan_(file)
            file.close()

    def scan_individual(self, list_file_name):
        '''
        Similar to scan, but instead of all the field, slice through individual 
        data point in the batch
        The batch dimension is set to be the first dimension by default
        '''
        for dat in self.scan(list_file_name):
            n_dims = {}
            n_batch = -1
            for field in self.field_list:
                if field != self.exception:
                    n_batch = dat[field].shape[0]
                    break
            for field in self.field_list:
                n_dims[field] = dat[field].ndim
            for i in range(n_batch):
                tmp = {}
                for field in self.field_list:
                    if field == self.exception:
                        tmp[field] = dat[field]
                    else:
                        if n_dims[field] == 1:
                            tmp[field] = dat[field][i]
                        else:
                            tmp[field] = dat[field][i,...]
                yield tmp

    def stack_(self, data_list, pad_value):
        '''
        Stack the list of data into single array (the batch dimension is the first)
        The value is pad in the first dimension with the pad value
        '''
        if isinstance(data_list[0], np.ndarray):
            n_dim = data_list[0].ndim
            n_mask = -1
            for dat in data_list:
                if dat.shape[0] > n_mask:
                    n_mask = dat.shape[0]
            new_list = []
            for dat in data_list:
                pad_width = [[0, 0] for i in range(n_dim)]
                pad_width[0][1] = n_mask - dat.shape[0]
                new_list.append(np.pad(dat, pad_width, constant_values=pad_value))
            return np.stack(new_list, axis=0)
        else:
            return np.array(data_list)

    def scan_stack(self, list_file_name, n_batch):
        '''
        Main scanner with n_batch data stacked
        '''
        storage = {}
        for field in self.field_list:
            storage[field] = []
        n_count = 0
        for dat in self.scan_individual(list_file_name):
            n_count += 1
            for field in self.field_list:
                storage[field].append(dat[field])
            if n_count == n_batch:
                result = {}
                for field in self.field_list:
                    pad_value = 0.0
                    if isinstance(storage[field][0], np.ndarray):
                        if storage[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                    np.int64, np.uint64):
                            pad_value = -1
                            for i in range(len(storage[field])):
                                storage[field][i] = storage[field][i].astype(np.int64)
                    result[field] = self.stack_(storage[field], pad_value)
                    storage[field] = []
                yield result
                n_count = 0
        if n_count > 0:
            result = {}
            for field in self.field_list:
                pad_value = 0.0
                if isinstance(storage[field][0], np.ndarray):
                    if storage[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                np.int64, np.uint64):
                        pad_value = -1
                        for i in range(len(storage[field])):
                            storage[field][i] = storage[field][i].astype(np.int64)
                result[field] = self.stack_(storage[field], pad_value)
                storage[field] = []
            yield result
                
    def generate_iterator(self, list_file_name, n_batch, output_name, testratio):
        '''
        Generate the batched masked data file for training
        Help to load the data faster for training
        '''
        file = h5py.File(output_name, 'w')
        file.create_group('train')
        file.create_group('test')
        n_train = 0
        n_test = 0
        for dat in self.scan_stack(list_file_name, n_batch):
            if random.random() < testratio:
                file.create_group(f'test/{n_test}')
                for field in self.field_list:
                    file.create_dataset(f'test/{n_test}/{field}', data=dat[field])
                n_test += 1
            else:
                file.create_group(f'train/{n_train}')
                for field in self.field_list:
                    file.create_dataset(f'train/{n_train}/{field}', data=dat[field])
                n_train += 1
        file.close()

    def generate_iterator_2(self, list_file_name, n_batch, output_name, testratio):
        '''
        Second generation of generate_iterator, allowing the randomized individual 
        structure instead of each dataset in hdf5 file
        '''
        file = h5py.File(output_name, 'w')
        file.create_group('train')
        file.create_group('test')
        n_train = 0
        n_test = 0
        n_count_train = 0
        n_count_test = 0
        storage_train = {}
        storage_test = {}
        for field in self.field_list:
            storage_train[field] = []
            storage_test[field] = []
        for dat in self.scan_individual(list_file_name):
            if random.random() < testratio:
                # Test part
                n_count_test += 1
                for field in self.field_list:
                    storage_test[field].append(dat[field])
                if n_count_test == n_batch:
                    file.create_group(f'test/{n_test}')
                    for field in self.field_list:
                        pad_value = 0.0
                        if isinstance(storage_test[field][0], np.ndarray):
                            if storage_test[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                        np.int64, np.uint64):
                                pad_value = -1
                                for i in range(len(storage_test[field])):
                                    storage_test[field][i] = storage_test[field][i].astype(np.int64)
                        output = self.stack_(storage_test[field], pad_value)
                        file.create_dataset(f'test/{n_test}/{field}', data=output)
                        storage_test[field] = []
                    n_count_test = 0
                    n_test += 1
            else:
                # Train part
                n_count_train += 1
                for field in self.field_list:
                    storage_train[field].append(dat[field])
                if n_count_train == n_batch:
                    file.create_group(f'train/{n_train}')
                    for field in self.field_list:
                        pad_value = 0.0
                        if isinstance(storage_train[field][0], np.ndarray):
                            if storage_train[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                        np.int64, np.uint64):
                                pad_value = -1
                                for i in range(len(storage_train[field])):
                                    storage_train[field][i] = storage_train[field][i].astype(np.int64)
                        output = self.stack_(storage_train[field], pad_value)
                        file.create_dataset(f'train/{n_train}/{field}', data=output)
                        storage_train[field] = []
                    n_count_train = 0
                    n_train += 1
        if n_count_train > 0:
            file.create_group(f'train/{n_train}')
            for field in self.field_list:
                pad_value = 0.0
                if isinstance(storage_train[field][0], np.ndarray):
                    if storage_train[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                np.int64, np.uint64):
                        pad_value = -1
                        for i in range(len(storage_train[field])):
                            storage_train[field][i] = storage_train[field][i].astype(np.int64)
                output = self.stack_(storage_train[field], pad_value)
                file.create_dataset(f'train/{n_train}/{field}', data=output)
        if n_count_test > 0:
            file.create_group(f'test/{n_test}')
            for field in self.field_list:
                pad_value = 0.0
                if isinstance(storage_test[field][0], np.ndarray):
                    if storage_test[field][0].dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                                                np.int64, np.uint64):
                        pad_value = -1
                        for i in range(len(storage_test[field])):
                            storage_test[field][i] = storage_test[field][i].astype(np.int64)
                output = self.stack_(storage_test[field], pad_value)
                file.create_dataset(f'test/{n_test}/{field}', data=output)

class DataIterator(Dataset):
    '''
    The data iterator for pre-defined hdf5 structure that split in batch and split train/test
    Always load to CPU following the guideline
    '''
    def __init__(self, destination, property_list):
        self.file = h5py.File(destination, 'r')
        self.property_list = property_list.copy()
        self.n_train = len(list(self.file['train'].keys()))
        self.n_test = len(list(self.file['test'].keys()))
        self.mode = 'all'

    def __del__(self):
        self.file.close()

    def __len__(self):
        if self.mode == 'all':
            return self.n_train + self.n_test
        if self.mode == 'train':
            return self.n_train
        if self.mode == 'test':
            return self.n_test

    def __getitem__(self, index):
        result = {}
        if self.mode == 'train':
            sub = self.file[f'train/{index}']
        elif self.mode == 'test':
            sub = self.file[f'test/{index}']
        else:
            if index < self.n_train:
                sub = self.file[f'train/{index}']
            else:
                sub = self.file[f'test/{index - self.n_train}']
        for property in self.property_list:
            result[property] = torch.tensor(np.array(sub[property]))
        return result  

    def dataloader(self, **kwargs):
        return DataLoader(self, batch_size=None, batch_sampler=None, pin_memory=True, **kwargs)

__all__ = ['H5PyScanner', 'DataIterator']