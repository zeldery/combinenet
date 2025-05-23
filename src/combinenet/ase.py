'''
Interface with ASE package
'''

import torch
from ase.calculators.calculator import Calculator as BaseCalculator
from .utils import HARTREE_TO_EV

class ASECalculator(BaseCalculator):
    '''
    
    '''
    implemented_properties = ['energy', 'forces', 'charges', 'stress']
    def __init__(self, model, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.model = model
        self.element_list = self.model.element_list.copy()
        self.model = self.model.to(device=device)

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        # All the information needed in atoms
        # All the things need to calculate in properties

        # CHANGE THIS WHEN IMPLEMENT PBC IN THE MODEL
        if atoms.get_pbc().any():
            raise NotImplementedError('Have not implement pbc yet')
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64, device=self.device)
        if 'forces' in properties:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

        energies = self.model.compute(atomic_numbers, positions)
        
        self.results['energy'] = energies.detach().cpu().item() * HARTREE_TO_EV

        if 'forces' in properties:
            forces = - torch.autograd.grad(energies, positions)[0]
            self.results['forces'] = forces.detach().cpu().numpy() * HARTREE_TO_EV
        
        if 'charges' in properties:
            charges = self.model.compute_charge(atomic_numbers, positions, torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.results['charges'] = charges.detach().cpu().numpy()
        
        if 'stress' in properties:
            raise NotImplementedError('Have not implement pbc yet')
        
'''
scaling = torch.eye(3, requires_grad=True, dtype=torch.float32, device=self.device)
cell = torch.matmul(cell, scaling)
volume = self.atoms.get_volume()
stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
self.results['stress'] = stress.cpu().numpy()
'''

