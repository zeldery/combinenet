'''
Interface with ASE package
'''

import numpy as np
import torch
from ase.calculators.calculator import Calculator as BaseCalculator
from .utils import HARTREE_TO_EV

class ASECalculator(BaseCalculator):
    '''
    Calculator wrap to integrate with ase package
    Work with ML models except Delta learning
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
        atoms.wrap()

        if atoms.get_pbc().any() and (not atoms.get_pbc().all()):
            raise NotImplementedError('Have not implement 1D and 2D periodic')
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        is_pbc = atoms.get_pbc().any()
        is_force = ('forces' in properties)
        is_stress = ('stress' in properties)
        if is_pbc:
            cell = torch.tensor(np.array(atoms.get_cell(complete=True)), dtype=torch.float32, device=self.device)
            if is_stress:
                scaling = torch.eye(3, requires_grad=True, dtype=torch.float32, device=self.device)
                volume = self.atoms.get_volume()
        
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64, device=self.device)
        if is_force:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

        if is_stress:
            positions = torch.matmul(positions, scaling)

        if is_pbc:
            energy = self.model.compute_pbc(atomic_numbers, positions, cell)
        else:
            energy = self.model.compute(atomic_numbers, positions)
        
        self.results['energy'] = energy.detach().cpu().item() * HARTREE_TO_EV

        if is_force:
            forces = - torch.autograd.grad(energy, positions)[0]
            self.results['forces'] = forces.detach().cpu().numpy() * HARTREE_TO_EV
        
        if 'charges' in properties:
            charges = self.model.compute_charge(atomic_numbers, positions, torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.results['charges'] = charges.detach().cpu().numpy()
        
        if is_stress:
            stress = torch.autograd.grad(energy, scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()
        
class ASEDeltaCalculator(BaseCalculator):
    '''
    Calculator wrapper to integrate with ase package for Delta Learning model
    '''
    implemented_properties = ['energy', 'forces']
    def __init__(self, runner, model=None, device=torch.device('cpu')):
        super().__init__()
        self.runner = runner
        self.model = model
        if model is not None:
            self.element_list = self.model.element_list.copy()
            self.model = self.model.to(device=device)
        self.device = device

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        atoms.wrap()
        if atoms.get_pbc().any() and (not atoms.get_pbc().all()):
            raise NotImplementedError('Have not implement 1D and 2D periodic')
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        is_pbc = atoms.get_pbc().any()
        is_force = ('forces' in properties)
        if is_pbc:
            cell = torch.tensor(np.array(atoms.get_cell(complete=True)), dtype=torch.float32, device=self.device)
            energy, forces = self.runner.run(atomic_numbers, positions, cell, [3, 3, 3])
        else:
            energy, forces = self.runner.run(atomic_numbers, positions)

        if self.model is None:
            self.results['energy'] = energy * HARTREE_TO_EV
            if is_force:
                self.results['forces'] = forces * HARTREE_TO_EV
            return # Stop here if no delta model presented
        
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64, device=self.device)
        if is_force:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)

        if is_pbc:
            if is_force:
                delta_e = self.model.compute_pbc(atomic_numbers, positions, cell)
                delta_f = -torch.autograd.grad(delta_e, positions)[0]
                self.results['energy'] = (energy + delta_e.detach().cpu().item()) * HARTREE_TO_EV
                self.results['forces'] = (forces + delta_f.detach().cpu().numpy()) * HARTREE_TO_EV
            else:
                delta_e = self.model.compute_pbc(atomic_numbers, positions, cell)
                self.results['energy'] = (energy + delta_e.detach().cpu().item()) * HARTREE_TO_EV
        else:
            if is_force:
                delta_e = self.model.compute(atomic_numbers, positions)
                delta_f = -torch.autograd.grad(delta_e, positions)[0]
                self.results['energy'] = (energy + delta_e.detach().cpu().item()) * HARTREE_TO_EV
                self.results['forces'] = (forces + delta_f.detach().cpu().numpy()) * HARTREE_TO_EV
            else:
                delta_e = self.model.compute(atomic_numbers, positions)
                self.results['energy'] = (energy + delta_e.detach().cpu().item()) * HARTREE_TO_EV
