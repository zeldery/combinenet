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
        is_pbc = atoms.get_pbc().any()
        is_force = ('forces' in properties)
        is_stress = ('stress' in properties) and is_pbc # Stress only defined with a cell

        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=self.device)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device,
                                 requires_grad=is_force)
        if is_pbc:
            cell = torch.tensor(np.array(atoms.get_cell(complete=True)), dtype=torch.float32, device=self.device)

        # Strain positions AND cell together; stress = dE/d(strain) / V at zero strain
        if is_stress:
            strain = torch.zeros((3, 3), dtype=torch.float32, device=self.device, requires_grad=True)
            deform = torch.eye(3, dtype=torch.float32, device=self.device) + strain
            pos_eval = positions @ deform
            cell = cell @ deform
        else:
            pos_eval = positions

        if is_pbc:
            energy = self.model.compute_pbc(atomic_numbers, pos_eval, cell)
        else:
            energy = self.model.compute(atomic_numbers, pos_eval)

        self.results['energy'] = energy.detach().cpu().item() * HARTREE_TO_EV

        # One autograd call for forces and stress, so the graph is not freed in between
        wrt = ([positions] if is_force else []) + ([strain] if is_stress else [])
        grads = torch.autograd.grad(energy, wrt) if wrt else []
        if is_force:
            self.results['forces'] = - grads[0].detach().cpu().numpy() * HARTREE_TO_EV
        if is_stress:
            stress = grads[-1].detach().cpu().numpy().astype(np.float64) / atoms.get_volume() * HARTREE_TO_EV
            stress = 0.5 * (stress + stress.T) # eV/A^3, symmetric
            self.results['stress'] = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] # Voigt: xx yy zz yz xz xy

        if 'charges' in properties:
            charges = self.model.compute_charge(atomic_numbers, positions, torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.results['charges'] = charges.detach().cpu().numpy()
        
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
