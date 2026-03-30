'''
Interface with dftbplus
Only work on Linux/MacOS
'''

from .utils import ELEMENT_DICTIONARY, ANGSTROM_TO_BOHR
import os
import numpy as np

ANGULAR_MOMENTUM_MAXIMUM = {'H': 'H = "s"', 'C':'C = "p"', 'N':'N = "p"', 'O':'O = "p"'}
HUBBARD_DERIVS = {'H': 'H = -0.1857', 'C':'C = -0.1492', 'N':'N = -0.1535', 'O':'O = -0.1575'}

class DFTBPlusRunner:
    def __init__(self, run_directory, slater_koster_path, run_command, tolerate=6, force=False, hessian=False):
        self.run_directory = run_directory
        self.slater_koster_path = slater_koster_path
        self.run_command = run_command
        self.tolerate=6
        self.force = force
        self.hessian = hessian

    def generate_input(self, atomic_numbers, positions, cell=None, kpoint=None, tolerate=6, force=False, hessian=False):
        '''
        Generate the input for DFTB+ package
        if cell is None, it will be generate for molecular system
        force: True or False for first derivative calculations
        hessian: True or False for second derivative calculations
        '''
        file = open('dftb_in.hsd', 'w')
        n = positions.shape[0]
        lst_elements = []
        internal_keys = {}
        i = 1
        for a in atomic_numbers:
            symbol = ELEMENT_DICTIONARY[a]
            if symbol in lst_elements:
                continue
            lst_elements.append(symbol)
            internal_keys[a] = i
            i += 1
        file.write('Geometry = GenFormat {\n')
        if cell is None:
            file.write(f'    {n} C\n')
        else:
            file.write(f'    {n} S\n')
        file.write(f"    {' '.join(lst_elements)}\n")
        for i in range(n):
            file.write(f"    {i+1} {internal_keys[atomic_numbers[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]} \n")
        if cell is not None:
            file.write('    0.0 0.0 0.0 \n')
            file.write(f'    {cell[0][0]} {cell[0][1]} {cell[0][2]}\n')
            file.write(f'    {cell[1][0]} {cell[1][1]} {cell[1][2]}\n')
            file.write(f'    {cell[2][0]} {cell[2][1]} {cell[2][2]}\n')
        file.write('}\n\n')
        file.write('''Hamiltonian = DFTB {
    SCC = Yes
    Filling = Fermi {
        Temperature [K] = 298
    }
    SlaterKosterFiles = Type2FileNames {
''')
        file.write(f'        Prefix = "{self.slater_koster_path}"')
        file.write('''
        Separator = "-"                     # Dash between type names
        Suffix = ".skf"                     # Suffix after second type name
    }
''')
        file.write(f'    SCCTolerance = 1.0E-{tolerate}\n')
        file.write('''    MaxSCCIterations = 1000
    HCorrection = Damping { Exponent = 4.00 }
    ThirdOrderFull = Yes
    MaxAngularMomentum = { 
''')
        for symbol in lst_elements:
            file.write(f'        {ANGULAR_MOMENTUM_MAXIMUM[symbol]}\n')
        file.write('''    }
    HubbardDerivs{
''')
        for symbol in lst_elements:
            file.write(f'        {HUBBARD_DERIVS[symbol]}\n')                
        file.write('    }\n')
        if cell is not None:
            file.write('    KPointsAndWeights = SupercellFolding {\n')
            file.write(f'        {kpoint[0]}   0   0\n')
            file.write(f'        0   {kpoint[1]}   0\n')
            file.write(f'        0   0   {kpoint[2]}\n')
            file.write(f'        0.5 0.5 0.5\n')
            file.write('    }\n')
        file.write('''}

Parallel {
    UseOmpThreads = Yes                   
}                   

ParserOptions = {
    ParserVersion = 5
}
                   
''')
        if force:
            file.write('''Analysis {
    CalculateForces = Yes
}
                       
''')
        if hessian:
            file.write('''Driver = SecondDerivatives {
    Delta = 1E-4
}
''')
        file.close()

    def collect(self, force=False, hessian=False):
        file = open('detailed.out', 'r')
        line = file.readline()
        while line != '':
            if 'Total energy:' in line:
                energy = float(line.split()[2])
            elif force and 'Total Forces' in line:
                forces = []
                line = file.readline()
                tmp = line.split()
                while len(tmp) == 4:
                    forces.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
                    line = file.readline()
                    tmp = line.split()
            line = file.readline()
        file.close()
        hessians = []
        if hessian:
            file = open('hessian.out')
            lines = file.readlines()
            file.close()
            for line in lines:
                hessians += line.split()
            hessians = [float(x) for x in hessians]
        outputs = [energy]
        if force:
            forces = np.array(forces, dtype=np.float32)
            forces *= ANGSTROM_TO_BOHR
            outputs.append(forces)
        if hessian:
            n = int(len(hessians)**0.5)
            hessians = np.array(hessians, dtype=np.float32).reshape(n, n)
            hessians *= ANGSTROM_TO_BOHR**2
            outputs.append(hessians)
        return outputs

    def run(self, atomic_numbers, positions, cell=None, kpoint=None, tolerate=None, force=None, hessian=None):
        path = os.getcwd()
        os.chdir(self.run_directory)
        if tolerate is None:
            tolerate = self.tolerate
        if force is None:
            force = self.force
        if hessian is None:
            hessian = self.hessian
        self.generate_input(atomic_numbers, positions, cell, kpoint, tolerate, force, hessian)
        os.system(self.run_command)
        outputs = self.collect(force, hessian)
        os.chdir(path)
        return outputs
