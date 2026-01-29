'''
Interface with dftbplus
Only work on Linux/MacOS
'''

from .utils import ELEMENT_DICTIONARY
import os
import numpy as np

ANGULAR_MOMENTUM_MAXIMUM = {'H': 'H = "s"', 'C':'C = "p"', 'N':'N = "p"', 'O':'O = "p"'}
HUBBARD_DERIVS = {'H': 'H = -0.1857', 'C':'C = -0.1492', 'N':'N = -0.1535', 'O':'O = -0.1575'}

class DFTBPlusRunner:
    def __init__(self, run_directory, slater_koster_path, run_command):
        self.run_directory = run_directory
        self.slater_koster_path = slater_koster_path
        self.run_command = run_command

    def generate_input(self, file, atomic_numbers, positions):
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
        file.write(f'{n} C\n')
        file.write(f"{' '.join(lst_elements)}\n")
        for i in range(n):
            file.write(f"{i+1} {internal_keys[atomic_numbers[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n")
        file.write('}\n')
        file.write('''Hamiltonian = DFTB {
    SCC = Yes
    Filling = Fermi {
        Temperature [K] = 400
    }
    SlaterKosterFiles = Type2FileNames {
''')
        file.write(f'        Prefix = "{self.slater_koster_path}"')
        file.write('''
        Separator = "-"                     # Dash between type names
        Suffix = ".skf"                     # Suffix after second type name
    }
    SCCTolerance = 1.0E-007  # Extremely small
    MaxSCCIterations = 1000
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
        file.write('''    }
                    
}
 
Options = {
  WriteAutotestTag = Yes
}

Analysis {
  CalculateForces = Yes
}
  
Parallel {
  UseOmpThreads = Yes                   
}                   

ParserOptions = {
  ParserVersion = 5
}

''')

    def generate_input_pbc(self, file, atomic_numbers, positions, cell, k_point):
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
        file.write(f'{n} S\n')
        file.write(f"{' '.join(lst_elements)}\n")
        for i in range(n):
            file.write(f"{i+1} {internal_keys[atomic_numbers[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n")
        file.write(f"0.0000000 0.0000000 0.0000000\n")
        file.write(f"{cell[0][0]} {cell[0][1]} {cell[0][2]}\n")
        file.write(f"{cell[1][0]} {cell[1][1]} {cell[1][2]}\n")
        file.write(f"{cell[2][0]} {cell[2][1]} {cell[2][2]}\n")
        file.write('}\n')
        file.write('''Hamiltonian = DFTB {
    SCC = Yes
    Filling = Fermi {
        Temperature [K] = 400
    }
    SlaterKosterFiles = Type2FileNames {
''')
        file.write(f'        Prefix = "{self.slater_koster_path}"')
        file.write('''
        Separator = "-"                     # Dash between type names
        Suffix = ".skf"                     # Suffix after second type name
    }
    SCCTolerance = 1.0E-007  # Extremely small
    MaxSCCIterations = 1000
    HCorrection = Damping { Exponent = 4.00 }
    ThirdOrderFull = Yes

    MaxAngularMomentum = { 
''')
        for symbol in lst_elements:
            file.write(f'        {ANGULAR_MOMENTUM_MAXIMUM[symbol]}\n')
        file.write('''    }
    KPointsAndWeights = SupercellFolding {
''')                   
        file.write(f'    {k_point[0]}   0   0')
        file.write(f'    0   {k_point[0]}   0')
        file.write(f'    0   0   {k_point[0]}')
        file.write(f'    0.5  0.5   0.5')
        file.write('''    }
    HubbardDerivs{
''')
        for symbol in lst_elements:
            file.write(f'        {HUBBARD_DERIVS[symbol]}\n')                
        file.write('''    }
                    
}
 
Options = {
  WriteAutotestTag = Yes
}

Analysis {
  CalculateForces = Yes
}
                   
Parallel {
  UseOmpThreads = Yes                   
}

ParserOptions = {
  ParserVersion = 5
}

''')

    def collect(self, file):
        line = file.readline()
        while line != '':
            if 'Total energy:' in line:
                energy = float(line.split()[2])
            elif 'Total Forces' in line:
                forces = []
                line = file.readline()
                tmp = line.split()
                while len(tmp) == 4:
                    forces.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
                    line = file.readline()
                    tmp = line.split()
            line = file.readline()
        forces = np.array(forces, dtype=np.float32)
        return energy, forces

    def run(self, atomic_numbers, positions, cell=None, k_point=None):
        path = os.getcwd()
        os.chdir(self.run_directory)
        with open('dftb_in.hsd', 'w') as f:
            if cell is None:
                self.generate_input(f, atomic_numbers, positions)
            else:
                self.generate_input_pbc(f, atomic_numbers, positions, cell, k_point)
        # Auto close the file
        os.system(self.run_command)
        with open('detailed.out', 'r') as f:
            energy, forces = self.collect(f)
        os.chdir(path)
        return energy, forces # Return 
