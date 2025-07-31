'''
Interface with dftbplus
Only work on Linux/MacOS
'''

from .utils import ELEMENT_DICTIONARY
import os

ANGULAR_MOMENTUM_MAXIMUM = {'H': 'H = "s"', 'C':'C = "p"', 'N':'N = "p"', 'O':'O = "p"'}
HUBBARD_DERIVS = {'H': 'H = -0.1857', 'C':'C = -0.1492', 'N':'N = -0.1535', 'O':'O = -0.1575'}

class DFTBPlusRunner:
    def __init__(self, run_directory, slater_koster_path, run_command):
        self.run_directory = run_directory
        self.slater_koster_path = slater_koster_path
        self.run_command = run_command

    def run(self, atomic_numbers, positions):
        path = os.getcwd()
        os.chdir(self.run_directory)
        with open('dftb_in.hsd', 'w') as f:
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
            f.write('Geometry = GenFormat {\n')
            f.write(f'{n} C\n')
            f.write(f"{' '.join(lst_elements)}\n")
            for i in range(n):
                f.write(f"{i+1} {internal_keys[atomic_numbers[i]]} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n")
            f.write('}\n')
            f.write('''Hamiltonian = DFTB {
    SCC = Yes
    Filling = Fermi {
        Temperature [K] = 400
    }
    SlaterKosterFiles = Type2FileNames {
''')
            f.write(f'        Prefix = "{self.slater_koster_path}"')
            f.write('''
        Separator = "-"                     # Dash between type names
        Suffix = ".skf"                     # Suffix after second type name
    }
    SCCTolerance = 1.0E-009  # Extremely small
    MaxSCCIterations = 1000
    HCorrection = Damping { Exponent = 4.00 }
    ThirdOrderFull = Yes

    MaxAngularMomentum = { 
''')
            for symbol in lst_elements:
                f.write(f'        {ANGULAR_MOMENTUM_MAXIMUM[symbol]}\n')
            f.write('''    }
    HubbardDerivs{
''')
            for symbol in lst_elements:
                f.write(f'        {HUBBARD_DERIVS[symbol]}\n')                
            f.write('''    }
                    
}
 
Options = {
  WriteAutotestTag = Yes
}

Analysis {
  CalculateForces = Yes
}

ParserOptions = {
  ParserVersion = 5
}

''')
        # Auto close the file
        os.system(self.run_command)
        with open('detailed.out', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Total energy:' in line:
                    data = float(line.split()[2])
                    break
        os.chdir(path)
        return data # Return 
