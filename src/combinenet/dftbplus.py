'''
Interface with dftbplus
Only work on Linux/MacOS
'''

from utils import ELEMENT_DICTIONARY
import os

ANGULAR_MOMENTUM_MAXIMUM = {'H': 'H = "s"', 'C':'C = "p"', 'N':'N = "p"', 'O':'O = "p"'}

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
    SlaterKosterFiles = Type2FileNames {
''')
            f.write(f'        Prefix = "{self.slater_koster_path}"')
            f.write('''
        Separator = "-"                     # Dash between type names
        Suffix = ".skf"                     # Suffix after second type name
    }
    RangeSeparated = LC {
        Screening = MatrixBased { }
    }
    SCCTolerance = 1.0E-008  # Extremely small
    MaxSCCIterations = 1000
    Mixer = Broyden { }

    MaxAngularMomentum = { 
''')
            for symbol in lst_elements:
                f.write(f'{ANGULAR_MOMENTUM[symbol]}\n') ## FIXXING 
            f.write('''
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
        os.system(self.run_command)
        # CONTINUE HERE

        os.chdir(path)
