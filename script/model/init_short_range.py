'''
Generate a ShortRangeModel with atomic energy shifts fitted from real data.

Reads one or more HDF5 files, fits per-element atomic energies via linear
regression (H, C, N, O), then constructs and saves a ShortRangeModel
initialised with those shifts.

Usage:
    python init_short_range.py -d file1.hdf5 [file2.hdf5 ...] [-o ani_model.pt]
'''

import argparse
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from combinenet.dataloader import H5PyScanner
from combinenet.combine import ShortRangeModel
from combinenet.sf import SymmetryFunction
from combinenet.net import IndexNetwork

ELEMENT_LIST = [1, 6, 7, 8]   # H, C, N, O

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', required=True,
                    help='HDF5 file(s) containing atomic_numbers and energies')
parser.add_argument('-o', '--out', type=str, default='ani_model.pt',
                    help='Output model file (default: ani_model.pt)')
args = parser.parse_args()

# ── Fit atomic energies ───────────────────────────────────────────────────────

print('Scanning data files...')
element_count = {k: [] for k in ELEMENT_LIST}
energies_list = []

scanner = H5PyScanner(['atomic_numbers', 'energies'], 'atomic_numbers')
for dat in scanner.scan_individual(args.data):
    for element in ELEMENT_LIST:
        element_count[element].append(
            (np.array(dat['atomic_numbers']) == element).sum()
        )
    energies_list.append(dat['energies'])

x = np.stack([np.array(element_count[e]) for e in ELEMENT_LIST], axis=1)
y = np.array(energies_list)

print(f'Structures: {x.shape[0]}   Total atoms: {x.sum()}')

reg = LinearRegression(fit_intercept=False)
reg.fit(x, y)
atomic_energies = reg.coef_.tolist()

print('Fitted atomic energies (Hartree):')
for elem, val in zip(ELEMENT_LIST, atomic_energies):
    print(f'  Z={elem}: {val:.10f}')

# ── Build ShortRangeModel ─────────────────────────────────────────────────────

symfunc = SymmetryFunction()
symfunc.set(4, [16.0], [0.9, 1.16875, 1.4375, 1.70625, 1.975, 2.24375,
            2.5125, 2.78125, 3.05, 3.31875, 3.5875, 3.85625,
            4.125, 4.39375, 4.6625, 4.93125], 5.2,
            [0.19634954, 0.58904862, 0.9817477, 1.3744468,
            1.7671459, 2.1598449, 2.552544, 2.9452431],
            [32.0], [8.0], [0.9, 1.55, 2.2, 2.85], 3.5)

neuralnet = IndexNetwork()
neuralnet.init(
    [[384, 160, 128, 96, 1], [384, 144, 112, 96, 1],
     [384, 128, 112, 96, 1], [384, 128, 112, 96, 1]],
    [['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu'],
     ['celu', 'celu', 'celu', 'celu'], ['celu', 'celu', 'celu', 'celu']],
    atomic_energies,
    [1.0, 1.0, 1.0, 1.0],
    torch.float64,
    True,
)

model = ShortRangeModel()
model.set([1, 6, 7, 8], symfunc, neuralnet)
model.write(args.out)

print(f'Model saved → {args.out}')
