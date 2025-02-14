==================================
Machine Learning Potential package
==================================

The package is used to create and test the machine learning potential (MLP), especially 
2nd and 4th generation neural network potential proposed by Behler.

The package includes the base of the code, with useful scripting files in script folder

Requirements
============

The package requires the following packages:

* ``numpy`` for array manipulation
* ``pandas`` for result storage in script
* ``torch`` for machine learning task
* ``h5py`` for large data manipulation
* ``scikit-learn`` for machine learning solving
* ``tqdm`` for display progress

Installation
============

One can install the package using ::

    pip install .

The script in the folder can be used separated with the package available.

Model supported
===============

The package supports the short-range neural network potential (NNP) with the 
addition of dispersion correction MLXDM and Charge-Equilibration scheme. All 
models are included in combine files, and can be imported as ::

    from mlpotential.combine import *

The list of supported models is:

* ``ShortRangeModel`` traditional short-range NNP
* ``ShortRangeEnsembleModel`` traditional short-range NNP with Ensemble
* ``ChargeModel`` Charge-equilibration scheme with NNP
* ``ChargeEnsembleModel`` Charge-equilibration scheme with Ensemble of NNP
* ``DispersionModel`` NNP with MLXDM
* ``DispersionEnsembleModel`` Ensemble of NNP with MLXDM
* ``ChargeDispersionModel`` Charge-equilibration scheme with NNP and MLXDM
* ``ChargeDispersionEnsembleModel`` Charge-equilibration scheme with Ensemble NNP and MLXDM

The workflow
============

For all model training, first prepare the data structure using ``dataloader`` module. See ``data_generation.py`` script 
for example.

1. Short-range model

To train the short-range model, initialize the model by ``model/init.py`` script, train it with ``train/train_energy.py`` 
or ``train/train_force.py``. For ensemble training, train a number of duplicate model (regenerate each for different initial 
parameters). Using ``model/merge.py`` to combine.

2. Charge model

To train the charge model, first initialize the charge model by ``model/init.py`` script. Train the charge model with 
``train/train_charge.py`` script. After that, modifiy the short-range model's mean and generate new data file with 
``charge_modification.py`` script. After that, train the short-range model with ``train/train_energy.py`` or 
``train/train_force.py``. Finally, combine the short-range part with the charge using ``model/merge.py`` script.

To utilize ensemble training, train duplicated model in the short-range training, similar to short-range model.

3. Dispersion model

For disperson model, create 4 copies of dipsersion model using ``model/init.py`` script. Train all the models with 
``train_xdm.py`` script, one each for m1, m2, m3, and v. After that, train the short-range model similar to section 1. 
then combine the model with ``model/merge.py``

To utilize ensemble training, train duplicated model in the short-range training, similar to short-range model.

Citations
=========

When using the package or script, please cite:

1. Tu, N. T. P.; Rezajooei, N.; Johnson, E. R.; Rowley, C. N. A Neural Network Potential with 
Rigorous Treatment of Long-Range Dispersion. Digital Discovery 2023, 2 (3), 718–727. 
https://doi.org/10.1039/D2DD00150K.

2. Tu, N. T. P.; Williamson, S.; Johnson, E. R.; Rowley, C. N. Modeling Intermolecular 
Interactions with Exchange-Hole Dipole Moment Dispersion Corrections to Neural Network Potentials. 
J. Phys. Chem. B 2024, 128 (35), 8290–8302. https://doi.org/10.1021/acs.jpcb.4c02882.

