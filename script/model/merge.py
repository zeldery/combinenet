import torch
from mlpotential.combine import *
from mlpotential.net import NetworkEnsemble

def short_ensemble(checkfile_list, output):
    nn_list = []
    for name in checkfile_list:
        chk = torch.load(name, map_location='cpu', weights_only=True)
        model = ShortRangeModel()
        model.load(chk['best_model'])
        nn_list.append(model.neural_network)
    ensemble = NetworkEnsemble()
    ensemble.set(nn_list)
    new_model = ShortRangeEnsembleModel()
    new_model.set(model.element_list, model.symmetry_function, ensemble)
    new_model.write(output)

def charge(charge_checkfile, short_checkfile, output):
    chk = torch.load(charge_checkfile, map_location='cpu', weights_only=True)
    e_model = ChargeModel()
    e_model.load(chk['best_model'])
    chk = torch.load(short_checkfile, map_location='cpu', weights_only=True)
    short = ShortRangeModel()
    short.load(chk['best_model'])
    e_model.short_network = short.neural_network
    e_model.write(output)

def charge_ensemble(charge_checkfile, short_checkfile_list, output):
    chk = torch.load(charge_checkfile, map_location='cpu', weights_only=True)
    e_model = ChargeModel()
    e_model.load(chk['best_model'])
    nn_list = []
    for name in short_checkfile_list:
        chk = torch.load(name, map_location='cpu', weights_only=True)
        model = ShortRangeModel()
        model.load(chk['best_model'])
        nn_list.append(model.neural_network)
    ensemble = NetworkEnsemble()
    ensemble.set(nn_list)
    new_model = ChargeEnsembleModel()
    new_model.set(e_model.element_list, e_model.symmetry_function, e_model.charge_model, ensemble)
    new_model.write(output)

def dispersion(m1_checkfile, m2_checkfile, m3_checkfile, v_checkfile, short_checkfile, output):
    chk = torch.load(m1_checkfile, map_location='cpu', weights_only=True)
    m1_model = DispersionModel()
    m1_model.load(chk['best_model'])
    chk = torch.load(m2_checkfile, map_location='cpu', weights_only=True)
    m2_model = DispersionModel()
    m2_model.load(chk['best_model'])
    chk = torch.load(m3_checkfile, map_location='cpu', weights_only=True)
    m3_model = DispersionModel()
    m3_model.load(chk['best_model'])
    chk = torch.load(v_checkfile, map_location='cpu', weights_only=True)
    v_model = DispersionModel()
    v_model.load(chk['best_model'])
    chk = torch.load(short_checkfile, map_location='cpu', weights_only=True)
    model = ShortRangeModel()
    model.load(chk['best_model'])
    # Use v_model for combined model
    v_model.dispersion_model.m1_net = m1_model.dispersion_model.m1_net
    v_model.dispersion_model.m2_net = m2_model.dispersion_model.m2_net
    v_model.dispersion_model.m3_net = m3_model.dispersion_model.m3_net
    v_model.short_network = model.neural_network
    v_model.write(output)

def dispersion_ensemble(m1_checkfile, m2_checkfile, m3_checkfile, v_checkfile, short_checkfile_list, output):
    chk = torch.load(m1_checkfile, map_location='cpu', weights_only=True)
    m1_model = DispersionModel()
    m1_model.load(chk['best_model'])
    chk = torch.load(m2_checkfile, map_location='cpu', weights_only=True)
    m2_model = DispersionModel()
    m2_model.load(chk['best_model'])
    chk = torch.load(m3_checkfile, map_location='cpu', weights_only=True)
    m3_model = DispersionModel()
    m3_model.load(chk['best_model'])
    chk = torch.load(v_checkfile, map_location='cpu', weights_only=True)
    v_model = DispersionModel()
    v_model.load(chk['best_model'])
    nn_list = []
    for name in short_checkfile_list:
        chk = torch.load(name, map_location='cpu', weights_only=True)
        model = ShortRangeModel()
        model.load(chk['best_model'])
        nn_list.append(model.neural_network)
    ensemble = NetworkEnsemble()
    ensemble.set(nn_list)
    # Use v_model for combined model
    v_model.dispersion_model.m1_net = m1_model.dispersion_model.m1_net
    v_model.dispersion_model.m2_net = m2_model.dispersion_model.m2_net
    v_model.dispersion_model.m3_net = m3_model.dispersion_model.m3_net
    new_model = DispersionEnsembleModel()
    new_model.set(v_model.element_list, v_model.symmetry_function, v_model.dispersion_model, ensemble)
    new_model.write(output)

def charge_disperesion(charge_checkfile, m1_checkfile, m2_checkfile, m3_checkfile, v_checkfile, short_checkfile, output):
    chk = torch.load(charge_checkfile, map_location='cpu', weights_only=True)
    e_model = ChargeModel()
    e_model.load(chk['best_model'])
    chk = torch.load(m1_checkfile, map_location='cpu', weights_only=True)
    m1_model = DispersionModel()
    m1_model.load(chk['best_model'])
    chk = torch.load(m2_checkfile, map_location='cpu', weights_only=True)
    m2_model = DispersionModel()
    m2_model.load(chk['best_model'])
    chk = torch.load(m3_checkfile, map_location='cpu', weights_only=True)
    m3_model = DispersionModel()
    m3_model.load(chk['best_model'])
    chk = torch.load(v_checkfile, map_location='cpu', weights_only=True)
    v_model = DispersionModel()
    v_model.load(chk['best_model'])
    v_model.dispersion_model.m1_net = m1_model.dispersion_model.m1_net
    v_model.dispersion_model.m2_net = m2_model.dispersion_model.m2_net
    v_model.dispersion_model.m3_net = m3_model.dispersion_model.m3_net
    chk = torch.load(short_checkfile, map_location='cpu', weights_only=True)
    short = ShortRangeModel()
    short.load(chk['best_model'])
    new_model = ChargeDispersionModel()
    new_model.set(e_model.element_list, e_model.symmetry_function, e_model.charge_model, v_model.dispersion_model, short.neural_network)
    new_model.write(output)

def charge_dispersion_ensemble(charge_checkfile, m1_checkfile, m2_checkfile, m3_checkfile, v_checkfile, short_checkfile_list, output):
    chk = torch.load(charge_checkfile, map_location='cpu', weights_only=True)
    e_model = ChargeModel()
    e_model.load(chk['best_model'])
    chk = torch.load(m1_checkfile, map_location='cpu', weights_only=True)
    m1_model = DispersionModel()
    m1_model.load(chk['best_model'])
    chk = torch.load(m2_checkfile, map_location='cpu', weights_only=True)
    m2_model = DispersionModel()
    m2_model.load(chk['best_model'])
    chk = torch.load(m3_checkfile, map_location='cpu', weights_only=True)
    m3_model = DispersionModel()
    m3_model.load(chk['best_model'])
    chk = torch.load(v_checkfile, map_location='cpu', weights_only=True)
    v_model = DispersionModel()
    v_model.load(chk['best_model'])
    v_model.dispersion_model.m1_net = m1_model.dispersion_model.m1_net
    v_model.dispersion_model.m2_net = m2_model.dispersion_model.m2_net
    v_model.dispersion_model.m3_net = m3_model.dispersion_model.m3_net
    nn_list = []
    for name in short_checkfile_list:
        chk = torch.load(name, map_location='cpu', weights_only=True)
        model = ShortRangeModel()
        model.load(chk['best_model'])
        nn_list.append(model.neural_network)
    ensemble = NetworkEnsemble()
    ensemble.set(nn_list)
    new_model = ChargeDispersionModel()
    new_model.set(e_model.element_list, e_model.symmetry_function, e_model.charge_model, v_model.dispersion_model, ensemble)
    new_model.write(output)

def main():
    pass

if __name__ == '__main__':
    main()
