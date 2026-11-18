import os
import py_compile

import numpy as np
import pyro
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import pytest
import torch

import AutoRegress
import c302
import DMM
from pyro_svi_c302 import evaluate_predictions
from utils import AtlasLoader, RandiTrialDataset, preprocess_gcamp_recording, responseGenerator
import vae


REAL_DATA_PATH = "/Users/yanlu/Google Drive/マイドライブ/exported_data"


def test_conductance_layer_uses_input_and_leak():
    pyro.clear_param_store()
    layer = c302.conductanceLayer(4)
    v_prev = torch.full((2, 4), -0.065)
    zero_in = torch.zeros(2, 4)
    pos_in = torch.full((2, 4), 0.2)

    # Fix stochastic parameters for deterministic check
    with pyro.poutine.condition(
        data={
            "E": torch.full((4,), -0.065),
            "G": torch.full((4,), 0.5),
            "C": torch.full((4,), 0.05),
        }
    ):
        v_zero = layer(v_prev, zero_in, dt=0.05)
    with pyro.poutine.condition(
        data={
            "E": torch.full((4,), -0.065),
            "G": torch.full((4,), 0.5),
            "C": torch.full((4,), 0.05),
        }
    ):
        v_pos = layer(v_prev, pos_in, dt=0.05)

    assert torch.allclose(v_zero, v_prev, atol=1e-6)
    assert torch.all(v_pos > v_prev)


def test_synapse_layers_accumulate_instead_of_overwriting():
    pyro.clear_param_store()
    # Two presynaptic neurons (0, 1) both targeting neuron 2
    gap = c302.GeneralSynapse([0, 1], [2, 2], [1.0, 2.0])
    wicks = c302.WicksSynapse([0, 1], [2, 2], [1.0, 2.0])
    v = torch.tensor([[-0.030, -0.020, -0.065]])

    with pyro.poutine.condition(data={"g_syn": torch.tensor([0.5, 0.5])}):
        gap_net = gap(v)
    expected_gap = 0.5 * 1.0 * (-0.030 - (-0.065)) + 0.5 * 2.0 * (-0.020 - (-0.065))
    assert torch.isclose(gap_net[0, 2], torch.tensor(expected_gap), atol=1e-6)

    with pyro.poutine.condition(
        data={
            "g_max": torch.tensor([1.0, 1.0]),
            "V_rest": torch.tensor([-0.060, -0.060]),
            "V_slope": torch.tensor([0.015, 0.015]),
            "E_rev": torch.tensor([0.0, 0.0]),
        }
    ):
        w_net = wicks(v)
    # Both synapses depolarize neuron 2; contribution must strictly exceed single synapse
    act0 = torch.sigmoid(torch.tensor((-0.030 + 0.060) / 0.015))
    act1 = torch.sigmoid(torch.tensor((-0.020 + 0.060) / 0.015))
    expected_w = (1.0 * act0 + 2.0 * act1) * (0.0 - (-0.065))
    assert torch.isclose(w_net[0, 2], expected_w, atol=1e-5)


def test_voltage_to_calcium_to_gcamp_conversion_and_gradients():
    pyro.clear_param_store()
    gcamp = c302.CalciumGCaMPLayer(3)
    v_rest = torch.full((2, 3), -0.065)
    v_seq_rest = v_rest.unsqueeze(1).repeat(1, 5, 1)
    v_step = torch.tensor([[-0.010, -0.065, -0.085], [-0.010, -0.065, -0.085]])
    v_seq_depol = v_step.unsqueeze(1).repeat(1, 20, 1).detach().requires_grad_(True)

    with pyro.poutine.condition(data={"offset": torch.zeros(3), "gain": torch.ones(3)}):
        y_rest = gcamp(v_seq_rest, v_rest=v_rest, dt=0.05)
    assert torch.allclose(y_rest, torch.zeros_like(y_rest), atol=1e-6)

    with pyro.poutine.condition(data={"offset": torch.zeros(3), "gain": torch.ones(3)}):
        y_seq = gcamp(v_seq_depol, v_rest=v_rest, dt=0.05)
    y_after = y_seq[:, -1, :]

    # Depolarized neuron > resting > hyperpolarized neuron
    assert torch.all(y_after[:, 0] > 0.05)
    assert torch.allclose(y_after[:, 1], torch.zeros(2), atol=1e-5)
    assert torch.all(y_after[:, 2] < 0.0)
    y_after.sum().backward()
    assert v_seq_depol.grad is not None and torch.isfinite(v_seq_depol.grad).all()


def test_read_connectome_and_sensory_batch_mode():
    pyro.clear_param_store()
    step_model = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_sensory_models=True,
        use_extrasynaptic=True,
        use_bnn_residual=True,
    )
    assert int(step_model.sensory_mask.sum().item()) == 86
    assert len(step_model.Neuron._neuron_List) == 86
    assert len(step_model.synapse.Generic_SRC) > 0

    rec_model = c302.RecurrentNematode(step_model, dt_obs=0.5, substeps=2)
    stim = torch.zeros(2, 4, 300)
    stim[:, 1, 10] = 1.0
    out = rec_model(ExternalInput=stim)
    assert out.shape == (2, 4, 300)
    assert torch.isfinite(out).all()


def test_preprocessing_despiking_baseline_and_negative_f0_handling():
    rng = np.random.default_rng(0)
    n_t = 200
    t = np.arange(n_t) * 0.5
    # Col 0: positive baseline with bleaching drift, calcium transient, bidirectional spikes, and NaN
    drift = 100.0 * np.exp(-t / 200.0) + 50.0
    transient = 45.0 * np.exp(-np.maximum(0.0, t - 40.0) / 8.0) * (t >= 40.0)
    col0 = drift + transient + rng.normal(0.0, 1.0, size=n_t)
    col0[10] = 900.0
    col0[25] = -400.0
    col0[15] = np.nan

    # Col 1: negative camera-offset baseline (like rec 50 col 105)
    col1 = -12.0 + rng.normal(0.0, 0.5, size=n_t)

    raw = np.stack([col0, col1], axis=1)
    norm, valid, noise = preprocess_gcamp_recording(raw, dt=0.5, norm_mode="dff_safe", drop_invalid_f0=True)

    assert not valid[15, 0]  # NaN preserved in obs_mask
    assert valid[10, 0] and abs(norm[10, 0]) < 1.0  # Positive spike removed
    assert valid[25, 0] and abs(norm[25, 0]) < 1.0  # Negative dropout removed
    assert norm[82, 0] > 0.15  # Calcium peak preserved in dF/F0
    assert not np.any(valid[:, 1])  # Negative F0 trace safely masked out in dff_safe

    # robust_z retains negative-offset trace without sign inversion
    norm_z, valid_z, _ = preprocess_gcamp_recording(raw, dt=0.5, norm_mode="robust_z")
    assert np.sum(valid_z[:, 1]) == n_t
    assert np.isfinite(norm_z).all()


def test_response_generator_maps_stimulus_correctly():
    rg = responseGenerator(folder="./wormfunconn/atlas/", strain="unc-31", size=40, stim_duration=1.0)
    x, y, m = rg.Dataset(return_mask=True)
    assert x.shape == y.shape == m.shape
    assert (x.sum(dim=(1, 2)) > 0).all()


def test_vae_and_dmm_and_autoregress_smoke():
    pyro.clear_param_store()
    step_model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    rec_model = c302.RecurrentNematode(step_model, dt_obs=0.5, substeps=2)
    vae_model = vae.VAElegans(
        rec_model,
        rec_model.model.Neuron.neuronSize,
        len(rec_model.model.synapse.Wicks_SRC),
        len(rec_model.model.synapse.Gap_Junction_SRC),
    )
    stim = torch.zeros(2, 6, 300)
    stim[:, 1, 0] = 1.0
    y = torch.zeros(2, 6, 300)
    mask = torch.zeros(2, 6, 300, dtype=torch.bool)
    mask[:, :, :10] = True
    noise = torch.full((2, 300), 0.05)

    svi_vae = SVI(vae_model, vae_model.guide, pyro.optim.ClippedAdam({"lr": 1e-3}), Trace_ELBO())
    loss_vae = svi_vae.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss_vae)
    rec = vae_model.reconstruct_img(y, ExternalInput=stim)
    assert rec.shape == (2, 6, 300)

    pyro.clear_param_store()
    dmm = DMM.DMM(input_dim=300, z_dim=16, emission_dim=32, transition_dim=32, rnn_dim=32, use_stim=True)
    lens = torch.tensor([6, 6], dtype=torch.long)
    rev = DMM.reverse_sequences_pad(torch.cat([y, stim], dim=-1), lens)
    svi_dmm = SVI(dmm.model, dmm.guide, pyro.optim.ClippedAdam({"lr": 1e-3}), Trace_ELBO())
    loss_dmm = svi_dmm.step(y, rev, mask, lens, 1.0, stim)
    assert np.isfinite(loss_dmm)

    ar = AutoRegress.AutoRegression(4, 2, [0, 1], [2, 3], [1.0, 1.0])
    ar_out = ar(torch.zeros(2, 5, 4))
    assert ar_out.shape == (2, 5, 4)


def test_tiny_mcmc_nuts_on_minimal_connectome():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([1], [2], [1.0], [0.0], [0.02]),
        "Generic": ([], [], []),
    }
    step_model = c302.NematodeForStep(3, {}, tiny_syn)
    rec_model = c302.RecurrentNematode(step_model, dt_obs=0.5, substeps=2)
    stim = torch.zeros(1, 5, 3)
    stim[:, 1, 0] = 1.0
    y = torch.zeros(1, 5, 3)
    mask = torch.ones(1, 5, 3, dtype=torch.bool)
    nuts = NUTS(rec_model, jit_compile=False, max_tree_depth=2)
    mcmc = MCMC(nuts, num_samples=2, warmup_steps=2, disable_progbar=True)
    mcmc.run(ExternalInput=stim, mask=mask, y=y)
    samples = mcmc.get_samples()
    assert len(samples) > 0


@pytest.mark.skipif(not os.path.isdir(REAL_DATA_PATH), reason="Randi exported_data not present")
def test_real_exported_data_svi_end_to_end():
    pyro.clear_param_store()
    loader = AtlasLoader(REAL_DATA_PATH, max_recordings=2, norm_mode="dff_safe")
    ds = loader.build_trial_dataset(pre_vols=5, post_vols=15)
    assert len(ds) > 0

    step_model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    rec_model = c302.RecurrentNematode(step_model, dt_obs=0.5, substeps=2)
    guide = AutoDiagonalNormal(rec_model, init_scale=0.02)
    svi = SVI(rec_model, guide, pyro.optim.ClippedAdam({"lr": 5e-3, "clip_norm": 10.0}), Trace_ELBO())

    stim, y, mask, noise = ds.stim_inputs[:4], ds.y[:4], ds.obs_mask[:4], ds.noise_scale[:4]
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)

    loader_eval = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [0, 1, 2, 3]), batch_size=2)
    metrics = evaluate_predictions(rec_model, loader_eval, torch.device("cpu"), guide=guide, num_samples=2)
    assert np.isfinite(metrics["rmse"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS accelerator not available")
def test_accelerator_device_execution_mps():
    pyro.clear_param_store()
    dev = torch.device("mps")
    step_model = c302.readConnectome("./data/CElegansNeuronTables.xls", use_bnn_residual=True)
    rec_model = c302.RecurrentNematode(step_model, dt_obs=0.5, substeps=2).to(dev)
    stim = torch.zeros(2, 5, 300, device=dev)
    stim[:, 1, 10] = 1.0
    y = torch.zeros(2, 5, 300, device=dev)
    mask = torch.ones(2, 5, 300, dtype=torch.bool, device=dev)
    noise = torch.full((2, 300), 0.05, device=dev)

    guide = AutoDiagonalNormal(rec_model, init_scale=0.02)
    svi = SVI(rec_model, guide, pyro.optim.ClippedAdam({"lr": 5e-3}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)
    assert guide.loc.device.type == "mps"


def test_randi_trial_dataset_save_load(tmp_path):
    stim = torch.zeros(2, 5, 10)
    stim[0, 1, 3] = 1.0
    y = torch.randn(2, 5, 10)
    mask = torch.ones(2, 5, 10, dtype=torch.bool)
    noise = torch.full((2, 10), 0.05)
    dts = torch.tensor([0.5, 0.5])
    meta = [{"rec_id": 0}, {"rec_id": 1}]
    ds = RandiTrialDataset(stim, y, mask, noise, dts, meta)
    out_path = os.path.join(tmp_path, "cache.npz")
    ds.save(out_path)
    loaded = RandiTrialDataset.load(out_path)
    assert torch.allclose(loaded.stim_inputs, stim)
    assert torch.allclose(loaded.y, y)
    assert torch.equal(loaded.obs_mask, mask)
    assert loaded.meta == meta


def test_all_python_modules_compile():
    for fname in [
        "c302.py",
        "utils.py",
        "pyro_svi_c302.py",
        "vae.py",
        "pyro_svi_vae.py",
        "DMM.py",
        "lightning_c302.py",
        "AutoRegress.py",
    ]:
        py_compile.compile(fname, doraise=True)
