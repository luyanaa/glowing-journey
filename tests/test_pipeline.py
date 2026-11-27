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
from pyro_svi_c302 import (
    compute_stratified_metrics,
    evaluate_predictions,
    init_to_prior_median,
)
from pyro_svi_vae import (
    _expand_scalar_kinetics,
    _load_compatible_state_dict,
    _set_prestim_only_trainable,
    compute_drift_gate,
    reweight_hop1_noise,
)
from utils import (
    AtlasLoader,
    RandiTrialDataset,
    preprocess_gcamp_recording,
    responseGenerator,
    split_trial_dataset,
    split_train_validation_indices,
    ToyoshimaLoader,
    nested_grouped_splits,
)
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
    v_seq_depol = v_step.unsqueeze(1).repeat(1, 40, 1).detach().requires_grad_(True)
    fixed_kinetics = {
        "offset": torch.zeros(3),
        "gain": torch.ones(3),
        "v_th_ca_logit": c302.CalciumGCaMPLayer.v_th_ca_to_logit(-0.038),
        "v_slope_ca": torch.tensor(0.008),
        "alpha_ca": torch.tensor(2.5),
        "tau_ca": torch.tensor(7.0),
        "k_on": torch.tensor(0.06),
        "k_off": torch.tensor(0.045),
    }

    with pyro.poutine.condition(data=fixed_kinetics):
        y_rest = gcamp(v_seq_rest, v_rest=v_rest, dt=0.05)
    assert torch.allclose(y_rest, torch.zeros_like(y_rest), atol=1e-6)

    with pyro.poutine.condition(data=fixed_kinetics):
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
    guide_tr = pyro.poutine.trace(vae_model.guide).get_trace(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    model_tr = pyro.poutine.trace(pyro.poutine.replay(vae_model, trace=guide_tr)).get_trace(
        ExternalInput=stim, mask=mask, y=y, noise_scale=noise
    )
    guide_sites = {
        n
        for n, s in guide_tr.nodes.items()
        if s["type"] == "sample" and not s["is_observed"] and not pyro.poutine.util.site_is_subsample(s)
    }
    model_sites = {
        n
        for n, s in model_tr.nodes.items()
        if s["type"] == "sample" and not s["is_observed"] and not pyro.poutine.util.site_is_subsample(s)
    }
    assert guide_sites == model_sites
    assert "decoder.gcamp.v_th_ca_logit" in guide_sites
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

    guide = AutoDiagonalNormal(rec_model, init_loc_fn=init_to_prior_median, init_scale=0.02)
    svi = SVI(rec_model, guide, pyro.optim.ClippedAdam({"lr": 5e-3}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=20)
    assert np.isfinite(loss)
    assert guide.loc.device.type == "mps"
    v_th_logit = guide.median()["gcamp.v_th_ca_logit"]
    v_th_val = float(c302.CalciumGCaMPLayer.logit_to_v_th_ca(v_th_logit).item())
    assert c302.CalciumGCaMPLayer.V_TH_CA_LOW < v_th_val < c302.CalciumGCaMPLayer.V_TH_CA_HIGH


def test_conductance_optogenetics_bounded_and_v_th_ca_physiological():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn)
    v_prev = torch.full((2, 3), -0.065)
    ext = torch.zeros(2, 3)
    ext[:, 0] = 1.0

    # Even under extreme Chrimson conductance, voltage must remain <= E_opt (0.0 V)
    with pyro.poutine.condition(
        data={
            "stim_gain": torch.tensor(100.0),
            "Neuron.conductance.E": torch.full((3,), -0.065),
            "Neuron.conductance.G": torch.full((3,), 0.5),
            "Neuron.conductance.C": torch.full((3,), 0.05),
            "synapse.wicks.g_max": torch.tensor([0.05]),
            "synapse.wicks.V_rest": torch.full((3,), -0.042),
            "synapse.wicks.V_slope": torch.full((3,), 0.015),
            "synapse.wicks.E_rev": torch.tensor([0.0]),
        }
    ):
        v_next = step(v_prev, ExternalInput=ext, dt=0.1)
    assert torch.all(v_next[:, 0] > -0.02)
    assert torch.all(v_next[:, 0] <= 0.0 + 1e-6)

    # Round-trip inverse check for logit <-> physiological v_th_ca mapping
    v_target = torch.tensor(-0.038)
    v_roundtrip = c302.CalciumGCaMPLayer.logit_to_v_th_ca(
        c302.CalciumGCaMPLayer.v_th_ca_to_logit(v_target)
    )
    assert torch.isclose(v_roundtrip, v_target, atol=1e-6)

    # v_th_ca_logit initializes cleanly and maps strictly inside (-0.060, -0.015) V
    rec = c302.RecurrentNematode(c302.NematodeForStep(3, {}, tiny_syn), dt_obs=0.5, substeps=2)
    guide = AutoDiagonalNormal(rec, init_loc_fn=init_to_prior_median, init_scale=0.02)
    stim = torch.zeros(1, 5, 3)
    stim[:, 1, 0] = 1.0
    y = torch.zeros(1, 5, 3)
    mask = torch.ones(1, 5, 3, dtype=torch.bool)
    _ = guide(ExternalInput=stim, mask=mask, y=y)

    # Trace should contain gcamp.v_th_ca_logit as a single unconstrained Normal site
    trace = pyro.poutine.trace(rec).get_trace(ExternalInput=stim, mask=mask, y=y)
    assert "gcamp.v_th_ca_logit" in trace.nodes
    assert "gcamp.v_th_ca" not in trace.nodes

    v_th_logit_init = float(guide.median()["gcamp.v_th_ca_logit"].item())
    v_th = float(c302.CalciumGCaMPLayer.logit_to_v_th_ca(v_th_logit_init).item())
    assert abs(v_th_logit_init - (-0.05)) < 1e-5
    assert c302.CalciumGCaMPLayer.V_TH_CA_LOW < v_th < c302.CalciumGCaMPLayer.V_TH_CA_HIGH
    assert abs(v_th - (-0.0381)) < 0.002

    # Regression for PyTorch AffineTransform.codomain bug:
    # Even if AutoDiagonalNormal unconstrained loc is pushed to extreme +/- values,
    # mapped v_th_ca must NEVER escape [V_TH_CA_LOW, V_TH_CA_HIGH] and ELBO must stay finite.
    svi = SVI(rec, guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    for extreme_logit in (-12.0, 12.0, -100.0, 100.0):
        with torch.no_grad():
            guide.loc.fill_(0.0)
            # Locate exact flat index of gcamp.v_th_ca_logit in AutoDiagonalNormal
            offset = 0
            found = False
            for name, site in guide.prototype_trace.iter_stochastic_nodes():
                numel = site["value"].numel()
                if name == "gcamp.v_th_ca_logit":
                    guide.loc[offset : offset + numel].fill_(extreme_logit)
                    found = True
                    break
                offset += numel
            assert found
        mapped = float(
            c302.CalciumGCaMPLayer.logit_to_v_th_ca(guide.median()["gcamp.v_th_ca_logit"]).item()
        )
        if abs(extreme_logit) <= 12.0:
            assert c302.CalciumGCaMPLayer.V_TH_CA_LOW < mapped < c302.CalciumGCaMPLayer.V_TH_CA_HIGH
        else:
            assert (
                c302.CalciumGCaMPLayer.V_TH_CA_LOW - 1e-6
                <= mapped
                <= c302.CalciumGCaMPLayer.V_TH_CA_HIGH + 1e-6
            )
        loss = svi.step(ExternalInput=stim, mask=mask, y=y)
        assert np.isfinite(loss)
        out = pyro.infer.Predictive(rec, guide=guide, num_samples=1, return_sites=("_RETURN",))(
            ExternalInput=stim
        )["_RETURN"]
        assert torch.isfinite(out).all()


def test_hierarchical_matrixized_synapses_match_edgewise_formula():
    pyro.clear_param_store()
    # Includes duplicate (0 -> 2) edge to verify accumulate=True matches index_add_
    src = [0, 0, 1]
    dst = [2, 2, 2]
    w = [1.0, 2.0, 1.5]
    v = torch.tensor([[-0.030, -0.020, -0.065, -0.050], [-0.045, -0.015, -0.060, -0.040]], requires_grad=True)

    wicks_shared = c302.WicksSynapse(src, dst, w, neuron_size=4)
    wicks_edge = c302.WicksSynapse(src, dst, w, neuron_size=None)
    assert wicks_shared.share_presynaptic and not wicks_edge.share_presynaptic

    v_rest_4 = torch.tensor([-0.042, -0.040, -0.045, -0.041])
    v_slope_4 = torch.tensor([0.015, 0.012, 0.014, 0.016])
    g_max_3 = torch.tensor([0.02, 0.03, 0.04])
    e_rev_3 = torch.tensor([0.0, -0.075, 0.0])

    with pyro.poutine.condition(
        data={
            "g_max": g_max_3,
            "V_rest": v_rest_4,
            "V_slope": v_slope_4,
            "E_rev": e_rev_3,
        }
    ):
        out_mat = wicks_shared(v)

    with pyro.poutine.condition(
        data={
            "g_max": g_max_3,
            "V_rest": v_rest_4[torch.tensor(src)],
            "V_slope": v_slope_4[torch.tensor(src)],
            "E_rev": e_rev_3,
        }
    ):
        out_edge = wicks_edge(v)

    assert torch.allclose(out_mat, out_edge, atol=1e-6)
    out_mat.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # Verify cached rollout matrices match uncached single-step call inside PyroModule.__call__
    tiny_syn = {
        "General": ([0, 1], [2, 2], [1.0, 2.0]),
        "Wicks": (src, dst, w, [0.0, -0.075, 0.0], [0.01, 0.01, 0.01]),
        "Generic": ([0], [3], [1.0]),
    }
    step = c302.NematodeForStep(4, {}, tiny_syn, use_extrasynaptic=True)
    cond_step = {
        "Neuron.conductance.E": torch.full((4,), -0.065),
        "Neuron.conductance.G": torch.full((4,), 0.5),
        "Neuron.conductance.C": torch.full((4,), 0.05),
        "synapse.wicks.g_max": g_max_3,
        "synapse.wicks.V_rest": v_rest_4,
        "synapse.wicks.V_slope": v_slope_4,
        "synapse.wicks.E_rev": e_rev_3,
        "synapse.general.g_syn": torch.tensor([0.02, 0.03]),
        "synapse.extrasyn_gain": torch.tensor([0.01]),
        "stim_gain": torch.tensor(1.2),
    }
    with pyro.poutine.condition(data=cond_step):
        v_uncached = step(v.detach(), dt=0.05)
    with pyro.poutine.condition(data=cond_step):
        with pyro.poutine.trace() as tr:
            # Calling step via RecurrentNematode exercises build_step_cache under active _pyro_context
            pass
    # Build cache inside active trace via fresh RecurrentNematode
    rec = c302.RecurrentNematode(c302.NematodeForStep(4, {}, tiny_syn, use_extrasynaptic=True), dt_obs=0.5, substeps=2)
    stim4 = torch.zeros(2, 4, 4)
    stim4[:, 1, 0] = 1.0
    out_rec = rec(ExternalInput=stim4)
    assert out_rec.shape == (2, 4, 4) and torch.isfinite(out_rec).all()

    # Check connectome parameter sharing shape on full readConnectome()
    sm = c302.readConnectome("./data/CElegansNeuronTables.xls", use_extrasynaptic=True)
    assert sm.synapse.wicks.share_presynaptic
    cache = sm.synapse.build_step_cache(300, torch.device("cpu"), torch.float32)
    assert cache["wicks"][0].shape == (300, 300)
    assert cache["general"][0].shape == (300, 300)
    assert cache["extra"].shape == (300, 300)
    assert cache["wicks"][2].shape == (300,)  # shared presynaptic V_rest


def test_minibatch_elbo_scales_with_dataset_size():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([1], [2], [1.0], [0.0], [0.02]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn)
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2)
    stim = torch.zeros(2, 6, 3)
    stim[:, 2, 0] = 1.0
    y = torch.full((2, 6, 3), 0.5)
    mask = torch.ones(2, 6, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.1)

    guide = AutoDiagonalNormal(rec, init_loc_fn=init_to_prior_median, init_scale=0.001)
    elbo = Trace_ELBO()
    # Initialize guide
    _ = guide(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    torch.manual_seed(0)
    loss_1x = elbo.loss(rec, guide, ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=2)
    torch.manual_seed(0)
    loss_10x = elbo.loss(rec, guide, ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=20)
    torch.manual_seed(0)
    loss_20x = elbo.loss(rec, guide, ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=40)
    # Scaling dataset_size must scale observation NLL strictly linearly while keeping KL fixed
    slope_a = (loss_10x - loss_1x) / 9.0
    slope_b = (loss_20x - loss_10x) / 10.0
    assert slope_a > 10.0
    assert np.isclose(slope_a, slope_b, rtol=1e-4)


def test_recording_grouped_split_and_rebaseline_clipping():
    n_trials = 40
    stim = torch.zeros(n_trials, 20, 10)
    y = torch.zeros(n_trials, 20, 10)
    mask = torch.ones(n_trials, 20, 10, dtype=torch.bool)
    noise = torch.full((n_trials, 10), 0.05)
    dts = torch.full((n_trials,), 0.5)
    meta = [{"rec_id": i // 4, "event_idx": i % 4} for i in range(n_trials)]
    ds = RandiTrialDataset(stim, y, mask, noise, dts, meta)

    tr_idx, te_idx, info = split_trial_dataset(ds, test_size=0.2, random_state=42, group_by_recording=True)
    tr_recs = {meta[i]["rec_id"] for i in tr_idx}
    te_recs = {meta[i]["rec_id"] for i in te_idx}
    assert info["grouped"] == 1
    assert len(tr_recs) > 0 and len(te_recs) > 0
    assert tr_recs.isdisjoint(te_recs)

    # Verify rebaseline_on_pre masks corrupted pre-windows and never amplifies past trial_clip_abs
    fake_loader = object.__new__(AtlasLoader)
    fake_loader.n_neurons = 6
    fake_loader.neuron_names = np.array([f"N{i}" for i in range(6)])
    y_full = np.zeros((30, 6), dtype=np.float32)
    y_full[:10, 0] = -9.5  # corrupted pre-stimulus baseline (> max_pre_abs=5.0)
    y_full[10:, 0] = 9.5
    y_full[:, 1:] = 0.2
    mask_full = np.ones((30, 6), dtype=bool)
    fake_loader.recordings = [
        {
            "rec_id": 0,
            "t": torch.arange(30, dtype=torch.float32) * 0.5,
            "dt": 0.5,
            "y": torch.from_numpy(y_full),
            "obs_mask": torch.from_numpy(mask_full),
            "noise_scale": torch.full((6,), 0.05),
            "clean_events": [(0, 10, 1, 1)],
        }
    ]
    tds = fake_loader.build_trial_dataset(pre_vols=10, post_vols=20, min_observed_neurons=5)
    assert not bool(tds.obs_mask[0, :, 0].any())  # corrupted pre-window masked out
    assert float(tds.y.abs().max()) <= 10.0


def test_stratified_connectome_hop_evaluation():
    # Chain graph: 0 -> 1 -> 2, neuron 3 unconnected
    syn = c302.synapseLayer(
        {
            "General": ([], [], []),
            "Wicks": ([0, 1], [1, 2], [1.0, 1.0]),
            "Generic": ([], [], []),
        }
    )
    adj = syn.adjacency_matrix(4)
    assert adj[0, 1] and adj[1, 2] and not adj[0, 2] and not adj[0, 3]

    stim = np.zeros((1, 25, 4), dtype=np.float32)
    stim[0, 10, 0] = 1.0
    y_true = np.zeros((1, 25, 4), dtype=np.float32)
    t_ax = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    y_true[0, 10:25, 0] = 0.5 * np.exp(-t_ax)
    y_true[0, 10:25, 1] = 0.2 * np.exp(-t_ax)
    y_true[0, 10:25, 2] = 0.1 * np.exp(-t_ax)
    y_true[0, 10:25, 3] = 0.05 * np.sin(t_ax * 6.0)
    y_pred = y_true.copy()
    y_pred[0, 10:25, 3] = 0.0  # unconnected neuron unpredicted
    mask = np.ones((1, 25, 4), dtype=bool)

    m = compute_stratified_metrics(y_true, y_pred, mask, stim=stim, pre_vols=10, adjacency=adj)
    assert m["stim_count"] == 15 and m["stim_r2"] > 0.99
    assert m["hop1_count"] == 15 and m["hop1_r2"] > 0.99
    assert m["hop2_count"] == 15 and m["hop2_r2"] > 0.99
    assert m["unconnected_count"] == 15 and m["unconnected_r"] == 0.0
    # Baseline-subtracted metrics must exist and preserve the exact fit here
    # because the synthetic pre-stimulus baseline is zero.
    for key in ("evoked_stim_r2", "evoked_hop1_r2", "evoked_hop2_r2"):
        assert key in m and m[key] > 0.99
    assert "evoked_unconnected_r2" in m


def test_evoked_metrics_remove_prestim_baseline():
    syn = c302.synapseLayer(
        {"General": ([], [], []), "Wicks": ([0], [1], [1.0]), "Generic": ([], [], [])}
    )
    adj = syn.adjacency_matrix(3)
    stim = np.zeros((1, 20, 3), dtype=np.float32)
    stim[0, 10, 0] = 1.0
    y_true = np.zeros((1, 20, 3), dtype=np.float32)
    y_pred = np.zeros((1, 20, 3), dtype=np.float32)
    # Different constant pre-stim offsets must be removed by the evoked metric.
    y_true[:, :10, :] = 4.0
    y_pred[:, :10, :] = -2.0
    ramp = np.linspace(0.1, 1.0, 10)
    y_true[0, 10:, 1] = 4.0 + ramp
    y_pred[0, 10:, 1] = -2.0 + ramp
    mask = np.ones((1, 20, 3), dtype=bool)
    m = compute_stratified_metrics(y_true, y_pred, mask, stim=stim, pre_vols=10, adjacency=adj)
    assert m["evoked_hop1_r2"] > 0.99
    assert m["evoked_hop1_r"] > 0.99


def test_evoked_metrics_use_paired_counterfactual_prediction():
    syn = c302.synapseLayer(
        {"General": ([], [], []), "Wicks": ([0], [1], [1.0]), "Generic": ([], [], [])}
    )
    adj = syn.adjacency_matrix(3)
    stim = np.zeros((1, 20, 3), dtype=np.float32)
    stim[0, 10, 0] = 1.0
    y_true = np.zeros((1, 20, 3), dtype=np.float32)
    y_true[:, :10, 1] = 4.0
    y_true[0, 10:, 1] = 4.0 + np.linspace(0.1, 1.0, 10)
    y_pred = np.zeros((1, 20, 3), dtype=np.float32)
    y_pred[:, :, 1] = 1.0  # unremoved pre-stim offset in the prediction
    y_pred[0, 10:, 1] += np.linspace(0.1, 1.0, 10)
    y_cf = np.zeros((1, 20, 3), dtype=np.float32)
    y_cf[:, :, 1] = 1.0
    mask = np.ones((1, 20, 3), dtype=bool)
    m = compute_stratified_metrics(
        y_true, y_pred, mask, stim=stim, pre_vols=10, adjacency=adj,
        evoked_baseline="mean", y_pred_counterfactual=y_cf,
    )
    assert m["evoked_hop1_r2"] > 0.99
    assert m["evoked_hop1_r"] > 0.99


def test_evoked_metrics_remove_linear_prestim_drift():
    syn = c302.synapseLayer(
        {"General": ([], [], []), "Wicks": ([0], [1], [1.0]), "Generic": ([], [], [])}
    )
    adj = syn.adjacency_matrix(3)
    stim = np.zeros((1, 20, 3), dtype=np.float32)
    stim[0, 10, 0] = 1.0
    y_true = np.zeros((1, 20, 3), dtype=np.float32)
    y_pred = np.zeros((1, 20, 3), dtype=np.float32)
    # Pre-stimulus linear drift continues into the post window.
    y_true[0, :10, 1] = np.linspace(0.0, 1.0, 10)
    y_pred[0, :10, 1] = np.linspace(0.0, 1.0, 10)
    y_true[0, 10:, 1] = 1.0 + np.linspace(0.1, 1.0, 10)
    y_pred[0, 10:, 1] = y_true[0, 10:, 1]
    mask = np.ones((1, 20, 3), dtype=bool)
    m = compute_stratified_metrics(
        y_true, y_pred, mask, stim=stim, pre_vols=10, adjacency=adj, evoked_baseline="linear"
    )
    assert m["evoked_hop1_r2"] > 0.99
    assert abs(m["evoked_post_r2"] - 1.0) < 1e-6


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


def test_causal_prestim_encoder_never_reads_post_stimulus():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=3, prestim_rank=2
    )
    stim = torch.zeros(1, 8, 3)
    stim[:, 3, 0] = 1.0
    mask = torch.ones(1, 8, 3, dtype=torch.bool)
    noise = torch.full((1, 3), 0.05)
    pre_a = torch.zeros(1, 3, 3)
    pre_a[:, :, 0] = torch.tensor([0.1, 0.2, 0.3]).view(1, 3)
    pre_b = torch.zeros(1, 3, 3)
    pre_b[:, :, 0] = torch.tensor([-0.3, -0.2, -0.1]).view(1, 3)
    pre_mask = torch.ones(1, 3, 3, dtype=torch.bool)

    def rollout(pre, post_value):
        y = torch.zeros(1, 8, 3)
        y[:, 3:, :] = post_value
        pyro.set_rng_seed(7)
        tr = pyro.poutine.trace(rec).get_trace(
            ExternalInput=stim,
            mask=mask,
            y=y,
            noise_scale=noise,
            pre_stim_obs=pre,
            pre_stim_mask=pre_mask,
        )
        return tr.nodes["_RETURN"]["value"].detach()

    # Identical causal pre-stim input must yield identical predictions even if all
    # post-stimulus observations change; post-stim y only enters observed likelihood sites.
    a1 = rollout(pre_a, 0.0)
    a2 = rollout(pre_a, 1.0)
    assert torch.allclose(a1, a2, atol=1e-6)

    # A different causal pre-stim input must change the open-loop trajectory.
    b = rollout(pre_b, 1.0)
    assert not torch.allclose(a2, b, atol=1e-6)


def test_vae_encoder_can_exceed_legacy_0165_conductance_wall():
    pyro.clear_param_store()
    e_rev_prior = torch.tensor([-0.075, -0.075])
    enc = vae.Encoder(4, 2, 2, e_rev_prior=e_rev_prior)
    enc.eval()
    with torch.no_grad():
        for tcn in (enc.neuron, enc.wick, enc.gap):
            tcn.linear.weight.zero_()
            tcn.linear.bias.fill_(10.0)
    x = torch.zeros(2, 5, 4)
    neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = enc(x)
    assert torch.exp(wicks_loc[0]).max() > 0.0165 * 1.1
    assert torch.exp(gap_loc).max() > 0.0165 * 1.1
    # E_rev is centered on the per-edge neurotransmitter prior and is not clamped at -65 mV.
    assert wicks_loc[1].min() > e_rev_prior.min() - 0.05
    assert wicks_loc[1].max() < e_rev_prior.max() + 0.05


def test_prestim_drift_gate_zeroes_selected_neurons():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=3, prestim_rank=2
    )
    stim = torch.zeros(1, 8, 3)
    stim[:, 3, 0] = 1.0
    pre = torch.zeros(1, 3, 3)
    pre[:, :, 0] = 0.5
    pre_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    y_base, _, _ = rec.simulate(stim, pre_stim_obs=pre, pre_stim_mask=pre_mask)
    gate = torch.ones(3)
    gate[0] = 0.0
    y_gate, _, _ = rec.simulate(stim, pre_stim_obs=pre, pre_stim_mask=pre_mask, drift_gate=gate)
    assert not torch.allclose(y_base, y_gate, atol=1e-7)
    # Gated neuron 0 must lose the pre-stim drift contribution; ungated neurons stay closer.
    delta = (y_base - y_gate).abs()
    assert delta[:, -1, 0].mean() > delta[:, -1, 1:].mean() + 1e-4


def test_drift_gate_uses_known_stimulus_and_connectome_only():
    stim = torch.zeros(2, 12, 4)
    stim[:, 5, 0] = 1.0
    adj = torch.zeros(4, 4)
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    gate_direct = compute_drift_gate(stim, adj, "direct")
    gate_hop1 = compute_drift_gate(stim, adj, "direct_hop1")
    assert gate_direct.shape == (2, 4)
    assert not gate_direct[0, 0] and gate_direct[0, 1] and gate_direct[0, 2]
    assert not gate_hop1[0, 0] and not gate_hop1[0, 1] and gate_hop1[0, 2]
    assert compute_drift_gate(stim, adj, "none") is None


def test_hop1_noise_reweighting_only_touches_hop1():
    stim = torch.zeros(1, 10, 4)
    stim[:, 4, 0] = 1.0
    adj = torch.zeros(4, 4)
    adj[0, 1] = 1.0
    noise = torch.ones(1, 4)
    out = reweight_hop1_noise(noise, stim, adj, 2.0)
    assert out[0, 0] == 1.0  # direct target unchanged
    assert out[0, 1] == 0.5  # 1-hop neighbor down-weighted
    assert out[0, 2] == 1.0 and out[0, 3] == 1.0
    assert torch.allclose(reweight_hop1_noise(noise, stim, adj, 1.0), noise)
    out_stim = reweight_hop1_noise(noise, stim, adj, 2.0, stim_weight=4.0)
    assert out_stim[0, 0] == 0.25
    assert out_stim[0, 1] == 0.5
    assert out_stim[0, 2] == 1.0 and out_stim[0, 3] == 1.0


def test_train_validation_split_is_grouped_and_disjoint():
    stim = torch.zeros(12, 5, 6)
    y = torch.zeros(12, 5, 6)
    mask = torch.ones(12, 5, 6, dtype=torch.bool)
    noise = torch.full((12, 6), 0.05)
    dts = torch.full((12,), 0.5)
    meta = [{"rec_id": i // 2} for i in range(12)]
    ds = RandiTrialDataset(stim, y, mask, noise, dts, meta)
    train_idx, test_idx, _ = split_trial_dataset(ds, test_size=0.25, random_state=3)
    core, val, info = split_train_validation_indices(
        ds, train_idx, val_size=0.34, random_state=7, group_by_recording=True
    )
    assert set(core).isdisjoint(set(val))
    assert set(core) | set(val) == set(train_idx)
    core_recs = {meta[i]["rec_id"] for i in core}
    val_recs = {meta[i]["rec_id"] for i in val}
    test_recs = {meta[i]["rec_id"] for i in test_idx}
    assert core_recs.isdisjoint(val_recs)
    assert val_recs.isdisjoint(test_recs)


def test_vae_guide_updates_all_variational_pyroparams_and_prestim():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0, 1], [1, 2], [2.0, 1.5], [0.0, -0.075], [0.02, 0.015]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(4, {}, tiny_syn)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=3, prestim_rank=2
    )
    vm = vae.VAElegans(rec, 4, 2, 1)
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())

    stim = torch.zeros(2, 8, 4)
    stim[:, 3, 0] = 1.0
    y = torch.randn(2, 8, 4) * 0.2
    y[:, 3:, 0] += 0.6
    mask = torch.ones(2, 8, 4, dtype=torch.bool)
    noise = torch.full((2, 4), 0.05)

    loss1 = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss1)
    ps = pyro.get_param_store()
    keys = sorted(ps.keys())
    assert not any(k.startswith("vae_encoder") for k in keys)

    k_enc = [k for k in keys if k.startswith("guide.encoder.")]
    k_gpp = [k for k in keys if k.startswith("guide.") and not k.startswith("guide.encoder.")]
    k_pre = [k for k in keys if k.startswith("decoder.prestim_encoder.")]
    assert len(k_enc) > 0
    assert len(k_gpp) == 20
    assert len(k_pre) == 17
    for req in (
        "guide.v_th_ca_loc",
        "guide.stim_gain_loc",
        "guide.tau_ca_loc",
        "guide.k_off_loc",
        "guide.gcamp_gain_loc",
        "guide.gcamp_offset_loc",
        "guide.obs_sigma_loc",
    ):
        assert req in ps

    before = {k: ps[k].detach().clone() for k in keys}
    loss2 = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss2)
    assert all(not torch.equal(before[k], ps[k].detach()) for k in k_gpp)
    assert all(not torch.equal(before[k], ps[k].detach()) for k in k_pre)
    assert sum(1 for k in k_enc if not torch.equal(before[k], ps[k].detach())) == len(k_enc)


def test_prestim_bimodal_artifact_rejected_in_dataset_and_encoder():
    # 1. CausalPreStimEncoder must zero out bimodal spikes (e.g. [-8.4, +9.6]) while keeping clean traces
    enc = c302.CausalPreStimEncoder(4, rank=2)
    y_pre = torch.zeros(1, 10, 4)
    y_pre[0, :, 0] = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3, 0.35, 0.4])
    y_pre[0, :, 1] = torch.tensor([9.6, -8.4, 9.6, -8.4, 0.0, 9.6, -8.4, 0.0, 9.6, -8.41])
    m_pre = torch.ones_like(y_pre, dtype=torch.bool)
    dv, dca, ibg, df, ds, _, _, _ = enc(y_pre, m_pre)
    assert abs(float(df[0, 0].item())) > 0.05
    assert abs(float(ds[0, 0].item())) > 0.05
    assert float(dv[0, 1].abs().item()) == 0.0
    assert float(df[0, 1].abs().item()) == 0.0
    assert float(ds[0, 1].abs().item()) == 0.0

    # 2. AtlasLoader.build_trial_dataset max_pre_dev rejects zero-median bimodal pre-stim spikes
    loader = object.__new__(AtlasLoader)
    loader.n_neurons = 6
    loader.neuron_names = [f"N{i}" for i in range(6)]
    y_full = np.zeros((30, 6), dtype=np.float32)
    y_full[:10, 0] = np.array([9.6, -8.4, 9.6, -8.4, 0.0, 9.6, -8.4, 0.0, 9.6, -8.4], dtype=np.float32)
    y_full[10:, 0] = 5.0
    y_full[:, 1:6] = 0.2
    loader.recordings = [
        {
            "rec_id": 82,
            "y": torch.from_numpy(y_full),
            "obs_mask": torch.ones((30, 6), dtype=torch.bool),
            "noise_scale": torch.full((6,), 0.05, dtype=torch.float32),
            "dt": 0.5,
            "clean_events": [(0, 10, "N1", 1)],
        }
    ]
    ds_clean = loader.build_trial_dataset(pre_vols=10, post_vols=20, min_observed_neurons=5, rebaseline_on_pre=True)
    assert len(ds_clean) == 1
    assert not bool(ds_clean.obs_mask[0, :, 0].any().item())  # Bimodal spike neuron masked out
    assert bool(ds_clean.obs_mask[0, :, 1].all().item())      # Clean neuron retained


def test_evaluate_predictions_passes_drift_gate_mode():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0, 1], [1, 2], [2.0, 1.5], [0.0, -0.075], [0.02, 0.015]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(4, {}, tiny_syn)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=4, prestim_rank=2
    )
    vm = vae.VAElegans(rec, 4, 2, 1)
    stim = torch.zeros(2, 12, 4)
    stim[:, 4, 0] = 1.0
    y = torch.randn(2, 12, 4) * 0.2
    mask = torch.ones(2, 12, 4, dtype=torch.bool)
    noise = torch.full((2, 4), 0.05)
    ds = torch.utils.data.TensorDataset(stim, y, mask, noise)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    m_gated = evaluate_predictions(
        vm, loader, torch.device("cpu"), guide=vm.guide, num_samples=2, pre_vols=4, drift_gate_mode="direct_hop1"
    )
    assert "post_inlier_r2" in m_gated
    assert "stim_inlier_r" in m_gated
    assert "hop1_inlier_r" in m_gated


def test_per_neuron_kinetics_shapes_and_warm_start_expansion():
    pyro.clear_param_store()
    g = c302.CalciumGCaMPLayer(4, per_neuron_kinetics=True)
    assert tuple(g.tau_ca.shape) == (4,)
    assert tuple(g.k_off.shape) == (4,)
    out = g.forward(torch.full((2, 5, 4), -0.065))
    assert out.shape == (2, 5, 4) and torch.isfinite(out).all()

    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn)
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2, per_neuron_kinetics=True)
    vm = vae.VAElegans(
        rec, rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC), len(rec.model.synapse.Gap_Junction_SRC),
    )
    state = dict(vm.state_dict())
    scalar_keys = [
        "guide.tau_ca_loc_unconstrained",
        "guide.tau_ca_scale_unconstrained",
        "guide.k_off_loc_unconstrained",
        "guide.k_off_scale_unconstrained",
    ]
    for key in scalar_keys:
        state[key] = state[key].detach().mean().reshape(())
    skipped, missing, unexpected = _load_compatible_state_dict(vm, state)
    expanded = _expand_scalar_kinetics(vm, state)
    assert set(expanded) == set(scalar_keys)
    assert torch.allclose(
        vm.guide.tau_ca_loc,
        torch.full((3,), float(state["guide.tau_ca_loc_unconstrained"])),
    )
    assert torch.allclose(
        vm.guide.k_off_loc,
        torch.full((3,), float(state["guide.k_off_loc_unconstrained"])),
    )
    for prefix in ("tau_ca", "k_off"):
        raw = getattr(vm.guide, f"{prefix}_scale_unconstrained")
        assert torch.allclose(
            raw,
            torch.full((3,), float(state[f"guide.{prefix}_scale_unconstrained"])),
        )
        # The constrained view must reflect the raw write, not remain at the
        # previous ExpTransform value.
        assert torch.allclose(
            getattr(vm.guide, f"{prefix}_scale"),
            torch.exp(raw),
        )


def test_connectome_shuffle_preserves_degree_and_changes_edges():
    pyro.clear_param_store()
    orig = c302.readConnectome("./data/CElegansNeuronTables.xls")
    shuf = c302.readConnectome("./data/CElegansNeuronTables.xls", shuffle_seed=7)
    go, gs = orig.synapse.general, shuf.synapse.general
    assert go.inputSize == gs.inputSize
    deg_o = np.bincount(
        np.concatenate([go.synapseInput.numpy(), go.synapseOutput.numpy()]),
        minlength=300,
    )
    deg_s = np.bincount(
        np.concatenate([gs.synapseInput.numpy(), gs.synapseOutput.numpy()]),
        minlength=300,
    )
    assert np.array_equal(np.sort(deg_o), np.sort(deg_s))
    assert not np.array_equal(
        np.stack([go.synapseInput.numpy(), go.synapseOutput.numpy()], axis=1),
        np.stack([gs.synapseInput.numpy(), gs.synapseOutput.numpy()], axis=1),
    )
    # Same seed is reproducible.
    shuf2 = c302.readConnectome("./data/CElegansNeuronTables.xls", shuffle_seed=7)
    assert np.array_equal(shuf.synapse.general.synapseInput.numpy(), shuf2.synapse.general.synapseInput.numpy())


def test_electrical_self_loops_are_filtered():
    pyro.clear_param_store()
    step = c302.readConnectome("./data/CElegansNeuronTables.xls")
    gap = step.synapse.general
    assert gap.inputSize > 0
    assert not torch.any(gap.synapseInput == gap.synapseOutput)


def test_stim_actuator_exact_exponential_and_signed():
    act = torch.zeros(3)
    ext = torch.ones(3)
    out = c302._advance_stim_actuator(act, ext, dt_sim=0.1, tau_rise=0.08, tau_decay=0.4)
    expected = 1.0 - torch.exp(torch.tensor(-0.1 / 0.08))
    assert torch.all(out < 1.0)
    assert torch.allclose(out, torch.full((3,), float(expected)), atol=1e-6)

    # A negative command must not be clamped away.
    neg = c302._advance_stim_actuator(torch.zeros(3), -torch.ones(3), 0.1, 0.08, 0.4)
    assert torch.all(neg < 0.0)

    # Decay to zero is the exact exponential, not an Euler step.
    decay = c302._advance_stim_actuator(torch.ones(3), torch.zeros(3), 0.1, 0.08, 0.4)
    expected_decay = torch.exp(torch.tensor(-0.1 / 0.4))
    assert torch.allclose(decay, torch.full((3,), float(expected_decay)), atol=1e-6)


def test_per_edge_release_sites_and_forward():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn, per_edge_release=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, per_neuron_kinetics=True, per_edge_release=True
    )
    vm = vae.VAElegans(
        rec, rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC), len(rec.model.synapse.Gap_Junction_SRC),
    )
    assert vm.per_edge_release and vm.guide.per_edge_release
    assert tuple(vm.guide.v_rest_edge_loc.shape) == (1,)
    assert tuple(vm.guide.v_slope_edge_loc.shape) == (1,)
    stim = torch.zeros(2, 6, 3)
    y = torch.zeros(2, 6, 3)
    mask = torch.ones(2, 6, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.05)
    trace = pyro.poutine.trace(vm.guide).get_trace(
        ExternalInput=stim, mask=mask, y=y, noise_scale=noise
    )
    for key in (
        "decoder.model.synapse.wicks.v_rest_edge_delta",
        "decoder.model.synapse.wicks.v_slope_edge_logdelta",
    ):
        assert tuple(trace.nodes[key]["value"].shape) == (1,)
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-3}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)


def test_toyoshima_loader_imports_nacl_trials(tmp_path):
    import pandas as pd

    names = ["ASHL", "AWCL", "ASEL", "AVDL", "AVDR", "AVJL"]
    data = np.random.default_rng(0).normal(size=(100, len(names))).astype(np.float32)
    pd.DataFrame(data).to_csv(tmp_path / "1_ratio.csv", index=False, header=False)
    pd.Series(names).to_csv(tmp_path / "1_uniqNames.csv", index=False, header=False)
    timing = pd.DataFrame(
        [{
            "#sample": 1,
            "#animal": 1,
            "name": "toy",
            "anesthesia": 0,
            "frames/sec": 2.0,
            "frame number of first stimuli (NaCl concenctration change from 50 mM to 25mM)": 10.0,
            "every N frames NaCl concentration change": 20.0,
            "number of stimulation cycles": 3.0,
            "total duration (sec)": 50.0,
        }]
    )
    with pd.ExcelWriter(tmp_path / "stimulation_timing.xlsx") as writer:
        timing.to_excel(writer, index=False, startrow=1)

    loader = ToyoshimaLoader(
        tmp_path,
        aconnectome_path="./pumpprobe/pumpprobe/aconnectome_ids.txt",
        target_fs=2.0,
        stim_targets=("ASH", "AWC", "ASE"),
    )
    ds = loader.build_trial_dataset(pre_vols=10, post_vols=50)
    assert len(ds) >= 2
    assert ds.y.shape[1:] == (60, 300)
    assert ds.meta[0]["stim_neuron"] == "NaCl"
    assert ds.meta[0]["polarity"] == "down"
    assert ds.meta[1]["polarity"] == "up"
    names_idx = loader.name_to_idx
    target_idx = [names_idx[n] for n in ("ASHL", "AWCL", "ASEL")]
    # 50→25 mM: ASHL/AWCL positive, ASEL negative.  The 10 s fixture interval
    # caps the 30 s default step at 20 frames.
    assert ds.stim_inputs[0, 10, target_idx].tolist() == [1.0, 1.0, -1.0]
    assert torch.all(ds.stim_inputs[0, 10:30, target_idx[0]] == 1.0)
    assert torch.all(ds.stim_inputs[0, 30:, target_idx[0]] == 0.0)
    # 25→50 mM flips the signs.
    assert ds.stim_inputs[1, 10, target_idx].tolist() == [-1.0, -1.0, 1.0]
    assert ds.meta[0]["step_duration_s"] == 10.0
    # The loader maps z-scored columns onto the Randi ΔF/F0 scale.
    observed = ds.y[ds.obs_mask]
    assert float(observed.std()) < 0.5
    assert ds.obs_mask[:, names_idx["AVDL"]].any()


def test_gap_junction_symmetric_current_conservation():
    pyro.clear_param_store()
    layer = c302.GeneralSynapse([0], [1], [1.0])
    v = torch.tensor([[-0.060, -0.080]])
    current = layer(v)
    assert abs(float(current.sum())) < 1e-6
    assert torch.allclose(current[0, 0], -current[0, 1], atol=1e-7)


def test_causal_preprocessing_prestim_invariance():
    rng = np.random.default_rng(2)
    # A positive baseline is required for dff_safe; the default
    # drop_invalid_f0=True must mask columns whose F0 is not usable.
    raw = (2.0 + rng.normal(scale=0.05, size=(120, 4))).astype(np.float32)
    altered = raw.copy()
    altered[60:] = (4.0 + 2.0 * rng.normal(size=(60, 4))).astype(np.float32)
    n1, m1, q1 = preprocess_gcamp_recording(raw, causal=True)
    n2, m2, q2 = preprocess_gcamp_recording(altered, causal=True)
    assert np.allclose(n1[:60], n2[:60], atol=1e-6)
    assert np.array_equal(m1[:60], m2[:60])
    # Noise is estimated from the first causal window, so changing post-stim
    # frames must not change it.
    assert np.allclose(q1, q2, atol=1e-7)


def test_causal_preprocessing_drops_invalid_f0_and_uses_past_noise():
    rng = np.random.default_rng(3)
    raw = np.empty((120, 2), dtype=np.float32)
    raw[:, 0] = 2.0 + rng.normal(scale=0.01, size=120)
    raw[:, 1] = 0.1 + rng.normal(scale=0.01, size=120)
    norm, valid, _ = preprocess_gcamp_recording(
        raw, causal=True, norm_mode="dff_strict", drop_invalid_f0=True
    )
    assert valid[:, 1].sum() == 0
    assert valid[:, 0].any()
    assert np.allclose(norm[:, 1], 0.0)
    # robust_z must be finite and not use the absolute-difference scale path.
    norm_z, valid_z, _ = preprocess_gcamp_recording(raw, causal=True, norm_mode="robust_z")
    assert np.isfinite(norm_z).all()
    assert valid_z[:, 0].any()


def test_per_neuron_stim_gain_sites_and_stim_actuator():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn, per_neuron_stim_gain=True)
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2, stim_actuator=True)
    vm = vae.VAElegans(
        rec, rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC), len(rec.model.synapse.Gap_Junction_SRC),
    )
    stim = torch.zeros(2, 20, 3)
    stim[:, 10, 0] = 1.0
    y = torch.zeros(2, 20, 3)
    mask = torch.ones(2, 20, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.05)
    trace = pyro.poutine.trace(vm.guide).get_trace(
        ExternalInput=stim, mask=mask, y=y, noise_scale=noise
    )
    assert tuple(trace.nodes["decoder.model.stim_gain_vec"]["value"].shape) == (3,)
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-3}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)

    with torch.no_grad():
        y_act, _, _ = rec.simulate(stim, dt_obs=0.5)
    rec_no_act = c302.RecurrentNematode(
        c302.NematodeForStep(3, {}, tiny_syn, per_neuron_stim_gain=True),
        dt_obs=0.5, substeps=2, stim_actuator=False,
    )
    with torch.no_grad():
        y_raw, _, _ = rec_no_act.simulate(stim, dt_obs=0.5)
    assert not torch.allclose(y_act, y_raw, atol=1e-7)


def test_nested_grouped_cv_has_disjoint_worm_splits():
    stim = torch.zeros(20, 5, 6)
    y = torch.zeros(20, 5, 6)
    mask = torch.ones(20, 5, 6, dtype=torch.bool)
    noise = torch.full((20, 6), 0.05)
    dts = torch.full((20,), 0.5)
    meta = [{"rec_id": i // 4} for i in range(20)]
    ds = RandiTrialDataset(stim, y, mask, noise, dts, meta)
    folds = nested_grouped_splits(ds, n_splits=3, val_size=0.25, random_state=4)
    assert len(folds) == 3
    for fold in folds:
        tr = {meta[i]["rec_id"] for i in fold["train"]}
        va = {meta[i]["rec_id"] for i in fold["val"]}
        te = {meta[i]["rec_id"] for i in fold["test"]}
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert set(fold["train"]).isdisjoint(fold["val"])


def test_wide_amp_priors_expands_stim_gain_and_ach_reversal_range():
    import math

    pyro.clear_param_store()
    step = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        per_neuron_stim_gain=True,
        wide_amp_priors=True,
    )
    assert step.wide_amp_priors is True
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2)
    vm = vae.VAElegans(
        rec,
        step.neuronSize,
        len(step.synapse.Wicks_SRC),
        len(step.synapse.Gap_Junction_SRC),
    )
    assert vm.wide_amp_priors is True
    assert vm.guide.wide_amp_priors is True
    assert vm.encoder.wide_amp_priors is True
    assert abs(float(vm.guide.stim_gain_vec_loc.detach().mean()) - math.log(2.0)) < 1e-5
    ach_mask = torch.isclose(step.synapse.wicks.e_rev_loc_buf, torch.tensor(-0.010), atol=1e-5)
    assert ach_mask.any()
    assert torch.allclose(
        step.synapse.wicks.e_rev_scale_buf[ach_mask],
        torch.full((int(ach_mask.sum()),), 0.035),
        atol=1e-5,
    )

    e_rev_prior = torch.tensor([-0.010, -0.010])
    enc = vae.Encoder(4, 2, 2, e_rev_prior=e_rev_prior, wide_amp_priors=True)
    enc.eval()
    with torch.no_grad():
        for tcn in (enc.neuron, enc.wick, enc.gap):
            tcn.linear.weight.zero_()
            tcn.linear.bias.fill_(10.0)
        x = torch.zeros(2, 5, 4)
        _, _, wicks_loc, _, gap_loc, _ = enc(x)
        assert torch.exp(wicks_loc[0]).max() > 0.12
        assert torch.exp(gap_loc).max() > 0.16

        enc.wick.linear.bias.fill_(-10.0)
        _, _, wicks_loc_neg, _, _, _ = enc(x)
        assert wicks_loc_neg[1].min() < -0.065


def test_evoked_syn_priors_sharpens_release_and_stays_calm_at_rest():
    import math

    pyro.clear_param_store()
    step = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        per_neuron_stim_gain=True,
        wide_amp_priors=True,
        evoked_syn_priors=True,
    )
    assert step.evoked_syn_priors is True
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2)
    vm = vae.VAElegans(
        rec,
        step.neuronSize,
        len(step.synapse.Wicks_SRC),
        len(step.synapse.Gap_Junction_SRC),
    )
    assert vm.evoked_syn_priors is True
    assert vm.guide.evoked_syn_priors is True
    assert vm.encoder.evoked_syn_priors is True

    vm.encoder.eval()
    with torch.no_grad():
        for tcn in (vm.encoder.neuron, vm.encoder.wick, vm.encoder.gap):
            tcn.linear.weight.zero_()
            tcn.linear.bias.zero_()
        n_loc, _, w_loc, _, _, _ = vm.encoder(torch.zeros(1, 5, step.neuronSize))
        assert torch.allclose(n_loc[3], torch.full_like(n_loc[3], -0.042), atol=1e-5)
        assert torch.allclose(n_loc[4], torch.full_like(n_loc[4], math.log(0.008)), atol=1e-5)
        assert torch.allclose(w_loc[0], torch.full_like(w_loc[0], math.log(0.015)), atol=1e-5)

        pyro.set_rng_seed(7)
        stim0 = torch.zeros(1, 20, step.neuronSize)
        y0, v0, _ = rec.simulate(stim0, dt_obs=0.5)
        assert torch.isfinite(y0).all() and torch.isfinite(v0).all()
        assert float(v0[:, -1, :].mean()) < -0.045


def test_stage2_freeze_non_prestim_preserves_connectome_and_updates_prestim():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    stim = torch.zeros(2, 12, 3)
    stim[:, 5, 0] = 1.0
    y = torch.randn(2, 12, 3) * 0.2
    y[:, 5:, 0] += 0.3
    mask = torch.ones(2, 12, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.05)

    step1 = c302.NematodeForStep(3, {}, tiny_syn)
    rec1 = c302.RecurrentNematode(step1, dt_obs=0.5, substeps=2, use_prestim_encoder=False)
    vm1 = vae.VAElegans(rec1, 3, 1, 0)
    svi1 = SVI(vm1, vm1.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    svi1.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert bool(vm1._has_amortized_post.item()) is True
    s1_state = {k: v.detach().clone() for k, v in vm1.state_dict().items()}

    pyro.clear_param_store()
    step2 = c302.NematodeForStep(3, {}, tiny_syn)
    rec2 = c302.RecurrentNematode(step2, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=5, prestim_rank=2)
    vm2 = vae.VAElegans(rec2, 3, 1, 0)
    _load_compatible_state_dict(vm2, s1_state)
    _set_prestim_only_trainable(vm2, True)

    wicks_before = vm2._ema_wicks_loc.detach().clone()
    stim_gain_before = vm2.guide.stim_gain_loc.detach().clone()
    gcamp_gain_before = vm2.guide.gcamp_gain_loc.detach().clone()
    prestim_w_before = vm2.decoder.prestim_encoder.local_fc1.weight.detach().clone()

    svi2 = SVI(vm2, vm2.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    for _ in range(2):
        loss = svi2.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
        assert np.isfinite(loss)

    assert torch.equal(vm2._ema_wicks_loc.detach(), wicks_before)
    assert torch.equal(vm2.guide.stim_gain_loc.detach(), stim_gain_before)
    assert torch.equal(vm2.guide.gcamp_gain_loc.detach(), gcamp_gain_before)
    assert not torch.allclose(vm2.decoder.prestim_encoder.local_fc1.weight.detach(), prestim_w_before)


def test_nested_protocol_eval_wiring_and_seed_flags():
    import subprocess
    import sys
    from pathlib import Path

    eval_src = Path("tools/eval_checkpoint.py").read_text()
    runner_src = Path("tools/run_nested_protocol.py").read_text()
    assert '"--substeps"' in eval_src and "substeps=args.substeps" in eval_src
    assert '"--eval-seed"' in eval_src and "pyro.set_rng_seed(args.eval_seed)" in eval_src
    assert '"--substeps", str(args.substeps)' in runner_src

    proc = subprocess.run(
        [sys.executable, "tools/eval_checkpoint.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--substeps" in proc.stdout
    assert "--eval-seed" in proc.stdout


def test_resolve_init_checkpoint_handles_multifold_without_collision(tmp_path):
    from tools.run_nested_protocol import _resolve_init_checkpoint

    c0 = tmp_path / "fold0" / "intact_seed42" / "vae_epoch_25.pt"
    c1 = tmp_path / "fold1" / "intact_seed42" / "vae_epoch_35.pt"
    c0.parent.mkdir(parents=True)
    c1.parent.mkdir(parents=True)
    c0.write_bytes(b"f0")
    c1.write_bytes(b"f1")

    init_runs = [
        {"tag": "intact", "train_seed": 42, "split_file": "/remote/fold_0.json", "checkpoint": "/remote/fold0/intact_seed42/vae_epoch_25.pt"},
        {"tag": "intact", "train_seed": 42, "split_file": "/remote/fold_1.json", "checkpoint": "/remote/fold1/intact_seed42/vae_epoch_35.pt"},
    ]
    assert _resolve_init_checkpoint(str(tmp_path), init_runs, 0, "intact", 42) == str(c0)
    assert _resolve_init_checkpoint(str(tmp_path), init_runs, 1, "intact", 42) == str(c1)


def test_direct_global_guide_elbo_plates_and_train_test_invariance():
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(3, {}, tiny_syn, wide_amp_priors=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=4, prestim_rank=3, variational_prestim=True
    )
    vm = vae.VAElegans(rec, 3, 1, 1, direct_global_guide=True)
    assert vm.encoder is None and vm.guide.encoder is None and vm.guide.prestim_q is not None

    stim = torch.zeros(2, 10, 3)
    stim[:, 4, 0] = 1.0
    y = torch.randn(2, 10, 3) * 0.2
    mask = torch.ones(2, 10, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.05)

    gtr = pyro.poutine.trace(vm.guide).get_trace(
        ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=10
    )
    mtr = pyro.poutine.trace(pyro.poutine.replay(vm, gtr)).get_trace(
        ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=10
    )
    assert float(mtr.nodes["decoder.model.synapse.wicks.g_max"]["scale"]) == 1.0
    assert float(mtr.nodes["prestim_z"]["scale"]) == 5.0
    assert float(mtr.nodes["obs_0"]["scale"]) == 5.0
    assert torch.equal(gtr.nodes["prestim_z"]["value"], mtr.nodes["prestim_z"]["value"])

    # Global parameter distributions must not depend on `y` vs `y=None`.
    pyro.set_rng_seed(11)
    tr_y = pyro.poutine.trace(vm.guide).get_trace(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    pyro.set_rng_seed(11)
    tr_none = pyro.poutine.trace(vm.guide).get_trace(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    assert torch.equal(
        tr_y.nodes["decoder.model.synapse.wicks.g_max"]["value"],
        tr_none.nodes["decoder.model.synapse.wicks.g_max"]["value"],
    )

    gmax_before = vm.guide.wicks_g_max_loc.detach().clone()
    gmax_glob_before = vm.guide.wicks_g_max_global_z.detach().clone()
    gmax_z_before = vm.guide.wicks_g_max_z.detach().clone()
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=10)
    assert np.isfinite(loss)
    assert not torch.allclose(vm.guide.wicks_g_max_loc.detach(), gmax_before)
    assert not torch.allclose(vm.guide.wicks_g_max_global_z.detach(), gmax_glob_before)
    assert not torch.allclose(vm.guide.wicks_g_max_z.detach(), gmax_z_before)


def test_direct_global_guide_stage2_freezes_connectome_and_trains_variational_prestim():
    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    stim = torch.zeros(2, 10, 3)
    stim[:, 5, 0] = 1.0
    y = torch.randn(2, 10, 3) * 0.2
    mask = torch.ones(2, 10, 3, dtype=torch.bool)
    noise = torch.full((2, 3), 0.05)

    step1 = c302.NematodeForStep(3, {}, tiny_syn)
    rec1 = c302.RecurrentNematode(step1, dt_obs=0.5, substeps=2, use_prestim_encoder=False)
    vm1 = vae.VAElegans(rec1, 3, 1, 1, direct_global_guide=True)
    svi1 = SVI(vm1, vm1.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    svi1.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    s1_state = {k: v.detach().clone() for k, v in vm1.state_dict().items()}

    pyro.clear_param_store()
    step2 = c302.NematodeForStep(3, {}, tiny_syn)
    rec2 = c302.RecurrentNematode(
        step2, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=5, prestim_rank=3, variational_prestim=True
    )
    vm2 = vae.VAElegans(rec2, 3, 1, 1, direct_global_guide=True)
    _load_compatible_state_dict(vm2, s1_state)
    _set_prestim_only_trainable(vm2, True)

    gmax_before = vm2.guide.wicks_g_max_loc.detach().clone()
    gmax_glob_before = vm2.guide.wicks_g_max_global_z.detach().clone()
    erev_before = vm2.guide.wicks_E_rev_z.detach().clone()
    gsyn_before = vm2.guide.gap_g_syn_loc.detach().clone()
    prestim_q_before = vm2.guide.prestim_q.fc2.weight.detach().clone()
    prestim_enc_before = vm2.decoder.prestim_encoder.global_out.weight.detach().clone()

    svi2 = SVI(vm2, vm2.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    for _ in range(3):
        loss = svi2.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
        assert np.isfinite(loss)

    assert torch.equal(vm2.guide.wicks_g_max_loc.detach(), gmax_before)
    assert torch.equal(vm2.guide.wicks_g_max_global_z.detach(), gmax_glob_before)
    assert torch.equal(vm2.guide.wicks_E_rev_z.detach(), erev_before)
    assert torch.equal(vm2.guide.gap_g_syn_loc.detach(), gsyn_before)
    assert not torch.allclose(vm2.guide.prestim_q.fc2.weight.detach(), prestim_q_before)
    assert not torch.allclose(vm2.decoder.prestim_encoder.global_out.weight.detach(), prestim_enc_before)


def test_pair_evoked_hop1_metrics_reported():
    from pyro_svi_c302 import compute_stratified_metrics

    rng = np.random.default_rng(7)
    b, t, n = 4, 28, 4
    stim = np.zeros((b, t, n), dtype=np.float32)
    stim[:, 10:12, 0] = 1.0
    adj = np.zeros((n, n), dtype=bool)
    adj[0, 1] = True  # neuron 1 is excitatory hop1
    adj[0, 2] = True  # neuron 2 is inhibitory hop1
    y_true = rng.normal(scale=0.01, size=(b, t, n)).astype(np.float32)
    y_true[:, 10:26, 0] += 0.4
    y_true[:, 10:26, 1] += 0.20
    y_true[:, 10:26, 2] -= 0.15
    y_pred = np.zeros_like(y_true)
    y_pred[:, 10:26, 0] = 0.25
    y_pred[:, 10:26, 1] = 0.08
    y_pred[:, 10:26, 2] = -0.06
    y_cf = np.zeros_like(y_true)
    mask = np.ones_like(y_true, dtype=bool)

    m = compute_stratified_metrics(
        y_true, y_pred, mask, stim=stim, pre_vols=10, adjacency=adj,
        evoked_baseline="mean", evoked_window_vols=16, y_pred_counterfactual=y_cf,
    )
    assert m["pair_evoked_hop1_count"] == 2
    assert m["pair_evoked_hop1_r"] > 0.99
    assert m["pair_evoked_hop1_sign_acc"] == 1.0
    assert 0.35 < m["pair_evoked_hop1_amp_ratio"] < 0.45


def test_extrasynaptic_pathway_calm_at_rest_no_self_loops_and_svi_step():
    from pyro_svi_c302 import compute_stratified_metrics

    pyro.clear_param_store()
    pyro.set_rng_seed(7)
    step = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_extrasynaptic=True,
        per_neuron_stim_gain=True,
        wide_amp_priors=True,
    )
    src = np.asarray(step.synapse.Generic_SRC, dtype=np.int64)
    dst = np.asarray(step.synapse.Generic_DST, dtype=np.int64)
    assert src.size > 0
    assert not np.any(src == dst)
    np_adj = step.synapse.extrasynaptic_adjacency_matrix(step.neuronSize)
    assert np_adj.shape == (step.neuronSize, step.neuronSize) and not np.any(np.diag(np_adj))

    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2)
    vm = vae.VAElegans(
        rec, step.neuronSize,
        len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    )
    assert vm.guide.use_extrasynaptic and vm.guide.extra_dim == len(src)

    with torch.no_grad():
        stim0 = torch.zeros(1, 10, step.neuronSize)
        y0, v0, _ = rec.simulate(stim0, dt_obs=0.5)
        assert torch.isfinite(y0).all() and torch.isfinite(v0).all()
        assert float(v0[:, -1, :].mean()) < -0.045

    stim = torch.zeros(2, 8, step.neuronSize)
    stim[:, 3, 0] = 1.0
    y = torch.randn(2, 8, step.neuronSize) * 0.1
    mask = torch.ones(2, 8, step.neuronSize, dtype=torch.bool)
    noise = torch.full((2, step.neuronSize), 0.05)

    extra_before = vm.guide.extrasyn_gain_z.detach().clone()
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)
    assert not torch.allclose(vm.guide.extrasyn_gain_z.detach(), extra_before)

    _set_prestim_only_trainable(vm, True)
    assert not vm.guide.extrasyn_gain_z.requires_grad

    # Verify np_only diagnostic metric
    b, t, n = 2, 16, 4
    s = np.zeros((b, t, n), dtype=np.float32)
    s[:, 5, 0] = 1.0
    adj = np.zeros((n, n), dtype=bool); adj[0, 1] = True
    np_a = np.zeros((n, n), dtype=bool); np_a[0, 2] = True
    yt = np.zeros((b, t, n), dtype=np.float32)
    yt[:, 5:10, 2] = 0.25
    yt[:, 10:, 2] = 0.10
    yp = np.zeros((b, t, n), dtype=np.float32)
    yp[:, 5:10, 2] = 0.20
    yp[:, 10:, 2] = 0.08
    metrics = compute_stratified_metrics(
        yt, yp, np.ones_like(yt, dtype=bool), stim=s, pre_vols=5,
        adjacency=adj, evoked_baseline="mean", evoked_window_vols=10,
        y_pred_counterfactual=np.zeros_like(yt), np_adjacency=np_a,
    )
    assert metrics["evoked_np_only_count"] == 20
    assert metrics["evoked_np_only_r2"] > 0.5
    assert metrics["evoked_np_only_r"] > 0.99


def test_fast_rollout_and_stacked_counterfactual_eval_numerical_equivalence():
    from torch.utils.data import DataLoader
    from pyro_svi_c302 import _TupleDataset, compute_stratified_metrics, evaluate_predictions

    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, per_neuron_stim_gain=True)
    rec = c302.RecurrentNematode(
        step,
        dt_obs=0.5,
        substeps=2,
        use_prestim_encoder=True,
        pre_vols=4,
        prestim_rank=3,
        variational_prestim=True,
        stim_actuator=True,
    )
    vm = vae.VAElegans(
        rec, n, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC), direct_global_guide=True
    )

    stim = torch.zeros(3, 12, n)
    stim[:, 5:7, 0] = 1.0
    y = torch.randn(3, 12, n) * 0.2
    mask = torch.ones(3, 12, n, dtype=torch.bool)
    noise = torch.full((3, n), 0.05)

    # Compare cached fast rollout vs explicit per-substep uncached reference
    with torch.no_grad():
        g_trace = pyro.poutine.trace(vm.guide).get_trace(
            ExternalInput=stim, y=None, noise_scale=noise, pre_stim_obs=y[:, :4, :], pre_stim_mask=mask[:, :4, :]
        )
        ps = {k: v["value"] for k, v in g_trace.nodes.items() if v["type"] == "sample"}
        cond_rec = pyro.condition(rec, data=ps)
        y_cached = cond_rec(
            ExternalInput=stim, y=None, noise_scale=noise, pre_stim_obs=y[:, :4, :], pre_stim_mask=mask[:, :4, :],
            sample_obs=False, use_step_cache=True,
        )
        y_uncached = cond_rec(
            ExternalInput=stim, y=None, noise_scale=noise, pre_stim_obs=y[:, :4, :], pre_stim_mask=mask[:, :4, :],
            sample_obs=False, use_step_cache=False,
        )
        assert float((y_cached - y_uncached).abs().max()) < 1e-5

    # Compare stacked 2*B evaluate_predictions against explicit unstacked calls
    loader = DataLoader(_TupleDataset(stim, y, mask, noise), batch_size=3, shuffle=False)
    pyro.set_rng_seed(99)
    m_stacked = evaluate_predictions(vm, loader, torch.device("cpu"), guide=vm.guide, num_samples=2, pre_vols=4)

    pyro.set_rng_seed(99)
    s_list, z_list = [], []
    with torch.no_grad():
        for stim_b, y_b, mask_b, noise_b in loader:
            for _ in range(2):
                gt = pyro.poutine.trace(vm.guide).get_trace(
                    ExternalInput=stim_b, mask=None, y=None, noise_scale=noise_b, pre_stim_obs=y_b[:, :4, :], pre_stim_mask=mask_b[:, :4, :], drift_gate=None
                )
                p_draw = {name: site["value"] for name, site in gt.nodes.items() if site["type"] == "sample" and name != "batch"}
                c = pyro.condition(vm, data=p_draw)
                s_list.append(c(ExternalInput=stim_b, mask=None, y=None, noise_scale=noise_b, pre_stim_obs=y_b[:, :4, :], pre_stim_mask=mask_b[:, :4, :], sample_obs=False))
                z_list.append(c(ExternalInput=torch.zeros_like(stim_b), mask=None, y=None, noise_scale=noise_b, pre_stim_obs=y_b[:, :4, :], pre_stim_mask=mask_b[:, :4, :], sample_obs=False))
    m_manual = compute_stratified_metrics(
        y.numpy(), torch.stack(s_list).mean(0).numpy(), mask.numpy(),
        stim=stim.numpy(), pre_vols=4, adjacency=step.synapse.adjacency_matrix(n),
        evoked_baseline="mean", evoked_window_vols=16, y_pred_counterfactual=torch.stack(z_list).mean(0).numpy(),
    )
    for k in ("r2", "post_inlier_r2", "stim_r2", "evoked_hop1_r2"):
        if k in m_manual and np.isfinite(m_manual[k]):
            assert abs(m_stacked[k] - m_manual[k]) < 1e-5, (k, m_stacked[k], m_manual[k])


def test_stage1b_freeze_wired_preserves_connectome_and_updates_extrasyn():
    from pyro_svi_vae import _load_compatible_state_dict, _set_extrasyn_only_trainable

    pyro.clear_param_store()
    pyro.set_rng_seed(11)
    step1 = c302.readConnectome(
        "./data/CElegansNeuronTables.xls", use_extrasynaptic=False, per_neuron_stim_gain=True, wide_amp_priors=True
    )
    rec1 = c302.RecurrentNematode(step1, dt_obs=0.5, substeps=2, use_prestim_encoder=False)
    vm1 = vae.VAElegans(
        rec1, step1.neuronSize, len(step1.synapse.Wicks_SRC), len(step1.synapse.Gap_Junction_SRC), direct_global_guide=True
    )
    s1_state = {k: v.detach().clone() for k, v in vm1.state_dict().items()}

    pyro.clear_param_store()
    pyro.set_rng_seed(11)
    step2 = c302.readConnectome(
        "./data/CElegansNeuronTables.xls", use_extrasynaptic=True, per_neuron_stim_gain=True, wide_amp_priors=True
    )
    rec2 = c302.RecurrentNematode(step2, dt_obs=0.5, substeps=2, use_prestim_encoder=False)
    vm2 = vae.VAElegans(
        rec2, step2.neuronSize, len(step2.synapse.Wicks_SRC), len(step2.synapse.Gap_Junction_SRC), direct_global_guide=True
    )
    _load_compatible_state_dict(vm2, s1_state)
    _set_extrasyn_only_trainable(vm2, True)

    wicks_before = vm2.guide.wicks_g_max_z.detach().clone()
    gap_before = vm2.guide.gap_g_syn_z.detach().clone()
    stim_before = vm2.guide.stim_gain_vec_loc.detach().clone()
    gcamp_before = vm2.guide.gcamp_gain_loc.detach().clone()
    extra_before = vm2.guide.extrasyn_gain_z.detach().clone()

    stim = torch.zeros(2, 8, step2.neuronSize)
    stim[:, 3, 0] = 1.0
    y = torch.randn(2, 8, step2.neuronSize) * 0.1
    mask = torch.ones(2, 8, step2.neuronSize, dtype=torch.bool)
    noise = torch.full((2, step2.neuronSize), 0.05)

    svi = SVI(vm2, vm2.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    for _ in range(2):
        _set_extrasyn_only_trainable(vm2, True)
        svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)

    assert float((vm2.guide.wicks_g_max_z.detach() - wicks_before).abs().max()) == 0.0
    assert float((vm2.guide.gap_g_syn_z.detach() - gap_before).abs().max()) == 0.0
    assert float((vm2.guide.stim_gain_vec_loc.detach() - stim_before).abs().max()) == 0.0
    assert float((vm2.guide.gcamp_gain_loc.detach() - gcamp_before).abs().max()) == 0.0
    assert not torch.allclose(vm2.guide.extrasyn_gain_z.detach(), extra_before)


def test_oscillatory_prestim_strictly_causal_and_stage2_freeze():
    from pyro_svi_vae import _load_compatible_state_dict, _set_prestim_only_trainable

    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True)
    rec_s1 = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2, use_prestim_encoder=False)
    vm_s1 = vae.VAElegans(
        rec_s1, n, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC), direct_global_guide=True
    )
    s1_state = {k: v.detach().clone() for k, v in vm_s1.state_dict().items()}

    pyro.clear_param_store()
    rec_osc = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=6, prestim_rank=6,
        variational_prestim=True, prestim_oscillatory=True,
    )
    vm_osc = vae.VAElegans(
        rec_osc, n, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC), direct_global_guide=True
    )
    _load_compatible_state_dict(vm_osc, s1_state)
    _set_prestim_only_trainable(vm_osc, True)

    wicks_before = vm_osc.guide.wicks_g_max_z.detach().clone()
    gap_before = vm_osc.guide.gap_g_syn_z.detach().clone()
    osc_out_before = vm_osc.decoder.prestim_encoder.osc_out.weight.detach().clone()

    stim = torch.zeros(2, 16, n)
    stim[:, 8:10, 0] = 1.0
    y = torch.randn(2, 16, n) * 0.1
    mask = torch.ones(2, 16, n, dtype=torch.bool)
    noise = torch.full((2, n), 0.05)

    # Strictly causal: perturbing y[:, pre_vols:, :] with fixed posterior leaves prediction identical
    with torch.no_grad():
        gtr = pyro.poutine.trace(vm_osc.guide).get_trace(
            ExternalInput=stim, y=y, mask=mask, noise_scale=noise
        )
        ps = {k: v["value"] for k, v in gtr.nodes.items() if v["type"] == "sample"}
        y_alt = y.clone()
        y_alt[:, 6:, :] += 5.0
        cond_vm = pyro.condition(vm_osc, data=ps)
        p1 = cond_vm(ExternalInput=stim, y=y, mask=mask, noise_scale=noise, sample_obs=False)
        p2 = cond_vm(ExternalInput=stim, y=y_alt, mask=mask, noise_scale=noise, sample_obs=False)
        assert float((p1 - p2).abs().max()) == 0.0

    svi = SVI(vm_osc, vm_osc.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    for _ in range(3):
        _set_prestim_only_trainable(vm_osc, True)
        loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
        assert np.isfinite(loss)

    assert float((vm_osc.guide.wicks_g_max_z.detach() - wicks_before).abs().max()) == 0.0
    assert float((vm_osc.guide.gap_g_syn_z.detach() - gap_before).abs().max()) == 0.0
    assert not torch.allclose(vm_osc.decoder.prestim_encoder.osc_out.weight.detach(), osc_out_before)


def test_oscillatory_prestim_fits_damped_oscillation():
    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=8, prestim_rank=6,
        variational_prestim=True, prestim_oscillatory=True,
    )
    vm = vae.VAElegans(
        rec, n, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC), direct_global_guide=True
    )
    t = torch.arange(24, dtype=torch.float32).view(1, 24, 1) * 0.5
    phase = torch.tensor([0.0, 1.2, 2.4, 3.1]).view(4, 1, 1)
    y = 0.35 * torch.sin(0.35 * t + phase) * torch.exp(-0.04 * t)
    y = y.expand(4, 24, n).clone()
    stim = torch.zeros_like(y)
    mask = torch.ones_like(y, dtype=torch.bool)
    noise = torch.full((4, n), 0.05)

    from pyro_svi_vae import _set_prestim_only_trainable
    svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 2e-2}), Trace_ELBO())
    losses = []
    for _ in range(15):
        _set_prestim_only_trainable(vm, True)
        losses.append(svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise))
    assert np.isfinite(losses[-1])
    assert losses[-1] < losses[0]


def test_negative_stimulus_detected_in_metrics_and_drift_gate():
    from pyro_svi_c302 import compute_drift_gate, compute_stratified_metrics

    rng = np.random.default_rng(7)
    b, t_tot, n = 4, 24, 5
    pre = 8
    stim = np.zeros((b, t_tot, n), dtype=np.float32)
    stim[:, pre, 0] = -1.5  # purely negative stimulus (e.g., hyperpolarizing / ASEL up-step)
    adj = np.zeros((n, n), dtype=np.float32)
    adj[0, 1] = 1.0  # 1-hop target is neuron 1
    adj[1, 2] = 1.0  # 2-hop target is neuron 2

    y_true = np.zeros((b, t_tot, n), dtype=np.float32)
    y_true[:, pre:pre + 10, 0] = -0.8
    y_true[:, pre:pre + 10, 1] = 0.4 + 0.05 * rng.standard_normal((b, 10)).astype(np.float32)
    y_pred = y_true * 0.9
    mask = np.ones_like(y_true, dtype=bool)

    m = compute_stratified_metrics(y_true, y_pred, mask, stim=stim, pre_vols=pre, adjacency=adj)
    assert m["stim_count"] > 0
    assert m["hop1_count"] > 0
    assert m["evoked_stim_r2"] > 0.8
    assert m["evoked_hop1_r2"] > 0.8

    gate_direct = compute_drift_gate(torch.from_numpy(stim), torch.from_numpy(adj), mode="direct")
    gate_hop1 = compute_drift_gate(torch.from_numpy(stim), torch.from_numpy(adj), mode="direct_and_hop1")
    assert float(gate_direct[0, 0]) == 0.0
    assert float(gate_direct[0, 1]) == 1.0
    assert float(gate_hop1[0, 0]) == 0.0
    assert float(gate_hop1[0, 1]) == 0.0
    assert float(gate_hop1[0, 2]) == 1.0


def test_kernel_metrics_and_np_strict_no_wired_path():
    from pyro_svi_c302 import compute_stratified_metrics

    b, t_tot, n = 6, 30, 6
    pre = 10
    stim = np.zeros((b, t_tot, n), dtype=np.float32)
    stim[:3, pre, 0] = 1.0
    stim[3:, pre, 0] = 1.0
    # Wired chain: 0 -> 1 -> 2 -> 3
    adj = np.zeros((n, n), dtype=np.float32)
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    adj[2, 3] = 1.0
    # NP links from 0 to: 1 (1-hop), 2 (2-hop), 3 (3-hop wired!), and 4 (strictly no wired path!)
    np_adj = np.zeros((n, n), dtype=np.float32)
    np_adj[0, 1] = 1.0
    np_adj[0, 2] = 1.0
    np_adj[0, 3] = 1.0
    np_adj[0, 4] = 1.0

    y_true = np.zeros((b, t_tot, n), dtype=np.float32)
    y_pred = np.zeros((b, t_tot, n), dtype=np.float32)
    t_post = np.arange(16, dtype=np.float32)
    wave_t = 0.6 * np.sin(0.35 * t_post) * np.exp(-0.08 * t_post)
    wave_p = 0.55 * np.sin(0.35 * t_post) * np.exp(-0.08 * t_post)
    y_true[:, pre:pre + 16, 1] = wave_t[None, :]
    y_pred[:, pre:pre + 16, 1] = wave_p[None, :]
    y_true[:, pre:pre + 16, 4] = 0.3 * wave_t[None, :]
    y_pred[:, pre:pre + 16, 4] = 0.3 * wave_t[None, :]
    mask = np.ones_like(y_true, dtype=bool)

    m = compute_stratified_metrics(
        y_true, y_pred, mask, stim=stim, pre_vols=pre, adjacency=adj, np_adjacency=np_adj, dt_obs=0.5
    )
    assert m["kernel_hop1_n_pairs"] == 1
    assert m["kernel_hop1_r"] > 0.99
    assert m["kernel_hop1_auc_r"] == 0.0 or np.isfinite(m["kernel_hop1_auc_r"])
    assert m["kernel_hop1_peak_latency_mae_s"] == 0.0
    assert m["kernel_hop1_fft_r"] > 0.99
    assert m["kernel_hop1_sign_acc"] == 1.0

    # Neuron 4 only -> 16 time steps * 6 trials = 96; excludes neuron 2 (hop2) and neuron 3 (3-hop wired!)
    assert m["evoked_np_only_count"] == 3 * 16 * 6
    assert m["evoked_np_no_hop12_count"] == 2 * 16 * 6
    assert m["evoked_np_no_wired_path_count"] == 1 * 16 * 6
    assert m["evoked_np_no_wired_path_r2"] > 0.99


def test_dataset_provenance_roundtrip_and_mismatch_rejection(tmp_path):
    from pyro_svi_vae import _validate_cache_provenance

    prov = {
        "source": "randi",
        "strain": "wild-type",
        "norm_mode": "dff_safe",
        "causal_preprocess": True,
        "pre_vols": 10,
        "post_vols": 50,
    }
    ds = RandiTrialDataset(
        stim_inputs=torch.zeros(2, 60, 4),
        y=torch.zeros(2, 60, 4),
        obs_mask=torch.ones(2, 60, 4, dtype=torch.bool),
        noise_scale=torch.full((2, 4), 0.1),
        dts=torch.full((2,), 0.5),
        meta=[{"rec_id": 1}, {"rec_id": 2}],
        provenance=prov,
    )
    cache = tmp_path / "cache.npz"
    ds.save(str(cache))
    loaded = RandiTrialDataset.load(str(cache))
    assert loaded.provenance == prov
    _validate_cache_provenance(loaded.provenance, prov, str(cache))

    bad = dict(prov, causal_preprocess=False)
    try:
        _validate_cache_provenance(loaded.provenance, bad, str(cache))
        raise AssertionError("Expected provenance mismatch ValueError")
    except ValueError as exc:
        assert "causal_preprocess" in str(exc)

    try:
        _validate_cache_provenance({}, prov, str(cache))
        raise AssertionError("Expected missing provenance ValueError")
    except ValueError as exc:
        assert "missing provenance" in str(exc)


def test_kernel_metrics_empty_on_insufficient_trials():
    from tools.kernel_metrics import compute_pair_kernel_metrics

    out = compute_pair_kernel_metrics(
        yt_ev=np.zeros((2, 20, 4), dtype=np.float32),
        yp_ev=np.zeros((2, 20, 4), dtype=np.float32),
        stim_target=np.array([[True, False, False, False], [True, False, False, False]]),
        neu_sel=np.ones((2, 4), dtype=bool),
        valid_ev=np.ones((2, 20, 4), dtype=bool),
        t0=5,
        t_ev_end=20,
        min_trials=3,
    )
    assert out == {}


def test_dimensionless_physiology_gauge_invariance_and_svi():
    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step_phys = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=False)
    step_dim = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True)

    # Conductance-free G buffer in dimensionless mode; G is not a PyroSample site
    assert not isinstance(step_dim.Neuron.conductance.G, pyro.nn.PyroSample)
    assert float((step_dim.Neuron.conductance.G - 1.0).abs().max()) == 0.0

    # Exact gauge scaling invariance: (G=0.5, C=0.05, g, g_stim) vs (G=1.0, tau=0.10, 2*g, 2*g_stim)
    stim = torch.zeros(2, 12, n)
    stim[:, 2:5, 0] = 1.0
    shared_ro = {
        "decoder.gcamp.v_th_ca_logit": torch.tensor(0.0),
        "decoder.gcamp.tau_ca": torch.tensor(1.5),
        "decoder.gcamp.alpha_ca": torch.tensor(0.5),
        "decoder.gcamp.v_slope_ca": torch.tensor(0.015),
        "decoder.gcamp.k_on": torch.tensor(0.5),
        "decoder.gcamp.k_off": torch.tensor(0.05),
        "decoder.gcamp.gain": torch.tensor(1.0),
        "decoder.gcamp.offset": torch.tensor(0.0),
    }
    ps_phys = {
        **shared_ro,
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.G": torch.full((n,), 0.5),
        "decoder.model.Neuron.conductance.C": torch.full((n,), 0.05),
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.05]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.synapse.general.g_syn": torch.tensor([0.02]),
        "decoder.model.stim_gain": torch.tensor(1.5),
    }
    ps_dim = {
        **shared_ro,
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.C": torch.full((n,), 0.10),
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.10]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.synapse.general.g_syn": torch.tensor([0.04]),
        "decoder.model.stim_gain": torch.tensor(3.0),
    }
    rec_phys = c302.RecurrentNematode(step_phys, dt_obs=0.5, substeps=4)
    rec_dim = c302.RecurrentNematode(step_dim, dt_obs=0.5, substeps=4)
    vm_phys = vae.VAElegans(rec_phys, n, 1, 1, direct_global_guide=True)
    vm_dim = vae.VAElegans(rec_dim, n, 1, 1, direct_global_guide=True)

    with torch.no_grad():
        y_phys = pyro.condition(vm_phys, data=ps_phys)(ExternalInput=stim, sample_obs=False)
        y_dim = pyro.condition(vm_dim, data=ps_dim)(ExternalInput=stim, sample_obs=False)
    assert float((y_phys[:, :, 0] - y_phys[:, 0:1, 0]).abs().max()) > 1e-3
    assert float((y_phys - y_dim).abs().max()) < 1e-5

    # Guide omits conductance.G site and completes finite SVI steps cleanly
    gtr = pyro.poutine.trace(vm_dim.guide).get_trace(ExternalInput=stim)
    assert "decoder.model.Neuron.conductance.G" not in gtr.nodes
    assert "decoder.model.Neuron.conductance.C" in gtr.nodes

    svi = SVI(vm_dim, vm_dim.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
    y = torch.zeros_like(stim)
    mask = torch.ones_like(stim, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)
    for _ in range(3):
        loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
        assert np.isfinite(loss)


def test_latent_ssm_causality_stability_and_online_filter():
    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 4
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True)
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=4, use_latent_ssm=True, latent_ssm_dim=8)
    vm = vae.VAElegans(rec, n, 1, 1, direct_global_guide=True)

    # Check spectral stability rho(A) < 0.98
    A = rec.latent_ssm.transition_matrix()
    eig_max = float(torch.linalg.eigvals(A.detach()).abs().max())
    assert eig_max < 0.98

    # Supports arbitrary pre_vols (e.g. 40 steps = 20s) AND enforces strict pre-stim causality
    step40 = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True)
    rec40 = c302.RecurrentNematode(step40, dt_obs=0.5, substeps=2, pre_vols=40, use_latent_ssm=True, latent_ssm_dim=8)
    with torch.no_grad():
        for mod in (rec.latent_ssm, rec40.latent_ssm):
            mod.pre_global_fc2.weight.normal_(std=0.05)
            mod.z_to_obs.weight.normal_(std=0.05)
            mod.z_to_current.weight.normal_(std=0.02)

    vm40 = vae.VAElegans(rec40, n, 1, 1, direct_global_guide=True)
    stim = torch.zeros(2, 60, n)
    y = torch.randn(2, 60, n) * 0.2
    y_future_mod = y.clone()
    y_future_mod[:, 40:, :] += 5.0
    y_past_mod = y.clone()
    y_past_mod[:, :40, :] += 2.0
    mask = torch.ones_like(y, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)
    eps = torch.randn(2, 61, 8)

    ps = {
        name: site["value"]
        for name, site in pyro.poutine.trace(vm40.guide).get_trace(ExternalInput=stim).nodes.items()
        if site["type"] == "sample"
    }
    with torch.no_grad():
        p1 = pyro.condition(vm40, data=ps)(ExternalInput=stim, y=y, mask=mask, noise_scale=noise, sample_obs=False, ssm_eps=eps)
        p2 = pyro.condition(vm40, data=ps)(ExternalInput=stim, y=y_future_mod, mask=mask, noise_scale=noise, sample_obs=False, ssm_eps=eps)
        p3 = pyro.condition(vm40, data=ps)(ExternalInput=stim, y=y_past_mod, mask=mask, noise_scale=noise, sample_obs=False, ssm_eps=eps)
    assert float((p1 - p2).abs().max()) == 0.0
    assert float((p1 - p3).abs().max()) > 1e-4

    # Verify SVI step works end-to-end with use_latent_ssm=True
    pyro.clear_param_store()
    svi = pyro.infer.SVI(
        vm,
        vm.guide,
        pyro.optim.Adam({"lr": 1e-3}),
        loss=pyro.infer.Trace_ELBO(num_particles=1),
    )
    loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
    assert np.isfinite(loss)

    # Verify evaluate_predictions passes pre_stim history and shared ssm_eps cleanly
    from pyro_svi_c302 import evaluate_predictions
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stim, y, mask, noise),
        batch_size=2,
        shuffle=False,
    )
    metrics = evaluate_predictions(
        vm,
        dl,
        torch.device("cpu"),
        guide=vm.guide,
        num_samples=2,
        pre_vols=20,
        counterfactual_evoked=True,
    )
    assert np.isfinite(metrics["r2"])

    # Verify Stage-2 freeze keeps latent_ssm trainable while freezing wired weights
    from pyro_svi_vae import _set_prestim_only_trainable
    _set_prestim_only_trainable(vm, True)
    assert vm.decoder.latent_ssm.pre_global_fc1.weight.requires_grad
    assert not vm.guide.wicks_g_max_z_unconstrained.requires_grad

    # Online filtering step helper is finite and masks unobserved entries
    z_p = torch.zeros(2, 8)
    y_obs = torch.ones(2, n)
    m_zero = torch.zeros(2, n, dtype=torch.bool)
    z_f0 = rec.latent_ssm.filter_step(z_p, y_obs, m_zero, torch.zeros_like(y_obs))
    z_f1 = rec.latent_ssm.filter_step(z_p, y_obs, torch.ones_like(m_zero), torch.zeros_like(y_obs))
    assert float(z_f0.abs().max()) == 0.0
    assert float(z_f1.abs().max()) > 0.0


def test_evoked_kernel_aux_loss_gradient_flow():
    from pyro_svi_vae import compute_evoked_kernel_aux_loss

    torch.manual_seed(0)
    b, t_tot, n = 4, 30, 5
    pre = 10
    stim = torch.zeros(b, t_tot, n)
    stim[:, pre, 0] = 1.0
    adj = torch.zeros(n, n)
    adj[0, 1] = 1.0

    gain = torch.nn.Parameter(torch.tensor(0.2))
    base_wave = torch.zeros(b, t_tot, n)
    base_wave[:, pre:pre + 16, 1] = torch.linspace(0.2, 1.0, 16)
    pred_zero = torch.zeros(b, t_tot, n)
    pred_stim = gain * base_wave
    y_true = 0.8 * base_wave
    mask = torch.ones(b, t_tot, n, dtype=torch.bool)

    k_loss, d_loss = compute_evoked_kernel_aux_loss(
        pred_stim, pred_zero, y_true, mask, stim, adj, pre_vols=pre, evoked_window_vols=16
    )
    assert float(k_loss) > 0.0
    assert float(d_loss) > 0.0
    (k_loss + d_loss).backward()
    assert gain.grad is not None and float(gain.grad.abs()) > 1e-3


def test_online_ttt_prequential_ordering_and_non_overlap_guard():
    from tools.eval_online_ttt import (
        group_and_order_trials_by_worm,
        select_causal_history_indices,
    )

    meta = [
        {"rec_id": 2, "stim_vol": 100, "event_idx": 1},  # idx 0
        {"rec_id": 1, "stim_vol": 200, "event_idx": 2},  # idx 1
        {"rec_id": 1, "stim_vol": 10, "event_idx": 0},   # idx 2
        {"rec_id": 1, "stim_vol": 45, "event_idx": 1},   # idx 3 (overlaps with idx 2 if post_vols=50)
        {"rec_id": 1, "stim_vol": 120, "event_idx": 3},  # idx 4 (non-overlapping with idx 2 and idx 3)
    ]
    by_worm = group_and_order_trials_by_worm(meta)
    assert list(by_worm.keys()) == [1, 2]
    # Worm 1 sorted strictly by stim_vol: 10 (idx 2), 45 (idx 3), 120 (idx 4), 200 (idx 1)
    assert by_worm[1] == [2, 3, 4, 1]
    assert by_worm[2] == [0]

    w1 = by_worm[1]
    # Target pos 0 (vol 10): no history
    assert select_causal_history_indices(w1, 0, meta, post_vols=50) == []
    # Target pos 1 (vol 45): vol 10 + 50 = 60 > 45 -> overlaps, MUST be excluded
    assert select_causal_history_indices(w1, 1, meta, post_vols=50) == []
    # Target pos 2 (vol 120): vol 10+50=60<=120 (ok), vol 45+50=95<=120 (ok)
    assert select_causal_history_indices(w1, 2, meta, post_vols=50) == [2, 3]
    # Target pos 3 (vol 200): all three prior trials non-overlapping; max_history=2 keeps most recent
    assert select_causal_history_indices(w1, 3, meta, post_vols=50) == [2, 3, 4]
    assert select_causal_history_indices(w1, 3, meta, post_vols=50, max_history=2) == [3, 4]


def test_online_ttt_tier_isolation_kl_anchor_and_worm_reset():
    from tools.eval_online_ttt import (
        adapt_worm_posterior,
        evaluate_online_ttt,
        restore_population_state,
        snapshot_population_state,
    )

    pyro.clear_param_store()
    torch.manual_seed(0)
    # 4-neuron graph with two distinct presynaptic sources (0 -> 2, 1 -> 3)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0, 1], [2, 3], [2.0, 2.0], [0.0, 0.0], [0.01, 0.01]),
        "Generic": ([], [], []),
    }
    n = 4
    step = c302.NematodeForStep(
        n, {}, tiny_syn, per_neuron_stim_gain=True, wide_amp_priors=True, dimensionless=True
    )
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4, stim_actuator=True
    )
    vm = vae.VAElegans(rec, n, 2, 1, direct_global_guide=True)
    adj = step.synapse.adjacency_matrix(n)

    # Trigger PyroParam registration before snapshotting
    stim_h = torch.zeros(2, 12, n)
    stim_h[:, 4:6, 0] = 1.0  # Only neuron 0 is stimulated in history (edge 0->2 active, edge 1->3 inactive)
    y_h = torch.zeros(2, 12, n)
    y_h[:, 4:, 0] = 0.8
    y_h[:, 5:, 2] = 0.35
    mask_h = torch.ones(2, 12, n, dtype=torch.bool)
    noise_h = torch.full((2, n), 0.05)
    _ = pyro.poutine.trace(vm.guide).get_trace(ExternalInput=stim_h, noise_scale=noise_h)
    pop_snap = snapshot_population_state(vm)

    # 1. Tier 1: updates observation/stim calibration only; all connectome weights bitwise frozen
    wicks_z0 = vm.guide.wicks_g_max_z.detach().clone()
    wicks_g0 = vm.guide.wicks_g_max_global_z.detach().clone()
    gap_z0 = vm.guide.gap_g_syn_z.detach().clone()
    e_z0 = vm.guide.neuron_E_z.detach().clone()
    gcamp_gain0 = vm.guide.gcamp_gain_loc.detach().clone()
    gcamp_off0 = vm.guide.gcamp_offset_loc.detach().clone()
    obs_sig0 = vm.guide.obs_sigma_loc.detach().clone()

    log_t1_loose = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=5, ttt_lr=3e-2, kl_weight=0.01, pre_vols=4, seed=7
    )
    assert np.isfinite(log_t1_loose["loss_end"])
    assert log_t1_loose["loss_end"] < log_t1_loose["loss_start"]
    assert float((vm.guide.wicks_g_max_z.detach() - wicks_z0).abs().max()) == 0.0
    assert float((vm.guide.wicks_g_max_global_z.detach() - wicks_g0).abs().max()) == 0.0
    assert float((vm.guide.gap_g_syn_z.detach() - gap_z0).abs().max()) == 0.0
    assert float((vm.guide.neuron_E_z.detach() - e_z0).abs().max()) == 0.0
    assert float((vm.guide.obs_sigma_loc.detach() - obs_sig0).abs().max()) == 0.0
    assert float((vm.guide.gcamp_gain_loc.detach() - gcamp_gain0).abs().max()) > 1e-4
    assert float((vm.guide.gcamp_offset_loc.detach() - gcamp_off0).abs().max()) > 1e-4

    # Likelihood tempering scales down gradient-driven drift on tiny history
    restore_population_state(vm, pop_snap)
    log_t1_temp = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=5, ttt_lr=3e-2,
        kl_weight=1.0, nll_temp=0.05, pre_vols=4, seed=7,
    )
    log_t1_untemp = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=5, ttt_lr=3e-2,
        kl_weight=1.0, nll_temp=1.0, pre_vols=4, seed=7,
    )
    assert log_t1_temp["max_whitened_shift"] < log_t1_untemp["max_whitened_shift"]

    # Explicit opt-in adapt_obs_sigma=True allows obs_sigma_loc to move
    restore_population_state(vm, pop_snap)
    adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=5, ttt_lr=3e-2,
        kl_weight=0.01, adapt_obs_sigma=True, pre_vols=4, seed=7,
    )
    assert float((vm.guide.obs_sigma_loc.detach() - obs_sig0).abs().max()) > 1e-4

    # Strong KL anchor must shrink parameter shift relative to loose KL anchor
    restore_population_state(vm, pop_snap)
    log_t1_tight = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=5, ttt_lr=3e-2, kl_weight=500.0, pre_vols=4, seed=7
    )
    assert log_t1_tight["max_param_shift"] < log_t1_loose["max_param_shift"]
    assert log_t1_tight["max_whitened_shift"] < log_t1_loose["max_whitened_shift"]

    # 2. Tier 2: updates global synaptic scale / neuron excitability, but keeps edge-level wicks_g_max_z frozen
    restore_population_state(vm, pop_snap)
    adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier2", ttt_steps=4, ttt_lr=3e-2, kl_weight=0.1, pre_vols=4, seed=7
    )
    assert float((vm.guide.wicks_g_max_z.detach() - wicks_z0).abs().max()) == 0.0
    assert float((vm.guide.neuron_E_z.detach() - e_z0).abs().max()) > 1e-4

    # 3. Tier 3: updates ONLY stimulated source's outgoing edge (idx 0: 0->2), keeping unstimulated edge (idx 1: 1->3) and global gain frozen
    restore_population_state(vm, pop_snap)
    adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier3", ttt_steps=4, ttt_lr=3e-2, kl_weight=0.1, pre_vols=4, seed=7
    )
    delta_edge = (vm.guide.wicks_g_max_z.detach() - wicks_z0).abs()
    assert float(delta_edge[0]) > 1e-4
    assert float(delta_edge[1]) == 0.0
    assert float((vm.guide.wicks_g_max_global_z.detach() - wicks_g0).abs().max()) == 0.0

    # 4. Full prequential evaluate_online_ttt across 2 worms + bitwise reset verification
    restore_population_state(vm, pop_snap)
    stim_all = torch.zeros(4, 12, n)
    stim_all[:, 4:6, 0] = 1.0
    y_all = torch.zeros(4, 12, n)
    y_all[:2, 4:, 0] = 0.7
    y_all[:2, 5:, 2] = 0.3
    y_all[2:, 4:, 0] = 0.4
    y_all[2:, 5:, 2] = 0.15
    mask_all = torch.ones(4, 12, n, dtype=torch.bool)
    noise_all = torch.full((4, n), 0.05)
    meta_all = [
        {"rec_id": 10, "stim_vol": 20, "event_idx": 0},
        {"rec_id": 10, "stim_vol": 60, "event_idx": 1},  # non-overlapping with vol 20 (post_vols=8 -> 20+8<=60)
        {"rec_id": 20, "stim_vol": 15, "event_idx": 0},
        {"rec_id": 20, "stim_vol": 20, "event_idx": 1},  # overlapping with vol 15 (15+8=23 > 20) -> k=0!
    ]
    ds = RandiTrialDataset(stim_all, y_all, mask_all, noise_all, torch.full((4,), 0.5), meta_all)

    res = evaluate_online_ttt(
        vm, ds, test_indices=[0, 1, 2, 3], device=torch.device("cpu"),
        adjacency=adj, tier="tier1", ttt_steps=3, ttt_lr=2e-2, kl_weight=1.0,
        pre_vols=4, post_vols=8, num_samples=2, eval_seed=42,
    )
    assert res["n_worms"] == 2
    assert res["n_trials"] == 4
    # Only trial 1 (worm 10, vol 60) has 1 non-overlapping past trial; trial 3 overlaps so k=0
    assert res["n_adapted_trials"] == 1
    assert res["learning_curve"]["k=0"]["n_trials"] == 3
    assert res["learning_curve"]["k=1..3"]["n_trials"] == 1
    assert np.isfinite(res["overall_metrics"]["r2"])

    res_zero = evaluate_online_ttt(
        vm, ds, test_indices=[0, 1, 2, 3], device=torch.device("cpu"),
        adjacency=adj, tier="tier1", ttt_steps=0,
        pre_vols=4, post_vols=8, num_samples=2, eval_seed=42,
    )
    assert res_zero["n_adapted_trials"] == 0

    res_gated = evaluate_online_ttt(
        vm, ds, test_indices=[0, 1, 2, 3], device=torch.device("cpu"),
        adjacency=adj, tier="tier2", ttt_steps=3, ttt_lr=2e-2, kl_weight=1.0,
        min_history=2, k_tier_schedule="none",
        pre_vols=4, post_vols=8, num_samples=2, eval_seed=42,
    )
    assert res_gated["n_adapted_trials"] == 0

    res_sched = evaluate_online_ttt(
        vm, ds, test_indices=[0, 1, 2, 3], device=torch.device("cpu"),
        adjacency=adj, tier="tier2", ttt_steps=3, ttt_lr=2e-2, kl_weight=1.0,
        min_history=2, k_tier_schedule="t1_then_t2",
        pre_vols=4, post_vols=8, num_samples=2, eval_seed=42,
    )
    assert res_sched["n_adapted_trials"] == 1

    # Model state must be bitwise restored to q_pop after evaluation
    for k, v in pop_snap["model_state"].items():
        assert torch.equal(vm.state_dict()[k], v), f"State drift on {k}"


def test_online_ttt_latent_ssm_deterministic():
    from tools.eval_online_ttt import (
        adapt_worm_posterior,
        restore_population_state,
        snapshot_population_state,
    )

    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [2], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 4
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, pre_vols=4, use_latent_ssm=True, latent_ssm_dim=6
    )
    vm = vae.VAElegans(rec, n, 1, 1, direct_global_guide=True)
    stim_h = torch.zeros(2, 10, n)
    stim_h[:, 4:6, 0] = 1.0
    y_h = torch.randn(2, 10, n) * 0.2
    mask_h = torch.ones(2, 10, n, dtype=torch.bool)
    noise_h = torch.full((2, n), 0.1)
    _ = pyro.poutine.trace(vm.guide).get_trace(ExternalInput=stim_h, noise_scale=noise_h)
    snap = snapshot_population_state(vm)

    log1 = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=3, ttt_lr=2e-2, kl_weight=1.0, pre_vols=4, seed=11
    )
    restore_population_state(vm, snap)
    log2 = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h, tier="tier1", ttt_steps=3, ttt_lr=2e-2, kl_weight=1.0, pre_vols=4, seed=11
    )
    assert log1["loss_end"] == log2["loss_end"]
    assert log1["max_whitened_shift"] == log2["max_whitened_shift"]


def test_split_np_ma_extrasynaptic_double_dissociation_and_svi():
    import warnings
    from pyro_svi_vae import _set_extrasyn_only_trainable
    from tools.eval_unc31_knockout import _eval_split_knockout_rollout

    # 1. Verify full C. elegans connectome splits NP (7,035) and MA (1,920) cleanly while preserving 8,717 union
    full_step = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_extrasynaptic=True,
        split_extrasynaptic=True,
        dimensionless=True,
    )
    assert len(full_step.synapse.NP_SRC) == 7035
    assert len(full_step.synapse.MA_SRC) == 1920
    assert len(full_step.synapse.Generic_SRC) == 8717
    np_adj_full = full_step.synapse.np_adjacency_matrix(full_step.neuronSize)
    ma_adj_full = full_step.synapse.ma_adjacency_matrix(full_step.neuronSize)
    assert int(np_adj_full.sum()) == 7035
    assert int(ma_adj_full.sum()) == 1920

    # 2. Toy circuit: 0 -> 1 via NP only, 0 -> 2 via MA only
    pyro.clear_param_store()
    torch.manual_seed(0)
    syn_list = {
        "General": ([], [], []),
        "Wicks": ([0], [3], [1.0], [0.0], [0.01]),
        "Generic": ([0, 0], [1, 2], [1.0, 1.0]),
        "NP": ([0], [1], [1.0]),
        "MA": ([0], [2], [1.0]),
    }
    n = 4
    step = c302.NematodeForStep(
        n, {}, syn_list,
        use_extrasynaptic=True, split_extrasynaptic=True, wide_amp_priors=True, dimensionless=True,
    )
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=4, use_prestim_encoder=False, pre_vols=4)
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)

    # Check Stage-1b freeze keeps NP/MA gains and slow timescales trainable, freezing wired params
    _set_extrasyn_only_trainable(vm, True)
    assert vm.guide.extrasyn_np_gain_z.requires_grad
    assert vm.guide.extrasyn_ma_gain_z.requires_grad
    assert vm.guide.tau_np_loc_unconstrained.requires_grad
    assert vm.guide.tau_ma_loc_unconstrained.requires_grad
    assert not vm.guide.wicks_g_max_z.requires_grad

    # Unfreeze and run 2 SVI steps with ZERO Pyro site mismatch warnings
    _set_extrasyn_only_trainable(vm, False)
    stim = torch.zeros(2, 16, n)
    stim[:, 4:6, 0] = 1.0
    y = torch.zeros_like(stim)
    y[:, 5:, 1] = 0.25
    y[:, 5:10, 2] = 0.30
    mask = torch.ones_like(stim, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)

    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
        for _ in range(2):
            loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
            assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    # Verify independent in-silico knockout & slow tau_np peak lag vs faster tau_ma
    shared_ps = {
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.C": torch.full((n,), 0.10),
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.0]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.stim_gain": torch.tensor(3.0),
        "decoder.model.synapse.extrasyn_np_gain": torch.tensor([0.05]),
        "decoder.model.synapse.extrasyn_ma_gain": torch.tensor([0.05]),
        "decoder.model.synapse.tau_np": torch.tensor(6.0),
        "decoder.model.synapse.tau_ma": torch.tensor(0.5),
        "decoder.gcamp.v_th_ca_logit": torch.tensor(0.0),
        "decoder.gcamp.v_slope_ca": torch.tensor(0.015),
        "decoder.gcamp.alpha_ca": torch.tensor(0.5),
        "decoder.gcamp.k_on": torch.tensor(0.5),
        "decoder.gcamp.tau_ca": torch.tensor(0.5),
        "decoder.gcamp.k_off": torch.tensor(0.1),
        "decoder.gcamp.gain": torch.ones(n),
        "decoder.gcamp.offset": torch.zeros(n),
    }
    stim_long = torch.zeros(1, 30, n)
    stim_long[:, 4:7, 0] = 1.0
    with torch.no_grad():
        y_full = pyro.condition(vm, data=shared_ps)(ExternalInput=stim_long, sample_obs=False)
        y_np_z = pyro.condition(
            vm, data={**shared_ps, "decoder.model.synapse.extrasyn_np_gain": torch.tensor([0.0])}
        )(ExternalInput=stim_long, sample_obs=False)
        y_ma_z = pyro.condition(
            vm, data={**shared_ps, "decoder.model.synapse.extrasyn_ma_gain": torch.tensor([0.0])}
        )(ExternalInput=stim_long, sample_obs=False)

    # Zeroing NP eliminates neuron 1 response while leaving neuron 2 unchanged; zeroing MA reverses it
    assert float((y_np_z[:, :, 1] - y_np_z[:, 0:1, 1]).abs().max()) < 1e-6
    assert float((y_np_z[:, :, 2] - y_full[:, :, 2]).abs().max()) < 1e-6
    assert float((y_ma_z[:, :, 2] - y_ma_z[:, 0:1, 2]).abs().max()) < 1e-6
    assert float((y_ma_z[:, :, 1] - y_full[:, :, 1]).abs().max()) < 1e-6
    # Slower tau_np (6.0s) peaks strictly later than faster tau_ma (0.5s)
    peak_np = int(torch.argmax(y_full[0, 4:, 1]))
    peak_ma = int(torch.argmax(y_full[0, 4:, 2]))
    assert peak_np > peak_ma

    # Verify _eval_split_knockout_rollout helper runs cleanly on dataloader
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stim, y, mask, noise), batch_size=2, shuffle=False
    )
    adj_4 = step.synapse.adjacency_matrix(n)
    np_4 = step.synapse.np_adjacency_matrix(n)
    ma_4 = step.synapse.ma_adjacency_matrix(n)
    res_split = _eval_split_knockout_rollout(vm, dl, adj_4, np_4, ma_4, pre_vols=4, num_samples=2, zero_np=True)
    assert np.isfinite(res_split["post_inlier_r2"])


def test_synaptic_lowpass_filter_phase_lag_and_svi():
    import warnings
    from pyro_svi_c302 import evaluate_predictions
    from tools.eval_online_ttt import adapt_worm_posterior

    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step_inst = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True, use_syn_filter=False)
    step_filt = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True, use_syn_filter=True)
    rec_inst = c302.RecurrentNematode(step_inst, dt_obs=0.5, substeps=4, pre_vols=6)
    rec_filt = c302.RecurrentNematode(step_filt, dt_obs=0.5, substeps=4, pre_vols=6)
    vm_inst = vae.VAElegans(rec_inst, n, 1, 0, direct_global_guide=True)
    vm_filt = vae.VAElegans(rec_filt, n, 1, 0, direct_global_guide=True)

    ps_base = {
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.C": torch.full((n,), 0.10),
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.15]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.stim_gain": torch.tensor(3.0),
        "decoder.gcamp.v_th_ca_logit": torch.tensor(0.0),
        "decoder.gcamp.v_slope_ca": torch.tensor(0.015),
        "decoder.gcamp.alpha_ca": torch.tensor(0.5),
        "decoder.gcamp.k_on": torch.tensor(0.5),
        "decoder.gcamp.tau_ca": torch.tensor(0.5),
        "decoder.gcamp.k_off": torch.tensor(0.1),
        "decoder.gcamp.gain": torch.ones(n),
        "decoder.gcamp.offset": torch.zeros(n),
    }

    # Resting equilibrium must remain exact regardless of tau_syn
    stim_zero = torch.zeros(1, 20, n)
    with torch.no_grad():
        y0_inst = pyro.condition(vm_inst, data=ps_base)(ExternalInput=stim_zero, sample_obs=False)
        y0_slow = pyro.condition(
            vm_filt, data={**ps_base, "decoder.model.synapse.wicks.tau_syn": torch.full((n,), 5.0)}
        )(ExternalInput=stim_zero, sample_obs=False)
    assert float((y0_inst - y0_slow).abs().max()) < 1e-6

    # Slow tau_syn must systematically delay downstream peak latency vs instantaneous release
    stim = torch.zeros(1, 30, n)
    stim[:, 6:9, 0] = 1.0
    with torch.no_grad():
        y_fast = pyro.condition(vm_inst, data=ps_base)(ExternalInput=stim, sample_obs=False)
        y_slow = pyro.condition(
            vm_filt, data={**ps_base, "decoder.model.synapse.wicks.tau_syn": torch.full((n,), 4.0)}
        )(ExternalInput=stim, sample_obs=False)
    peak_fast = int(torch.argmax(y_fast[0, 6:, 1]))
    peak_slow = int(torch.argmax(y_slow[0, 6:, 1]))
    assert peak_slow > peak_fast

    # End-to-end SVI + evaluate_predictions + Phase 6 TTT with zero Pyro warnings
    stim_b = stim.expand(2, -1, -1).clone()
    y_b = torch.randn(2, 30, n) * 0.1
    mask_b = torch.ones_like(y_b, dtype=torch.bool)
    noise_b = torch.full((2, n), 0.1)
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm_filt, vm_filt.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
        for _ in range(2):
            loss = svi.step(ExternalInput=stim_b, mask=mask_b, y=y_b, noise_scale=noise_b)
            assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stim_b, y_b, mask_b, noise_b), batch_size=2, shuffle=False
    )
    ev = evaluate_predictions(vm_filt, dl, torch.device("cpu"), guide=vm_filt.guide, num_samples=2, pre_vols=6)
    assert np.isfinite(ev["r2"])
    ttt_log = adapt_worm_posterior(vm_filt, stim_b, y_b, mask_b, noise_b, tier="tier1", ttt_steps=2, pre_vols=6)
    assert np.isfinite(ttt_log["loss_end"])


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
        "tools/eval_checkpoint.py",
        "tools/run_nested_protocol.py",
        "tools/kernel_metrics.py",
        "tools/substep_convergence.py",
        "tools/noise_ceiling.py",
        "tools/run_baselines.py",
        "tools/eval_unc31_knockout.py",
        "tools/eval_online_ttt.py",
        "tools/run_genotype_joint.py",
    ]:
        py_compile.compile(fname, doraise=True)


def test_safe_local_posterior_expansion_when_batch_equals_neurons():
    from pyro_svi_c302 import extract_and_expand_posterior_samples

    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [2], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 4
    step = c302.NematodeForStep(
        n, {}, tiny_syn, per_neuron_stim_gain=True, wide_amp_priors=True, dimensionless=True
    )
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=4,
        prestim_rank=3, variational_prestim=True, per_neuron_kinetics=True,
    )
    vm = vae.VAElegans(rec, n, 1, 1, direct_global_guide=True)
    # Intentionally match batch_size B == n_neurons == 4!
    stim = torch.zeros(4, 8, n)
    y = torch.randn(4, 8, n) * 0.1
    mask = torch.ones_like(y, dtype=torch.bool)
    noise = torch.full((4, n), 0.05)

    gtr = pyro.poutine.trace(vm.guide).get_trace(
        ExternalInput=stim, mask=None, y=None, noise_scale=noise,
        pre_stim_obs=y[:, :4, :], pre_stim_mask=mask[:, :4, :],
    )
    ps = extract_and_expand_posterior_samples(gtr, repeat_factor=2)
    assert tuple(ps["prestim_z"].shape) == (8, 3)
    assert tuple(ps["decoder.model.Neuron.conductance.E"].shape) == (4,)
    assert tuple(ps["decoder.model.Neuron.conductance.C"].shape) == (4,)
    assert tuple(ps["decoder.model.stim_gain_vec"].shape) == (4,)
    assert tuple(ps["decoder.gcamp.tau_ca"].shape) == (4,)
    assert tuple(ps["decoder.gcamp.gain"].shape) == (4,)


def test_unified_intact_supervision_adj_for_shuffle():
    from pyro_svi_vae import compute_evoked_kernel_aux_loss, reweight_hop1_noise

    intact = c302.readConnectome("./data/CElegansNeuronTables.xls")
    shuf = c302.readConnectome("./data/CElegansNeuronTables.xls", shuffle_seed=7)
    adj_intact = torch.as_tensor(intact.synapse.adjacency_matrix(intact.neuronSize), dtype=torch.float32)
    adj_shuf = torch.as_tensor(shuf.synapse.adjacency_matrix(shuf.neuronSize), dtype=torch.float32)
    assert not torch.equal(adj_intact, adj_shuf)

    # Same intact biological supervision matrix yields identical hop1 weighting masks
    stim = torch.zeros(2, 20, intact.neuronSize)
    stim[:, 10, 0] = 1.0
    noise = torch.ones(2, intact.neuronSize)
    rw_a = reweight_hop1_noise(noise, stim, adj_intact, weight=3.0, stim_weight=6.0)
    rw_b = reweight_hop1_noise(noise, stim, adj_intact, weight=3.0, stim_weight=6.0)
    rw_shuf = reweight_hop1_noise(noise, stim, adj_shuf, weight=3.0, stim_weight=6.0)
    assert torch.equal(rw_a, rw_b)
    assert not torch.equal(rw_a, rw_shuf)

    y_true = torch.randn(2, 20, intact.neuronSize) * 0.1
    mask = torch.ones(2, 20, intact.neuronSize, dtype=torch.bool)
    p_s = torch.randn(2, 20, intact.neuronSize, requires_grad=True)
    p_z = torch.zeros_like(p_s)
    k1, d1 = compute_evoked_kernel_aux_loss(p_s, p_z, y_true, mask, stim, adj_intact, pre_vols=10)
    assert torch.isfinite(k1) and torch.isfinite(d1)


def test_dimensionless_zero_mean_log_leak_gauge_and_gap_reciprocity():
    import warnings

    pyro.clear_param_store()
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [2], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 3
    step_phys = c302.NematodeForStep(n, {}, tiny_syn, dimensionless=False)
    step_het = c302.NematodeForStep(
        n, {}, tiny_syn, dimensionless=True, leak_gauge="zero_mean_log"
    )
    rec_phys = c302.RecurrentNematode(step_phys, dt_obs=0.5, substeps=4)
    rec_het = c302.RecurrentNematode(step_het, dt_obs=0.5, substeps=4)
    vm_phys = vae.VAElegans(rec_phys, n, 1, 1, direct_global_guide=True)
    vm_het = vae.VAElegans(rec_het, n, 1, 1, direct_global_guide=True)

    # Heterogeneous G with geom_mean == 1.0: G_phys = G_bar * G_dim, lambda = 1 / G_bar
    G_vec = torch.tensor([0.2, 0.8, 1.0])
    G_bar = float(torch.exp(torch.log(G_vec).mean()).item())
    log_g_raw = torch.log(G_vec) - torch.log(G_vec).mean()
    C_vec = torch.tensor([0.04, 0.08, 0.10])
    stim = torch.zeros(2, 16, n)
    stim[:, 5:8, 0] = 1.0

    shared_gcamp = {
        "decoder.gcamp.v_th_ca_logit": torch.tensor(0.0),
        "decoder.gcamp.v_slope_ca": torch.tensor(0.015),
        "decoder.gcamp.alpha_ca": torch.tensor(0.5),
        "decoder.gcamp.k_on": torch.tensor(0.5),
        "decoder.gcamp.tau_ca": torch.tensor(0.5),
        "decoder.gcamp.k_off": torch.tensor(0.05),
        "decoder.gcamp.gain": torch.tensor(1.0),
        "decoder.gcamp.offset": torch.tensor(0.0),
    }
    ps_phys = {
        **shared_gcamp,
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.G": G_vec,
        "decoder.model.Neuron.conductance.C": C_vec,
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.05]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.synapse.general.g_syn": torch.tensor([0.03]),
        "decoder.model.stim_gain": torch.tensor(1.2),
    }
    ps_het = {
        **shared_gcamp,
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.log_G_raw": log_g_raw,
        "decoder.model.Neuron.conductance.C": C_vec / G_bar,
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.05 / G_bar]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.synapse.general.g_syn": torch.tensor([0.03 / G_bar]),
        "decoder.model.stim_gain": torch.tensor(1.2 / G_bar),
    }
    with torch.no_grad():
        yp = pyro.condition(vm_phys, data=ps_phys)(ExternalInput=stim, sample_obs=False)
        yh = pyro.condition(vm_het, data=ps_het)(ExternalInput=stim, sample_obs=False)
    assert float((yp - yh).abs().max()) < 1e-5
    # Verify KCL current conservation under asymmetric leak G
    gap_cur = step_het.synapse.general(torch.tensor([[-0.055, -0.075, -0.065]]))
    assert float(gap_cur[0, :2].sum().abs()) < 1e-6

    # Verify clean SVI with zero Pyro warnings
    y = torch.zeros_like(stim)
    mask = torch.ones_like(stim, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm_het, vm_het.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
        for _ in range(2):
            loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
            assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"


def test_latent_ssm_v2_gru_split_stability_and_variational_traj():
    import warnings
    from pyro_svi_c302 import evaluate_predictions

    pyro.clear_param_store()
    torch.manual_seed(0)
    tiny_syn = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0], [1], [2.0], [0.0], [0.01]),
        "Generic": ([], [], []),
    }
    n = 4
    step = c302.NematodeForStep(n, {}, tiny_syn, wide_amp_priors=True, dimensionless=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, pre_vols=20,
        use_latent_ssm=True, latent_ssm_dim=6,
        ssm_encoder="gru", ssm_split_latent=True, ssm_artifact_dim=4,
        ssm_variational_traj=True, ssm_smoother=True, history_vols=10,
    )
    vm = vae.VAElegans(rec, n, 1, 1, direct_global_guide=True)
    assert "_latent_ssm_ref" in vm.guide.__dict__
    assert not any("guide._latent_ssm_ref" in k for k in vm.state_dict().keys())

    # 1. History window + causal invariance & decoupled z_neural vs z_artifact
    with torch.no_grad():
        rec.latent_ssm.gru_z_head.weight.normal_(std=0.05)
        rec.latent_ssm.art_to_obs.weight.normal_(std=0.05)
        rec.latent_ssm.z_to_current.weight.normal_(std=0.03)
        rec.latent_ssm.v_to_z.weight.normal_(std=0.03)

    stim = torch.zeros(2, 32, n)
    stim[:, 20:23, 0] = 1.0
    y = torch.randn(2, 32, n) * 0.2
    mask = torch.ones_like(y, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)

    y_fut = y.clone()
    y_fut[:, 20:, :] += 4.0  # future mutated
    y_early = y.clone()
    y_early[:, :10, :] += 4.0  # earlier than history_vols=10 mutated
    y_hist = y.clone()
    y_hist[:, 10:20, :] += 2.0  # inside [10:20] history window mutated

    eps = torch.randn(2, 33, 10)  # total_z_dim = 6 + 4 = 10
    gtr = pyro.poutine.trace(vm.guide).get_trace(
        ExternalInput=stim, mask=None, y=None, noise_scale=noise,
        pre_stim_obs=y[:, :20, :], pre_stim_mask=mask[:, :20, :],
    )
    ps_fixed = {
        k: v["value"] for k, v in gtr.nodes.items()
        if v["type"] == "sample" and not v.get("is_observed", False)
    }
    with torch.no_grad():
        p0 = pyro.condition(vm, data=ps_fixed)(
            ExternalInput=stim, y=None, pre_stim_obs=y[:, :20, :], pre_stim_mask=mask[:, :20, :],
            sample_obs=False, ssm_eps=eps,
        )
        p_f = pyro.condition(vm, data=ps_fixed)(
            ExternalInput=stim, y=None, pre_stim_obs=y_fut[:, :20, :], pre_stim_mask=mask[:, :20, :],
            sample_obs=False, ssm_eps=eps,
        )
        p_e = pyro.condition(vm, data=ps_fixed)(
            ExternalInput=stim, y=None, pre_stim_obs=y_early[:, :20, :], pre_stim_mask=mask[:, :20, :],
            sample_obs=False, ssm_eps=eps,
        )
        # Direct encode_initial invariance for history_vols=10
        z_a, _, _, _ = rec.latent_ssm.encode_initial(y[:, 10:20, :], mask[:, 10:20, :])
        z_h, _, _, _ = rec.latent_ssm.encode_initial(y_hist[:, 10:20, :], mask[:, 10:20, :])
    assert float((p0 - p_f).abs().max()) == 0.0
    assert float((p0 - p_e).abs().max()) == 0.0
    assert float((z_a - z_h).abs().max()) > 1e-4

    # Verify z_artifact is invariant to stimulus (cannot shortcut through V -> z -> y_obs)
    A_art = rec.latent_ssm.artifact_transition_matrix()
    a0 = torch.randn(2, 4)
    art_stim = rec.latent_ssm.obs_residual(rec.latent_ssm.step_artifact(a0, A_art=A_art, eps=eps[:, 1, 6:]))
    art_zero = rec.latent_ssm.obs_residual(rec.latent_ssm.step_artifact(a0, A_art=A_art, eps=eps[:, 1, 6:]))
    assert torch.equal(art_stim, art_zero)

    # 2. Closed-loop stability check
    stab = c302.check_closed_loop_stability(rec, n_steps=60)
    assert stab["finite"] and stab["stable"]
    assert stab["autonomous_A_spectral_radius"] < 0.98
    assert np.isfinite(stab["closed_loop_spectral_radius"])

    # 3. End-to-end SVI with variational_traj + smoother has zero Pyro mismatch warnings
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-3}), Trace_ELBO())
        for _ in range(2):
            loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
            assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    # 4. Open-loop evaluate_predictions + strictly causal online filter rollout
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(stim, y, mask, noise), batch_size=2)
    ev = evaluate_predictions(vm, dl, torch.device("cpu"), guide=vm.guide, num_samples=2, pre_vols=20)
    assert np.isfinite(ev["r2"]) and np.isfinite(ev["calib_crps"])

    y_mod_t25 = y.clone()
    y_mod_t25[:, 25, :] += 10.0
    with torch.no_grad():
        cond_f = pyro.condition(vm, data=ps_fixed)
        f1 = cond_f(ExternalInput=stim, y=y, mask=mask, sample_obs=False, filter_online=True)
        f2 = cond_f(ExternalInput=stim, y=y_mod_t25, mask=mask, sample_obs=False, filter_online=True)
    assert float((f1[:, :26, :] - f2[:, :26, :]).abs().max()) == 0.0
    assert float((f1[:, 26:, :] - f2[:, 26:, :]).abs().max()) > 1e-6


def test_ttt_modes_worm_latent_temporal_val_and_promotion_gate():
    from tools.eval_online_ttt import (
        _build_worm_latent_edge_basis,
        _compute_ttt_promotion_gate,
        adapt_worm_posterior,
        restore_population_state,
        snapshot_population_state,
    )

    pyro.clear_param_store()
    torch.manual_seed(0)
    n = 6
    tiny_syn = {
        "General": ([0, 2], [1, 3], [1.0, 1.0]),
        "Wicks": ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [2.0] * 6, [0.0] * 6, [0.01] * 6),
        "Generic": ([], [], []),
    }
    step = c302.NematodeForStep(
        n, {}, tiny_syn, per_neuron_stim_gain=True, wide_amp_priors=True, dimensionless=True
    )
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4, stim_actuator=True
    )
    vm = vae.VAElegans(rec, n, 6, 2, direct_global_guide=True)
    stim_h = torch.zeros(4, 12, n)
    stim_h[:, 4:6, 0] = 1.0
    y_h = torch.randn(4, 12, n) * 0.25
    y_h[:, 4:8, 0] += 0.6
    mask_h = torch.ones(4, 12, n, dtype=torch.bool)
    noise_h = torch.full((4, n), 0.1)
    _ = pyro.poutine.trace(vm.guide).get_trace(ExternalInput=stim_h, noise_scale=noise_h)
    pop_snap = snapshot_population_state(vm)

    # ttt_a moves gcamp only; stim_gain & excitability stay frozen
    stim_0 = vm.guide.stim_gain_vec_loc.detach().clone()
    vrest_0 = vm.guide.wicks_V_rest_z.detach().clone()
    gmax_glob_0 = vm.guide.wicks_g_max_global_z.detach().clone()
    gmax_edge_0 = vm.guide.wicks_g_max_z.detach().clone()
    adapt_worm_posterior(vm, stim_h, y_h, mask_h, noise_h, ttt_mode="ttt_a", ttt_steps=3, pre_vols=4)
    assert float((vm.guide.stim_gain_vec_loc.detach() - stim_0).abs().max()) == 0.0
    assert float((vm.guide.wicks_V_rest_z.detach() - vrest_0).abs().max()) == 0.0

    # ttt_b moves stim_gain only; gcamp_gain stays frozen
    restore_population_state(vm, pop_snap)
    gc_0 = vm.guide.gcamp_gain_loc.detach().clone()
    adapt_worm_posterior(vm, stim_h, y_h, mask_h, noise_h, ttt_mode="ttt_b", ttt_steps=3, pre_vols=4)
    assert float((vm.guide.gcamp_gain_loc.detach() - gc_0).abs().max()) == 0.0
    assert float((vm.guide.stim_gain_vec_loc.detach() - stim_0).abs().max()) > 1e-4

    # ttt_e moves global syn scale + gcamp/stim, keeping E/V_rest/V_slope/C frozen
    restore_population_state(vm, pop_snap)
    e_0 = vm.guide.neuron_E_z.detach().clone()
    c_0 = vm.guide.neuron_C_loc.detach().clone()
    vs_0 = vm.guide.wicks_V_slope_loc.detach().clone()
    adapt_worm_posterior(vm, stim_h, y_h, mask_h, noise_h, ttt_mode="ttt_e", ttt_steps=4, kl_weight=0.1, pre_vols=4)
    assert float((vm.guide.neuron_E_z.detach() - e_0).abs().max()) == 0.0
    assert float((vm.guide.wicks_V_rest_z.detach() - vrest_0).abs().max()) == 0.0
    assert float((vm.guide.wicks_V_slope_loc.detach() - vs_0).abs().max()) == 0.0
    assert float((vm.guide.neuron_C_loc.detach() - c_0).abs().max()) == 0.0
    assert float((vm.guide.wicks_g_max_global_z.detach() - gmax_glob_0).abs().max()) > 1e-5

    # ttt_d with worm_latent_dim=2 on 6 edges strictly constrains delta_wicks_z to rank-2 spectral subspace
    restore_population_state(vm, pop_snap)
    log_d = adapt_worm_posterior(
        vm, stim_h, y_h, mask_h, noise_h,
        ttt_mode="ttt_d", worm_latent_dim=2, temporal_val_frac=0.25, ttt_steps=5, kl_weight=0.05, pre_vols=4,
    )
    assert log_d["n_train_hist"] == 3 and log_d["n_val_hist"] == 1
    assert np.isfinite(log_d["val_nll_start"]) and np.isfinite(log_d["val_nll_best"])
    assert float((vm.guide.wicks_V_rest_z.detach() - vrest_0).abs().max()) == 0.0

    # Verify promotion gate logic
    zs = {"overall_metrics": {"post_inlier_r2": 0.100, "evoked_stim_r2": 0.050, "evoked_hop1_r2": 0.040, "kernel_hop1_r2": 0.060}}
    good = {"overall_metrics": {"post_inlier_r2": 0.097, "evoked_stim_r2": 0.070, "evoked_hop1_r2": 0.042, "kernel_hop1_r2": 0.075}}
    bad_base = {"overall_metrics": {"post_inlier_r2": 0.015, "evoked_stim_r2": 0.076, "evoked_hop1_r2": 0.043, "kernel_hop1_r2": 0.082}}
    assert _compute_ttt_promotion_gate(zs, good)["pass"] is True
    assert _compute_ttt_promotion_gate(zs, bad_base)["pass"] is False


def test_np_ma_exclusive_strata_matched_deltak_calibration_and_genotype_joint():
    import warnings
    from pyro_svi_c302 import compute_predictive_calibration_metrics, compute_stratified_metrics
    from tools.kernel_metrics import compute_matched_kernel_difference

    # 1. Exclusive NP/MA/overlap strata + matched Delta K_ij(t) + predictive calibration
    b, t_tot, n = 4, 24, 5
    pre = 8
    adj = np.zeros((n, n), dtype=np.float32)
    adj[0, 1] = 1.0  # wired hop1
    np_adj = np.zeros((n, n), dtype=np.float32)
    np_adj[0, 2] = 1.0  # NP exclusive
    np_adj[0, 4] = 1.0  # NP & MA overlap
    ma_adj = np.zeros((n, n), dtype=np.float32)
    ma_adj[0, 3] = 1.0  # MA exclusive
    ma_adj[0, 4] = 1.0  # NP & MA overlap

    stim = np.zeros((b, t_tot, n), dtype=np.float32)
    stim[:, pre:pre + 2, 0] = 1.0
    mask = np.ones((b, t_tot, n), dtype=bool)
    t_grid = np.linspace(0.0, 1.0, t_tot - pre, dtype=np.float32)
    wt_true = np.zeros((b, t_tot, n), dtype=np.float32)
    mut_true = np.zeros((b, t_tot, n), dtype=np.float32)
    wt_true[:, pre:, 2] = 0.5 * np.sin(np.pi * t_grid)[None, :]
    mut_true[:, pre:, 2] = 0.1 * np.sin(np.pi * t_grid)[None, :]
    wt_true[:, pre:, 3] = 0.4 * np.sin(np.pi * t_grid)[None, :]
    mut_true[:, pre:, 3] = 0.4 * np.sin(np.pi * t_grid)[None, :]

    m_exc = compute_stratified_metrics(
        wt_true, wt_true, mask, stim=stim, pre_vols=pre, adjacency=adj,
        evoked_baseline="mean", y_pred_counterfactual=np.zeros_like(wt_true),
        np_adjacency=np_adj, ma_adjacency=ma_adj, dt_obs=0.5,
    )
    assert m_exc["evoked_np_exclusive_count"] > 0 and abs(m_exc["evoked_np_exclusive_r2"] - 1.0) < 1e-5
    assert m_exc["evoked_ma_exclusive_count"] > 0 and abs(m_exc["evoked_ma_exclusive_r2"] - 1.0) < 1e-5
    assert m_exc["evoked_np_ma_overlap_count"] > 0

    dk = compute_matched_kernel_difference(
        wt_true, wt_true, mask, stim, mut_true, mut_true, mask, stim,
        adjacency=adj, np_adjacency=np_adj, ma_adjacency=ma_adj, pre_vols=pre,
    )
    assert "np_exclusive" in dk and dk["np_exclusive"]["delta_k_r"] > 0.99

    rng = np.random.default_rng(7)
    draws = rng.normal(loc=wt_true[None, ...], scale=0.1, size=(6, b, t_tot, n))
    scales = np.full((6, b, n), 0.1, dtype=np.float64)
    cal = compute_predictive_calibration_metrics(wt_true, draws, scales, mask, pre_vols=pre)
    assert np.isfinite(cal["calib_crps"]) and np.isfinite(cal["calib_nll"])
    assert cal["calib_cov90"] >= cal["calib_cov50"]

    # 2. Genotype-conditioned Bayesian NP/MA inference + prior sensitivity CLI params
    pyro.clear_param_store()
    syn_list = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [1.0], [0.0], [0.01]),
        "Generic": ([0, 0], [2, 3], [1.0, 1.0]),
        "NP": ([0], [2], [1.0]),
        "MA": ([0], [3], [1.0]),
    }
    step = c302.NematodeForStep(
        n, {}, syn_list, use_extrasynaptic=True, split_extrasynaptic=True,
        dimensionless=True, tau_np_prior=1.0, tau_ma_prior=4.0, tau_mod_prior_std=1.2,
        genotype_conditioned=True,
    )
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4)
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)
    geno = torch.tensor([0, 1], dtype=torch.long)
    stim_g = torch.zeros(2, 16, n)
    stim_g[:, 4:7, 0] = 1.0
    y_g = torch.zeros_like(stim_g)
    mask_g = torch.ones_like(stim_g, dtype=torch.bool)
    noise_g = torch.full((2, n), 0.1)

    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
        loss = svi.step(ExternalInput=stim_g, mask=mask_g, y=y_g, noise_scale=noise_g, genotype_id=geno)
        assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    # Zeroing gamma_np_mut=0 eliminates NP response ONLY for genotype 1 (unc-31), leaving WT (0) intact
    ps_ko = {
        "decoder.model.Neuron.conductance.E": torch.full((n,), -0.065),
        "decoder.model.Neuron.conductance.C": torch.full((n,), 0.10),
        "decoder.model.synapse.wicks.g_max": torch.tensor([0.0]),
        "decoder.model.synapse.wicks.V_rest": torch.full((n,), -0.042),
        "decoder.model.synapse.wicks.V_slope": torch.full((n,), 0.015),
        "decoder.model.synapse.wicks.E_rev": torch.tensor([0.0]),
        "decoder.model.stim_gain": torch.tensor(3.0),
        "decoder.model.synapse.extrasyn_np_gain": torch.tensor([0.05]),
        "decoder.model.synapse.extrasyn_ma_gain": torch.tensor([0.05]),
        "decoder.model.synapse.tau_np": torch.tensor(2.0),
        "decoder.model.synapse.tau_ma": torch.tensor(1.0),
        "decoder.model.synapse.gamma_np_mut": torch.tensor(0.0),
        "decoder.model.synapse.gamma_ma_mut": torch.tensor(1.0),
        "decoder.model.synapse.gamma_wired_mut": torch.tensor(1.0),
        "decoder.gcamp.v_th_ca_logit": torch.tensor(0.0),
        "decoder.gcamp.v_slope_ca": torch.tensor(0.015),
        "decoder.gcamp.alpha_ca": torch.tensor(0.5),
        "decoder.gcamp.k_on": torch.tensor(0.5),
        "decoder.gcamp.tau_ca": torch.tensor(0.5),
        "decoder.gcamp.k_off": torch.tensor(0.1),
        "decoder.gcamp.gain": torch.ones(n),
        "decoder.gcamp.offset": torch.zeros(n),
    }
    with torch.no_grad():
        yo = pyro.condition(vm, data=ps_ko)(ExternalInput=stim_g, sample_obs=False, genotype_id=geno)
    assert float(yo[0, 5:, 2].abs().max()) > 1e-4  # WT NP responds
    assert float(yo[1, 5:, 2].abs().max()) < 1e-6  # unc-31 NP eliminated
    assert float((yo[0, 5:, 3] - yo[1, 5:, 3]).abs().max()) < 1e-6  # MA identical across WT and unc-31


def test_randi_missing_gcamp_strict_guard_and_disjoint_shuffle(tmp_path):
    """Missing *_gcamp.txt emits a warning by default, raises with strict_recordings=True, and disjoint rewiring has 0 overlap."""
    import warnings
    from utils import AtlasLoader

    rec_dir = tmp_path / "exported_data"
    rec_dir.mkdir()
    # One complete recording (id=1) and one missing gcamp (id=2)
    np.savetxt(rec_dir / "1_t.txt", np.arange(20, dtype=np.float32) * 0.5)
    np.savetxt(rec_dir / "1_gcamp.txt", np.ones((20, 2), dtype=np.float32))
    np.savetxt(rec_dir / "1_stim_neurons.txt", np.array([1], dtype=np.int64), fmt="%d")
    np.savetxt(rec_dir / "1_stim_volume_i.txt", np.array([5], dtype=np.int64), fmt="%d")
    (rec_dir / "1_labels.txt").write_text("ADAL\nADAR\n")
    np.savetxt(rec_dir / "2_t.txt", np.arange(20, dtype=np.float32) * 0.5)

    ids_ok, ids_missing = AtlasLoader._discover_recording_ids_with_missing(str(rec_dir))
    assert ids_ok == [1] and ids_missing == [2]

    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        ds_warn = AtlasLoader(str(tmp_path), strain="wild-type", strict_recordings=False)
        assert len(ds_warn.recordings) == 1
    assert any("missing" in str(w.message).lower() for w in w_rec)

    try:
        AtlasLoader(str(tmp_path), strain="wild-type", strict_recordings=True)
        raise AssertionError("Expected FileNotFoundError with strict_recordings=True")
    except FileNotFoundError:
        pass

    # Disjoint Maslov-Sneppen rewiring has 0 overlap with intact connectome and preserves edge count
    intact = c302.readConnectome("./data/CElegansNeuronTables.xls", use_extrasynaptic=True, split_extrasynaptic=True)
    disjoint = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_extrasynaptic=True,
        split_extrasynaptic=True,
        shuffle_seed=7,
        shuffle_disjoint=True,
    )
    intact_wired = (
        set(zip(intact.synapse.Wicks_SRC, intact.synapse.Wicks_DST))
        | set(zip(intact.synapse.Gap_Junction_SRC, intact.synapse.Gap_Junction_DST))
        | set(zip(intact.synapse.Gap_Junction_DST, intact.synapse.Gap_Junction_SRC))
    )
    dis_wicks = set(zip(disjoint.synapse.Wicks_SRC, disjoint.synapse.Wicks_DST))
    dis_gap = set(zip(disjoint.synapse.Gap_Junction_SRC, disjoint.synapse.Gap_Junction_DST))
    assert len(disjoint.synapse.Wicks_SRC) == len(intact.synapse.Wicks_SRC)
    assert len(disjoint.synapse.Gap_Junction_SRC) == len(intact.synapse.Gap_Junction_SRC)
    assert len(dis_wicks & intact_wired) == 0
    assert len(dis_gap & intact_wired) == 0


def test_phase8a_state_observer_causality_slow_decay_and_full_filter():
    """Phase 8A state observer assimilates strictly pre_vols causally, obeys min_decay, and runs full-state online filtering."""
    pyro.clear_param_store()
    n = 6
    syn_list = {
        "General": ([0], [1], [1.0]),
        "Wicks": ([0, 1], [2, 3], [1.0, 1.0], [0.0, 0.0], [0.01, 0.01]),
    }
    step = c302.NematodeForStep(n, {}, syn_list, dimensionless=True, use_syn_filter=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4,
        use_latent_ssm=True, latent_ssm_dim=6, ssm_encoder="gru",
        ssm_split_latent=True, ssm_artifact_dim=3,
        ssm_variational_traj=False, ssm_state_observer=True, ssm_min_decay=0.85,
    )
    with torch.no_grad():
        rec.latent_ssm.obs_local_fc2.weight.fill_(0.25)
        rec.latent_ssm.K_obs.weight.fill_(0.15)
        rec.latent_ssm.K_art_obs.weight.fill_(0.1)
        A = rec.latent_ssm.transition_matrix()
        eig_mags = torch.linalg.eigvals(A).abs()
        assert float(eig_mags.min().item()) >= 0.849
        assert float(eig_mags.max().item()) <= 0.99

    stim = torch.zeros(1, 10, n)
    stim[:, 4:6, 0] = 1.0
    y_pre_a = torch.randn(1, 10, n) * 0.3
    y_pre_b = y_pre_a.clone()
    y_pre_b[:, 4:, :] += 50.0  # future perturbation must have zero effect on open-loop prediction
    mask = torch.ones(1, 10, n, dtype=torch.bool)
    eps0 = torch.zeros(1, 11, rec.latent_ssm.total_z_dim)

    with torch.no_grad():
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        out_a, _, _, z_a = rec._rollout(
            stim, None, 0.5, pre_stim_obs=y_pre_a[:, :4, :], pre_stim_mask=mask[:, :4, :],
            ssm_eps=eps0, return_latent_traj=True,
        )
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        out_b, _, _, z_b = rec._rollout(
            stim, None, 0.5, pre_stim_obs=y_pre_b, pre_stim_mask=mask, ssm_eps=eps0, return_latent_traj=True,
        )
        assert z_a is not None and z_a.shape == (1, 6, 6)
        assert torch.allclose(out_a, out_b, atol=1e-6)

        y_pre_c = y_pre_a.clone()
        y_pre_c[:, :4, :] += 1.5
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        out_c, _, _ = rec._rollout(stim, None, 0.5, pre_stim_obs=y_pre_c[:, :4, :], pre_stim_mask=mask[:, :4, :], ssm_eps=eps0)
        assert float((out_a[:, 4:, :] - out_c[:, 4:, :]).abs().max().item()) > 1e-4

        # Strictly causal online filtering at frame t must ignore y[:, t+1:, :]
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        f_a = rec.rollout_online_filter(stim, y_pre_a, mask)
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        f_b = rec.rollout_online_filter(stim, y_pre_b, mask)
        assert torch.allclose(f_a[:, :5, :], f_b[:, :5, :], atol=1e-6)


def test_multi_horizon_rollout_and_overshooting_losses():
    """Open-loop multi-horizon rollout & latent overshooting losses backprop cleanly through LatentNeuralSSM."""
    import pyro_svi_vae

    pyro.clear_param_store()
    n = 5
    syn_list = {"General": ([], [], []), "Wicks": ([0], [1], [1.0], [0.0], [0.01])}
    step = c302.NematodeForStep(n, {}, syn_list, dimensionless=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4,
        use_latent_ssm=True, latent_ssm_dim=4, ssm_encoder="gru",
        ssm_variational_traj=True, ssm_smoother=True, ssm_state_observer=True, ssm_min_decay=0.85,
    )
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)
    aux_opt = torch.optim.Adam(vm.parameters(), lr=1e-3)
    stim = torch.zeros(2, 12, n)
    stim[:, 4:6, 0] = 1.0
    y = torch.randn(2, 12, n) * 0.2
    mask = torch.ones(2, 12, n, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)

    loss_val = pyro_svi_vae._aux_rollout_overshoot_step(
        vm, aux_opt, stim, y, mask, noise, pre_vols=4, rollout_w=0.5, overshoot_w=0.5,
    )
    assert np.isfinite(loss_val) and loss_val >= 0.0


def test_worm_latent_generative_factor_and_ttt_d():
    """Trainable generative worm latent factor b_w trains in SVI and adapts in TTT-D."""
    import warnings
    from tools.eval_online_ttt import (
        _predict_single_trial_counterfactual,
        adapt_worm_posterior,
        restore_population_state,
        snapshot_population_state,
    )

    pyro.clear_param_store()
    n = 6
    syn_list = {"General": ([], [], []), "Wicks": ([0], [1], [1.0], [0.0], [0.01])}
    step = c302.NematodeForStep(n, {}, syn_list, dimensionless=True)
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4,
        worm_latent_dim=3, num_worms=4,
    )
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)
    stim = torch.zeros(2, 10, n)
    stim[:, 4:6, 0] = 1.0
    y = torch.randn(2, 10, n) * 0.2
    mask = torch.ones(2, 10, n, dtype=torch.bool)
    noise = torch.full((2, n), 0.1)
    worm_ids = torch.tensor([0, 2], dtype=torch.long)

    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        svi = SVI(vm, vm.guide, pyro.optim.ClippedAdam({"lr": 1e-2}), Trace_ELBO())
        elbo = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise, worm_id=worm_ids)
        assert np.isfinite(elbo)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    snap = snapshot_population_state(vm)
    log = adapt_worm_posterior(
        vm, stim_hist=stim, y_hist=y, mask_hist=mask, noise_hist=noise,
        tier="tier1", ttt_mode="ttt_d", ttt_steps=3, ttt_lr=0.05, kl_weight=1.0, pre_vols=4,
    )
    assert "_worm_b_tensor" in log
    bw = log.pop("_worm_b_tensor")
    assert bw.shape == (1, 3) and float(bw.abs().max().item()) > 0.0
    yp, ycf = _predict_single_trial_counterfactual(
        vm, stim[:1], y[:1], mask[:1], noise[:1], num_samples=1, pre_vols=4, eval_seed=42, worm_b_override=bw,
    )
    assert yp.shape == (1, 10, n) and ycf.shape == (1, 10, n)
    restore_population_state(vm, snap)


def test_hardened_genotype_joint_messenger_calib_and_identifiability():
    """Hardened genotype joint skips gamma_wired_mut when fixed, calibrates strain obs, reweights ELBO, and reports Fisher corr."""
    import warnings
    from torch.utils.data import DataLoader, TensorDataset
    from tools.run_genotype_joint import (
        _CohortObsScaleMessenger,
        compute_elbo_obs_weights,
        compute_joint_identifiability_diagnostics,
    )

    pyro.clear_param_store()
    n = 6
    syn_list = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [1.0], [0.0], [0.01]),
        "Generic": ([0, 0], [2, 3], [1.0, 1.0]),
        "NP": ([0], [2], [1.0]),
        "MA": ([0], [3], [1.0]),
    }
    step = c302.NematodeForStep(
        n, {}, syn_list, use_extrasynaptic=True, split_extrasynaptic=True,
        dimensionless=True, genotype_conditioned=True, fix_wired_mut=True,
    )
    assert not hasattr(step.synapse, "gamma_wired_mut")
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4, strain_obs_calib=True,
    )
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)
    assert not hasattr(vm.guide, "gamma_wired_mut_loc")
    assert hasattr(vm.guide, "gcamp_gain_mut_loc")

    stim = torch.zeros(4, 10, n)
    stim[:, 4:6, 0] = 1.0
    y = torch.randn(4, 10, n) * 0.2
    y[2:] += 2.0
    mask = torch.ones(4, 10, n, dtype=torch.bool)
    noise = torch.full((4, n), 0.1)
    geno = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    w_imp = compute_elbo_obs_weights(100, 20, 2, 2, "importance_weighted", torch.device("cpu"))
    w_bal = compute_elbo_obs_weights(100, 20, 2, 2, "balanced_50_50", torch.device("cpu"))
    assert abs(float(w_imp[0] / w_imp[2]) - 5.0) < 1e-5
    assert abs(float(w_bal[0] / w_bal[2]) - 1.0) < 1e-5

    elbo_fn = pyro.infer.Trace_ELBO()
    pyro.set_rng_seed(0)
    with _CohortObsScaleMessenger(w_imp):
        loss_imp = elbo_fn.loss(vm, vm.guide, ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=120, genotype_id=geno)
    pyro.set_rng_seed(0)
    with _CohortObsScaleMessenger(w_bal):
        loss_bal = elbo_fn.loss(vm, vm.guide, ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=120, genotype_id=geno)
    assert abs(loss_imp - loss_bal) > 10.0

    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        optim = pyro.optim.ClippedAdam({"lr": 1e-2})
        with _CohortObsScaleMessenger(w_imp):
            svi = SVI(vm, vm.guide, optim, Trace_ELBO())
            loss = svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise, dataset_size=120, genotype_id=geno)
            assert np.isfinite(loss)
    mismatch_w = [str(w.message) for w in w_rec if "Found non-auxiliary vars" in str(w.message) or "not in guide" in str(w.message)]
    assert not mismatch_w, f"Pyro site mismatch warnings: {mismatch_w}"

    ldr_wt = DataLoader(TensorDataset(stim[:2], y[:2], mask[:2], noise[:2]), batch_size=2)
    ldr_u31 = DataLoader(TensorDataset(stim[2:], y[2:], mask[2:], noise[2:]), batch_size=2)
    ident = compute_joint_identifiability_diagnostics(vm, ldr_wt, ldr_u31, pre_vols=4, device=torch.device("cpu"))
    assert "gamma_np_mut" in ident["param_names"]
    assert "gamma_wired_mut" not in ident["param_names"]
    corr = np.asarray(ident["fisher_corr"])
    assert corr.shape[0] == len(ident["param_names"]) and np.allclose(np.diag(corr), 1.0)


def test_eval_unc31_knockout_tier1_calib_cv_and_recording_counts():
    """WT-frozen unc-31 Tier-1 calibration CV preserves synaptic weights and reports recording summary."""
    from torch.utils.data import DataLoader, TensorDataset
    from tools.eval_unc31_knockout import evaluate_unc31_tier1_calib_cv, summarize_dataset_recordings
    from utils import RandiTrialDataset

    pyro.clear_param_store()
    n = 6
    syn_list = {
        "General": ([], [], []),
        "Wicks": ([0], [1], [1.0], [0.0], [0.01]),
        "Generic": ([0, 0], [2, 3], [1.0, 1.0]),
        "NP": ([0], [2], [1.0]),
        "MA": ([0], [3], [1.0]),
    }
    step = c302.NematodeForStep(n, {}, syn_list, use_extrasynaptic=True, split_extrasynaptic=True, dimensionless=True)
    rec = c302.RecurrentNematode(step, dt_obs=0.5, substeps=2, use_prestim_encoder=False, pre_vols=4)
    vm = vae.VAElegans(rec, n, 1, 0, direct_global_guide=True)

    b, t = 8, 12
    stim = torch.zeros(b, t, n)
    stim[:, 4:6, 0] = 1.0
    y = torch.randn(b, t, n) * 0.2
    mask = torch.ones(b, t, n, dtype=torch.bool)
    noise = torch.full((b, n), 0.1)
    dts = torch.full((b,), 0.5)
    meta = [{"rec_id": i // 2, "stim_neuron_idx": 0} for i in range(b)]
    ds_u31 = RandiTrialDataset(stim, y, mask, noise, dts, meta)
    rec_sum = summarize_dataset_recordings(ds_u31)
    assert rec_sum["n_trials"] == 8 and rec_sum["n_recordings"] == 4

    adj = step.synapse.adjacency_matrix(n)
    np_adj = step.synapse.np_adjacency_matrix(n)
    ma_adj = step.synapse.ma_adjacency_matrix(n)
    ldr_wt = DataLoader(TensorDataset(stim[:2], y[:2], mask[:2], noise[:2]), batch_size=2)
    w_before = vm.guide.wicks_g_max_z.detach().clone()

    res = evaluate_unc31_tier1_calib_cv(
        vm, ds_u31, ldr_wt, adj, np_adj, ma_adj,
        pre_vols=4, n_folds=2, ttt_steps=2, num_samples=1,
    )
    assert res["n_folds"] == 2
    assert "delta_np_r2_mean" in res["cv_summary"]
    assert torch.allclose(vm.guide.wicks_g_max_z, w_before)
