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
    ]:
        py_compile.compile(fname, doraise=True)
