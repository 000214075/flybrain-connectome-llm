"""Tests for connectome loading, the neuron model, and the language model.

Run with: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import (  # noqa: E402
    Connectome,
    build_pathway,
    build_synthetic_pathway,
    shuffle_pathway,
)
from flybrain.model import Block, FlyBrainLM, ModelConfig, build_rope_cache  # noqa: E402
from flybrain.neurons import LIFLayer, k_wta, surrogate_spike  # noqa: E402
from flybrain.wholebrain import BrainPathway, WholeBrainCircuit, shuffled_edges  # noqa: E402


def tiny_connectome() -> Connectome:
    """6 neurons labelled the way MaleCNS labels them.

    10,11 = antennal lobe projection neurons, 20,21 = Kenyon cells,
    30 = MBON, 40 = APL, 50 = a SEZ projection neuron (must NOT count as a PN),
    60 = a Kenyon cell known only by its `type`.
    """
    ids = np.array([10, 11, 20, 21, 30, 40, 50, 60], dtype=np.int64)
    edges = [
        (10, 20, 5.0),
        (10, 21, 2.0),
        (11, 20, 3.0),
        (20, 30, 7.0),
        (21, 30, 4.0),
        (40, 20, 9.0),
        (40, 21, 1.0),
        (60, 30, 6.0),
        (50, 30, 8.0),
    ]
    index = {int(v): i for i, v in enumerate(ids)}
    pre = np.array([index[a] for a, _, _ in edges], dtype=np.int64)
    post = np.array([index[b] for _, b, _ in edges], dtype=np.int64)
    weight = np.array([w for _, _, w in edges], dtype=np.float32)
    labels = {
        10: ("alpn", "da1_lpn"),
        11: ("alpn", "da2_vpn"),
        20: ("kenyon_cell", "kcg-m"),
        21: ("kenyon_cell", "kcab-c"),
        30: ("mbon", "mbon01"),
        40: ("apl",),
        50: ("sezpn", "sez_projection"),
        60: ("kcg-s1",),
    }
    return Connectome(ids, pre, post, weight, labels, {}, name="tiny")


# --------------------------------------------------------------------------
# connectome
# --------------------------------------------------------------------------


def test_class_matching_and_adjacency():
    connectome = tiny_connectome()
    assert list(connectome.by_class("pn")) == [10, 11]
    assert list(connectome.by_class("kc")) == [20, 21, 60]
    assert list(connectome.by_class("mbon")) == [30]
    assert list(connectome.by_class("apl")) == [40]

    pn_to_kc = connectome.adjacency(
        connectome.by_class("pn"), np.array([20, 21]), normalise=None
    )
    assert pn_to_kc.shape == (2, 2)
    assert pn_to_kc[0, 0] == 5.0 and pn_to_kc[0, 1] == 2.0
    assert pn_to_kc[1, 0] == 3.0 and pn_to_kc[1, 1] == 0.0


def test_class_rules_exclude_things_that_merely_look_like_pns():
    connectome = tiny_connectome()
    # SEZ projection neurons are not antennal lobe PNs and must not be collected.
    assert 50 not in connectome.by_class("pn").tolist()
    # A Kenyon cell whose only label is its type is still a Kenyon cell.
    assert 60 in connectome.by_class("kc").tolist()
    assert 60 not in connectome.by_class("mbon").tolist()


def test_adjacency_normalisation_modes():
    connectome = tiny_connectome()
    pn = connectome.by_class("pn")
    kc = np.array([20, 21])  # the two Kenyon cells that receive PN input
    normalised = connectome.adjacency(pn, kc, normalise="pre")
    np.testing.assert_allclose(normalised.sum(axis=1), [1.0, 1.0], rtol=1e-6)
    by_post = connectome.adjacency(pn, kc, normalise="post")
    np.testing.assert_allclose(by_post.sum(axis=0), [1.0, 1.0], rtol=1e-6)


def test_min_weight_filters_synapses():
    connectome = tiny_connectome()
    pn = connectome.by_class("pn")
    kc = np.array([20, 21])
    pn_to_kc = connectome.adjacency(pn, kc, min_weight=1.0, normalise=None)
    strict = connectome.adjacency(pn, kc, min_weight=4.0, normalise=None)
    assert pn_to_kc[1, 0] == 3.0  # present at threshold 1
    assert strict[1, 0] == 0.0  # filtered at threshold 4


def test_build_pathway_uses_real_synapses():
    pathway = build_pathway(tiny_connectome())
    assert pathway.n_pn == 2
    assert pathway.n_kc == 3  # includes the type-only Kenyon cell
    assert pathway.kc_to_mbon.shape == (3, 1)
    # APL strength is normalised so the strongest Kenyon cell gets 1.0
    assert pathway.apl_to_kc.shape == (3,)
    assert pathway.apl_to_kc.max() == pytest.approx(1.0)
    assert pathway.meta["pn_to_kc_nnz"] == 3
    # 2 of the 3 Kenyon cells receive PN input; the meta value is rounded.
    assert pathway.meta["kc_connected_fraction"] == pytest.approx(2 / 3, abs=1e-4)


def test_synthetic_pathway_sparsity():
    pathway = build_synthetic_pathway(20, 100, 4, kc_fan_in=5, seed=0)
    assert pathway.pn_to_kc.shape == (20, 100)
    assert (pathway.pn_to_kc > 0).sum() == 100 * 5
    # Every KC receives the same number of inputs, so the fan-in is controlled.
    assert set((pathway.pn_to_kc > 0).sum(axis=0).tolist()) == {5}


def test_shuffle_preserves_degree_but_changes_wiring():
    pathway = build_synthetic_pathway(30, 200, 6, kc_fan_in=4, seed=1)
    shuffled = shuffle_pathway(pathway, seed=7)
    # The control must be the same size and density, only rewired.
    assert (shuffled.pn_to_kc > 0).sum() == (pathway.pn_to_kc > 0).sum()
    np.testing.assert_array_equal(
        (shuffled.pn_to_kc > 0).sum(axis=1), (pathway.pn_to_kc > 0).sum(axis=1)
    )
    assert (shuffled.kc_to_mbon > 0).sum() == (pathway.kc_to_mbon > 0).sum()
    assert not np.array_equal(shuffled.pn_to_kc, pathway.pn_to_kc)
    assert not np.array_equal(shuffled.kc_to_mbon, pathway.kc_to_mbon)


def test_whole_brain_shuffle_keeps_every_degree_and_rejects_bad_edges():
    """The whole-brain control must differ from the connectome *only* in wiring.

    Synapse counts are reused unchanged by `scripts/shuffle_wholebrain.py`, and
    `build_wholebrain.py` normalised them by postsynaptic in-degree, so a control
    that changed either degree sequence would not be comparable at all.
    """
    rng = np.random.default_rng(0)
    n = 60
    pairs = set()
    while len(pairs) < 400:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            pairs.add((a, b))
    pre = np.array([a for a, _ in sorted(pairs)], dtype=np.int64)
    post = np.array([b for _, b in sorted(pairs)], dtype=np.int64)
    # Every neuron needs a nonzero out-degree, otherwise a swap can leave a cell
    # with nothing to send and the test would trivially hold.
    assert np.bincount(pre, minlength=n).min() > 0

    shuffled = shuffled_edges(pre, post, n, seed=5, rounds=4, proposals=2_000)
    np.testing.assert_array_equal(np.sort(post), np.sort(shuffled))
    np.testing.assert_array_equal(
        np.bincount(post, minlength=n), np.bincount(shuffled, minlength=n)
    )
    # Only the postsynaptic side moves, so out-degrees and the sign each neuron
    # transmits with (both indexed by `pre`) cannot change.
    assert shuffled.shape == post.shape
    assert not np.array_equal(post, shuffled)
    assert (pre == shuffled).sum() == 0, "a shuffle must not create self-loops"
    flat = pre * n + shuffled
    assert np.unique(flat).size == flat.size, "a shuffle must not create parallel edges"
    # Same seed, same control.
    np.testing.assert_array_equal(shuffled, shuffled_edges(pre, post, n, seed=5, rounds=4, proposals=2_000))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_whole_brain_shuffle_stays_simple_on_a_dense_graph(seed):
    """Dense graphs are where two swaps can propose the same replacement edge.

    Rejecting each proposal on its own is not enough there: swap A and swap B in
    the same round do not see each other's new edge, so the round has to be
    de-duplicated as a whole. Running many proposals per round is what makes the
    collision likely.
    """
    rng = np.random.default_rng(seed)
    n = 40
    pairs = set()
    while len(pairs) < 600:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            pairs.add((a, b))
    pre = np.array([a for a, _ in sorted(pairs)], dtype=np.int64)
    post = np.array([b for _, b in sorted(pairs)], dtype=np.int64)

    shuffled = shuffled_edges(pre, post, n, seed=seed, rounds=6, proposals=8_000)
    flat = pre * n + shuffled
    assert np.unique(flat).size == flat.size, "a shuffle must not create parallel edges"
    assert (pre == shuffled).sum() == 0, "a shuffle must not create self-loops"
    np.testing.assert_array_equal(
        np.bincount(post, minlength=n), np.bincount(shuffled, minlength=n)
    )


def test_pathway_roundtrip(tmp_path):
    pathway = build_synthetic_pathway(12, 40, 3, seed=3)
    path = str(tmp_path / "pathway.npz")
    pathway.save(path)
    from flybrain.connectome import Pathway

    loaded = Pathway.load(path)
    np.testing.assert_array_equal(loaded.pn_to_kc, pathway.pn_to_kc)
    np.testing.assert_array_equal(loaded.kc_to_mbon, pathway.kc_to_mbon)
    assert loaded.n_kc == pathway.n_kc


# --------------------------------------------------------------------------
# neurons
# --------------------------------------------------------------------------


def test_surrogate_spike_forward_is_binary_and_backward_is_finite():
    v = torch.linspace(-2, 2, 41, requires_grad=True)
    spikes = surrogate_spike(v)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    spikes.sum().backward()
    assert torch.isfinite(v.grad).all()
    assert (v.grad >= 0).all()  # a larger potential must not reduce spiking


def test_k_wta_keeps_exactly_k():
    x = torch.randn(5, 20)
    out = k_wta(x, 4)
    assert (out != 0).sum(dim=-1).tolist() == [4] * 5
    # The surviving values are the largest of each row.
    assert torch.allclose(out.sort(dim=-1).values[:, -4:], x.sort(dim=-1).values[:, -4:])


def test_lif_integrates_and_fires_sparsely():
    lif = LIFLayer(32, steps=4, tau_m=5.0)
    weak = torch.full((2, 32), 0.05)
    strong = torch.full((2, 32), 5.0)
    weak_spikes = lif(weak)
    strong_spikes = lif(strong)
    weak_mean = float(weak_spikes.detach().mean())
    strong_mean = float(strong_spikes.detach().mean())
    assert weak_spikes.max() <= 1.0  # a sub-threshold current fires at most once
    assert strong_mean > weak_mean
    assert strong_spikes.max() <= 4.0  # at most one spike per step


def test_lif_non_spiking_mode_is_a_saturating_rate():
    """The rate path must be bounded, or a recurrent loop using it explodes.

    It used to return `relu(current)`. In a recurrent connectome that is positive
    feedback with no brake: measured on the connectome-only model, the loss reached
    6.2e5 with a gradient norm of 1.9e8 inside fifty steps.
    """
    lif = LIFLayer(8, steps=3)
    current = torch.randn(2, 8) * 40.0
    out = lif(current, spiking=False)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0
    # Still monotone in the input: a bigger drive means a higher rate.
    bigger = lif(current + 1.0, spiking=False)
    assert bool((bigger >= out).all())


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["mb", "mb-shuffled", "swiglu"])
def test_model_forward_backward(mode):
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, context=16, ffn_mode=mode, mb_steps=2
    )
    pathway = build_synthetic_pathway(8, 40, 3, kc_fan_in=3, seed=0)
    model = FlyBrainLM(cfg, None if mode == "swiglu" else pathway)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert torch.isfinite(loss)
    loss.backward()
    # Every learnable projection must receive gradient, including the KC output
    # path, otherwise the connectome module would be silently frozen.
    for name in ("pn_proj.weight", "mbon_proj.weight", "raw_apl_gain"):
        if mode == "swiglu":
            continue
        param = dict(model.named_parameters())[f"blocks.0.ffn.{name}"]
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all()


def test_kv_cache_matches_full_forward():
    """Incremental decoding must reproduce the single-pass logits exactly."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=48, d_model=32, n_layers=2, n_heads=4, context=32, ffn_mode="mb", mb_steps=1
    )
    pathway = build_synthetic_pathway(8, 24, 3, kc_fan_in=3, seed=0)
    model = FlyBrainLM(cfg, pathway).eval()

    tokens = torch.randint(0, 48, (1, 12))
    with torch.no_grad():
        full_logits = model(tokens)

        # Feed the same tokens one at a time, keeping the cache.
        cache = None
        step_logits = []
        for i in range(tokens.shape[1]):
            logits, cache = model(tokens[:, i : i + 1], cache=cache, use_cache=True)
            step_logits.append(logits[:, -1, :])
        incremental = torch.stack(step_logits, dim=1)

    torch.testing.assert_close(incremental, full_logits, rtol=2e-3, atol=2e-3)


def test_rope_offset_matches_positional_encoding():
    """A chunk decoded at an offset must match the same chunk in a full pass."""
    torch.manual_seed(1)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=16, ffn_mode="swiglu")
    model = FlyBrainLM(cfg, None).eval()
    tokens = torch.randint(0, 32, (1, 10))
    with torch.no_grad():
        full = model(tokens)
        _, cache = model(tokens[:, :6], use_cache=True)
        tail, _ = model(tokens[:, 6:], cache=cache, use_cache=True)
    torch.testing.assert_close(tail, full[:, 6:], rtol=2e-3, atol=2e-3)


def test_connectome_wiring_is_frozen():
    """Training must not modify the connectome matrices themselves."""
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=8, ffn_mode="mb")
    pathway = build_synthetic_pathway(8, 20, 2, kc_fan_in=3, seed=0)
    model = FlyBrainLM(cfg, pathway)
    layer = model.blocks[0].ffn
    before = layer.pn_to_kc.clone()
    trainable = {id(p) for p in model.parameters()}
    assert id(layer.pn_to_kc) not in trainable
    x = torch.randint(0, 32, (1, 8))
    _, loss = model(x, torch.randint(0, 32, (1, 8)))
    loss.backward()
    assert torch.equal(before, layer.pn_to_kc)


def test_config_json_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=3, mb_max_kc=100)
    path = str(tmp_path / "model.json")
    cfg.to_json(path)
    loaded = ModelConfig.from_json(path)
    assert loaded == cfg


def test_context_overflow_is_rejected():
    cfg = ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, context=8, ffn_mode="swiglu")
    model = FlyBrainLM(cfg, None)
    with pytest.raises(ValueError, match="exceeds context"):
        model(torch.randint(0, 16, (1, 9)))


# --------------------------------------------------------------------------
# whole brain circuit
# --------------------------------------------------------------------------


def tiny_brain_pathway(
    n_neurons: int = 24, d_model: int = 32, *, chunks: int = 1, iters: int = 2, rank: int = 8
):
    """A small brain circuit with real inhibition, for exercising the module."""
    rng = np.random.default_rng(0)
    sources = rng.integers(0, n_neurons, size=80)
    targets = rng.integers(0, n_neurons, size=80)
    keep = sources != targets
    pre = torch.from_numpy(sources[keep].astype(np.int64))
    post = torch.from_numpy(targets[keep].astype(np.int64))
    weight = torch.full((int(keep.sum()),), 0.05, dtype=torch.float32)
    sign = torch.ones(n_neurons, dtype=torch.float32)
    sign[::3] = -1.0  # every third neuron is inhibitory
    circuit = WholeBrainCircuit(pre, post, weight, sign)
    pathway = BrainPathway(circuit, d_model, rank=rank, chunks=chunks, iters=iters)
    return pathway


def test_whole_brain_is_alive_at_initialisation():
    """The circuit must actually fire before training.

    A silent circuit is a dead one: the surrogate spike gradient is only non-zero
    near threshold, so if the initial drive landed below it the brain would never
    receive any gradient and would sit inert inside the model forever.
    """
    torch.manual_seed(0)
    pathway = tiny_brain_pathway(n_neurons=64, d_model=32, chunks=2, iters=2)
    x = torch.randn(4, 8, 32)
    _, state = pathway.forward_sequence(x, chunk_size=4)
    stats = pathway.activity_stats(state)
    assert stats["firing_fraction"] > 0.0, f"the whole brain is silent: {stats}"
    assert stats["max_state"] > 0.0

    # A rate state is in (0, 1] for every neuron including silent ones, so "fraction
    # above zero" would read 100% no matter what the circuit did. The rate mode has
    # to report something that can actually fall.
    pathway.spiking = False
    with torch.no_grad():
        rate = torch.zeros(4, 64)
    rate_stats = pathway.activity_stats(rate)
    assert "firing_fraction" not in rate_stats
    assert rate_stats["half_active_fraction"] == 0.0


def test_whole_brain_touches_every_neuron():
    pathway = tiny_brain_pathway(n_neurons=24)
    x = torch.randn(2, 8, 32)
    out, state = pathway.forward_sequence(x, chunk_size=1)
    assert out.shape == x.shape
    # The state has an entry for every neuron in the connectome, not a subset.
    assert state.shape == (2, 24)
    # The synaptic sweep reads and writes the whole population.
    activity = torch.rand(2, 24)
    current = pathway.circuit.synaptic_current(activity, pathway.circuit.effective_synapses())
    assert current.shape == activity.shape
    assert torch.isfinite(current).all()


def test_whole_brain_is_causal():
    """No token may be influenced by a later token through the brain.

    The brain reads out before folding in each chunk, so with a chunk size of one
    a token's readout depends only on the tokens before it. If the order were
    reversed this test would fail, and the model would be scoring itself on tokens
    it had already seen.
    """
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=16, ffn_mode="swiglu")
    brain = tiny_brain_pathway(n_neurons=24, d_model=32, chunks=16, iters=2)
    model = FlyBrainLM(cfg, None, brain).eval()

    tokens = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        baseline = model(tokens)

    perturbed = tokens.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 7) % 32  # change only the last token
    with torch.no_grad():
        changed = model(perturbed)

    # Every position before the perturbation must be bit-identical.
    torch.testing.assert_close(changed[:, :-1], baseline[:, :-1], rtol=0, atol=0)
    assert not torch.equal(changed[:, -1], baseline[:, -1])


def test_whole_brain_cache_matches_full_forward():
    """With one token per chunk, incremental decoding must match a full pass."""
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=16, ffn_mode="swiglu")
    brain = tiny_brain_pathway(n_neurons=20, d_model=32, chunks=16, iters=2)
    model = FlyBrainLM(cfg, None, brain).eval()

    tokens = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        full = model(tokens)
        cache = None
        steps = []
        for i in range(tokens.shape[1]):
            logits, cache = model(tokens[:, i : i + 1], cache=cache, use_cache=True)
            steps.append(logits[:, -1, :])
        incremental = torch.stack(steps, dim=1)

    torch.testing.assert_close(incremental, full, rtol=2e-3, atol=2e-3)


def test_whole_brain_inhibition_is_signed_and_not_trainable():
    """The transmitter-derived sign is fixed; only the magnitude may be learned."""
    pathway = tiny_brain_pathway(n_neurons=12)
    circuit = pathway.circuit
    before = circuit.sign.clone()
    assert (circuit.sign < 0).any() and (circuit.sign > 0).any()

    x = torch.randn(2, 4, 32)
    out, _ = pathway.forward_sequence(x, chunk_size=1)
    out.sum().backward()

    assert torch.equal(before, circuit.sign)  # a buffer, untouched by the optimizer
    assert circuit.sign.requires_grad is False
    assert circuit.raw_gain.grad is not None  # magnitude is learned
    assert torch.isfinite(circuit.raw_gain.grad).all()


def test_the_control_checker_rejects_a_run_whose_schedule_differs(tmp_path):
    """A matched-steps comparison is not a matched comparison.

    This project drew a wrong conclusion from exactly this mistake: two sweep kernels
    were compared "at matched steps", but their arms had different `max_steps`
    (600 vs 150), so the cosine schedules diverged right after warmup -- at step 150
    one arm was at lr 2.8e-5 and the other at its 6.0e-6 floor. The whole 0.267 gap
    was the schedule, not the kernel, and it took a second look to catch.

    `scripts/compare_wiring_arms.py` exists to catch it (it did, printing
    `NOT A MATCHED CONTROL`), so its allow-list is pinned here: the wiring it is
    ablating and the bookkeeping fields may differ, and *nothing else* may.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "compare_wiring_arms", Path(__file__).resolve().parent.parent / "scripts" / "compare_wiring_arms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = {
        "run_name": "a",
        "out_dir": "checkpoints/a",
        "lr": 3e-5,
        "warmup_steps": 40,
        "max_steps": 600,
        "model": {"brain_path": "data/wholebrain.npz", "brain_impl": "csr"},
    }

    def written(tmp_path, name, **overrides):
        import json

        payload = json.loads(json.dumps(base))
        for key, value in overrides.items():
            if key == "brain_path":
                payload["model"]["brain_path"] = value
            elif key == "brain_impl":
                payload["model"]["brain_impl"] = value
            else:
                payload[key] = value
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    # Differing only in the thing under test, plus bookkeeping: a real control.
    left = written(tmp_path, "left")
    right = written(tmp_path, "right", brain_path="data/wholebrain_shuffled.npz", out_dir="checkpoints/b")
    assert module.check_matched(left, right) == []

    # A different schedule is a different experiment, however equal the step counts
    # one chooses to compare at.
    schedule = written(tmp_path, "schedule", max_steps=150)
    problems = module.check_matched(left, schedule)
    assert problems, "a differing max_steps made the comparison non-matched and went unreported"
    assert any("max_steps" in item for item in problems)

    # ...and so is a different learning rate, or a different kernel when the point of
    # the comparison is not the kernel.
    assert module.check_matched(left, written(tmp_path, "lr", lr=1e-4))
    assert module.check_matched(left, written(tmp_path, "impl", brain_impl="index_add"))


def test_both_sweep_implementations_agree_when_given_the_right_layout():
    """`prepare_synapses`, not `effective_synapses`, is what a sweep consumes.

    The connectome has two sweep implementations, a gather/scatter and a sparse CSR
    matmul. The CSR structure is sorted by *postsynaptic* neuron, so the values it is
    given must be permuted into that same edge order -- which is what
    `prepare_synapses` does and `effective_synapses` does not. Handing the CSR path
    the un-permuted values silently applies each edge's weight, transmitter sign and
    release gain to a *different* edge; measured on the real connectome, the two paths
    then disagree on 99.9% of the population. Both language models route their sweeps
    through `advance_drive`, which passes `prepare_synapses`, so this is the contract
    they depend on -- and getting it wrong looks exactly like a working model.
    """
    torch.manual_seed(11)
    brain = tiny_brain_pathway(n_neurons=40, d_model=32, chunks=1, iters=1)
    circuit = brain.circuit
    state = torch.rand(3, 40)
    drive = torch.randn(3, 40)

    outputs = {}
    for impl in ("index_add", "csr"):
        circuit.impl = impl
        circuit.zero_grad(set_to_none=True)
        with torch.no_grad():
            outputs[impl] = circuit.step(
                state, drive, circuit.prepare_synapses(), spiking=False
            ).clone()

    # Without this the test could pass on a graph whose edge order happens to be
    # already sorted, and would then be testing nothing.
    order = torch.argsort(circuit.post.long(), stable=True)
    assert not torch.equal(order, torch.arange(order.numel())), (
        "this circuit's edges are already in CSR order; the layout question is moot"
    )

    torch.testing.assert_close(outputs["index_add"], outputs["csr"], rtol=1e-5, atol=1e-5)


def test_the_transposed_csr_structure_pairs_each_edge_with_its_own_value():
    """The transpose must be built in the same edge order as the values fed to it.

    `_ConnectomeSweep` keeps a second CSR tensor for W^T and hands it `values[order_t]`,
    where `values` has been permuted by `prepare_synapses`. Building `order_t` from the
    raw `pre` array instead of the permuted one pairs every edge's value with a
    different edge. The forward pass is unaffected and so is the gradient onto
    `raw_gain` -- only the gradient that flows back into the incoming population state
    is wrong, and only from the second token onward. That combination is why it can
    train to a checkpoint without ever looking broken.
    """
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    n_neurons = 24
    sources = rng.integers(0, n_neurons, size=160)
    targets = rng.integers(0, n_neurons, size=160)
    keep = sources != targets
    circuit = WholeBrainCircuit(
        torch.from_numpy(sources[keep].astype(np.int64)),
        torch.from_numpy(targets[keep].astype(np.int64)),
        torch.rand(int(keep.sum())),
        torch.where(torch.rand(n_neurons) < 0.3, -1.0, 1.0),
    )
    circuit._ensure_csr()

    # The connectome as a dense matrix, straight from the raw edge list.
    dense = torch.zeros(n_neurons, n_neurons)
    dense.index_put_(
        (circuit.post.long(), circuit.pre.long()),
        circuit.weight * circuit.sign[circuit.pre.long()],
        accumulate=True,
    )
    base = (circuit.weight_perm * circuit.sign[circuit.pre_perm]).detach()
    transpose = torch.sparse_csr_tensor(
        circuit.csr_crow_t,
        circuit.csr_col_t,
        base[circuit.order_t],
        size=(n_neurons, n_neurons),
        check_invariants=False,
    )
    torch.testing.assert_close(transpose.to_dense(), dense.t(), rtol=1e-6, atol=1e-6)

    # The gradient this path produces, against the same dense arithmetic. Without the
    # `requires_grad_` the sweep skips the transpose entirely and the test is vacuous.
    # The matrix carries the release gain as well as the counts, since the values the
    # sweep is given do.
    values = circuit.prepare_synapses().detach()
    dense_gain = torch.zeros(n_neurons, n_neurons)
    dense_gain.index_put_(
        (circuit.post_perm.long(), circuit.pre_perm.long()), values, accumulate=True
    )
    activity = torch.rand(3, n_neurons)
    grad_out = torch.randn(3, n_neurons)
    act = activity.clone().requires_grad_(True)
    out = circuit.synaptic_current_csr(act, values)
    (out * grad_out).sum().backward()
    torch.testing.assert_close(
        act.grad, (dense_gain.t() @ grad_out.t()).t(), rtol=1e-5, atol=1e-5
    )


def test_the_scaled_sweep_matches_the_shipped_one_on_both_gradients():
    """Moving the release gain off the edges must change nothing but the cost.

    An edge's value is `weight * sign[pre] * release_gain[pre]`, and `weight` and
    `sign` are frozen, so `gain[pre]` can be applied to the activity instead of to
    the edges. The scaled kernel relies on that to keep its sparse matrix constant,
    which is what lets it skip both the per-sweep rebuild and the per-edge gradient.
    The two routes therefore have to be the same function, including the gradient
    that flows through the incoming population state -- the part the forward pass
    cannot check. Both gradients are asserted, because the scaled path computes them
    by a different route than the one it replaces.
    """
    torch.manual_seed(23)
    rng = np.random.default_rng(23)
    n_neurons = 32
    sources = rng.integers(0, n_neurons, size=220)
    targets = rng.integers(0, n_neurons, size=220)
    keep = sources != targets
    circuit = WholeBrainCircuit(
        torch.from_numpy(sources[keep].astype(np.int64)),
        torch.from_numpy(targets[keep].astype(np.int64)),
        torch.rand(int(keep.sum())),
        torch.where(torch.rand(n_neurons) < 0.3, -1.0, 1.0),
    )

    state = torch.rand(3, n_neurons)
    drive = torch.randn(3, n_neurons)
    out_weight = torch.randn(3, n_neurons)

    for spiking in (False, True):
        results = {}
        for impl in ("csr", "csr-scaled"):
            circuit.impl = impl
            circuit.zero_grad(set_to_none=True)
            act = state.clone().requires_grad_(True)
            synapses = circuit.prepare_synapses() if circuit.needs_edge_values else None
            out = circuit.step(act, drive, synapses, spiking=spiking)
            (out * out_weight).sum().backward()
            results[impl] = (out.detach(), act.grad, circuit.raw_gain.grad)

        # A gain gradient of all zeros would satisfy a loose comparison without
        # saying anything, so the two gradient tensors are required to be live.
        assert float(results["csr"][2].abs().max()) > 0
        for got, want in zip(results["csr-scaled"], results["csr"]):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def _port_circuit(n_neurons: int = 24, n_edges: int = 160, seed: int = 7) -> WholeBrainCircuit:
    g = torch.Generator().manual_seed(seed)
    pre = torch.randint(0, n_neurons, (n_edges,), generator=g)
    post = torch.randint(0, n_neurons, (n_edges,), generator=g)
    weight = torch.rand(n_edges, generator=g) + 0.1
    sign = torch.where(torch.rand(n_neurons, generator=g) < 0.3, -1.0, 1.0)
    return WholeBrainCircuit(pre, post, weight, sign, impl="csr")


def _write_mask(path, mask: np.ndarray) -> str:
    np.savez_compressed(path, mask=mask.astype(bool), neuron_ids=np.arange(mask.size))
    return str(path)


def test_absent_port_masks_leave_the_model_bit_identical(tmp_path):
    """The `all` arm of the port experiment must be the model that already exists.

    Not "close to" it: a mask of ones multiplied in would be a different graph, and
    every number produced before the masks existed would stop being comparable.
    """
    from flybrain.wholebrain import BrainPathway

    circuit = _port_circuit()
    torch.manual_seed(3)
    plain = BrainPathway(circuit, 8, rank=4, chunks=1, iters=1, spiking=False)
    torch.manual_seed(3)
    same = BrainPathway(circuit, 8, rank=4, chunks=1, iters=1, spiking=False)

    tokens = torch.randn(2, 3, 8)
    torch.manual_seed(11)
    a = plain.drive_from_tokens(tokens)
    torch.manual_seed(11)
    b = same.drive_from_tokens(tokens)
    assert torch.equal(a, b), "two identically-configured pathways disagree"
    # Rebuild the drive by hand: with no mask the pipeline must reduce to exactly
    # this expression, so the comparison is against the definition, not against
    # another call of the same function.
    reference = plain.norm_drive(
        plain.up(plain.down(plain.norm_in(tokens.reshape(6, -1))))
    ) * plain.drive_scale
    assert torch.equal(a, reference.reshape(2, 3, -1)), "the unmasked drive is not the plain pipeline"

    # And the read half, which is masked at a different place.
    state = torch.rand(2, circuit.n_neurons)
    assert torch.equal(plain.read_latent(state), plain.read_down(plain.read_state(state)))

    # A pathway given no mask must not have gained a buffer, or an old checkpoint
    # would stop loading.
    assert "input_mask" not in plain._buffers
    assert "output_mask" not in plain._buffers


def test_input_ports_zero_the_drive_everywhere_else(tmp_path):
    """Text must reach the port neurons and nobody else."""
    from flybrain.wholebrain import BrainPathway

    n = 24
    circuit = _port_circuit(n_neurons=n)
    mask = np.zeros(n, dtype=bool)
    mask[[1, 4, 9]] = True
    path = _write_mask(tmp_path / "in.npz", mask)

    torch.manual_seed(3)
    masked = BrainPathway(
        circuit, 8, rank=4, chunks=1, iters=1, spiking=False, input_mask_path=path
    )
    drive = masked.drive_from_tokens(torch.randn(2, 3, 8))
    assert drive.shape == (2, 3, n)
    off = drive[..., ~torch.from_numpy(mask)]
    assert float(off.abs().max()) == 0.0, "a non-port neuron received drive"
    on = drive[..., torch.from_numpy(mask)]
    assert float(on.abs().max()) > 0.0, "the port neurons received nothing"


def test_output_ports_hide_everything_else_from_the_read_out(tmp_path):
    """Changing a non-port neuron's state must not change what the model reads."""
    from flybrain.wholebrain import BrainPathway

    n = 24
    circuit = _port_circuit(n_neurons=n)
    mask = np.zeros(n, dtype=bool)
    mask[[2, 5, 17]] = True
    path = _write_mask(tmp_path / "out.npz", mask)

    torch.manual_seed(3)
    masked = BrainPathway(
        circuit, 8, rank=4, chunks=1, iters=1, spiking=False, output_mask_path=path
    )
    state = torch.rand(2, n)
    before = masked.read_latent(state)
    state[:, ~torch.from_numpy(mask)] += 5.0
    after = masked.read_latent(state)
    assert torch.equal(before, after), "the read-out reacted to a hidden neuron"


def test_a_mask_of_the_wrong_length_is_rejected_rather_than_broadcast(tmp_path):
    """A misaligned mask would silently index the wrong neurons."""
    from flybrain.wholebrain import BrainPathway, load_port_mask

    path = _write_mask(tmp_path / "short.npz", np.zeros(5, dtype=bool))
    with pytest.raises(ValueError, match="expected"):
        load_port_mask(path, 24)
    with pytest.raises(ValueError, match="expected"):
        BrainPathway(
            _port_circuit(), 8, rank=4, chunks=1, iters=1, spiking=False, input_mask_path=path
        )


def test_port_masks_stay_out_of_the_state_dict(tmp_path):
    """Otherwise every existing checkpoint would fail to load."""
    from flybrain.wholebrain import BrainPathway

    n = 24
    mask = np.zeros(n, dtype=bool)
    mask[:4] = True
    path = _write_mask(tmp_path / "m.npz", mask)
    torch.manual_seed(3)
    masked = BrainPathway(
        _port_circuit(n_neurons=n), 8, rank=4, chunks=1, iters=1, spiking=False,
        input_mask_path=path, output_mask_path=path,
    )
    keys = set(masked.state_dict())
    assert not any("mask" in k for k in keys), keys


def test_the_connectome_edges_are_not_in_the_optimiser():
    """The real synapse counts must survive training untouched.

    `weight` holds the measured contact count per edge, so it is a buffer rather
    than a parameter: what the fly's wiring *is* is not something the optimiser is
    allowed to move. The unknown biophysics -- how much each cell releases -- is
    modelled separately by the per-neuron `raw_gain`. That split is what makes
    "the trained model is the connectome" a checkable statement rather than a
    slogan, so it is asserted in three places: the tensor, the module's parameter
    list, and the optimiser's parameter groups.
    """
    from flybrain.train import TrainConfig, build_optimizer

    pathway = tiny_brain_pathway(n_neurons=12)
    circuit = pathway.circuit
    assert circuit.weight.requires_grad is False
    assert not any(name.startswith("weight") for name, _ in circuit.named_parameters())

    optimizer = build_optimizer(pathway, TrainConfig(lr=1e-3))
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(circuit.weight) not in optimized

    before = circuit.weight.clone()
    x = torch.randn(2, 4, 32)
    out, _ = pathway.forward_sequence(x, chunk_size=1)
    out.sum().backward()
    optimizer.step()
    assert torch.equal(before, circuit.weight), "the connectome's counts were modified"


def test_brain_cannot_be_silenced_by_the_optimiser():
    """Drive, integration rate and release magnitude are floored.

    Left unconstrained, training drives these to zero within a few hundred steps
    (measured on the real model: 15.9% firing down to 0.7%), because suppressing a
    noisy-looking module is cheaper than learning to use it. The whole connectome
    then stops participating even though the code still "runs" it.
    """
    pathway = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=2, iters=2)
    with torch.no_grad():
        pathway.raw_drive_scale.fill_(-50.0)  # softplus(-50) ~ 0
        pathway.circuit.raw_gain.fill_(-50.0)
        pathway.circuit.raw_step.fill_(-50.0)

    assert float(pathway.drive_scale) >= pathway.drive_floor
    assert float(pathway.circuit.release_gain.min()) >= pathway.circuit.gain_floor
    assert float(pathway.circuit.step_size) >= pathway.circuit.step_floor

    # With every factor pushed to its floor the population must still fire.
    _, state = pathway.forward_sequence(torch.randn(4, 8, 32), chunk_size=4)
    assert float(state.max()) > 0.0, "the floored circuit still went silent"


def test_brain_activity_is_exposed_for_regularisation():
    """The trainer needs the firing fraction as a differentiable quantity."""
    pathway = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=2, iters=2)
    x = torch.randn(2, 8, 32)
    out, _ = pathway.forward_sequence(x, chunk_size=4)
    assert pathway.last_activity is not None
    assert pathway.last_activity.requires_grad, "the activity term must carry gradient"
    assert 0.0 <= float(pathway.last_activity.detach()) <= 1.0
    out.sum().backward()
    assert torch.isfinite(pathway.circuit.raw_gain.grad).all()


def test_brain_has_a_learned_resting_state():
    """Every token must receive a brain read-out, including the first chunk.

    Starting the connectome from zeros makes the first chunk's read-out exactly
    zero, so those tokens get nothing from the brain at all. A learned spontaneous
    activity gives them a real, trainable contribution instead.
    """
    pathway = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=2, iters=1)
    rest = pathway.initial_state(4, torch.device("cpu"), torch.float32)
    assert rest.shape == (4, 32)
    assert (rest > 0).all() and (rest < 1).all(), "resting activity is a firing probability"

    readout = pathway._read(rest)
    assert readout.abs().sum() > 0, "the first chunk would receive nothing from the brain"
    assert readout.requires_grad, "the resting state must be learnable"

    readout.sum().backward()
    assert pathway.raw_rest.grad is not None
    assert pathway.raw_rest.grad.abs().sum() > 0


def test_whole_brain_gradients_reach_the_transformer():
    """The brain must actually influence the loss, not sit beside it."""
    torch.manual_seed(5)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=8, ffn_mode="swiglu")
    brain = tiny_brain_pathway(n_neurons=20, d_model=32, chunks=2, iters=2)
    model = FlyBrainLM(cfg, None, brain)
    x = torch.randint(0, 32, (2, 8))
    _, loss = model(x, torch.randint(0, 32, (2, 8)))
    loss.backward()
    for name in ("brain.down.weight", "brain.up.weight", "brain.read_down.weight", "brain.read_up.weight"):
        param = dict(model.named_parameters())[name]
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} has a zero gradient"


def test_brain_in_layers_modulates_every_block():
    """Interleaved, the brain reaches every layer -- appended, it reaches none.

    Placed after the stack the connectome's output only enters the language-model
    head, so all 164,587 neurons run without any transformer layer ever being
    affected by them. The utilisation number that matters is the share of the
    network the brain actually drives.
    """
    cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=3,
        n_heads=4,
        context=8,
        ffn_mode="swiglu",
        brain_in_layers=True,
        brain_chunks=2,
    )
    torch.manual_seed(8)
    brain = tiny_brain_pathway(n_neurons=24, d_model=32, chunks=2, iters=1)
    model = FlyBrainLM(cfg, None, brain).eval()

    seen: list = []
    handles = [
        block.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs.get("bias")), with_kwargs=True
        )
        for block in model.blocks
    ]
    try:
        with torch.no_grad():
            model(torch.randint(0, 32, (2, 8)))
    finally:
        for handle in handles:
            handle.remove()

    # 3 blocks x 2 chunks, and every one of them was handed a live brain read-out.
    assert len(seen) == 6, f"expected the brain at every block, saw {len(seen)} calls"
    assert all(bias is not None for bias in seen), "a block ran without the brain"
    assert all(float(bias.abs().sum()) > 0 for bias in seen), "the brain bias was identically zero"


def test_interleaved_brain_matches_incremental_decoding():
    """Chunking the stack must not change what the model computes.

    Threading the key/value state across chunks makes the split exact: attention is
    causal, so a chunk can only see what came before it anyway. That is what lets
    training interleave the connectome with every block without changing the model.
    """
    torch.manual_seed(9)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        context=16,
        ffn_mode="swiglu",
        brain_in_layers=True,
        brain_chunks=16,
    )
    brain = tiny_brain_pathway(n_neurons=24, d_model=32, chunks=16, iters=1)
    model = FlyBrainLM(cfg, None, brain).eval()

    tokens = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        full = model(tokens)
        cache = None
        steps = []
        for i in range(tokens.shape[1]):
            logits, cache = model(tokens[:, i : i + 1], cache=cache, use_cache=True)
            steps.append(logits[:, -1, :])
    torch.testing.assert_close(torch.stack(steps, dim=1), full, rtol=1e-4, atol=1e-5)


def test_interleaved_brain_stays_causal():
    """A chunk must be read out before it is folded into the brain."""
    torch.manual_seed(10)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        context=16,
        ffn_mode="swiglu",
        brain_in_layers=True,
        brain_chunks=8,
    )
    brain = tiny_brain_pathway(n_neurons=24, d_model=32, chunks=8, iters=1)
    model = FlyBrainLM(cfg, None, brain).eval()

    tokens = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        baseline = model(tokens)
    perturbed = tokens.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 5) % 32
    with torch.no_grad():
        changed = model(perturbed)

    torch.testing.assert_close(changed[:, :-1], baseline[:, :-1], rtol=0, atol=0)
    assert not torch.equal(changed[:, -1], baseline[:, -1])


def test_widening_the_brain_rank_keeps_what_was_learned():
    """Raising the read-out bandwidth must extend the trained brain, not erase it.

    The rank is the brain's bandwidth: at 64, at most 64 directions of a
    164,587-neuron state can reach the language model. Widening it is only useful
    if the trained slice survives, and only works if the new slice is non-zero --
    zero would have a zero gradient and stay dead for the whole run.
    """
    from dataclasses import replace

    from flybrain.wholebrain import expand_state_dict_rank

    torch.manual_seed(11)
    narrow_cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context=8, ffn_mode="swiglu")
    narrow_cfg = replace(narrow_cfg, brain_rank=8, brain_in_layers=True)
    torch.manual_seed(12)
    narrow = FlyBrainLM(narrow_cfg, None, tiny_brain_pathway(n_neurons=24, d_model=32, rank=8))
    wide = FlyBrainLM(
        replace(narrow_cfg, brain_rank=16), None, tiny_brain_pathway(n_neurons=24, d_model=32, rank=16)
    )
    tokens = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        reference = narrow(tokens)

    # A zero-filled new slice reproduces the narrow model exactly.
    exact = expand_state_dict_rank(narrow.state_dict(), wide, std_scale=0.0)
    wide.load_state_dict(exact, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(wide(tokens), reference, rtol=1e-5, atol=1e-6)

    # The default initialisation perturbs the read-out only slightly, and stays live.
    widened = expand_state_dict_rank(narrow.state_dict(), wide)
    assert not torch.equal(widened["brain.up.weight"], exact["brain.up.weight"])
    assert float(widened["brain.up.weight"][:, 8:].abs().sum()) > 0, "the added units start dead"
    wide.load_state_dict(widened, strict=True)

    probe = narrow.brain.initial_state(2, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        trained = narrow.brain.read(probe)
        widened_read = wide.brain.read(probe)
    drift = float((widened_read - trained).abs().max() / trained.abs().max())
    assert drift < 0.05, f"widening moved the trained read-out by {drift:.1%}"

    loss = wide(tokens, torch.randint(0, 32, (1, 8)))[1]
    loss.backward()
    assert float(wide.brain.up.weight.grad[:, 8:].abs().sum()) > 0, "the new units get no gradient"
    assert float(wide.brain.read_down.weight.grad[8:, :].abs().sum()) > 0


def test_the_token_drive_cannot_collapse_to_one_direction():
    """The drive is the brain's input bandwidth, and training destroys it silently.

    Measured on this project's own checkpoint after 1,500 steps: `down` stayed full
    rank (64) while `up` fell to rank 2, so `up(down(x))` had effective rank 1 --
    every one of the 164,587 neurons was driven by the same scalar, and the
    connectome could only act as a gain knob on a fixed pattern.
    """
    from flybrain.wholebrain import effective_rank

    pathway = tiny_brain_pathway(n_neurons=64, d_model=32, rank=8)
    healthy = float(pathway.drive_orthogonality())
    # Random directions over only 64 neurons still correlate at ~1/sqrt(64).
    assert healthy < 0.5, f"independent directions should score near zero, got {healthy}"

    with torch.no_grad():
        pathway.up.weight.copy_(pathway.up.weight[:, :1].expand(-1, 8))  # collapse
    collapsed = float(pathway.drive_orthogonality())
    assert collapsed > 0.5, f"a collapsed drive must be penalised, got {collapsed}"
    assert effective_rank(pathway.up.weight.t()) == 1

    # And the penalty must carry a gradient that pushes the directions apart.
    penalty = pathway.drive_orthogonality()
    penalty.backward()
    assert pathway.up.weight.grad is not None
    assert float(pathway.up.weight.grad.abs().sum()) > 0


def test_synapse_use_measures_transmitting_synapses():
    """Utilisation of the connectome, weighted by synaptic strength."""
    pathway = tiny_brain_pathway(n_neurons=24, d_model=16)
    silent = torch.zeros(2, 24)
    assert pathway.synapse_use(silent) == 0.0
    firing = torch.ones(2, 24)
    assert pathway.synapse_use(firing) == pytest.approx(1.0)
    half = torch.zeros(2, 24)
    half[:, 0] = 1.0 * (pathway.circuit.out_strength()[0] > 0)
    share = pathway.synapse_use(half)
    assert 0.0 <= share <= 1.0


def test_the_connectome_sweep_is_not_corrupted_by_autocast():
    """The sweep must run outside bf16 autocast, or its gradients come back garbage.

    Measured on the real 164,587-neuron connectome: with the sweep left inside CPU
    bf16 autocast, `raw_gain` came back with a gradient of 4.9e29 on 3 of 12
    *identical* steps, and one run died with `INDICES element is out of DATA
    bounds`. With autocast disabled around the sweep the same 12 steps were
    bit-identical. The loss never changed, so only the backward was affected --
    which is exactly the kind of silent corruption a training run cannot see.
    """
    torch.manual_seed(6)
    pathway = tiny_brain_pathway(n_neurons=48, d_model=32, chunks=2, iters=1)
    x = torch.randn(2, 8, 32)

    magnitudes = []
    for _ in range(12):
        for param in pathway.parameters():
            param.grad = None
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out, _ = pathway.forward_sequence(x, chunk_size=4)
        out.sum().backward()
        magnitudes.append(float(pathway.circuit.raw_gain.grad.abs().max()))

    assert all(m == magnitudes[0] for m in magnitudes), (
        "the sweep's gradient changed between identical steps under autocast: "
        f"{magnitudes}"
    )


def test_the_scatter_sweep_matches_autograd_but_saves_no_gather():
    """The recomputing backward must give the same gradients as plain autograd.

    Eight sweeps did not fit in 24 GiB because autograd keeps a (batch, edges)
    gather per edge chunk -- 1.6 GiB per sweep. The custom Function saves the
    population state instead (~10 MiB) and gathers again in the backward, which is
    only safe if it produces identical gradients.
    """
    torch.manual_seed(21)
    rng = np.random.default_rng(3)
    sources = rng.integers(0, 40, size=200)
    targets = rng.integers(0, 40, size=200)
    keep = sources != targets
    pre = torch.from_numpy(sources[keep].astype(np.int64))
    post = torch.from_numpy(targets[keep].astype(np.int64))
    weight = torch.rand(int(keep.sum()))
    sign = torch.ones(40)
    circuit = WholeBrainCircuit(pre, post, weight, sign, edge_chunk=64)

    activity = torch.rand(3, 40)
    values = torch.rand(int(keep.sum())) * 0.1

    # Reference: the same arithmetic with autograd's own bookkeeping.
    a = activity.clone().requires_grad_(True)
    v = values.clone().requires_grad_(True)
    plain = torch.zeros_like(a)
    for start in range(0, pre.numel(), 64):
        stop = min(start + 64, pre.numel())
        plain.index_add_(1, post[start:stop], a[:, pre[start:stop]] * v[start:stop])
    plain.sum().backward()

    a2 = activity.clone().requires_grad_(True)
    v2 = values.clone().requires_grad_(True)
    fast = circuit.synaptic_current_index_add(a2, v2)
    fast.sum().backward()

    torch.testing.assert_close(fast, plain, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(a2.grad, a.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v2.grad, v.grad, rtol=1e-5, atol=1e-6)


def test_participation_ratio_counts_independent_directions():
    """The metric that the factor ranks miss: do the tokens differ to the brain?"""
    from flybrain.wholebrain import BrainPathway

    same = torch.randn(1, 64).expand(8, -1).clone()
    assert float(BrainPathway.participation_ratio(same)) == pytest.approx(1.0, abs=1e-3)

    independent = torch.randn(8, 64)
    assert float(BrainPathway.participation_ratio(independent)) > 6.0


def test_participation_penalty_punishes_a_shared_drive():
    """A drive the whole batch shares is a scalar gain knob, and must be scored as one."""
    pathway = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=2, iters=1)
    x = torch.randn(6, 8, 32)

    with torch.no_grad():
        pathway._last_drive = torch.randn(1, 32).expand(6, 32).clone()
        pathway._last_readout = torch.randn(1, 32).expand(6, 32).clone()
    shared = float(pathway.participation_penalty())

    with torch.no_grad():
        pathway._last_drive = torch.randn(6, 32)
        pathway._last_readout = torch.randn(6, 32)
    spread = float(pathway.participation_penalty())

    assert shared > spread, f"a shared drive scored {shared}, spread {spread}"
    assert shared > 1.5, "a fully collapsed pair of interfaces should score near the maximum"

    # And it must carry gradient to the weights that produced the drive.
    out, _ = pathway.forward_sequence(x, chunk_size=4)
    pathway.participation_penalty().backward()
    assert pathway.up.weight.grad is not None
    assert float(pathway.up.weight.grad.abs().sum()) > 0


def test_the_readout_can_see_below_the_spike_threshold():
    """Two runs with identical spikes but different membrane states must differ.

    That is the whole point of the analogue read-out: on the real model the spikes
    were bit-identical across eight different documents, so the language model was
    handed one constant vector by all 164,587 neurons no matter what it read.
    """
    torch.manual_seed(31)
    brain = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=2, iters=1)
    state = torch.zeros(4, 32)
    state[:, :8] = 1.0  # the same spikes in both cases

    brain.circuit.last_analog = torch.zeros(4, 32)
    quiet = brain.read(state).detach()

    brain.circuit.last_analog = torch.rand(4, 32)
    excited = brain.read(state).detach()

    assert float(brain.analog_read) > 0, "the analogue read-out must be on by default"
    assert not torch.allclose(quiet, excited), "the sub-threshold state never reached the read-out"

    brain.analog_readout = False
    brain.circuit.last_analog = torch.rand(4, 32)
    off = brain.read(state).detach()
    torch.testing.assert_close(off, quiet, rtol=0, atol=0)

    # And it must not leak: two identical sequences score identically regardless of
    # what ran before them.
    brain.analog_readout = True
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=16, ffn_mode="swiglu")
    model = FlyBrainLM(cfg, None, brain).eval()
    tokens = torch.randint(0, 32, (4, 16))
    with torch.no_grad():
        first = model(tokens)
        model(torch.randint(0, 32, (4, 16)))
        second = model(tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_off_domain_text_is_rejected_before_it_reaches_the_corpus():
    """The guard that keeps a fetched page's side content out of the corpus.

    Fetched pages carry navigation, unrelated news and other people's research. A
    teacher asked to ground itself in a passage will write a confident question
    about whatever it was handed -- measured: the unfiltered corpus contained
    question/answer pairs about human biomarkers and cable theory for a model that
    is supposed to know about Kenyon cells.
    """
    from flybrain.data import is_domain_text

    assert is_domain_text("果蝇蘑菇体的Kenyon细胞接收投射神经元的稀疏输入，每个KC约有5个PN输入。")
    assert is_domain_text("FlyWire released a whole-brain connectome of 139,255 neurons.")
    assert is_domain_text("The leaky integrate-and-fire neuron emits a spike when the membrane potential crosses threshold.")
    assert not is_domain_text("资料中谁说识别生物标志物可以促进更早诊断和更好治疗？")
    assert not is_domain_text("Click here to subscribe to our newsletter about cooking and travel.")
    # A passing mention of "brain" is not enough on its own.
    assert not is_domain_text("The brain is mentioned once in this otherwise unrelated paragraph.")


def test_the_pure_model_contains_no_transformer_machinery():
    """The whole point of `arch="connectome"`: nothing but the connectome at work.

    The hybrid model is a transformer whose feed-forward block happens to be a
    mushroom body -- attention, rotary positions, RMSNorm, a residual stream and a
    key/value cache are still what model the sequence. This asserts the pure model
    has none of those, by name and by behaviour.
    """
    from flybrain.connectome_lm import ConnectomeLM, build_lm

    torch.manual_seed(41)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        context=8,
        ffn_mode="swiglu",
        arch="connectome",
    )
    brain = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=1, iters=1)
    model = build_lm(cfg, None, brain)
    assert isinstance(model, ConnectomeLM)

    names = [name for name, _ in model.named_modules()]
    forbidden = ("attn", "rope", "norm1", "norm2", "norm_f", "blocks")
    for token in forbidden:
        assert not any(token in name for name in names), f"found {token} in {names}"

    # And the machinery is not hiding in the parameter names either.
    params = [name for name, _ in model.named_parameters()]
    assert not any("q_proj" in n or "kv" in n or "rope" in n for n in params), params

    tokens = torch.randint(0, 64, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 64)

    # Causality is the ordering of the recurrence; there is no mask anywhere.
    # `logits[:, k]` is produced after folding token k in and before token k+1
    # exists, so it must depend on the prefix `0..k` and on nothing after it.
    for position in range(tokens.shape[1]):
        later = tokens.clone()
        later[:, position] = (later[:, position] + 7) % 64
        moved = model(later)
        # Changing token k cannot move any prediction made before k...
        torch.testing.assert_close(
            moved[:, :position], logits[:, :position], rtol=0, atol=0
        )
        # ...and must move the prediction made immediately after it, which is the
        # whole reason the alignment has to be fold-then-read: reading before
        # folding hid token k from the prediction of token k+1.
        assert not torch.equal(moved[:, position], logits[:, position]), (
            f"token {position} does not reach the prediction of the next token"
        )


def test_holding_distinct_drives_reaches_a_distinct_state_in_either_mode():
    """The pure model needs the connectome to remember, so this is the load-bearing test.

    A correction of an earlier version of this test, which asserted that spiking
    states collapse onto one attractor while rate states do not. That claim came
    from reading `(state - state[0]).abs().max()` on binary states, where the
    statistic is 0 or 1 and says nothing about magnitude. Re-measured on the real
    164,587-neuron circuit with unambiguously scaled statistics
    (scripts/probe_connectome_memory.py), the circuit keeps distinct drives distinct
    in *both* modes, which is what this asserts.
    """
    torch.manual_seed(43)
    brain = tiny_brain_pathway(n_neurons=64, d_model=32, chunks=1, iters=1)

    drives = torch.randn(6, 64)
    drives = drives - drives.mean(dim=1, keepdim=True)
    drives = drives / drives.norm(dim=1, keepdim=True) * (64**0.5)
    drives = drives * brain.drive_scale

    def separation(spiking: bool, steps: int = 10) -> float:
        """Mean pairwise distance between streams, as a share of their own spread."""
        brain.spiking = spiking
        brain.reset_stats()
        brain.start_sequence()
        with torch.no_grad():
            state = brain.initial_state(6, torch.device("cpu"), drives.dtype)
            for _ in range(steps):
                state = brain.advance_drive(state, drives)
        pairwise = torch.cdist(state.float(), state.float()).mean()
        independent = state.float().std() * (2**0.5)
        return float(pairwise / independent) if float(independent) > 0 else 0.0

    for spiking in (True, False):
        ratio = separation(spiking)
        assert ratio > 0.5, (
            f"with spiking={spiking} the connectome collapsed distinct drives onto "
            f"one state: mean pairwise distance is {ratio:.3f} of independent patterns"
        )


def test_the_current_read_keeps_more_of_the_input_than_the_spike_read():
    """Why the pure model reads the pre-activation instead of the rate.

    A spike and a rate both pass the synaptic current through a threshold, and this
    connectome drives most of its cells far from that threshold -- measured on the
    real circuit, one sweep leaves ~13.5% of neurons within half a threshold of
    firing. The linear pre-activation is not saturating, so it retains the
    difference between two inputs that the thresholded state compresses. Measured
    on the real 164,587-neuron model at initialisation, the read-out latent spread
    between four distinct streams was 103 (spike), 143 (rate), 190 (current).
    """
    torch.manual_seed(29)
    brain = tiny_brain_pathway(n_neurons=96, d_model=32, chunks=1, iters=1)
    brain.spiking = False
    brain.reset_stats()
    brain.start_sequence()

    drives = torch.randn(4, 96)
    drives = drives - drives.mean(dim=1, keepdim=True)
    drives = drives / drives.norm(dim=1, keepdim=True) * (96**0.5) * brain.drive_scale
    with torch.no_grad():
        state = brain.initial_state(4, torch.device("cpu"), drives.dtype)
        for _ in range(6):
            state = brain.advance_drive(state, drives)

    def latent_spread(mode: str) -> float:
        brain.analog_mode = mode
        with torch.no_grad():
            latent = brain.read_latent(state)
        return float(torch.cdist(latent.float(), latent.float(), p=1).mean())

    spike = latent_spread("spike")
    current = latent_spread("current")
    assert current > spike, (
        f"the linear current must carry more of the input than the thresholded "
        f"state: spike {spike:.4g}, current {current:.4g}"
    )


def test_generation_and_training_agree_on_the_alignment():
    """`generate` folds the prompt and then reads; `forward` must do the same thing.

    They disagreed: `forward` read the population state *before* folding each token
    in, so training asked the model to predict the token after next while generation
    predicted the next one. A checkpoint that scored 4.5 in training was generating
    from a different model than the one that was scored.
    """
    from flybrain.connectome_lm import ConnectomeLM

    torch.manual_seed(37)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        context=8,
        arch="connectome",
        brain_chunks=1,
        brain_iters=1,
    )
    brain = tiny_brain_pathway(n_neurons=32, d_model=32, chunks=1, iters=1)
    model = ConnectomeLM(cfg, brain).eval()
    prompt = torch.randint(0, 64, (1, 5))

    with torch.no_grad():
        # The training path: the last position's logits predict the next token.
        trained = model(prompt)[:, -1]

        # The generation path, exactly as `ConnectomeLM.generate` steps it: fold
        # every prompt token, then read once.
        brain.reset_stats()
        brain.start_sequence()
        state = brain.initial_state(1, torch.device("cpu"), model.embed.weight.dtype)
        drives = model._drives(prompt)
        for position in range(prompt.shape[1]):
            state = brain.advance_drive(state, drives[:, position])
        generated = model.head(brain.read_latent(state))

    torch.testing.assert_close(trained, generated, rtol=0, atol=0)


def test_a_read_out_mode_that_is_not_one_of_the_three_is_rejected():
    """A typo must not silently disable the analogue read-out."""
    circuit = WholeBrainCircuit(
        torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0.5, 0.5]), torch.tensor([1.0, 1.0])
    )
    with pytest.raises(ValueError, match="analog_mode"):
        BrainPathway(circuit, 8, rank=4, analog_mode="pre")
    assert BrainPathway(circuit, 8, rank=4, analog_mode="current").analog_mode == "current"


def test_incremental_decoding_of_the_pure_model_matches_one_full_pass():
    """The streaming CLI and web server feed one token at a time; it must score the same.

    The whole model state between steps is the population vector, handed over in
    `ModelCache.brain`. If the incremental and the one-shot path disagreed, the
    interface and the training loss would be measuring different models.
    """
    from flybrain.connectome_lm import ConnectomeLM

    torch.manual_seed(17)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=1, n_heads=2, context=8, arch="connectome"
    )
    brain = tiny_brain_pathway(n_neurons=48, d_model=32, chunks=1, iters=1)
    model = ConnectomeLM(cfg, brain).eval()

    tokens = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        full = model(tokens)[:, -1, :]
        cache = None
        for position in range(tokens.shape[1]):
            logits, cache = model(tokens[:, position : position + 1], cache=cache, use_cache=True)
        incremental = logits[:, -1, :]
    torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-4)


def test_the_pure_model_survives_a_checkpoint_round_trip(tmp_path):
    """`flybrain-chat` and the server reload a checkpoint and nothing else.

    The pure architecture has no transformer blocks, so the old loader's assumption
    that a connectome wiring buffer lives at `blocks.0.ffn.pn_to_kc` does not hold.
    """
    from dataclasses import asdict

    from flybrain.connectome_lm import ConnectomeLM
    from flybrain.generate import load_model

    torch.manual_seed(23)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        context=8,
        arch="connectome",
        brain_chunks=1,
        brain_iters=1,
    )
    brain = tiny_brain_pathway(
        n_neurons=32, d_model=32, chunks=cfg.brain_chunks, iters=cfg.brain_iters
    )
    model = ConnectomeLM(cfg, brain).eval()
    tokens = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        expected = model(tokens)

    path = tmp_path / "pure.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(cfg)}, path)
    reloaded, reloaded_cfg = load_model(str(path), device="cpu")
    assert reloaded_cfg.arch == "connectome"
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tokens), expected, rtol=0, atol=0)


def test_a_block_without_a_bias_is_unchanged():
    """The interleaved path must be opt-in: no brain means the old arithmetic."""
    cfg = ModelConfig(vocab_size=16, d_model=16, n_layers=1, n_heads=2, context=4, ffn_mode="swiglu")
    block = Block(cfg, None)
    x = torch.randn(2, 4, 16)
    cos, sin = build_rope_cache(4, 8, 10_000.0, torch.device("cpu"), torch.float32)
    plain, _ = block(x, cos, sin)
    same, _ = block(x, cos, sin, None, False, None)
    assert torch.equal(plain, same)

    biased, _ = block(x, cos, sin, None, False, torch.randn(2, 1, 16))
    assert not torch.allclose(plain, biased), "the brain bias had no effect on the block"


# --------------------------------------------------------------------------
# chat formatting and checkpoints
# --------------------------------------------------------------------------

TOKENIZER_PATH = "data/tokenized/tokenizer.json"
needs_tokenizer = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH), reason="tokenizer has not been trained yet"
)


@needs_tokenizer
def test_format_chat_folds_system_prompt_without_a_system_token():
    """The vocabulary has no <system> token, so a system turn must not require one."""
    from flybrain.generate import format_chat
    from flybrain.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    assert tokenizer.token_to_id("<system>") is None

    ids = format_chat(
        [
            {"role": "system", "content": "你是果蝇神经科学专家"},
            {"role": "user", "content": "蘑菇体是什么"},
        ],
        tokenizer,
    )
    assert ids[0] == tokenizer.token_to_id("<bos>")
    assert ids[-1] == tokenizer.token_to_id("<assistant>")
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    assert "果蝇神经科学专家" in rendered
    assert "蘑菇体是什么" in rendered


@needs_tokenizer
def test_format_chat_multi_turn_structure():
    from flybrain.generate import format_chat
    from flybrain.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    user_id = tokenizer.token_to_id("<user>")
    assistant_id = tokenizer.token_to_id("<assistant>")
    ids = format_chat(
        [
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": "第一个回答"},
            {"role": "user", "content": "第二个问题"},
        ],
        tokenizer,
    )
    # user, assistant, user, then the generation prompt.
    assert [i for i in ids if i in (user_id, assistant_id)] == [
        user_id,
        assistant_id,
        user_id,
        assistant_id,
    ]


def test_checkpoint_roundtrip_keeps_the_whole_brain(tmp_path):
    """A saved model must reload with its connectome and brain wiring intact."""
    from flybrain.generate import load_model

    torch.manual_seed(6)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=4, context=8, ffn_mode="swiglu")
    brain = tiny_brain_pathway(n_neurons=20, d_model=32, chunks=2, iters=2)
    model = FlyBrainLM(cfg, None, brain).eval()
    tokens = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        before = model(tokens)

    path = str(tmp_path / "ckpt.pt")
    torch.save(
        {"model": model.state_dict(), "step": 0, "best_val": 1.0, "model_config": asdict(cfg)},
        path,
    )
    loaded, loaded_cfg = load_model(path, device="cpu")
    assert loaded.brain is not None, "the whole-brain module was dropped on reload"
    assert loaded.brain.circuit.n_neurons == 20
    with torch.no_grad():
        after = loaded(tokens)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


@needs_tokenizer
def test_generate_stream_produces_text_and_respects_the_budget():
    from flybrain.generate import SamplingConfig, generate_stream
    from flybrain.tokenizer import load_tokenizer

    torch.manual_seed(7)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    # The model's vocabulary must cover the tokenizer's ids, so reuse its size.
    cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=32,
        n_layers=1,
        n_heads=4,
        context=32,
        ffn_mode="swiglu",
    )
    model = FlyBrainLM(cfg, None).eval()
    prompt = tokenizer.encode("苍蝇").ids

    text = "".join(
        generate_stream(
            model,
            tokenizer,
            prompt,
            cfg=SamplingConfig(max_new_tokens=12, temperature=0.7),
            device="cpu",
            dtype=torch.float32,
        )
    )
    assert isinstance(text, str)
