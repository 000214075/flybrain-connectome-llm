"""Diagnose why the whole-brain circuit is silent at initialisation.

Prints the actual magnitudes at each stage so the firing threshold can be set from
measurement instead of arithmetic on paper.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch.nn.functional as F  # noqa: E402

from flybrain.wholebrain import BrainPathway, WholeBrainCircuit  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    n_neurons = 64
    sources = rng.integers(0, n_neurons, size=80)
    targets = rng.integers(0, n_neurons, size=80)
    keep = sources != targets
    circuit = WholeBrainCircuit(
        torch.from_numpy(sources[keep].astype(np.int64)),
        torch.from_numpy(targets[keep].astype(np.int64)),
        torch.full((int(keep.sum()),), 0.05),
        torch.ones(n_neurons),
    )
    pathway = BrainPathway(circuit, 32, rank=8, chunks=2, iters=2)

    x = torch.randn(4, 8, 32)
    pooled = pathway.norm_in(x).mean(dim=1)
    raw = pathway.up(pathway.down(pooled))
    drive = pathway._drive(x)

    print("--- drive path ---")
    print(f"pooled   std {pooled.std():.4f}")
    print(f"raw up   std {raw.std():.4f}  max {raw.abs().max():.4f}")
    print(f"drive    std {drive.std():.4f}  max {drive.abs().max():.4f}")
    print(f"drive_scale {float(pathway.drive_scale):.4f}")

    decay = circuit.lif.decay
    print("\n--- lif ---")
    print(f"raw_tau {float(circuit.lif.raw_tau):.4f} -> decay {float(decay):.4f} (dt/tau)")
    print(f"v_th {circuit.lif.v_th}   step_size {float(circuit.step_size):.4f}")

    state = circuit.initial_state(4, x.device, x.dtype)
    synapses = circuit.effective_synapses()
    for step in range(4):
        current = circuit.synaptic_current(state, synapses) + drive
        scaled = current * circuit.step_size
        v_before = state
        lif = circuit.lif(scaled, spiking=True)
        print(
            f"\nstep {step}: syn+drive std {current.std():.4f} scaled std {scaled.std():.4f} "
            f"max {scaled.abs().max():.4f}\n"
            f"        above-threshold fraction {(scaled > circuit.lif.v_th).float().mean():.4f}"
            f"  spikes {float(lif.sum()):.1f}"
        )
        state = circuit(state, drive, synapses, spiking=True)

    print(f"\nfinal state: active {(state > 0).float().mean():.4f} mean {state.mean():.4f}")

    print("\n--- what a rate-coded (non-spiking) brain would give ---")
    state = circuit.initial_state(4, x.device, x.dtype)
    for step in range(4):
        current = circuit.synaptic_current(state, synapses) + drive
        state = torch.relu(current * circuit.step_size)
    print(f"rate state: nonzero {(state > 0).float().mean():.4f} mean {state.mean():.4f}")

    print("\n--- membrane potential trajectory (spiking off) ---")
    v = torch.zeros(4, n_neurons)
    for step in range(8):
        current = torch.zeros(4, n_neurons) + drive
        v = v + decay * (-v + current * circuit.step_size)
        print(f"  step {step}: v std {v.std():.4f} max {v.abs().max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
