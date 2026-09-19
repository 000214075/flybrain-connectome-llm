"""Full training pipeline for the FlyBrain language model.

Covers the whole loop: data loading, bf16 mixed precision, warmup + cosine
schedule, gradient accumulation and clipping, periodic validation perplexity,
sampling previews, resumable checkpoints and a metrics log.

Every run is bf16 by default: on the RX 7900 XTX fp32 matmul measures ~3 TFLOPS
against ~77 TFLOPS for bf16, so fp32 training is not a viable option here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from flybrain.connectome import Pathway, build_synthetic_pathway, shuffle_pathway
from flybrain.data import TokenWindowDataset
from flybrain.device import describe_memory, device_report, pick_device, tune_rocm
from flybrain.connectome_lm import build_lm
from flybrain.model import FlyBrainLM, ModelConfig
from flybrain.tokenizer import load_tokenizer


@dataclass
class TrainConfig:
    run_name: str = "flybrain"
    data_dir: str = "data/tokenized"
    token_dtype: str = "uint16"
    tokenizer_path: str = "data/tokenized/tokenizer.json"
    out_dir: str = "checkpoints/run"
    model: ModelConfig = field(default_factory=ModelConfig)
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    batch_size: int = 32
    grad_accum: int = 4
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 40
    log_interval: int = 20
    checkpoint_interval: int = 500
    sample_interval: int = 500
    sample_prompt: str = "苍蝇的大脑"
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    compile: bool = False
    num_workers: int = 0
    seed: int = 1337
    resume: str = "auto"  # "auto" | "none" | path
    max_minutes: float = 0.0  # 0 = no wall-clock limit
    build_pathway_from: str = ""  # connectome npz; empty means synthetic wiring
    # Homeostatic term keeping the whole brain firing. Without it the optimiser
    # silences the circuit early in training: its output looks like noise, and
    # suppressing it is cheaper than learning to use it. Measured on this project,
    # the firing fraction fell from 15.9% to 0.7% within 300 steps when unregularised.
    brain_firing_target: float = 0.05
    brain_firing_penalty: float = 2.0
    # Keeps the token drive from collapsing onto a single direction. Measured on
    # this project's own checkpoint: after 1,500 steps the drive map had effective
    # rank 1, so all 164,587 neurons were driven by one scalar per chunk.
    brain_drive_orthogonality: float = 0.0
    # Keeps the two interfaces from acting as one vector shared by the whole batch:
    # the factor ranks can be healthy while the composed map is still rank 1.
    brain_participation_penalty: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        model_raw = raw.pop("model", {})
        cfg = cls(**raw)
        base = asdict(ModelConfig())
        base.update({k: v for k, v in model_raw.items() if k in base})
        cfg.model = ModelConfig(**base)
        return cfg


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_pathway(cfg: TrainConfig) -> Pathway | None:
    """Real wiring when a connectome export exists, otherwise synthetic wiring."""
    mode = cfg.model.ffn_mode
    if mode == "swiglu":
        return None
    if cfg.build_pathway_from and os.path.exists(cfg.build_pathway_from):
        pathway = Pathway.load(cfg.build_pathway_from)
    else:
        pathway = build_synthetic_pathway(
            cfg.model.mb_max_pn or 240,
            cfg.model.mb_max_kc or 2400,
            cfg.model.mb_max_mbon or 34,
            seed=cfg.seed,
        )
    if mode == "mb-shuffled":
        pathway = shuffle_pathway(pathway, seed=cfg.seed)
    return pathway


def load_brain(cfg: TrainConfig):
    """Build the whole-brain module when a connectome export is configured.

    Prints the size of what it is about to run, so a training log always records
    whether the full brain was actually part of the model.
    """
    if not cfg.model.brain_path:
        return None
    if not os.path.exists(cfg.model.brain_path):
        print(
            f"warning: brain_path {cfg.model.brain_path!r} not found; "
            "training without the whole-brain circuit"
        )
        return None
    from flybrain.wholebrain import BrainPathway, load_whole_brain

    circuit, meta, _ = load_whole_brain(
        cfg.model.brain_path,
        device="cpu",
        edge_chunk=cfg.model.brain_edge_chunk,
        impl=cfg.model.brain_impl,
    )
    pathway = BrainPathway(
        circuit,
        cfg.model.d_model,
        rank=cfg.model.brain_rank,
        chunks=cfg.model.brain_chunks,
        iters=cfg.model.brain_iters,
        spiking=cfg.model.brain_spiking,
        analog_readout=cfg.model.brain_analog_readout,
        analog_mode=getattr(cfg.model, "brain_analog_mode", "rate"),
        input_mask_path=getattr(cfg.model, "brain_input_mask", "") or None,
        output_mask_path=getattr(cfg.model, "brain_output_mask", "") or None,
    )
    if getattr(cfg.model, "arch", "transformer") == "connectome":
        # There is no stack to interleave with: the connectome *is* the sequence
        # model, swept once per token, so the whole brain is on the output path of
        # every token by construction rather than by placement.
        placement = "one sweep per token, the whole sequence model"
    elif cfg.model.brain_in_layers:
        placement = f"interleaved with all {cfg.model.n_layers} blocks"
    else:
        placement = "appended after the stack"
    print(
        f"whole brain: {meta.n_neurons:,} neurons, {meta.nnz:,} synapses "
        f"({meta.excitatory:,} excitatory / {meta.inhibitory:,} inhibitory), "
        f"{cfg.model.brain_chunks} chunks x {cfg.model.brain_iters} sweeps per pass, "
        f"rank {cfg.model.brain_rank}, {placement}, impl={cfg.model.brain_impl}"
    )
    return pathway


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matrices, never to norms or gains."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("apl_to_kc"):
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=use_fused
    )


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to `min_lr_ratio * lr`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * coeff)


@torch.no_grad()
def evaluate(model: FlyBrainLM, loader, device, *, iters: int, dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path: str, model, optimizer, cfg: TrainConfig, step: int, best_val: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "config": asdict(cfg),
            "model_config": asdict(cfg.model),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer) -> tuple[int, float]:
    """Restore a run, tolerating parameters that did not exist when it was saved.

    Loading is non-strict on purpose: adding a parameter to the model (a learned
    resting state, say) must not make every earlier checkpoint unloadable. The
    keys that were absent are reported rather than silently ignored.
    """
    from flybrain.wholebrain import expand_state_dict_rank

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = expand_state_dict_rank(ckpt["model"], model)
    if any(saved[k].shape != ckpt["model"][k].shape for k in saved):
        print(f"widened the brain bandwidth from the checkpoint to rank {model.cfg.brain_rank}")
    result = model.load_state_dict(saved, strict=False)
    if result.missing_keys:
        print(f"checkpoint is missing weights for: {', '.join(result.missing_keys)}")
        print("  (initialised from scratch; everything else was restored)")
    if result.unexpected_keys:
        print(f"checkpoint has weights the model no longer uses: {', '.join(result.unexpected_keys)}")
    if "optimizer" in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"could not restore optimizer state ({exc}); continuing with a fresh optimiser")
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", float("inf")))


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        tune_rocm(verbose=True)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.device = pick_device()
        self.dtype = resolve_dtype(cfg.dtype)
        if self.dtype != torch.float32 and not torch.cuda.is_available():
            print("warning: mixed precision requested without an accelerator; falling back to fp32")
            self.dtype = torch.float32

        os.makedirs(cfg.out_dir, exist_ok=True)
        self.metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
        self.tokenizer = load_tokenizer(cfg.tokenizer_path) if os.path.exists(cfg.tokenizer_path) else None

        train_path = os.path.join(cfg.data_dir, "train.bin")
        val_path = os.path.join(cfg.data_dir, "val.bin")
        self.train_ds = TokenWindowDataset(train_path, cfg.model.context, token_dtype=cfg.token_dtype, seed=cfg.seed)
        self.val_ds = TokenWindowDataset(val_path, cfg.model.context, token_dtype=cfg.token_dtype, seed=cfg.seed + 1)
        loader_kwargs = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, shuffle=False, **loader_kwargs)

        pathway = None if getattr(cfg.model, "arch", "transformer") == "connectome" else load_pathway(cfg)
        brain = load_brain(cfg)
        self.model = build_lm(cfg.model, pathway, brain).to(self.device)
        if cfg.compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.optimizer = build_optimizer(self.model, cfg)
        self.step = 0
        self.best_val = float("inf")
        self.stop_requested = False

        cfg.model.to_json(os.path.join(cfg.out_dir, "model_config.json"))
        with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2, ensure_ascii=False, default=str)

    def maybe_resume(self) -> None:
        if self.cfg.resume == "none":
            return
        if self.cfg.resume == "weights":
            # Fine-tuning from a pretrained checkpoint: take the weights, start a
            # fresh optimiser and step counter so the new run gets its own learning
            # rate schedule instead of inheriting the end of the previous one.
            path = os.path.join(self.cfg.out_dir, "best.pt")
            if not os.path.exists(path):
                path = os.path.join(self.cfg.out_dir, "last.pt")
            if not os.path.exists(path):
                print(f"weights resume requested but no checkpoint in {self.cfg.out_dir}; starting fresh")
                return
            state = torch.load(path, map_location="cpu", weights_only=False)
            from flybrain.wholebrain import expand_state_dict_rank

            result = self.model.load_state_dict(
                expand_state_dict_rank(state["model"], self.model), strict=False
            )
            if result.missing_keys:
                print(f"pretrained checkpoint is missing: {', '.join(result.missing_keys)}")
            print(f"loaded weights from {path}; step reset to 0 with a fresh optimiser")
            return
        path = self.cfg.resume
        if path == "auto":
            path = os.path.join(self.cfg.out_dir, "last.pt")
        if os.path.exists(path):
            self.step, self.best_val = load_checkpoint(path, self.model, self.optimizer)
            print(f"resumed from {path} at step {self.step} (best val {self.best_val:.4f})")

    def log(self, record: dict[str, Any]) -> None:
        record = {"step": self.step, "wall": time.time(), **record}
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            print(f"\nreceived signal {signum}; finishing the current step then saving")
            self.stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    @torch.no_grad()
    def brain_health(self) -> dict:
        """Fraction of the connectome that is firing right now.

        A whole-brain module that has gone silent is invisible in the loss curve:
        it simply stops contributing. This makes that failure mode visible in the
        log, and it is how the silent-brain regression was originally found.
        """
        if self.model.brain is None:
            return {}
        try:
            x, _ = next(iter(self.train_loader))
        except StopIteration:
            return {}
        # Eight streams, not two: a participation ratio needs a batch to span, and
        # with two streams the ceiling is one independent direction, so the number
        # would read 1.0 no matter how the brain behaved.
        probe = x[: min(8, x.shape[0])].to(self.device)
        with torch.no_grad():
            if getattr(self.model.cfg, "arch", "transformer") == "connectome":
                # The connectome model has no key/value cache: its whole state is the
                # population vector, and `forward` returns just the logits.
                self.model(probe)
                state = self.model.brain.last_state
            else:
                _, cache = self.model(probe, use_cache=True)
                state = cache.brain
        if state is None:
            return {}
        stats = self.model.brain.activity_stats(state)
        stats["neurons"] = self.model.brain.circuit.n_neurons
        stats["synapses"] = int(self.model.brain.circuit.pre.numel())
        # The read-out scale is the signature of the one pathology this model has
        # shown: summing 164,587 neurons through a 256-wide read-out makes an
        # untrained read-out already too loud (report 3.3.2), and if training inflates
        # it further the loss on unseen data climbs *above* the uniform guess while the
        # training loss keeps falling. Logged so that failure is visible as a number
        # rather than only as a bad validation curve.
        stats["readout_std"] = round(float(self.model.brain.read_latent(state).std()), 4)
        # How many of the probe's streams the whole brain actually answers
        # differently, at both interfaces: the number that says whether the
        # connectome is carrying the batch or just idling through it.
        for name, ratio in (
            ("drive_participation", self.model.brain.drive_participation()),
            ("readout_participation", self.model.brain.readout_participation()),
        ):
            if ratio is not None:
                stats[name] = round(float(ratio), 2)
        return stats

    def brain_penalty(self) -> Tensor:
        """Loss terms that keep the whole brain firing and keep its input connected.

        Returns a zero tensor when there is no brain, so the training step does not
        need to branch on it.
        """
        brain = getattr(self.model, "brain", None)
        empty = torch.zeros((), device=self.device)
        if brain is None or brain.last_activity is None:
            return empty
        total = empty
        if self.cfg.brain_firing_penalty > 0:
            deviation = brain.last_activity.float() - self.cfg.brain_firing_target
            total = total + self.cfg.brain_firing_penalty * deviation.pow(2)
        if self.cfg.brain_drive_orthogonality > 0:
            total = total + self.cfg.brain_drive_orthogonality * brain.drive_orthogonality()
        if self.cfg.brain_participation_penalty > 0:
            total = total + self.cfg.brain_participation_penalty * brain.participation_penalty()
        return total

    @torch.no_grad()
    def preview(self, max_new_tokens: int = 60) -> str:
        if self.tokenizer is None:
            return "(no tokenizer)"
        model = self.model
        was_training = model.training
        model.eval()
        ids = self.tokenizer.encode(self.cfg.sample_prompt).ids
        idx = torch.tensor([ids], device=self.device)
        for _ in range(max_new_tokens):
            logits = model(idx[:, -self.cfg.model.context :])
            logits = logits[:, -1, :].float()
            probs = F.softmax(logits / 0.8, dim=-1)
            top = torch.topk(probs, 40, dim=-1)
            choice = torch.multinomial(top.values, 1)
            idx = torch.cat([idx, top.indices.gather(-1, choice)], dim=-1)
        if was_training:
            model.train()
        return self.tokenizer.decode(idx[0].tolist())

    def train(self) -> dict:
        cfg = self.cfg
        self.install_signal_handlers()
        self.maybe_resume()
        print(json.dumps(device_report(), indent=2, ensure_ascii=False))
        print(f"params: {self.model.param_count():,}")
        print(f"train windows/epoch: {len(self.train_ds):,}  val windows: {len(self.val_ds):,}")

        model = self.model
        model.train()
        tokens_per_step = cfg.batch_size * cfg.model.context * cfg.grad_accum
        self.log({"event": "start", "device": device_report(), "params": model.param_count()})
        started = time.time()
        accum_loss = 0.0
        accum_count = 0
        micro = 0
        data_iter = iter(self.train_loader)

        while self.step < cfg.max_steps and not self.stop_requested:
            lr = lr_at(self.step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)

            step_started = time.perf_counter()
            for _ in range(cfg.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    self.train_ds.epoch += 1
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
                ):
                    _, loss = model(x, y)
                loss = loss + self.brain_penalty()
                (loss / cfg.grad_accum).backward()
                accum_loss += float(loss.item())
                accum_count += 1
                micro += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.step += 1
            step_seconds = time.perf_counter() - step_started

            if self.step % cfg.log_interval == 0 or self.step == 1:
                mean_loss = accum_loss / max(1, accum_count)
                brain = getattr(self.model, "brain", None)
                activity = (
                    round(float(brain.mean_activity.detach().float().item()), 4)
                    if brain is not None and brain.mean_activity is not None
                    else None
                )
                synapses = (
                    round(brain.synapse_use(brain.last_state), 4)
                    if brain is not None and brain.last_state is not None
                    else None
                )
                record = {
                    "event": "train",
                    "loss": round(mean_loss, 4),
                    "ppl": round(math.exp(min(mean_loss, 20)), 3),
                    "lr": round(lr, 6),
                    "grad_norm": round(float(grad_norm), 3),
                    "tokens_per_second": round(tokens_per_step / max(step_seconds, 1e-9), 1),
                    "brain_activity": activity,
                    "brain_synapse_use": synapses,
                    "brain_sweeps": int(getattr(brain, "sweeps", 0)) if brain is not None else 0,
                    "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    if torch.cuda.is_available()
                    else 0.0,
                }
                self.log(record)
                brain_note = ""
                if activity is not None:
                    brain_note = f" brain {100 * activity:.1f}%"
                    if synapses is not None:
                        brain_note += f" syn {100 * synapses:.1f}%"
                    brain_note += f" {record['brain_sweeps']}sweeps"
                print(
                    f"step {self.step:>6}/{cfg.max_steps} loss {mean_loss:.4f} "
                    f"ppl {record['ppl']:.2f} lr {lr:.2e} gnorm {record['grad_norm']:.2f} "
                    f"{record['tokens_per_second']:.0f} tok/s{brain_note}",
                    flush=True,
                )
                accum_loss, accum_count = 0.0, 0

            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val_loss = evaluate(
                    model, self.val_loader, self.device, iters=cfg.eval_iters, dtype=self.dtype
                )
                val_ppl = math.exp(min(val_loss, 20))
                improved = val_loss < self.best_val
                self.best_val = min(self.best_val, val_loss)
                brain_stats = self.brain_health()
                self.log(
                    {
                        "event": "eval",
                        "val_loss": round(val_loss, 4),
                        "val_ppl": round(val_ppl, 3),
                        "brain": brain_stats,
                    }
                )
                print(f"  eval val_loss {val_loss:.4f} val_ppl {val_ppl:.2f} best {self.best_val:.4f}")
                if brain_stats:
                    brain = self.model.brain
                    down_rank, up_rank = brain.drive_bandwidth()
                    use = brain.synapse_use(brain.last_state) if brain.last_state is not None else 0.0
                    drive_pr = brain.drive_participation()
                    read_pr = brain.readout_participation()
                    shape = ""
                    if drive_pr is not None and read_pr is not None:
                        shape = (
                            f", batch participation {float(drive_pr):.1f}/{float(read_pr):.1f}"
                            f" of {brain._last_drive.shape[0]}"
                        )
                    print(
                        f"  whole brain: {brain_stats['neurons']:,} neurons, "
                        f"{brain_stats['synapses']:,} synapses, "
                        f"{100 * brain_stats.get('firing_fraction', brain_stats.get('half_active_fraction', 0.0)):.2f}% active, "
                        f"drive rank {down_rank}/{up_rank}, synapse use {100 * use:.1f}%"
                        f"{shape}"
                    )
                if improved:
                    save_checkpoint(
                        os.path.join(cfg.out_dir, "best.pt"), model, self.optimizer, cfg, self.step, self.best_val
                    )

            if cfg.sample_interval and self.step % cfg.sample_interval == 0:
                text = self.preview()
                self.log({"event": "sample", "text": text})
                print(f"  sample: {text[:160]!r}")

            if self.step % cfg.checkpoint_interval == 0 or self.step == cfg.max_steps:
                save_checkpoint(
                    os.path.join(cfg.out_dir, "last.pt"), model, self.optimizer, cfg, self.step, self.best_val
                )

            if cfg.max_minutes and (time.time() - started) / 60.0 >= cfg.max_minutes:
                print(f"wall-clock budget of {cfg.max_minutes} minutes reached; stopping")
                self.stop_requested = True

        save_checkpoint(
            os.path.join(cfg.out_dir, "last.pt"), model, self.optimizer, cfg, self.step, self.best_val
        )
        summary = {
            "steps": self.step,
            "best_val_loss": self.best_val,
            "minutes": round((time.time() - started) / 60, 2),
            "params": model.param_count(),
            "memory": describe_memory(),
        }
        self.log({"event": "done", **summary})
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a training config JSON")
    parser.add_argument("--max-steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--max-minutes", type=float, default=None, help="override the wall-clock budget")
    parser.add_argument("--seed", type=int, default=None, help="override the seed (init and data order)")
    parser.add_argument("--lr", type=float, default=None, help="override the peak learning rate")
    parser.add_argument("--resume", default=None, help="override the resume setting")
    parser.add_argument("--out-dir", default=None, help="override the output directory")
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_minutes is not None:
        cfg.max_minutes = args.max_minutes
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.lr = args.lr
    if args.resume is not None:
        cfg.resume = args.resume
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
