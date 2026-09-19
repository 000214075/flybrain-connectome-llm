"""Interactive command-line chat with the trained FlyBrain model.

    python -m flybrain.cli --checkpoint checkpoints/run/best.pt
    python -m flybrain.cli --checkpoint ... --once "苍蝇大脑有多少神经元？"
"""

from __future__ import annotations

import argparse
import sys
import time

from flybrain.device import tune_rocm
from flybrain.generate import SamplingConfig, chat_reply, load_model
from flybrain.tokenizer import load_tokenizer

HELP = """commands:
  /reset          clear the conversation
  /temp <float>   set sampling temperature
  /max <int>      set max new tokens
  /topk <int>     set top-k
  /system <text>  set the system prompt
  /help           show this help
  /quit           exit
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenized/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--system", default="你是一只果蝇大脑神经网络语言模型，用中文准确回答关于果蝇大脑与连接组的问题。")
    parser.add_argument("--once", default=None, help="answer a single prompt and exit")
    args = parser.parse_args()

    tune_rocm()
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if not torch.cuda.is_available():
        args.device, dtype = "cpu", torch.float32
        print("no accelerator found; running on CPU")

    tokenizer = load_tokenizer(args.tokenizer)
    model, model_cfg = load_model(args.checkpoint, device=args.device)
    print(f"loaded {args.checkpoint}")
    if getattr(model_cfg, "arch", "transformer") == "connectome":
        print(
            f"  pure connectome model: no transformer stack, no attention, no "
            f"positional encoding; {model_cfg.d_model} -> "
            f"{model.brain.circuit.n_neurons:,} neurons -> vocabulary"
        )
    else:
        print(
            f"  params {model.param_count():,}  depth {model_cfg.n_layers}  width {model_cfg.d_model}  "
            f"context {model_cfg.context}  ffn {model_cfg.ffn_mode}"
        )
        if model_cfg.ffn_mode.startswith("mb"):
            layer = model.blocks[0].ffn
            print(
                f"  mushroom body: {layer.n_pn} PN -> {layer.n_kc} KC (k={layer.k} active) "
                f"-> {layer.n_mbon} MBON, {'spiking' if layer.spiking else 'rate'} LIF"
            )
    if model.brain is not None:
        circuit = model.brain.circuit
        print(
            f"  whole brain: {circuit.n_neurons:,} neurons, {circuit.pre.numel():,} synapses, "
            f"{model_cfg.brain_chunks} chunks x {model_cfg.brain_iters} sweeps per pass, "
            f"{'spiking' if model_cfg.brain_spiking else 'rate'} LIF -- every token passes through it"
        )

    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    history: list[dict] = [{"role": "system", "content": args.system}] if args.system else []

    def answer(prompt: str) -> None:
        history.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        tokens = 0
        pieces: list[str] = []
        for delta in chat_reply(model, tokenizer, history, cfg=sampling, device=args.device, dtype=dtype):
            print(delta, end="", flush=True)
            pieces.append(delta)
            tokens += 1
        elapsed = time.perf_counter() - started
        reply = "".join(pieces)
        history.append({"role": "assistant", "content": reply})
        if elapsed > 0:
            print(f"\n[生成 {tokens} 个 token，用时 {elapsed:.1f}s]\n")

    if args.once:
        print(f"user> {args.once}")
        print("assistant> ", end="")
        answer(args.once)
        return 0

    print(HELP)
    while True:
        try:
            line = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            command, _, value = line[1:].partition(" ")
            value = value.strip()
            if command in ("quit", "exit"):
                return 0
            if command == "help":
                print(HELP)
            elif command == "reset":
                history = [{"role": "system", "content": args.system}] if args.system else []
                print("conversation cleared")
            elif command == "temp":
                sampling.temperature = float(value)
                print(f"temperature = {sampling.temperature}")
            elif command == "max":
                sampling.max_new_tokens = int(value)
                print(f"max_new_tokens = {sampling.max_new_tokens}")
            elif command == "topk":
                sampling.top_k = int(value)
                print(f"top_k = {sampling.top_k}")
            elif command == "system":
                args.system = value
                history = [{"role": "system", "content": value}] if value else []
                print("system prompt updated")
            else:
                print(f"unknown command: /{command}")
            continue
        print("assistant> ", end="")
        answer(line)


if __name__ == "__main__":
    raise SystemExit(main())
