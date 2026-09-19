"""FastAPI server providing a streaming chat interface to the trained model.

    python -m flybrain.serve --checkpoint checkpoints/run/best.pt --port 8000

`GET /` serves a small self-contained chat page; `POST /api/chat` streams
server-sent events so the browser shows tokens as they are produced.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator

from pydantic import BaseModel

from flybrain.device import device_report, tune_rocm
from flybrain.generate import SamplingConfig, chat_reply, load_model
from flybrain.tokenizer import load_tokenizer


class ChatRequest(BaseModel):
    """Request body for `/api/chat`.

    Defined at module scope on purpose. With `from __future__ import annotations`
    in effect, annotations are strings resolved against the module globals, so a
    model declared inside a function cannot be resolved by FastAPI -- it then
    treats the parameter as a query field and every request fails validation.
    """

    message: str
    history: list[dict] = []
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    max_new_tokens: int = 256

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlyBrain-LLM 对话</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background: #0d1117; color: #e6edf3; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 12px 18px; border-bottom: 1px solid #21262d; display: flex;
           align-items: baseline; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header span { font-size: 12px; color: #8b949e; }
  #log { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 820px; padding: 10px 14px; border-radius: 10px; white-space: pre-wrap;
         line-height: 1.6; word-break: break-word; }
  .user { align-self: flex-end; background: #1f6feb33; border: 1px solid #1f6feb66; }
  .assistant { align-self: flex-start; background: #161b22; border: 1px solid #21262d; }
  .meta { font-size: 11px; color: #8b949e; margin-top: 6px; }
  footer { border-top: 1px solid #21262d; padding: 12px 18px; display: flex; gap: 10px;
           align-items: flex-end; }
  textarea { flex: 1; resize: none; background: #161b22; color: #e6edf3; border: 1px solid #30363d;
             border-radius: 8px; padding: 10px; font: inherit; min-height: 44px; max-height: 200px; }
  button { background: #238636; color: #fff; border: 0; border-radius: 8px; padding: 10px 18px;
           font: inherit; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .controls { display: flex; gap: 14px; align-items: center; font-size: 12px; color: #8b949e; }
  input[type=range] { width: 110px; }
</style>
</head>
<body>
<header>
  <h1>FlyBrain-LLM</h1>
  <span id="info">果蝇大脑连接组接线的语言模型</span>
</header>
<div id="log"></div>
<footer>
  <div style="flex:1">
    <textarea id="input" rows="1" placeholder="问点关于苍蝇大脑的问题…（Enter 发送，Shift+Enter 换行）"></textarea>
    <div class="controls">
      <label>温度 <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.8">
        <span id="tempv">0.8</span></label>
      <label>最多生成 <input type="number" id="maxnew" value="256" min="16" max="1024" step="16"
        style="width:64px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px"></label>
      <button id="clear" style="background:#30363d;padding:6px 12px">清空</button>
    </div>
  </div>
  <button id="send">发送</button>
</footer>
<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const send = document.getElementById('send');
const temp = document.getElementById('temp');
const tempv = document.getElementById('tempv');
const maxnew = document.getElementById('maxnew');
let history = [];

temp.oninput = () => tempv.textContent = temp.value;

function bubble(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

async function ask() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  send.disabled = true;
  bubble('user', text);
  const out = bubble('assistant', '');
  const meta = document.createElement('div');
  meta.className = 'meta';
  const started = performance.now();
  let chars = 0;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, history, temperature: parseFloat(temp.value),
                            max_new_tokens: parseInt(maxnew.value)})
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status + ': ' + await resp.text());
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\\n\\n');
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.split('\\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.delta) { out.textContent += payload.delta; chars += payload.delta.length;
                             log.scrollTop = log.scrollHeight; }
        if (payload.error) { out.textContent += '\\n[错误] ' + payload.error; }
        if (payload.done && payload.history) history = payload.history;
      }
    }
  } catch (err) {
    out.textContent += '\\n[请求失败] ' + err.message;
  }
  const secs = (performance.now() - started) / 1000;
  meta.textContent = chars + ' 字 / ' + secs.toFixed(1) + ' 秒';
  out.appendChild(meta);
  send.disabled = false;
  input.focus();
}

send.onclick = ask;
document.getElementById('clear').onclick = () => { history = []; log.innerHTML = ''; };
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
input.focus();
</script>
</body>
</html>
"""


def build_app(checkpoint: str, tokenizer_path: str, device: str, dtype_name: str, system: str):
    """Create the FastAPI app, loading the model once at startup."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    tune_rocm()
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    if not torch.cuda.is_available():
        device, dtype = "cpu", torch.float32

    tokenizer = load_tokenizer(tokenizer_path)
    model, model_cfg = load_model(checkpoint, device=device)
    report = device_report()

    app = FastAPI(title="FlyBrain-LLM")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/health")
    def health() -> dict:
        arch = getattr(model_cfg, "arch", "transformer")
        return {
            "status": "ok",
            "arch": arch,
            "params": model.param_count(),
            "ffn_mode": model_cfg.ffn_mode if arch != "connectome" else None,
            "layers": model_cfg.n_layers if arch != "connectome" else 0,
            "context": model_cfg.context,
            "neurons": model.brain.circuit.n_neurons if model.brain is not None else 0,
            "synapses": int(model.brain.circuit.pre.numel()) if model.brain is not None else 0,
            "device": report.get("name", device),
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in request.history]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": request.message})
        sampling = SamplingConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        def event_stream() -> Iterator[str]:
            collected: list[str] = []
            try:
                for delta in chat_reply(
                    model, tokenizer, messages, cfg=sampling, device=device, dtype=dtype
                ):
                    collected.append(delta)
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            except Exception as exc:  # surface the failure to the browser
                yield f"data: {json.dumps({'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False)}\n\n"
            reply = "".join(collected)
            new_history = [m for m in messages if m.get("role") != "system"]
            new_history.append({"role": "assistant", "content": reply})
            yield f"data: {json.dumps({'done': True, 'history': new_history[-40:]}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.model = model
    app.state.config = model_cfg
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenized/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--system", default="你是一只果蝇大脑神经网络语言模型，用中文准确回答关于果蝇大脑与连接组的问题。")
    args = parser.parse_args()

    import uvicorn

    app = build_app(args.checkpoint, args.tokenizer, args.device, args.dtype, args.system)
    print(f"serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
