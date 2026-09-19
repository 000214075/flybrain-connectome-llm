# v1.0 — 纯连接组模型权重（仅模型本体）

这些是 [flybrain-connectome-llm](https://github.com/000214075/flybrain-connectome-llm) 的检查点，
**只保留模型本体与训练元数据**（`model` / `model_config` / `config` / `step` / `best_val`），
训练时的 AdamW 动量状态已被剥离，体积约为原始 `best.pt` 的 40%。

> ⚠️ **不能用于断点续训**（没有优化器状态），但可直接推理、采样、微调。
> ⚠️ **没有一条权重能正常对话**：最好的一条 ppl 30.811，采样输出是语料模板碎片。
> 它们适合用来研究、复现与二次开发，不适合当聊天模型用。原因见 README
> 「局限」一节：18.4M 训练 token ÷ 186M 参数 = 0.10 token/参数。

## 加载

```powershell
# 命令行采样
python -m flybrain.generate --checkpoint canonical-rank512.pt --prompt "苍蝇大脑"

# 网页对话（FastAPI + SSE）
flybrain-serve --checkpoint canonical-rank512.pt --tokenizer data\tokenized_domain\tokenizer.json
```

检查点里自带 `model_config` 与连接组张量，所以**不需要**再下载 MaleCNS 数据集就能加载。
每个文件的 SHA256、步数、参数量都记在附件 `manifest.json` 里。

## 交付模型

| 附件 | 运行 | step | 参数量 | 固定 24 批 val_loss / ppl |
|---|---|---|---|---|
| **`canonical-rank512.pt`** | **canonical 长训练（18000 步预算）** | **17400** | **186,168,794** | **3.4279 / 30.811** |

这是本项目的最终交付：18 个张量全部是 `brain.circuit.*` 与读出/嵌入层，
没有注意力、位置编码、归一化层、残差流、KV 缓存。评估用同一条固定 24 批命令
（`--batches 24 --batch 8 --context 32`），并已对本附件**重新跑过一遍确认逐位一致**。

## 读出秩扫描（同一配方、同为 600 步，可直接横比）

| 附件 | step | 参数量 | 固定 24 批 val_loss / ppl |
|---|---|---|---|
| `rank128.pt` | 600 | 53,081,306 | 5.8264 / 339.1 |
| `rank256.pt` | 600 | 97,443,802 | 5.3059 / 201.5 |
| `rank512.pt` | 600 | 186,168,794 | 4.8979 / 134.0 |
| `rank1024.pt` | 600 | 363,618,778 | **4.5997 / 99.45** |

结论：**读出侧在 1024 秩仍未饱和**（512→1024 还有 0.2982 的差距）。注意
`rank512.pt` 与 `canonical-rank512.pt` 参数量相同但不是同一次运行——前者是 600 步的扫描臂，
后者是 18000 步预算的长训练。

## 生物电刺激五臂（同为 450 步、同一配方，只差端口掩码，可直接横比）

`brain_spiking: true`（脉冲 LIF + SuperSpike 替代梯度），掩码之外的一切逐字段相同。

| 附件 | 刺激/读出端口 | step | 固定 24 批 val_loss / ppl |
|---|---|---|---|
| `biospike-nomask.pt` | 全部 164,587 个神经元 | 450 | **4.9549 / 141.9** |
| `biospike-inonly.pt` | 只有 15,760 个感觉神经元 | 450 | 5.4531 / 233.5 |
| `biospike-outonly.pt` | 只在 2,239 个输出神经元读出 | 450 | 6.2474 / 516.6 |
| `biospike-random.pt` | 等数量随机端口（对照） | 450 | 6.7510 / 854.9 |
| `biospike-sensory.pt` | 真实感觉输入 + 真实输出读出 | 450 | 7.2254 / 1373.9 |

结论（详见 `research/biological_training.md`）：**真实解剖端口比等数量随机端口更差**
（+0.4744 ≈ 12× seed 散布），而且这个代价是超加性的——输入掩码 +0.4982、
输出掩码 +1.2925，合起来却是 +2.271，而随机端口的同样组合精确落在加性预测上
（6.7510 vs 6.7456）。也就是说差异出在**真实端口集本身**，不在"掩码"这件事上。

## 端口掩码对照（模拟模式，同为 500 步，可直接横比）

| 附件 | 刺激/读出端口 | step | 固定 24 批 val_loss / ppl |
|---|---|---|---|
| `ports-all.pt` | 全部神经元（等价于无掩码） | 500 | **5.3785 / 216.7** |
| `ports-random-sensory.pt` | 等数量随机端口 | 500 | 5.5953 / 269.2 |
| `ports-sensory.pt` | 真实感觉神经元 | 500 | 5.6194 / 275.7 |
| `ports-output.pt` | 真实输出神经元 | 500 | 6.3483 / 571.5 |

## 没有发布的检查点，以及为什么

**度保持打乱接线的对照臂没有发布。** `checkpoints/flybrain-connectome-shuffled` 只跑到
**150 步**，而它配对的那条真实接线臂（492 步）在 canonical 晋升时被覆盖掉了。
两个文件步数不同，放在一起会诱导出一个报告里明确排除掉的对比——
真正的同步数对照结论在 `reports/FINAL_REPORT.md` §7.6：第 50/100/150 步的差是
**0.0010 / 0.0012 / 0.0016**，即"这个预算下没有可测量差异"，而这个差值只有
seed 散布（0.0395）的 0.04 倍。

想复现这个对照，用仓库里的
[`scripts/shuffle_wholebrain.py`](https://github.com/000214075/flybrain-connectome-llm/blob/main/scripts/shuffle_wholebrain.py)
生成打乱接线（只置换突触后端点，逐位保持入/出度与递质符号），再跑
`configs/train_connectome_only.json` 与 `configs/train_connectome_shuffled.json` 两条臂。

## 许可

权重以 **The Unlicense** 奉献至公有领域，与仓库代码一致，可自由用于任何目的。
但请注意：**MaleCNS / FlyWire 数据集与 DeepSeek 生成的语料不在授权范围内**，
本附件只包含从它们派生的模型权重。

---

## English summary

Weight-only checkpoints (optimiser state stripped, ~40% of the original size) for a
language model whose entire sequence path is the Janelia MaleCNS connectome —
164,587 neurons and 25,563,096 synapses, with no attention, positional encoding,
normalisation, residual stream or KV cache. `canonical-rank512.pt` is the deliverable
(step 17400, 186.2M parameters, val ppl 30.811 on a fixed 24-batch held-out set).
The rank sweep (128/256/512/1024, all at step 600) and the five spiking arms (all at
step 450) are step-matched and directly comparable with each other. The degree-preserving
rewiring control is deliberately not published because it is not step-matched with its
partner. None of these weights can hold a conversation; they are research artefacts.
Licensed under The Unlicense.
