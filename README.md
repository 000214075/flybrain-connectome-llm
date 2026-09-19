# FlyBrain-LLM：用真实苍蝇大脑连接组接线、并在 GPU 上运行整个大脑的语言模型

本项目在一台普通 Windows 11 电脑（AMD Ryzen 9 7950X + Radeon RX 7900 XTX）
上完成整条链路：下载真实果蝇连接组 → 用真实接线确定模型内部网络 →
把**全部 164,587 个神经元、2556 万条突触**放进语言模型的前向与输出路径 →
训练一个中文语言模型 → 提供命令行与网页对话界面。全程使用 AMD ROCm 生态。

**最终交付的模型是纯连接组模型**（`arch: "connectome"`）：没有注意力、没有位置编码、
没有归一化层、没有残差流、没有 KV 缓存、没有 transformer block，
序列建模整个交给连接组自身的递归。它**已训练完成**，但它**不是本仓库语言能力
最好的模型**——混合模型（`arch: "transformer"`）的困惑度更低。两个模型的
实测对比与原因见下面「架构」一节和 `reports/FINAL_REPORT.md`。

---

## 先澄清三件事

1. **DeepSeek API 不能训练模型**，它只能推理。所以它的角色是**教师模型**：生成蝇脑领域
   语料（讲解、问答、多轮对话），再用这些语料训练本地模型。这叫知识蒸馏，
   **没有任何权重从 API 传过来**。

2. **连接组不是语言模型**，它是一张接线图。这里用它做一件有据可依的事：
   **固定模型中某个子层的连接结构**，只让少量参数可学习。这是
   Lappalainen et al., *Nature* 2024（doi:10.1038/s41586-024-07939-3）验证过的方法。

3. **本项目不预设"蝇脑接线一定有用"**。已有开源项目 `nftechie/flm` 把雄性蝇脑连接组
   接到 1.2B 模型上，作者自己承认对照组更好。所以这里内置**三臂对照实验**，
   结论如实报告，包括负结果。

---

## 真实数据

| 数据集 | 内容 | 规模 | 许可 |
|---|---|---|---|
| **Janelia MaleCNS v1.0** | 雄蝇全中枢神经系统连接组 + 细胞类型 + 神经递质预测 | 164,587 神经元 / 2556 万条带类型突触 | CC-BY-4.0 |
| 中文维基百科 + TinyStories | 通用中英文语料 | 2.09 亿 token | CC-BY-SA / CDLA |

从 MaleCNS 提取出的蘑菇体回路（数字直接来自数据）：

```
投射神经元 PN      692        Kenyon 细胞 KC    4,064      MBON 97      APL 2（双侧一对）
PN→KC 突触      22,586        KC→MBON 突触     61,210
每个 KC 平均输入   5.56   ← 文献约 5–10，吻合
有 PN 输入的 KC  93.8%
```

全脑回路（`data/wholebrain.npz`，127 MiB）：

```
神经元        164,587（全部）
突触        25,563,096（全部唯一连接）
兴奋性       106,791（乙酰胆碱）
抑制性        57,255（GABA 22,055 / 谷氨酸 29,296 / 组胺 5,904）   ← 抑制是真的
神经调质         541（多巴胺 / 章鱼胺 / 血清素，不参与快速突触电流）
平均入/出度    155.3（最大入度 11,526）
```

---

## 架构

仓库里有**两个模型**，都跑在同一个 164,587 神经元连接组上。选哪个由配置里的
`model.arch` 决定。

| `arch` | 序列建模靠什么 | 同一留出集 val_ppl | 交付定位 |
|---|---|---|---|
| `"connectome"` | **只有连接组自身**：每个 token 完整扫一遍 2556 万条突触 | **30.811** | **最终交付的模型** |
| `"transformer"` | 注意力 + RoPE + RMSNorm + 残差 + KV 缓存 | **10.42** | 早期版本，保留可复现 |

两者都在 context 32、`data/tokenized_domain/val.bin` 上用同一套窗口打分。

**"整脑在环"的指标上纯模型更好（每 token 100% 覆盖、突触传递 45.1% vs 11.3%、
批参与度 6.48/7 vs 5.39/8）；语言能力上混合模型更好。** 原因是代价结构：
纯模型每个 token 必须完整扫过全部突触，混合模型把 512 个 token 池成 4 块只扫 4 次。
详见 `reports/FINAL_REPORT.md` §3.3 与 §7.5。

### A. 纯连接组模型（`arch: "connectome"`）

**除文本进出接口之外，没有任何现代 LLM 部件**：没有注意力，没有位置编码，
没有归一化层，没有残差流，没有 KV 缓存，没有 transformer block。词的顺序、
"上一个 token 重要"这些事，只能来自连接组自己的递归：

```
token id ──► 嵌入 (vocab → 256) ──► 感官投射 (256 → 164,587) ──► 驱动
每个位置 t：
    ① 读出：logits = 词表头( 低秩读出( state ) )      ← 先读，再写
    ② 演进：state = LIF( W · state + drive_t )
              W = 25,563,096 条真实突触（106,791 兴奋 / 57,255 抑制）
```

因果性不靠掩码：**先读后写**这个顺序本身就是因果——位置 t 的预测只能看到 t 之前
写进去的东西。蘑菇体在这个模型里也不是单独模块，它的 Kenyon 细胞与 MBON
就是同一个连接组里的细胞。

剩下的只有接口：一个嵌入表和一个到词表的线性映射。任何模型（生物与否）都得有
文本进出口，而这两个都不跨位置混合信息，所以都不承担序列建模。

`scripts/diagnose_readout.py` 是这套设计的检查工具，它回答纯递归模型唯一不能
缺的性质：**不同输入跑完连接组之后还分不分得开**。

### B. transformer 混合模型（`arch: "transformer"`）

**这是标准 decoder-only transformer**（RoPE、RMSNorm、KV 缓存），
有两处接线来自数据。它**不是**"基于苍蝇大脑的 LLM"，而是"带苍蝇接线的
transformer"，两者必须区分清楚。

### 1. 蘑菇体前馈子层（每个 block）

```
x ──► PN 投影（可学习）                d_model → 692
   ──► PN→KC 固定稀疏矩阵（真实突触）    692 → 4064
   ──► APL 全局抑制（1 个可学习标量）     按真实 APL→KC 突触数加权
   ──► LIF 脉冲神经元 + k-WTA            约 5% 的 KC 同时激活
   ──► KC→MBON 固定稀疏矩阵（真实突触）   4064 → 97
   ──► MBON 投影（可学习）               97 → d_model
```

### 2. 全脑回路（与 transformer 堆栈交错，覆盖每个 token、每个 block）
```
每个 token 块 x (B,c,d)：
    ① 读出：state(164,587) ──► 低秩投影 ──► d_model ──► 送进**全部 8 个 block**
    ② 栈内计算：8 个 block 依次处理该块（注意力对前面所有块可见）
    ③ 写入：state = LIF( W · state + drive(x) )
              W = 25,563,096 条真实突触，含 57,255 个抑制性神经元
```

**每一次前向传播，GPU 都会完整扫过全部 164,587 个神经元和 2556 万条突触**
（`brain_chunks × brain_iters` 次，最终配置为 4 次；显存与时间的实测上限见报告 §5）。
训练日志每步打印三个实测利用率：

```
brain 10.7% syn 11.3% 4sweeps
      │        │      └── 本次前向对连接组的扫描次数
      │        └── 真正传递了脉冲的突触占比（按突触强度加权）
      └── 放电神经元占比
```

**交错接线是关键**：把连接组接在堆栈**之后**时，它虽然完整运行了 2556 万条突触，
输出却只进入最后的语言模型头——实测 **0/8 个 block 受它影响**。
交错之后，每个 token 块进入每个 block 之前都会拿到一次脑读出，
脑回路真正参与整条输出路径（实测 **8/8 个 block**）。
块内注意力通过键值状态向前传递，因此分块处理与整段处理在数学上完全等价。

**放电比例是受硬约束保护的**：训练早期，模型的梯度倾向于把脑回路压到静默
（实测从 15.9% 一路掉到 0.7%），因为压制一个看起来像噪声的模块比学会利用它更省事。
现在有三个机制阻止这件事：

1. **放电稳态正则**（`brain_firing_penalty`）：惩罚放电比例偏离目标（默认 5%）。
2. **三个增益都有下限**：突触释放幅度、膜电位积分速率、token 驱动——训练可以把它们
   调高，但乘积不可能把回路压到阈值以下。
3. **每步记录放电比例**，静默会被立刻发现。

**驱动带宽也要盯着**：`up(down(x))` 是 token 上下文进入脑回路的唯一通道。
实测 1500 步之后 `down` 仍是满秩（64），但 `up` 塌到了**秩 2**，合成映射**有效秩 1**
——即 164,587 个神经元实际上只被**一个标量**驱动，连接组只能当增益旋钮用。
现在有 `brain_drive_orthogonality` 正则项阻止这种塌缩，
`scripts/measure_brain_utilization.py` 会把两个因子的有效秩都测出来。

**脑在跑 ≠ 脑在传信息**（这一轮最关键的发现）：把深度覆盖、梯度可达性、突触扫描
全部推到 100% 之后，还剩下一个问题——脑的输出到底区分不区分 token。
新增的 `batch participation` 指标（一批 8 条真实文档，脑给出多少个独立方向）测出：
**8 条毫不相干的文档跑完整个连接组之后，164,587 个神经元的脉冲图案逐位完全相同**，
读出参与度 **1.00 / 8**——整个脑对语言模型只能贡献一个**常量**。
根因是脉冲的量化：驱动差异远小于一个阈值的间隔，被阈值全部抹平。两个修复：
`brain_analog_readout`（把阈下膜电流也送进读出，参与度 1.00 → 2.23）
与 `brain_participation_penalty`（直接惩罚"给整批发同一个向量"，再训 150 步到 **4.39 / 8**）。

可训练参数：输入/输出投影、APL 增益、LIF 时间常数、每个神经元的释放强度。
**接线本身是苍蝇的，不是学出来的。** 神经元的兴奋/抑制符号来自真实神经递质且
**固定不可学**（`test_whole_brain_inhibition_is_signed_and_not_trainable`），
只有释放幅度可学——因为突触强度正是连接组里没有的未知生物物理量。

### 三臂对照

| 臂 | `ffn_mode` | 含义 |
|---|---|---|
| A | `mb` | **真实连接组接线** |
| B | `mb-shuffled` | 保持每个神经元连接数量不变，随机重连（配置模型构造） |
| C | `swiglu` | 标准前馈，不含连接组 |

实测 A 与 B 耗时几乎相同（37,343 vs 37,264 tok/s），所以这是**计算量严格匹配**的对照。

---

## 实测性能与 GPU 利用率

硬件：Radeon RX 7900 XTX（gfx1100，48 CU，23.98 GB）、ROCm 7.10、torch 2.9.1。

| 指标 | 实测值 |
|---|---|
| bf16 矩阵乘 4096³ | **77.2 TFLOPS** |
| fp32 矩阵乘 4096³ | 2.98 TFLOPS（仅为 bf16 的 1/26，故全程 bf16） |
| **训练中 GPU 计算引擎利用率** | **平均 86.0%，中位 87.3%，峰值 99.6%** |
| 训练吞吐（全脑 + 蘑菇体，4135 万参数） | **28,278 token/s** |
| 单次全脑扫描（16 流，前向） | 3.4 ms（CSR）／10.2 ms（gather-scatter） |
| 单次全脑扫描（16 流，前向+反向） | 39.8 ms（CSR 自定义反向）／77.2 ms |

利用率由 Windows 性能计数器实测（`scripts/gpu_util.py`），不是从吞吐推算的。

### 为了榨干这张 AMD 卡做的优化（全部基于实测）

1. **bf16 全流程**。fp32 矩阵乘慢 26 倍，等于不可用。
2. **打开 AOTriton 注意力内核**：`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`。
   默认关闭时 Flash / 高效注意力都不可用，只有 MATH 回退路径；打开后两者都可用。
3. **稀疏扫描用 CSR 矩阵乘而非 `index_add_`**：实测 368 GB/s vs 112 GB/s，
   快 3.3 倍。
4. **`check_invariants=False`**：`torch.sparse_csr_tensor` 默认会校验索引结构，
   在 2556 万个索引上要花约 30 ms，是它所保护的矩阵乘的 4 倍。
5. **手写反向传播**：`torch.sparse.mm` 的反向会稠密化 164587²，申请约 101 GiB 而 OOM。
   改为：输入梯度走转置 CSR 矩阵乘，权重梯度走分块归约（无原子操作）。
   端到端因此快 1.94 倍。
6. **全脑固定 fp32**：这套构建的 hipSPARSE 不支持 bf16 稀疏乘
   （`hipsparseSpMM not implemented`），且脑状态只有几十 MB，用 fp32 代价可忽略。
7. **分块读出**：把读出放在更新之前，既保证因果性，又让全脑扫描次数与读出次数解耦。
   `chunks=2, iters=1` 比 `chunks=4, iters=1` 快一倍（28.3k vs 15.3k tok/s）。

---

## 安装（原生 Windows + AMD ROCm，不需要 WSL）

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio `
  --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/
.venv\Scripts\python.exe -m pip install numpy pandas pyarrow tokenizers requests tqdm `
  fastapi "uvicorn[standard]" pytest
.venv\Scripts\python.exe -m pip install -e .
```

一键脚本 `scripts/setup_windows_amd.ps1`。

> **关于 WSL**：本机 WSL 里预装了 ROCm 7.2.4，但那是**标准 Linux 版**，依赖
> `/dev/kfd` 与 `amdgpu` 内核模块——WSL 没有。WSL 只通过 `/dev/dxg` 暴露 GPU，
> 而桥接库 `librocdxg.so` 缺失，`rocminfo` 报 `hsa_init Failed`。
> 因此走原生 Windows 路线，实测可用。

---

## 完整流程

```powershell
$py = ".venv\Scripts\python.exe"

# 1. 数据
$py scripts\download_parallel.py <url> <dest>       # 连接组（8 线程分块，绕开单连接限速）
$py scripts\download.py configs\corpus.json data    # 语料

# 2. 连接组 → 回路
$py scripts\connectome_tool.py inspect --data data\connectome\malecns   # 先看真实列名
$py scripts\connectome_tool.py build --source malecns `
      --data data\connectome\malecns --out data\pathway_malecns.npz     # 蘑菇体回路
$py scripts\build_wholebrain.py --data data\connectome\malecns `
      --out data\wholebrain.npz                                         # 全脑 + 递质符号

# 3. 分词器 + token 打包
$py scripts\prepare_data.py --corpus data\corpus --out data\tokenized --vocab-size 16384

# 4. 抓取公开的果蝇脑资料（FlyWire / 连接组文献 / 中文报道）
$py scripts\fetch_sources.py                       # 默认走 127.0.0.1:6696 代理
$py scripts\fetch_sources.py --list                # 看抓取清单

# 5. DeepSeek 教师语料——需要有效密钥
$env:DEEPSEEK_API_KEY = "sk-..."
$py -m flybrain.teacher_cli --probe
# 自述式（教师凭自己的知识写）
$py -m flybrain.teacher_cli --out data\corpus\domain\flybrain.jsonl --records 300
# 有据式（每条都锚定在抓取到的真实段落上，准确得多）
$py -m flybrain.teacher_cli --out data\corpus\domain\grounded.jsonl `
    --records 420 --mode docqa --max-tokens 2048 --workers 12

# 6. 混合语料 + 领域微调数据（源文重复 8 次、教师记录 60 次、通用文本占 25%）
$py scripts\build_domain_corpus.py --repeat 60 --source-repeat 8 --base-fraction 0.25
$py scripts\prepare_data.py --corpus data\corpus_domain --out data\tokenized_domain `
    --skip-tokenizer --tokenizer data\tokenized\tokenizer.json

# 7. 训练
$py -m flybrain.train --config configs\train_full.json        # 基座
$py -m flybrain.train --config configs\train_domain_v2.json   # 领域微调（也用于热启动加宽秩）

# 8. 对话
$py -m flybrain.cli   --checkpoint checkpoints\flybrain-full\best.pt
$py -m flybrain.serve --checkpoint checkpoints\flybrain-full\best.pt --port 8000

# 9. 纯连接组模型（最终交付，不含任何现代 LLM 部件）
$py -m flybrain.train --config configs\train_connectome_only.json
$py -m flybrain.cli   --checkpoint checkpoints\flybrain-connectome\best.pt
$py -m flybrain.serve --checkpoint checkpoints\flybrain-connectome\best.pt --port 8000
```

> **为什么教师语料要"有据"**：让教师凭记忆写果蝇脑，得到的是流畅但真伪不明的文字；
> 而自由问答模式在 `deepseek-flash` 上**根本产不出东西**——它是推理模型，
> 推理过程会把整个 token 预算吃光，实测 4096 也仍然是空回答。
> 把每条记录锚定在一段抓取到的公开段落上同时解决两个问题：
> 事实来自资料，而回答只需要复述眼前的内容，预算够用。统计上，
> 有据模式在 12 并发下约 **36 条/分钟**，纯自述模式约 9 条/分钟。

## 训练流程包含什么

`src/flybrain/train.py`：bf16 混合精度、梯度累积与裁剪、线性 warmup + 余弦退火、
权重衰减只作用于矩阵、定期验证困惑度、自动保存最优检查点、断点续训、
定期采样预览、JSONL 指标日志、`Ctrl+C` 优雅保存、`--max-minutes` 墙钟预算、
**以及每 150 步一次的全脑放电比例监测**。

## 项目结构

```
src/flybrain/
  connectome.py    连接组加载、细胞类型规则、蘑菇体回路提取、对照重连
  wholebrain.py    全脑回路：CSR 稀疏扫描 + 自定义反向 + 神经元动力学
  neurons.py       LIF 脉冲神经元 + SuperSpike 替代梯度 + k-WTA
  model.py         transformer 混合模型（蘑菇体 + 全脑模块，含统一缓存）
  connectome_lm.py 纯连接组模型：无注意力/位置编码/归一化/残差/KV 缓存的序列模型
  data.py          语料读取（txt/jsonl/parquet）、token 打包、窗口数据集
  tokenizer.py     字节级 BPE（中英文无损）
  train.py         完整训练流程
  generate.py      流式采样、聊天模板、检查点加载
  cli.py / serve.py  命令行对话 / FastAPI + SSE 网页对话
  teacher.py        DeepSeek 教师语料生成
  device.py         ROCm 环境调优与设备自检
scripts/           数据下载、回路构建、基准测量、性能剖析、显存诊断
tests/             66 项测试
```

## 自动循环（AutoLoop）

本项目装了一个"自己往下跑"的循环：**每次问答彻底结束后，自动把北极星要求再发一遍**，
由 Stop 钩子（turn 内续跑）与调度器（1 小时补位、durable）两个引擎配合，
单引擎做不到"不停"——细节与原因见 [`autoloop/README.md`](autoloop/README.md)。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Status    # 看状态
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Disable   # 停
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Enable    # 再开
```

**要正常聊天先 `-Disable`**，否则它会拦住每一次收工。另注：Stop 钩子是刚写入磁盘的，
grok 在会话启动时读钩子，所以当前会话里需要 `/hooks` → 按 `r` 重载（或重启会话）才生效；
调度器那半边已经生效。


## 测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

其中几项专门守住容易静默出错的地方：

- `test_kv_cache_matches_full_forward`：逐 token 增量解码必须与整段前向一致
- `test_whole_brain_cache_matches_full_forward`：脑状态跨调用延续必须一致
- `test_whole_brain_is_causal`：改动最后一个 token，之前所有位置必须**逐位不变**
- `test_whole_brain_is_alive_at_initialisation`：脑回路初始化时必须放电
- `test_whole_brain_inhibition_is_signed_and_not_trainable`：抑制符号不可被训练抹掉
- `test_the_connectome_edges_are_not_in_the_optimiser`：2556 万条突触计数不在
  `named_parameters()` 也不在优化器参数组里，跑完一次 backward + step 后逐位不变
- `test_the_pure_model_contains_no_transformer_machinery`：纯模型的模块与参数名里
  不得出现 `attn`/`rope`/`norm1`/`norm2`/`norm_f`/`blocks`/`q_proj`/`kv`，
  且改 token k **必须**影响紧随其后的那个预测
- `test_generation_and_training_agree_on_the_alignment`：`forward` 与 `generate`
  两条路径逐位一致（防住本项目真实出现过的目标错位缺陷）
- `test_incremental_decoding_of_the_pure_model_matches_one_full_pass`：
  纯模型逐 token 解码与整段前向逐位一致
- `test_the_pure_model_survives_a_checkpoint_round_trip`：纯架构检查点能被
  `flybrain-chat` / `flybrain-serve` 的加载器重建
- `test_both_sweep_implementations_agree_when_given_the_right_layout`：
  连接组的两个扫描实现（gather/scatter 与稀疏 CSR）在 fp32 舍入内必须一致；
  它同时守住一个**极易踩的 API 契约**——CSR 的边按突触后排序，所以必须喂
  `prepare_synapses()`（已置换）而不是 `effective_synapses()`（未置换），
  喂错了会在 99.9% 的输出上悄悄算错（我自己就踩了这个坑）

---

## 局限（如实说明）

- **纯连接组模型的语言能力仍明显更弱**：同一留出集上 val_ppl **30.811**
  （rank512、step 17400、186,168,794 参数），而混合模型是 **10.42**。根因是代价结构——
  纯模型每个 token 都要完整扫过 2556 万条突触，混合模型把 512 个 token 池成 4 块
  只扫 4 次。同样 40 分钟，混合模型能看约 2000 万 token，纯模型只有约 82 万。
  历史上优化过三轮：扫描改用稀疏 CSR 内核（**356 tok/s vs `index_add` 的 218 tok/s，
  1.63 倍**，两种内核在同步调度表下 450 步内质量无差异）、把余弦调度表铺满预算
  （原来 600 步的表在第 600 步就把学习率衰减到地板，浪费了 10 分钟预算）、
  以及把学习率从 3e-5 提到 **6e-5**（贡献最大的一段）。这三处合起来把同一个
  40 分钟预算下的困惑度从 274.6 降到 **180.4（−34.3%）**，492 → 804 步。
  其后把预算拉到 18000 步、读出秩从 256 提到 512，同一份固定 24 批评估下
  困惑度才从 37.90（rank256）走到 **30.811（−18.7%）**。即便如此，与混合模型的
  差距仍是数量级的。
- **它至今不能正常对话（最后一轮实测）**：采样输出是训练语料里的排版模板碎片，
  常在冒号处断掉，不是回答。根因已定位，而且不是"训练时间不够"：
  **18.4M 训练 token ÷ 186.17M 参数 = 0.10 token/参数**，比让模型学会对话大致
  需要的量少约两个数量级。所以沿用同一配方继续训练不会有质变，先要解决语料规模。
  样本原文见 `reports/serve_chat_samples.md`。
- **学习率的"3e-5 上限"曾是个假象，已更正**：那个上限出自一个**恒定学习率、
  没有 warmup、context 8 / batch 8** 的探针，和真实配方不是一回事。用真实配方
  （warmup 40 + 余弦、context 32、batch 32）、同调度表、同 seed、同步数重测后：
  **6e-5 比 3e-5 好 0.2453**（同一份固定 24 批，第 200 步），是实测 seed 散布
  0.0395 的 **6.2 倍**；1e-4 与 6e-5 不可分辨；**2e-4 才真的更差**。
  而且四条臂**没有任何一步**的验证损失高于均匀猜测 9.7041——旧记录里
  "损失反而高于均匀猜测（22.1 / 12.6）"同样来自那个探针。
  细节见 `reports/FINAL_REPORT.md` §7.8。
- **两种扫描内核可以互换，450 步内质量没有可测量差异**：gather/scatter 与
  稀疏 CSR 在单次扫描上差 1e-7（fp32 舍入），在**同为 600 步调度表**的两条臂上
  跑 450 步后验证损失平均只差 **+0.0013、最大 0.0023**——是换 seed 散布
  （0.0395）的 3%，远在噪声之下。早先报告里"轨迹会在约 100 步后分开
  （第 150 步差 0.267）"的结论**已撤回**：那次对照的两条臂调度表长度不同
  （`max_steps` 600 vs 150），差距主要来自退火差异而非内核。
- **混合模型的语言能力受规模限制**：6264 万参数、约 1 亿 token 训练量，
  能产出领域词汇但成句仍不连贯，不可能有大模型的知识广度。
- **显存是主要瓶颈**：全脑模块会让 PyTorch 缓存分配器预留约 22 GiB（存活张量仅
  1.4 GiB）。纯模型 batch 32 时峰值 18.99 GiB，batch 64 会打满显存并把每 token
  代价从 5.08 ms 抬到 33.4 ms。这是当前吞吐进一步上升的主要障碍。
- **分块近似的代价（仅混合模型）**：读出发生在处理当前块之前，所以一个块内的
  token 看不到自己对脑状态的影响（这正是因果性的来源）。`chunks=2` 时第一块读出
  为零向量。纯连接组模型没有这个问题：每个 token 都是先写入再读出，覆盖率 100%。
  这是状态空间模型一类工作的标准折中。
- **脑回路用了全部神经元与突触，但神经递质只有符号**：真实的受体动力学、突触
  可塑性、神经调质的体积传递都没有建模。神经调质神经元不参与快速突触电流
  （符号 0），这是一个明确的简化。
- **KC 输入处加了 RMSNorm**，不符合生物学（真实 Kenyon 细胞没有这种归一化），
  是为了让固定扇入的回路在深层可训练。代码里已标注。
- **PN 与 MBON 投影是学出来的**，所以"输入输出如何编码"仍是模型自己学的，
  只有中间的接线路是苍蝇的。

---

## 复现实验

```powershell
# 三臂对照
$py -m flybrain.train --config configs\train_mb.json
$py -m flybrain.train --config configs\train_mb_shuffled.json
$py -m flybrain.train --config configs\train_swiglu.json

# 性能测量
$py scripts\bench_gpu.py                       # 硬件能力与矩阵乘上限
$py scripts\bench_wholebrain.py                # 全脑扫描吞吐
$py scripts\bench_sweep_impl.py                # CSR vs gather-scatter
$py scripts\bench_configs.py                   # 各前馈模式对比
$py scripts\gpu_util.py --seconds 40           # 实测 GPU 利用率
$py scripts\probe_memory_cause.py              # 显存占用归因
$py scripts\measure_brain_utilization.py `     # 混合模型全脑利用率（10 项实测指标）
    --config configs\train_domain_ft.json `
    --checkpoint checkpoints\flybrain-full\best.pt

# 纯连接组模型的诊断与测量
$py scripts\diagnose_readout.py                # 三种读法的输入保真度与梯度可达性
$py scripts\bench_connectome_lm.py             # batch/context 的代价曲线
$py scripts\measure_connectome_utilization.py ` # 连接组利用率（含突触传递、批参与度）
    --checkpoint checkpoints\flybrain-connectome\best.pt
$py scripts\tune_connectome_lr.py --lrs 3e-5   # 用新鲜数据扫学习率（别用固定 batch）
```

---

# 开源发布信息

## 许可证

本仓库**自己的代码与文档**采用 [The Unlicense](LICENSE)：奉献至公有领域，任何人可为
任何目的（含商业用途）复制、修改、出售、再分发，**无需署名、无需保留任何声明**。
两点如实说明：Unlicense 的"公有领域"声明在所有司法辖区未必可执行（兜底条款已写入
`LICENSE` 正文），且它**不授予专利权**。

**第三方数据不在授予范围内**，见下面「第三方数据」一节。

## 运行环境（钉版）

整套链路是在下面这一台机器上跑通的，版本是实测可用的组合：

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 |
| CPU | AMD Ryzen 9 7950X |
| GPU | AMD Radeon RX 7900 XTX（`gfx1100`，24 GB，960 GB/s，**96 MB Infinity Cache**） |
| GPU 栈 | 原生 Windows ROCm（TheRock 轮子），非 WSL |
| PyTorch | `2.9.1+rocm7.10.0a20251120` |
| Python | `.venv`（见 `pyproject.toml`） |
| Shell | PowerShell 5.1 |

必须设置的环境变量（不设会静默走慢路径或直接报错）：

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'
```

一键环境自检与安装：[`scripts/setup_windows_amd.ps1`](scripts/setup_windows_amd.ps1)，
设备与矩阵乘上限测量：`scripts\bench_gpu.py`。

这套栈有四个**踩过才知道**的约束，写在最前面以免你重复花时间：

1. **Windows ROCm 没有 Triton**，所以 `torch.compile` 用不了；所有内核优化都靠手写
   算子（`csr-scaled`）与调度，不是靠编译器。
2. **hipSPARSE 缺 bf16 的 SpMM**（`hipsparseSpMM not implemented`），稀疏乘法得自己写。
3. **96 MB Infinity Cache 对本项目格外重要**：这个工作负载是"按行不规则 gather"，
   显存带宽吃不满，靠大缓存兜住行长尾。换卡时这一项比峰值带宽更能预测吞吐。
4. **训练输出不要写 C:**。一个训练臂靠页文件预留会吃掉 25–33 GB，C: 会被顶到 0；
   本项目的检查点都写在 E:/I:。

## 权重

训练时的 `best.pt` 里有一半以上体积是 AdamW 动量，对推理和微调都没用，所以
发布用的权重**只保留模型本体与元数据**（`model` / `model_config` / `config` /
`step` / `best_val` 五个键），体积约为原来的 40%。它们**不能用于断点续训**。

权重发布在 [`Releases`](https://github.com/000214075/flybrain-connectome-llm/releases/latest)
（v1.0，14 个文件，合计约 10 GB），不随 git 仓库分发。附件 `manifest.json` 记录每个文件的
来源运行、步数、参数量、验证损失与 SHA256。三组权重各自**同步数、可直接横比**：

| 组 | 文件 | step | 固定 24 批 val_loss / ppl |
|---|---|---|---|
| 交付模型 | `canonical-rank512.pt`（186,168,794 参数） | 17400 | **3.4279 / 30.811** |
| 读出秩扫描 | `rank128/256/512/1024.pt` | 600 | 5.8264 / 5.3059 / 4.8979 / **4.5997** |
| 生物电刺激五臂 | `biospike-{nomask,inonly,outonly,random,sensory}.pt` | 450 | 4.9549 / 5.4531 / 6.2474 / 6.7510 / 7.2254 |
| 端口掩码对照 | `ports-{all,random-sensory,sensory,output}.pt` | 500 | 5.3785 / 5.5953 / 5.6194 / 6.3483 |

剥离优化器**没有改变模型本身**：对发布用的 `canonical-rank512.pt` 重跑同一条固定 24 批
命令，结果仍是 **3.4279 / 30.811**，与原始 `best.pt` 逐位一致。

**度保持打乱接线的对照臂（step 150）在存档组里，但不要拿它跟别的臂横比**：它配对的
真实接线臂（492 步）在 canonical 晋升时被覆盖，两者步数不同，放在一起会诱导出一个报告里
明确排除掉的对比（同步数结论见 `reports/FINAL_REPORT.md` §7.6，第 50/100/150 步的差是
0.0010 / 0.0012 / 0.0016）。用 `scripts/shuffle_wholebrain.py` 可以重建它。

另有 [`v1.0-checkpoints-extra`](https://github.com/000214075/flybrain-connectome-llm/releases/tag/v1.0-checkpoints-extra)：
把其余 **27 条实验臂**的权重一并存档（学习率、随机种子、扫描内核、数据加载、配方演化各组），
约 17 GB。其中 `hybrid-transformer.pt` 是**全项目语言能力最好的模型**（固定 24 批
ppl **10.42**），留作纯连接组模型的对照。那一组**不是**精选对照集——各臂步数与调度不同，
**不要跨臂比 `best_val`**。每次训练的配置与指标曲线在 [`runs/`](runs/)，日志在 [`logs/`](logs/)。

加载方式与仓库内检查点完全一致：

```powershell
$py -m flybrain.generate --checkpoint canonical-rank512.pt --prompt "苍蝇大脑"
flybrain-serve --checkpoint canonical-rank512.pt --tokenizer data\tokenized_domain\tokenizer.json
```

## 进度与突破（截至 2026-09-19）

**已经站得住的：**

| 项目 | 数字 | 依据 |
|---|---|---|
| 交付模型是纯连接组、无任何现代 LLM 部件 | **186,168,794 参数，18 个张量全部是 `brain.circuit.*`**；无注意力/RoPE/RMSNorm/残差/KV 缓存 | `tests/test_the_pure_model_contains_no_transformer_machinery.py` |
| 自研稀疏内核 `csr-scaled` | 比 `csr` 快 **2.18×**、比 `index_add` 快 **4.58×**，且与朴素实现数值严格等价 | `scripts/verify_sweep_impls.py` |
| 扫描成本模型 | `ms/扫描 ≈ 5.1 + 0.44 × 批宽`；预测每步 1227 ms vs 实测 **1217 ms（误差 0.8%）** | `research/training_efficiency.md` |
| 性能瓶颈已定位 | 固定项只跑到 **~40 GB/s**（峰值的 4%，行长尾所致，最大入度 11,526）；每列项 **464 GB/s**（48%，近上限） | 同上 |
| 当前最优检查点 | 固定 24 批下 **3.4279 / ppl 30.811**（rank512、step 17400），旧 canonical 3.6349 / 37.90，**−18.7%** | `reports/eval_*_c32.json` |
| 测量纪律 | 同一检查点重测**逐位相同**（<0.0001）；seed 散布 **0.0395**，是所有"有无差异"结论的判据尺度 | `scripts/seed_variance.py` |

**有信息量但是否定的（同样值得借鉴，别重复踩）：**

- **度保持重接线：没测出差异。** 把连接组的突触目标整体置换、严格保持入/出度
  （逐边权重多重集相等、源神经元 100% 不变），固定 24 批下差异 **0.0016**，
  只有 seed 散布的 **0.04×**。
- **真实解剖端口比等数量随机端口更差**：+0.4744 ≈ **12× seed 散布**。而且这件事
  是**超加性**的——输入端掩码 +0.4982、输出端掩码 +1.2925、两者合起来 +2.271
  > 两者之和 1.791；随机端口的同样组合却精确落在加性预测上（6.7510 vs 6.7456，
  差 0.0054），说明差异出在**真实端口集本身**。
- **零训练接线测试**：稀疏（15,760 个感觉神经元）驱动下，真实接线激活了驱动集
  之外 **1000 个神经元（6.3%）**，打乱接线只有 **280 个（1.8%）**——真实接线
  扩散得**更多**，与"真实接线把活动约束在核心回路"的预期相反。
- **刺激侧从未按生物电标定**：一次扫描后只有 **13.55%** 的神经元落在阈值 ±0.5 的
  响应带内，所以早期"8 条无关文档跑出逐位相同的脉冲图案"不是脉冲本身的问题，
  而是刺激幅度远小于阈值间隔。
- **脉冲 vs 模拟是"更少步数、更好质量"的交换**：同样 20 分钟预算，脉冲臂只跑到
  **450 步**而模拟臂跑到 **600 步**（少 25%）；但固定 24 批下脉冲的 `nomask` 臂
  **4.9549** 仍优于模拟参考点 **5.3059**。注意这两者步数不同，**不是等步对照**。
- **翻倍参数是净亏**：rank256（97.4M）→ rank512（186.2M）让"每参数见过多少 token"
  从 0.19 掉到 **0.10**，这也是它至今不会对话的根本原因。

详细的推导、被推翻的预测、以及若干次自我更正都写在
[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) 与
[`research/`](research/) 里：

| 想了解 | 读 |
|---|---|
| 全部实测数据、逐条结论与更正记录 | `reports/FINAL_REPORT.md`（§7.x 是实验日志） |
| 训练效率与成本模型 | `research/training_efficiency.md` |
| 生物电刺激路线（诊断、五臂实验、文献） | `research/biological_training.md` |
| 文献综述与引用（含 FLM / FlyToLLM 对比） | `research/literature_flybrain_llm.md` |
| 下一批实验的设计与判据 | `research/next_experiments.md` |
| 纯模型的真实输出样本 | `reports/serve_chat_samples.md` |

## 第三方数据

本仓库**不包含**第三方数据集的原始文件，只包含下载与构建脚本（`scripts/download.py`、
`scripts/build_wholebrain.py`、`scripts/prepare_data.py`）。但**从 MaleCNS 构建出的连接组
张量单独发布**在 [`v1.0-derived-connectome`](https://github.com/000214075/flybrain-connectome-llm/releases/tag/v1.0-derived-connectome)
（`wholebrain.npz` 与两份打乱对照，共约 405 MB）——因为上游下载地址将来可能失效，而
CC-BY-4.0 允许派生分发。**这三个张量按 CC-BY-4.0 使用，需保留署名**；仓库自己的代码是
The Unlicense。请遵守各自条款：

| 数据 | 来源 | 用途 | 许可（以其发布页为准） |
|---|---|---|---|
| MaleCNS v1.0 | Janelia | 全脑连接组：164,587 神经元 / 25,563,096 突触 | CC-BY-4.0 |
| FlyWire | FlyWire 联盟 | 补充连接组与注释 | 见发布页 |
| 中文维基百科 + TinyStories | 公开语料 | 通用语言预训练 | CC-BY-SA / CDLA |
| DeepSeek 生成语料 | DeepSeek API | 蝇脑领域语料（教师蒸馏） | 受其服务条款约束，不随仓库或附件分发 |

连接组的引用（本项目使用的就是这一份）：

> *Sexual dimorphism in the complete Drosophila male central nervous system*,
> Cell, 2026-09-03；bioRxiv `10.1101/2025.10.09.680999`；PubMed 42691995。

本项目与 Janelia、FlyWire、DeepSeek 均无隶属关系。使用教师模型生成语料需要你自己的
API 密钥：`$env:DEEPSEEK_API_KEY = "sk-..."`（本仓库不含任何密钥）。
