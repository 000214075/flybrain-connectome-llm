# 下一批实验：设计、代码位置、判据

## ⚠️ 当前状态（每轮动手前先读这一段；最后更新 2026-09-19 17:20）

**✅ rank512 长训练已完成，且已替换 canonical**（2026-09-19 17:14 停，**跑满 18000 步**）：
`best.pt` = **`step 17400`**，同一条固定 24 批命令得 **3.4279 / ppl 30.811**（旧 canonical 3.6349 / 37.90，`step 16800`）
——**驱动脚本与主代理各独立跑一次，逐位复现**（4.9 s / 5.1 s；`reports/eval_flybrain-connectome-rank512-long_c32.json`
与 `reports/eval_rank512long_verify_c32.json`）。旧 canonical **已改名为 `best-rank256-16800-3.6349.pt` 留在同目录，未删除**；
`best.pt`/`last.pt`/`model_config.json`/`train_config.json`/`metrics.jsonl` 已同步换成 rank512 那套
（**186,168,794 参数**——比旧的 97.4M 翻了一倍，所以"每参数见过多少 token"反而更低）。
从 canonical 路径复核过（`reports/eval_canonical_promoted_c32.json` = 3.4279 / 30.811），
`flybrain.serve` 已重启并确认加载新模型（health: params 186,168,794）。**不要再重跑这条臂。**
**⚠️ 交互质量没有跨档**：用新模型采样仍是语料模板碎片（口径见 `reports/serve_chat_samples.md`）。
（历史留档：那条臂的锁是 `reports/rank_sweep.lock`、日志 `reports/rank_sweep-20260919-111404.log`、
输出 `E:\flybrain-connectome\rank512-long`；它最终**跑满 18000 步**，`best.pt` 停在 `step 17400`——
当时按 1.1833 s/步 外推的"照片终点"两难，结果是**跑完**这一边。）
**GPU 现在是空的**（那条臂 17:14 已停、锁已释放）——所以**下一步可以直接起一个 GPU 任务**，
但闸门仍会把空闲的 `flybrain.serve` 算作作业（见下面的闸门授权）；**一次有界评估可以与它并存，第二个训练作业不行**。
**⚠️ 提交内存（14:29 那条臂还在跑时实测：上限 80.84 GB / 可用仅 8.01 GB；private 合计 47.76 GB = 臂主进程 6.12 +
4 个 worker 各 4.88 + serve 6.73）⇒ 臂与 serve 都活着时不要起第三个进程**，尤其任何带 dataloader worker 的
（§7.33 那次把 C: 顶到 0.000 GB 的探针就是这么叠加出来的）。需要跑固定 24 批评估时，**最稳的顺序是等臂停再跑**。

**交互层（主代理追加）**：`src/flybrain/serve.py` 的 Web 对话界面**正在 8000 端口运行**
（WMI 拉起，本会话重启不会杀掉；`--checkpoint checkpoints\flybrain-connectome\best.pt`，
`--tokenizer data\tokenized_domain\tokenizer.json`）。它只在**生成回复的那几秒**用 GPU，
其余时间空转，所以不算"第二个 GPU 任务"；但**不要同时压它**（每次回复实测 0.6–3.1 秒）。
纯连接组模型的输出质量**首次落地**在 [`reports/serve_chat_samples.md`](../reports/serve_chat_samples.md)：
链路完整（FastAPI + SSE + 浏览器逐字），但输出是语料里的评分/指令模板片段、**常在冒号处断掉，不是对话**；
已实测**排除**"系统提示配置"这个原因（带/不带 `--system` 都出现同一句被记住的语料模板），
所以归因指向训练状态（ppl 37.90、<45.6% 个 epoch、context 32）。
⇒ 长训练停机后，**把该服务指向新的 rank512 臂检查点再采样一次**，是判断"训练能否改善交互"的唯一实测手段。
（另外：4173 不是本项目——那是 `Desktop\weixin` 的 NexusChat 预览服务，它没有任何 LLM 功能。）

**⚠️ 训练效率：瓶颈已量化**，新增 [`research/training_efficiency.md`](../research/training_efficiency.md)
（含文献，见文献 §7.9）。扫描成本实测 **ms/扫描 ≈ 5.1 + 0.44 × 批宽**：固定项 5.1 ms 只跑出 **~40 GB/s**
（本卡 960 GB/s 峰值的 4%，行长尾所致：入度 median 100 / max 11,526）；每列项 0.44 ms ≈ **464 GB/s**（已近上限）。
交付模型 `connectome_lm.py:112-122` **逐位置递推** ⇒ **前向 32 次扫描 + 反向 32 次 = 每步 64 次扫描**
（训练日志每步的 `32sweeps` 与之一致）。
成本模型 `64 × (5.1 + 0.44 × 32) = 1227 ms` vs **实测步时 1217 ms（误差 0.8%）** ⇒ **扫描就是这一步的几乎全部成本**。
**⚠️ 一条必须记住的自我更正**：我一度写成"`brain_chunks: 1` 让上下文被 `mean(dim=1)` 平均成一次扫描、
各位置 logits 相同"——**已作废**（那行属于 `model.py` 的 transformer 路径；交付模型走
`connectome_lm.py` 自己的 forward，逐位置传二维驱动，不经 mean-pooling，日志的 `32sweeps` 是反证）。
⇒ 成本**几乎正比于 token 数**，加批宽几乎不提高吞吐（batch 64 仅 +14%），
**真正的大杠杆是减少边数**（按键保留每神经元 top-k 输入 ⇒ 成本近似线性下降，k=64 时约 0.41×）。
**卡空闲后的优先顺序**：① 剪枝 top-k ∈ {32,64,128} 的 600 步同预算对照（速度收益 + **同一条固定 24 批命令**测质量代价）；
② 按入度分箱调度能否把固定项 5.1 ms 压到 ~1 ms（预期 −21%）；③ 批宽 64（+14%，需重调 lr）；
④ 用 `torch.profiler` 对真实一步做分解，直接确认 64 次扫描（目前是"日志计数 + 成本模型对账"）。
**不要**在没有空闲 GPU 时做这些——它们都要独占显卡。

**⚠️ 生物电刺激路线（主代理追加，应"用正确的生物电刺激而不是 LLM 训练方式"这条要求）**：新增
[`research/biological_training.md`](../research/biological_training.md)（逐条诊断 + 分阶段方案 S1–S4 + 文献 §7.10）。
诊断（都有出处）：交付模型当前 **`brain_spiking: false`**（`configs/train_connectome_rank512_long.json:29`，跑的是模拟 LIF）、
**刺激打在全部 164,587 个神经元上**（未设 `brain_input_mask`；该配置项已实现且有 5 个测试）、
读出是**学出来的稠密线性层**、学习规则是 **BPTT + AdamW**（`train.py:181/205`）、还靠辅助损失"逼大脑放电"（`train.py:395`）。
而**本项目自己测过**纯脉冲：8 条无关文档跑完整张连接组后，**164,587 个神经元的脉冲图案逐位相同**（批内独立方向 1.000），
根因是**刺激幅度远小于阈值间隔**——一次扫描后只有 **13.55%** 的神经元落在阈值 ±0.5 的响应带内。
⇒ **不是"脉冲不行"，是刺激没有按生物电标定，而刺激侧从来没被调过**（过去只在读出侧打补丁）。
**已生成三条臂**（`scripts/make_bio_configs.py`，脚本自断言"只差端口"）：`configs/train_biospike-{nomask,sensory,random}.json`，
各 **600 步 / 20 分钟 / seed 1337 / 全部 `brain_spiking: true`**，写 **I: 盘**（C: 15.29 GB、E: 6.73 GB、I: **25.61 GB**）。
参考点 = 已有的 600 步 rank256 臂（**模拟模式**、无掩码）= **5.3059 / 201.53** ⇒ 这三条臂同时给出"开脉冲的代价"。
**必需的对照**是 `biospike-random`（等量随机端口）：没有它，端口结果不能归因于解剖学（实验 A 的教训）。
**判据不是 loss**：先跑批参与度诊断，看脉冲状态的批内独立方向能否从 **1.000** 升到 3–6；
升不上去说明刺激幅度仍不够，进 S2（扫 `drive_scale`，目标把 13.55% 的响应带占比提到 40–70%）。

**🆕 更便宜、应当先做的一步（报告 §7.44，文献 §7.15 提案）：零训练"路由"测量。**
已发表近邻工作 arXiv `2606.17745`（幼虫冻结 rate operator、度/权匹配重连对照）把问题拆成两半：
**粗略动力学由度/权统计决定、输入路由与主导模式由精确接线决定**（真实接线在稀疏驱动下把活动限制在核心 1/5，重连 2/3、随机图全部）。
本项目对得上的一测：同一份 `data/wholebrain.npz`（真实）+ 一份**度/权匹配的重连版**，在**稀疏驱动**下测
①被激活神经元占比、②`drive`/`readout` 参与度、③与随机图对比。**成本 = 几次前向，不需要训练、不需要 20 分钟预算。**
**判据**：真实接线的激活占比显著低于重连版 ⇒ "刺激没标定"可能**主要是路由问题**（这会改写 S2 的方向）；
两者接近 ⇒ 支持"幅度"那条诊断。与 S2 **不互斥**，但更便宜，**先做**。
**重连版可能已经在手上**：`data/wholebrain_shuffled.npz`（142.2 MB）与 `data/wholebrain_shuffled_sorted.npz`（135.9 MB）
⇒ 把 `brain_path` 指向它就能跑几次前向；**但先核它是否是"度与权重匹配"的重连**（Therianos 的对照是这样造的），
若只是朴素置换则需另造度匹配版，否则对比被混杂。

**⚠️ 18:2x 只读实测（报告 §7.41）——三个必须先知道的事实**
1. **`max_minutes: 20` 只够跑 450 步**（脉冲臂实测 2.6–2.8 s/步，比 rate 臂的 1.2 s/步慢 ~2.2×；前两个臂的 `done` 都写 `steps: 450`）
   ⇒ 三者彼此可比，但**不能**与 600 步的 rank256 参考臂（5.3059）混进同一张表。要跑满 600 步，`max_minutes` 给 ~30。
2. **批参与度诊断已经有数了**（`I:\flybrain-bio\<arm>\metrics.jsonl` 的 eval 记录，同一步、同一条仪表）：

| step | nomask（loss / 放电率 / 读出参与度） | sensory | random |
|---|---|---|---|
| 250 | 5.6492 / .369 / **6.25** | 7.5908 / .062 / **1.05** | 7.1972 / .038 / 1.82 |
| 450 | **5.2149** / .369 / **6.31** | **7.3638** / .053 / **1.33** | **6.8922** / .035 / **3.75** |

   固定 24 批（三条都跑完，**都是 `checkpoint_step 450`**）：`eval_biospike-nomask_c32.json` = **4.9549 / 141.86**、
   `eval_biospike-random_c32.json` = **6.7510 / 854.92**、`eval_biospike-sensory_c32.json` = **7.2254 / 1373.88**。
   ⇒ **真实端口臂是三个里最差的，等量随机端口的对照比它好 0.4744**（同仪表、同一步、同 seed、同预算）。
   按"参与度 3–6"这条判据：**nomask 与 random 在 450 步达标（6.31 / 3.75），sensory 不达标（1.33，step 250 起一直在 1.0–1.3）**，
   loss 与诊断同向。注意**塌陷不是"有输出掩码就必然发生"**：random 同样有 2,239 个输出端口的掩码，先塌（1.82）后回升（3.75），
   所以差别在于**被掩的是不是真实输出端口**。
   **怀疑对象（只读源码）**：`wholebrain.py:752-757` 的 `_output_ports()` 把 164,587 维状态里 98.6% 清零，
   而 `:797-803` 的 `read_latent()` 正是 `readout_participation()`（`:920-924`）的输入。
   **但两臂的输入掩码与输出掩码是同时加的 ⇒ 归因仍不成立**，缺的对照（只加输入 / 只加输出）见上一段。
3. **⚠️ 两个掩码臂同时改了输入掩码与输出掩码**（输出掩码把读出限制到 **2,239 个神经元 = 1.36%**；输入掩码 15,760 = 9.6%）
   ⇒ 现在**"限制输入到真实感觉端口"与"读出塌成一条方向"是混在一起的**。**缺的对照 = 只加输入掩码 / 只加输出掩码各一条**。
   另外同一步（step 250）**random 7.1972 比 sensory 7.5908 低 0.39**，且 random 的读出参与度同样塌 ⇒ **"真实端口更好"目前没有被支持**。
   一个正面线索：nomask（脉冲）450 步的 4.9549 比 rate 参考臂 600 步的 5.3059 还低 0.35（预算不同，只能当线索）。

**✅ 闸门授权（主代理拍板，2026-09-19 16:30）——`flybrain.serve` 空闲时不算阻挡 GPU 的作业**：
**一次有界的评估可以与空闲的 serve 并存**。依据：① serve 只在生成回复的那 0.6–3.1 秒用 GPU，其余时间空转；
② 实测它起来之后那条臂的吞吐 **883.5 → 894.5 tok/s，没有变慢**；③ 评估本身有界（24 批 × batch 8 × context 32，
历史记录里 24 批前向只用 4.5–6.2 秒）；④ 闸门要防的是**两个训练作业**抢卡，一次有界评估不是那种作业。
**例外只限评估与短探针**：**第二个长训练作业仍然绝对禁止**（那才是 §7.6 里吞吐掉 7 倍的场景）。
**✅ 追加授权（主代理，17:25）：有界的"训练探针"也可与空闲 serve 并存**——例如 §7.32/§7.35 那个
**1 步**的 nw 机制定位实验（两份 config、`out_dir` 指 I:、比 `last.pt` 逐位）。依据：① 它跑 1 步就结束，
不是持续占用；② serve 空闲；③ 当前实测提交余量 **16.58 GB**（上限 60.83，页文件 30.4 GB 只用 1.33 GB），
C: **55.96 GB** 空闲，页文件有空间长。**仍然要做的事**：探针若带 `num_workers: 2`，会起 4 个 worker
（每个约 4.88 GB），提交会被顶到上限附近、页文件临时长大——**一次只跑一个探针，别并排跑**。
**rank2048 不在本次授权里**（它是长训练作业）：要跑就等 S1 三条臂跑完，且 `out_dir` **必须写 I:**
（每份 checkpoint 约 8.9 GB × 2 = 17.8 GB，E: 只有 6.73 GB 放不下）。
**安全阀**：若评估报显存不足，就停掉 serve 再重试。
**而且已经自动化，不必再等人**：`scripts/eval_when_free.ps1` 已在跑（16:29:59 起，WMI 拉起、本会话重启不会杀掉）——
它轮询等到**没有任何 `flybrain.train` 进程**，再等 25 秒让驱动收尾，**若驱动已写过 10 分钟内的结果就跳过**（避免重复评估），
否则用**与 3.6349 完全相同的那条命令**评估 `E:\flybrain-connectome\rank512-long\best.pt`，写
`reports\eval_flybrain-connectome-rank512-long_c32.json`，日志在 `reports\eval_when_free.log`。
**它只评估，绝不替换或删除任何 checkpoint**——替换 canonical 是另一步，且**必须先给旧 `best.pt`（`step=16800`）改名留存**。

**中途读数（报告 §7.36，唯一合法口径）**：两个 run 的进程内 eval 参数完全一样（`eval_interval 200` / `eval_iters 4` /
`batch 32` / `val.bin` / `shuffle=False` / `seed 1337`），用 `scripts/compare_inarm_evals.py` 比：
**33 个可比步里 rank512 赢 32 步，平均 −0.1344**（step 6600：canonical 4.2980 vs rank512 **4.1208**）。**只是中途外推，不是结论。**
**停机后可立刻套的判据（§7.39 用五对同 checkpoint 数据修正）**：`best.pt` 里的 `best_val` 是**进程内** eval，固定 24 批是
另一条仪表；两者的偏移（固定 − 进程内）在**五个同 checkpoint 的测量**上是 **−0.1613 … −0.2633**，而且
**随模型变好单调收窄**（rank128/600 步 −0.2633 → rank256 −0.2374 → rank512 −0.2237 → rank1024 −0.2014 → canonical/16800 步 −0.1613）。
所以判据**不是**"`best_val` ≤ 3.7962 就够了"，而是**一个区间**：rank512 现在是 **3.6447（`step=15600`，比 canonical 的 3.7962
好 0.1515，同一条仪表）**，搬到固定 24 批上**预计领先 0.05–0.25（中心 ≈0.15，即 ≈3.45–3.60）**——**仍是外推**。
**唯一有效的数字是那条 `--batches 24` 命令。**
**`best.pt` 是"进程内最好那一步"的存档、不是最后一步**（canonical 跑到 18000 步、best 停在 16800 步）⇒
报告里必须写 **checkpoint 里的 `step` 字段**；`scripts/eval_checkpoint.py` 本轮已加上 `checkpoint_step` 输出
（`torch.load(..., mmap=True)` 读，失败返回 `None`）。

**⚠️ 下一轮的第一个障碍：`flybrain.serve` 活着时闸门永远是 BUSY**（`autoloop_busy_check.ps1` 按设计把 serve 也算作项目作业），
所以**臂停了闸门也不会变 FREE**，而规则要求"跑 GPU 任务前先过闸门"⇒ 评估会被挡。需要人/主代理二选一：
(a) 跑评估时**临时停掉 serve**（pid 27156 启动器 / 35016 应用进程）——评估本身 **≈40–60 s**（19 条历史 `eval_*_c32.json` 里
24 批前向只用 **4.5–6.2 s**）；(b) 明确**授权"短评估与空闲 serve 并存"**（依据：§7.37 实测 serve 起来后那条臂吞吐
883.5 → 894.5 tok/s，**没有变慢**）。**在此之前不要绕过闸门去跑。**

**评估命令（只差闸门）**：

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'
.venv\Scripts\python.exe -u scripts\eval_checkpoint.py `
    --checkpoint E:\flybrain-connectome\rank512-long\best.pt `
    --tokens data\tokenized_domain\val.bin --batches 24 --batch 8 --context 32 `
    --out reports\eval_flybrain-connectome-rank512-long_c32.json
```

**替换 canonical 的条件与保护**：只有固定 24 批 **< 3.6349** 才替换；**替换前必须先把旧 `best.pt`（1407 MB，`step=16800`）
改名留存**（直接覆盖会毁掉唯一的旧 canonical 存档，§7.14）。从 E: 拷 2422 MB 到 C: 会净增约 1 GB（C: 现有 15.2 GB，够）。

**已完成，不要重跑**：

* **实验 B（`brain_rank` 扫描）**：同一个 600 步预算下 rank128 = 5.8264/339.1、
  rank256 = 5.3059/201.5、rank512 = 4.8979/134.0、rank1024 = 4.5997/99.45（报告 §7.22）
  → **读出未饱和**（递减但每档增益仍是判据阈值 0.04 的 7 倍以上）。**四条臂都已跑完。**
  下一步该测的是 **rank2048**（同 600 步预算；判据：增益 ≤0.04 才算饱和）。
  **⚠️ 先看磁盘（§7.36 实测）**：checkpoint 体积近似线性于 rank —— rank256 = 1.375 GB/个、rank512 = 2.36 GB/个、
  rank1024 = 4.45 GB/个 ⇒ **rank2048 一个臂要约 17 GB**（只有 best.pt/last.pt，原地覆盖）。当前 **E: 只剩 6.73 GB、I: 只剩 25.61 GB**
  ⇒ **rank2048 的 `out_dir` 必须写 I:，E: 放不下**；I: 也只够一个臂，再要第二个必须由人决定清理（§7.14）。
* **实验 D（`num_workers` 等价性）**：**结论=保持 `num_workers: 2`，不要改**（报告 §7.32 是定论）。
  两条臂的 `metrics.jsonl` 确实不同（eq2 与 eq0 **从第 35 步**开始分：8.0824 vs 8.0662），
  而**同配置重跑两次逐步完全相同**（`reports/workers_repro_compare.txt`），所以差异是真的。
  **但差异的原因不是数据顺序**——前 70 个批次的 `x` 哈希在 `nw=2 / nw=0 / nw=2` 下**逐批一致**
  （`scripts/check_worker_data_order.py`），所以只能来自**计算路径的数值差**，**机制未定**。
  → 为省内存改掉交付物的数值轨迹不划算；那 19.5 GB 省不下来。
  **未测**：eq0 自己重跑是否也可复现（若不可复现，正确表述是"nw=0 更不确定"，**不是**"nw=0 改变结果"）。
  **注意**：那个顺序探针**不免费**——它起 2 个 worker，实测把页文件从 37.5 顶到 50.5 GB、
  把 C: 从 10.8 GB 顶到 **0.000 GB 约 40 秒**。**臂在跑时不要起任何带 worker 的进程。**
  **本轮新增（报告 §7.35，12:2x–12:5x，只读文件 + 读 torch 源码）**：nw=2 与 nw=0 的差异
  **可复现且由配置决定**（第二次 nw=2 与第一次逐位相同，且与 eq0 的差异也逐位相同）；
  **RNG 已被排除**（torch 2.9.1：`dataloader.py:696-701` 两种 iterator 都只从全局 RNG 取 1 个数；训练路径不消耗全局 RNG）。
  **判据修正**：step 30 的 eval 记录 `val_loss` 相同，但 `readout_std`/`drive_participation` 已不同 →
  发散**早于 step 35**，只是被 4 位小数藏住。**因此**"<0.04 不算效果"只在**收敛段**成立；
  早期步数（loss≈7）上，只改投递路径 60 步就能长到 **0.31**。
  **最便宜的定位实验（需卡；配方已写好，见下）**：同 seed 只跑 **1 步**，nw=2 与 nw=0 各一次，比 `state_dict` 逐位。
  **✅ 已跑（2026-09-19 19:28–19:33，报告 §7.43）：第 1 步之后 nw=2 与 nw=0 的 `model` 段 18/18 张量逐位相同**
  （`last.pt` 与 `best.pt` 都比过；`best_val` 9.748783826828003 两边相同；train loss 都 9.7739、eval 都 9.7488）
  ⇒ **投递路径在第 1 步不改变数值**（连同 §7.32 的数据顺序、§7.35 的 RNG，三个候选都被排除）。
  探针配置 = `configs\workers_probe_nw{2,0}.json`（只差 `num_workers`/`out_dir`/`run_name`），输出保留在 `I:\flybrain-probe\`。
  **剩下的候选是"第一次 eval 之后才出现的东西"**（推断，非观测）：实验 D 里 step 30 的 eval 之前 loss 逐位相同、之后不同，
  而 nw=2 在第一次 eval 时**第一次去起 val loader 的 worker 进程**（nw=0 不起任何进程）。
  **下一轮收口实验（配方）**：同一对配置跑 **31 步**（第一次 eval 之后 1 步）比 `last.pt` —— 若已不同，触发点就在 step 30 的 eval；
  再跑 **29 步**对（eval 尚未发生）作对照。四份约 10–12 分钟卡时间。
  **⚠️ 先算磁盘**：每个探针写 `best.pt`+`last.pt` = 2.75 GB，四份 = **11 GB**，而 **I: 只剩 6.37 GB**（本轮探针占了 5.5 GB）、
  E: 6.73 GB ⇒ 要么由人决定清 I:，要么把探针改成只写 `last.pt`（需要改 `train.py`，不要擅自改）。

**磁盘规则（报告 §7.30 / §7.31 实测）**：C 盘**空闲底 ≈36 GB**，而**一个训练臂吃掉约 25 GB**
（页文件 `C:\pagefile.sys`，**按分配量占盘**，训练一停自己就回来）。所以：空闲时 **≳40 GB** 才够起一个臂；
**同一时刻只允许一个臂**；新训练的输出一律写 **E: 或 I:**。臂在跑时 C: 常见只有 2–6 GB，**这是正常的**。

**臂在跑时不要起带 dataloader worker 的探针/脚本**：实测一次这样的探针把页文件从 37.5 顶到 50.5 GB、
把 C: 顶到 **0.000 GB 约 40 秒**（训练臂没死，但没必要冒这个险）。

**报告编号**：§7.22–§7.34 与 §16 已被占用，新开 **§7.35 起**。
主代理与循环**都在**改 `reports/FINAL_REPORT.md`：只追加自己的新节，**不要重复别人已写过的结论**；
写数字前**先从文件重算**，不要依赖任何子任务/看守的状态字段。

---

这份文档是为了让**下一轮（包括定时任务里的那一轮）不必重新推导**就能直接动手。
每条都写清楚：为什么做、改哪里、用什么判据、以及**什么结果算阴性**。

**通用判据**：一律用固定 24 批留出集，同一条命令：

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'
.venv\Scripts\python.exe -u scripts\eval_checkpoint.py --checkpoint <路径> `
    --tokens data\tokenized_domain\val.bin --batches 24 --batch 8 --context 32 `
    --out reports\eval_<名字>_c32.json
```

当前基准：`checkpoints/flybrain-connectome/best.pt` = **val_loss 3.6349 / ppl 37.90**
（18000 步 / 330.4 分钟，见报告 §7.20；**2026-09-19 10:5x 更正**：这里原先写的
`4.6834 / ppl 108.14` 是**更早的** canonical，已经在 §7.20 被替换）。
**只有更好才替换 canonical。** 噪声底参考：同 seed 重跑差 0.0001–0.0002；
换 seed 在同一份 24 批上的散布 **0.0395**（即小于约 0.04 的差异不要当作效果）。
**2026-09-19 17:1x 复测的第三个数（报告 §7.40）**：**同一条命令重跑同一个 checkpoint 逐位相同**
（rank512 的 `best.pt` 两次都是 3.4279 / 30.811，`seconds` 4.9 vs 5.1 ⇒ 不是缓存）⇒ **该口径的重复性 < 0.0001**；
旧 canonical 今天复测也与上午记的 3.6349 逐位相同。所以"0.04"那条要写清是**换 seed/换臂**的散布。

**GPU 纪律**：这台机器只有一块卡。跑任何 GPU 任务前先跑
`scripts\autoloop_busy_check.ps1 -QuietMinutes 0`，退出码 3 就不要跑。

**磁盘纪律（2026-09-19 10:5x 更新，见报告 §7.26）**：C: 的可用空间**不是慢性泄漏**，
它是"**有几个 flybrain 训练臂活着**"的函数——页文件 `C:\pagefile.sys` 按 38.6–39.5 GB 的
**分配量**占盘（实际只用 0.81 GB），一个训练臂在跑时页文件约 +20 GB、C: 掉到 4–11 GB，
臂退出后页文件缩回去、C: 自己回到 **36 GB**（本轮实测的空闲底：页文件 18.97 GB / C: 36.50 GB）。
所以"15 GB 门槛"要按**同一时刻只有一个臂**执行，
而且**不要把训练中途的 C: 读数当成趋势**。另外：新臂的 `out_dir` **直接用绝对 E: 路径**
（如 `E:/flybrain-connectome/…`），不要再用 §7.23 的目录 junction。

---

## 实验 A：文本入口 / 读出端口的位置 —— **已完成，阴性（见报告 §7.13）**

> **结论（2026-09-19）**：500 步同调度筛选，固定 24 批：
> `all` **5.3785** < `random_sensory` 5.5953 ≈ `sensory` 5.6194 < `output` 6.3483。
> **现状（致密到全部神经元）胜出**；按解剖选端口反而更差（+0.241，seed 散布的 6.1 倍）；
> FlyToLLM 那条"随机优于真实感觉端口"**没有复现**（差 0.024 = 0.61 倍 seed 散布，不可分辨）。
> **不要改模型**。仍未验证：跑到 9000 步后排序是否改变（需要把 `sensory` 臂跑到 171 分钟）。
> 下面保留原始设计以便追溯。

### 原始设计

**为什么**：`research/literature_flybrain_llm.md` §1.1 记录，FlyToLLM 在**同一份
MaleCNS v1.0 数据**上做了预注册对照，结论是**入口位置的效应大于接线本身**：
真实接线 + 真实感觉端口 = 3.959，真实接线 + 等量随机端口 = **3.689**（随机更好），
理由是感觉神经元在解剖上处于外周、局部抑制之后。而本项目现在是**第三种做法**：
致密投影到**全部 164,587 个神经元**，读出也用**全部**神经元。

**已经完成的部分**：`scripts/build_port_masks.py` 已经建好掩码（98 项测试里 5 项守住其性质）：

| 掩码 | 神经元数 | 内容 |
|---|---|---|
| `data/ports_sensory.npz` | **15,760** | 真实感觉类：`vnc_sensory`、`cb_sensory`、`ol_sensory`、`sensory_ascending`、`sensory_descending` + 三个 `_tbc` |
| `data/ports_output.npz` | **2,239** | 下行/运动/传出：`descending_neuron`、`vnc_motor`、`cb_motor`、`vnc_efferent`、`cb_efferent`、`efferent_ascending`、`efferent_descending` |
| `data/ports_random_sensory.npz` | 15,760 | **等数量**随机对照 |
| `data/ports_random_output.npz` | 2,239 | **等数量**随机对照 |

全部对齐 `data/wholebrain.npz` 的神经元顺序（有测试守住）。已核对：感觉集与输出集**不相交**；
164,403 / 164,587 个神经元有 `superclass` 标注（184 个没有）。
值得记一笔的巧合：本项目的输出集是 **2,239** 个，FlyToLLM 报的是 **2,241** 个——
两个独立项目从同一份解剖里数出了几乎相同的数字，说明这个集合的定义没有歧义。

**要改的代码**：

1. `src/flybrain/wholebrain.py` 的 `BrainPathway.__init__`（约 555 行）加两个可选参数：
   `input_mask_path: str | None = None`、`output_mask_path: str | None = None`。
   读到掩码后 `register_buffer("input_mask", ..., persistent=False)`；**`None` 时必须逐位等同现状**
   （不要构造全 1 掩码再相乘——那会让 `all` 这一臂与今天的结果出现浮点级差异，破坏可比性）。
2. 入口：`drive_from_tokens`（750 行）与 `drive`（705 行）在 `norm_drive(...) * drive_scale`
   **之后**乘掩码。放在之后是因为 `norm_drive` 是 LayerNorm，掩码放在前面会改变归一化的统计量，
   那就不再是"文本只能到达这些神经元"，而变成"换了一种输入缩放"。
3. 出口：`read_latent`（742 行）与 `read_state`（713 行）在 `read_down(state)` **之前**乘掩码。
4. `src/flybrain/model.py` 的 `ModelConfig` 加 `brain_input_ports: str = "all"`、
   `brain_output_ports: str = "all"`（取值 `all` / `sensory` / `output` / `random_sensory` /
   `random_output`），并在 `ConnectomeLM` 构造 `BrainPathway` 处解析成路径。
5. 新配置：复制 `configs/train_connectome_scaled.json`，只改 `max_steps` 与上面两个字段。

**必须加的测试**（否则"默认不变"只是口头承诺）：
* 掩码为 `None` 时，前向输出与改动前**逐位相同**（用固定 seed 的小电路）；
* 给一个掩码后，掩码外的神经元**确实**收到零驱动（只有掩码内的驱动器非零）；
* `BrainPathway` 的状态字典里不出现掩码（`persistent=False`，否则旧检查点加载会失败）。

**怎么跑**：四条臂，**表同步数、同 seed、同一份留出数据**，只在最后用固定 24 批评：
`all`（现状，3 小时内会有一个新基准）／`sensory`／`random_sensory`／`output`。
为了便宜，先用 `max_steps: 400, max_minutes: 12` 做筛选，只把有希望的臂放到 40 分钟。

**什么算阴性**：如果 `sensory` 与 `random_sensory` 的差小于 0.04（噪声底），
那就是**入口位置也不重要**——这同样要写进报告，而且它有独立价值：
它会说明 FlyToLLM 的结论在**不同任务与不同规模**下不成立。
**不要**因为"文献说要更好"就把噪声当成效果。

**风险**：低。入口/出口都只是线性层，吞吐影响可忽略（15,760/164,587 ≈ 9.6% 的驱动非零，
稀疏掩码上乘法的代价可忽略）。

---

## 实验 B：读出流形维数 `brain_rank`（BCI 派生，便宜）—— **已完成，结论：未饱和**

> **✅ 2026-09-19 10:45:48 三条臂全部跑完。** 结果见报告 **§7.25**，原始数据：
> `reports/eval_rank{128,256,512}_c32.json`。**固定 24 批**（同一条命令）：
>
> | 臂 | 参数量 | val_loss | val_ppl |
> |---|---|---|---|
> | rank128 | 53,081,306 | **5.8264** | **339.138** |
> | rank256 | 97,443,802 | **5.3059** | **201.528** |
> | rank512 | 186,168,794 | **4.8979** | **134.009** |
>
> 128→256 = −0.5205（13.2× seed 散布）、256→512 = −0.4080（10.3× seed 散布）
> → **未饱和**，按下面的判据应当继续往上试。**下一臂 rank 1024 已由另一个轮次在 10:46 起跑**
> （配置写绝对 E: 路径，`max_steps 600` / `max_minutes 20` / `seed 1337`，与三条臂同协议；
> 见报告 §7.27）——**不要重复启动**，先看忙闲闸门。
>
> **⚠️ 这个结果不能当成生物学结论**：三条臂按构造就差**参数量**（差的就是读出矩阵），
> 所以它证明的是"读出容量更大更好"，**不**证明果蝇脑是高维的，也不支撑/推翻 §5 的 BCI 论证。
>
> 用法：`scripts/run_rank_sweep.ps1`（`-Ranks 128,256,512`，**串行**，本机只有一块卡）。
> 先看 `reports/rank_sweep.lock`（里面有活着的 PID 就是在跑），再看最新的 `reports/rank_sweep-*.log`。
> 脚本带单实例锁：带着活 PID 再启动会直接退出（码 3），不会变成两个任务抢一块 GPU。
> 配置由 `scripts/make_rank_configs.py` 生成（已断言三条臂只差 `model.brain_rank` 与运行标识）。

**为什么**：`research/literature_flybrain_llm.md` §5。BCI 群体解码文献的共识是
神经活动落在远低于神经元数的低维流形上，且对其**线性**读出已接近最优。
本项目已经是一个线性群体读出（`read_down: 164,587 → rank`），而实测参与度
（驱动 6.45/7、读出 5.83/7）说明有效维数很低——但也可能正是"rank 太小"造成的假象。

**改哪里**：只改配置的 `model.brain_rank`，分别是 **128 / 256（现状）/ 512**。
不动代码。

**怎么跑**：`max_steps: 600`、同 seed、同 lr 表，三条臂，固定 24 批评。

**什么算阴性**：512 与 256 的差落在 0.04 之内 → 读出已经饱和，**记为已测且无效**，
不要再回头改 rank。若 128 明显更差而 512 明显更好，说明还没饱和，可以再往上试。
**实测结论：后者**（差 0.4080）→ 往上是 rank **1024**（已在跑，见上）。
往上的边界：rank1024 的参数量约 373M，checkpoint 会到 GB 级，**必须写 E:/I:，不要写 C:**；
显存也要重新看（rank512 已用到 `torch allocated 4.31 GiB`，rank1024 预计再翻一倍）。

---

## 实验 C：把"步数"与"梯度修正"分开（**诚实性所需**）

**为什么**：报告 §7.10 的 180.35 → 108.14 **同时**改了两件事：内核 `csr → csr-scaled`
（已证明数值恒等）与**转置 CSR 梯度缺陷的修复**（**确实改数值**）。
现在的写法已经明说"两者没有分离"，但应当把它测出来。

**怎么跑**：用 `configs/train_connectome_scaled.json` 把 `brain_impl` 改成 **`csr`**
（其余逐字段相同、同 seed、同 `max_steps: 1200`、同 40 分钟），得到
"旧内核 + 已修梯度"的固定 24 批数字。与两个已知点比较：

| 点 | 内核 | 梯度 | 步数 |
|---|---|---|---|
| 旧 canonical | `csr` | **有缺陷** | 804 |
| **本轮要补的** | `csr` | **已修** | ~804 |
| run A | `csr-scaled` | 已修 | 1200 |
| 新 canonical | `csr-scaled` | 已修 | 2021 |

`csr` 与 `csr-scaled` 数值恒等（有三处一致性的测试），所以第 2 行与第 1 行之差
= **梯度修正的贡献**；第 4 行与第 2 行之差 = **算力的贡献**。

**注意**：`csr` 的墙钟比 `csr-scaled` 慢 2.18×，所以 40 分钟只能跑到约 804 步——
这正好与原 canonical 同步数，是这次对照能成立的原因。

---

## 实验 D：`num_workers` 降到 0 是否**顺序不变**、能否省下 19.5 GB（本机内存压力）

**为什么**：报告 §7.24 实测——每个训练臂提交内存约 **25 GB**，其中 **19.5 GB 是 4 个
dataloader worker**（每个 4.88 GB），而语料只有 **80 MB**。这 19.5 GB 纯粹是"新进程重新
import torch/ROCm"的代价。本机提交内存因此被打到 **99.4%（62.21/62.59 GB）**，页文件被撑到
35.5 GB 并吃掉 C 盘。省掉这 19.5 GB 对整台机器都是实质改善。

**怎么跑**（两条臂，同 seed、同 `max_steps`（比如 30 步就够）、同其它一切，只差 `num_workers`）：

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'
# 臂 1: num_workers=2（现状）  臂 2: num_workers=0
.venv\Scripts\python.exe -u -m flybrain.train --config configs\train_workers2.json --resume none
.venv\Scripts\python.exe -u -m flybrain.train --config configs\train_workers0.json --resume none
```

比较三件事：① `metrics.jsonl` 里每步的 `loss` 是否**逐步相同**；② 每步的
`tokens_per_second`（worker=0 会不会拖慢数据供给）；③ **进程树的总 `PrivateBytes`**
（用 `Win32_PerfFormattedData_PerfProc_Process` 求和，注意要把
`multiprocessing.spawn` 出来的子进程一起算）。

**代码读下来的预期（不是结论）**：顺序**应当**不变——`TokenWindowDataset` 的文档字符串说
"Consecutive windows are reconstructed deterministically from `(epoch, index)` so that resuming
reproduces the same data order"，`__getitem__` 只依赖 `(seed, epoch, index)`；`shuffle=True`
的采样器在**主进程**里跑。**但这没有实测**。

**什么算阴性**：若 `num_workers=0` 的 loss 曲线与 `num_workers=2` 不是逐步相同，
**不能**把 canonical 配方改成 0（历史 arms 不可比），只能说明"顺序确实受 worker 数影响"，
并把这个坑写进报告。

**注意**：不要在三臂扫描（实验 B）**跑的过程中**改这条——那会让"只差 brain_rank"的断言失效。

---

## 不要做的事

* **不要**把 BPTT 换成三因子/资格迹学习（`research/literature_flybrain_llm.md` §3 给了三条理由）。
* **不要**再测批量：32 → 64 只快 18% 但显存 8.35 → 14.67 GiB，128 直接 OOM（已实测）。
* **不要**声称"真实接线更好"：三项独立工作（本项目 §7.6、FLM、FlyToLLM）都不支持，
  而且本项目的措辞必须是"没测出差异"，不是"测出更差"。
* **不要**在两个 GPU 任务同时在跑的时候做任何计时对比。
