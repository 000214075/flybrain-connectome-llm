# 文献调研：把果蝇连接组当作语言模型的基底

本文档记录本项目（`C:\Users\heyiy\Desktop\cangying`）在"可靠的生物学基础"方面所做的文献核对。
每条结论都标注了**实际读到的程度**，不把"看到标题"写成"读过原文"。

检索工具：本仓库新增的本地搜索服务器 `tools/localsearch/`
（`python tools/localsearch/cli.py "查询词" --full`）。它在本机运行、不需要任何 API key，
并且会在结果与查询无关时明确报告 `reliable: false`，而不是静默返回不相关的页面——
这正是它被做出来的原因（见 §8）。

---

## 1. 最重要的一条：已经有人做过"果蝇连接组 + 大语言模型"，而且是负结果

这必须先说，因为它直接约束本项目可以诚实声称什么。

**FLM — Fly Language Model**（GitHub: `FLModel/flm`，同一内容镜像在 `nftechie/flm`）

- **数据集与本项目相同**：MaleCNS v1.0。README 原文写作 "166,700 retained nodes and
  25,582,938 directed connections"（本项目用 164,587 neurons / 25,563,096 synapses，
  差异来自各自不同的节点保留策略）。
- **架构**：一个**冻结**的预训练语言模型（Liquid AI LFM2.5-1.2B-Instruct）。token embedding
  驱动整张固定图，一个 **278,528 参数的 adapter** 读取图的状态并修正下一个 token 的 logits。
  **只有 adapter 训练**，图与语言主干全程冻结。
- **图的递推**：`x = tanh(W @ (0.6*x + 0.4*input))`，`W[post, pre]` 是**入度归一化后的解剖接触数**。
- README 明确写着：**"These are abstract numerical states, not simulated action potentials.
  The model does not infer transmitter signs, dopamine, or biological time from the wiring."**
- **控制实验**：仓库并行训练了一个**参数对齐的 direct-input adapter** 作为对照。原文：
  **"Its matched direct-input control performed slightly better; it does not establish an
  advantage from fly anatomy."**
- 还有一句：**"Removing graph edges removes the residual exactly. Relabeling is an interface
  control, not proof that this topology beats arbitrary wiring."**
- 论文：`https://artificialscientific.com/papers/flies-are-all-you-need`

**读到的程度**：README 全文（4,573 字符）**已读**。
论文原文**未读**——README 称论文描述的是"另一项独立的冻结研究"，其 per-token 结果与拟合产物
不在仓库里。因此上面所有关于论文的结论都只来自 README 的自述，**不是**我对论文的判断。

**对本项目的意义（重要）**：

本项目 §7.6 的接线消融实验（真实连接组 vs 保度数打乱接线）发现**没有可测量的优势**
（匹配步数下差异 0.001–0.002，仅为种子噪声的 5%）。做那个实验时它只是一个**孤立**的阴性结果。
现在 FLM 用**同一份数据集**、一个**独立**的架构、一个**参数对齐的 direct-input 对照**，
得到了方向一致的结论：**接线本身没有带来优势。**

两层含义：

1. **诚实性**：本项目**不能**声称"因为用了真正的果蝇接线，所以模型更好"。两边的证据都不支持。
   §7.6 那条阴性结果应当从"孤例"升级为"与一项独立工作一致"。
2. **本项目与 FLM 的真实差异** —— 也就是本项目仍然有意义的那些部分：

   | | FLM | 本项目（纯连接组模型） |
   |---|---|---|
   | 语言能力来源 | 冻结的预训练 1.2B LLM | **没有**预训练 LLM，全靠自己学 |
   | 可训练参数 | 278,528（只有 adapter） | **97,443,802**（整个模型） |
   | 图是否训练 | 完全冻结 | 每个神经元的释放增益可训练 |
   | 神经元状态 | 抽象数值（`tanh` 递推） | **LIF 动力学 + SuperSpike 代理梯度** |
   | 递质符号 | 明确**不用** | **用**（106,791 兴奋 / 57,255 抑制） |
   | 生物时间 | 明确**不建模** | 有膜时间常数与阈值 |
   | 序列混合机制 | 连接组 + 预训练 transformer | **只有连接组**（无 attention/RoPE/残差/KV cache） |

   FLM 回答的是"把连接组接到一个**已经会说话**的大脑上，接线有没有用"；
   本项目回答的是"**只有一个连接组**，能不能学会说话"。后者更难，也是用户要求的那一个问题。

### 1.1 第二个独立项目，而且它做了**预注册的对照实验**：FlyToLLM

**FlyToLLM**（GitHub `ArtyomITA/flytollm`，README 为意大利语，头部有英文 TL;DR）
是第三个用 **male-cns:v1.0** 做语言模型的项目，也是**方法上最严谨**的一个。
**读到的程度：README 全文 20,216 字符已读**（论文/仓库内的 `reports/SUITE_TEST_FASE_6B.md`
**未读**，下面所有数字都来自 README 的自述表格）。

**它怎么做的**（与本项目的对照很重要）：

| | FlyToLLM | **本项目** |
|---|---|---|
| 神经元 | LIF、二值脉冲、代理梯度 | LIF、SuperSpike、另有速率通路 |
| 递质符号 | **固定**（Dale 定律），来自神经递质 | **固定**，来自神经递质（一致） |
| 突触强度 | 由实测接触数**初始化**，然后**可训练** | 接触数**冻结为 buffer**，只有释放增益可训练 |
| 文本入口 | 真实 **17,937 个感觉神经元**（标准配置改为等量随机节点） | **全部 164,587 个神经元**（致密投影） |
| 读出 | **2,241 个下行/运动/传出神经元**，聚成 77 个群体 | 全体神经元 → rank 256 → 词表 |
| 序列记忆 | 图**外**的小因果注意力（4 头，128 token 缓存） | **没有**（这是本项目的硬要求） |
| 规模 | **5.46M 参数**，几乎全是突触 | 97.4M 参数 |
| 硬件 | 一张 GTX 1080（8 GB） | RX 7900 XTX |
| 任务 | TinyStories，词表 4096 | 领域语料，词表 16384 |

**它的实测结论（留出 16 篇故事的交叉熵，越低越好）**：

| 配置 | CE |
|---|---|
| 真实接线 + 真实感觉端口，**26,247** update | **3.634** |
| 真实接线 + 真实感觉端口，8,000 update | 3.959 |
| 真实接线 + **随机端口**，8,000 update | 3.689 |
| **保度数打乱接线**，8,000 update | **3.814** |
| Configuration model（拓扑破坏），8,000 update | 3.648 |
| Erdős–Rényi（同等边数），8,000 update | 3.697 |
| 超类内打乱（保留分块矩阵），2000 update | 5.005 |
| 权重（接触数）在边上置换，2000 update | 5.103（**无影响**） |
| Dale 符号在节点间置换，2000 update / 8000 | 4.645 / 3.723 |
| 故意翻转 6% 的符号（估计的分类错误率），2000 | 4.982（**无损害**） |
| GRU 2×256，同数据 | **3.429** |
| Transformer 2 层，同数据 | 3.563 |
| 三元 Kneser-Ney，同数据 | **3.410** |

**必须逐条说清楚这些数字意味着什么、以及不能拿来比什么**：

1. **不能拿它的 CE 与本项目的困惑度比。** 数据不同（TinyStories vs 领域语料）、
   词表不同（4096 vs 16384）、参数量差 18 倍。它的 3.6 与本项目的 4.68（即 ppl 108）
   **没有任何可比性**，把它写成"别人做到 3.6 而我们 108"是错的。
2. **它的接线对照方向与本项目、与 FLM 都不同**：保度数打乱 = **3.814**，
   而真实接线 = **3.959**——也就是**真实接线更差**。README 的英文 TL;DR 直接写
   "every null model beats the real wiring"。配上本项目 §7.6（无可测差异）与
   FLM（direct-input 对照略好），现在是**三项独立工作**指向同一结论。
3. **一个对"冻结接触数"很关键的对照**：它把接触数在边上**置换**后**完全没有影响**
   (5.103)，并明确解释原因是"**i pesi sono allenabili**"（权重可训练），
   所以解剖信息不在权重里。**这条对本项目不成立**：本项目的接触数是**冻结的 buffer**，
   解剖信息**确实**留在权重里（报告 §3.3.3 用测试守住）。
   所以"置换权重无影响"不能拿来推断本项目。
4. **Dale 符号是承重的**：置换符号在 2000 步从基准 5.087 掉到 4.645（变好）——
   即真实符号模式是一个**约束/刹车**。本项目用了真实的递质符号，方向一致。
   同时它对 6% 的符号翻转完全免疫，说明这个约束是鲁棒的。
5. **文本入口位置的效应比接线本身大**：真实感觉神经元在解剖上处于**外周、局部抑制之后**，
   把入口换成等量随机节点反而更好（3.689 vs 3.959），README 称换端口可"拿回与上限之间 87% 的差距"。
6. **它必须把序列记忆放在图外**（小注意力），理由是"连接组没有'句子里第几个位置'这个轴"。
   本项目**按用户的硬要求不设这个外部注意力**——这正是本项目最难、也最独特的地方，
   同时是它困惑度远高于对方的**结构性原因之一**（不是全部：数据和规模也不同）。

### 1.3 把五项工作放在一起：**连接组在它演化出来要做的任务上有效，在任意任务上不见得有效**（本轮新增，两项引用均已独立核实）

到本轮为止，用同一份或同类果蝇全脑连接组做的**可比较**工作已经有五项，方向分成两半：

| 工作 | 任务 | 有对照吗 | 结果 |
|---|---|---|---|
| 本项目 §7.6 | 语言建模 | 保度数打乱 | **无差异**（0.0016，种子散布的 4%） |
| FLM | 语言建模（冻结 1.2B LLM） | 参数对齐的直接输入 | **对照略好** |
| FlyToLLM | 语言建模（TinyStories） | 保度数打乱 / 配置模型 / ER | **所有零模型都更好** |
| **FlyGM** | **全身运动控制（GNN + RL）** | **保度数打乱** | **连接组更好**（样本效率更高） |
| **Liew et al.** | **视叶朝向图（脉冲神经元）** | —— | **朝向图从连接组约束的动力学中涌现** |
| **FLYNN** | **机器人导航（MuJoCo）** | 手工设计网络 | **连接组拓扑在 OOD 与感觉丧失下鲁棒性显著更强** |
| **Eon Systems** | **具身行为（虚拟身体）** | —— | **仅凭连接 + 神经递质身份即复现生物神经响应，91% 一致** |

- **FlyGM**：Jin et al.，*Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion
  Control in Fruit Fly*，arXiv **2602.17997**（2026-02）。**核实过**：arXiv HTML、BAAI hub、
  chatpaper、arxivlens、以及 NeurIPS 2025 页面都能独立找到同一标题。摘要原文提到
  "we compare it against a **degree-preserving** …"——**与本项目用的是同一类对照**，
  而且它报告的是**正面**结果。仅摘要与 README 已读，**全文未读**。
- **Liew et al.**：*Connectome-Based Modelling Reveals Orientation Maps in the Drosophila
  Optic Lobe*，arXiv **2609.01330**，NeurIPS 2025。**核实过**：mlanthology、arXiv、
  Semantic Scholar、proceedings.com、以及一份 PDF 都能独立找到。
  它的做法与本项目**最接近**（"integrates a complete fruit fly brain connectome with
  biologically grounded spiking neuron models"），而结论是**结构性的**：
  朝向图不是由连接组"附带产生"的噪声，而是从连接组约束的动力学里**涌现**出来的。
  仅摘要与片段已读，**全文未读**。

**由此得到的、本项目应当采用的表述**：

> 连接组带来的优势**看起来是任务相关的**：在它演化出来要解决的**感觉处理与运动控制**任务上，
> 结构会承重（FlyGM 的样本效率、Liew 的朝向图）；在**语言建模**这种它从未演化去做的任务上，
> 三项独立工作都测不到优势。

这个表述能同时容纳全部五项证据，而且**不需要**把任何一项解释掉。
它也**不**替本项目辩护：本项目的任务恰恰是语言建模，所以按这条规律，**测到"没优势"是预期之内**。

**必须写明的边界**：
- 两项正面结果**全文都未读**，只有摘要与片段；它们的任务与词表/指标都与本项目不同，
  **不能**用它们的数字或结论替代本项目的实测。
- "任务相关性"是**对这些工作的归纳，不是一条被检验过的定律**。
  它给出一句可检验的预言（把同一套连接组模型放到感觉/运动任务上应当表现出优势），
  但本项目**没有做**那个实验。

### 1.5 URCHIN：最小化的 Dale 律约束脉冲语言模型（本轮新增，**摘要与全文 HTML 已读**）

**URCHIN**（Unified Recurrent Connectome with Horizontal Integrate-and-fire Neurons），
Po-Han Chiang，arXiv **2609.13899**（2026-09-12），已被 BabyLM Workshop @ EMNLP 2026 接收。
**核实过**：arXiv HTML、PDF、alphaXiv、HuggingFace（`phclab/URCHIN_BabyLM2026_Strict` 与
`URCHINBABY/URCHIN_BabyLM2026_Strict`）都能独立找到。

**它做了什么**：
- **单层 128 个 LIF 神经元**，无 attention，4.23M 参数（其中脉冲核心仅 34K）；
- **Dale 律**约束的 E/I 结构（80:20 兴奋/抑制比），全连接横向（lateral）拓扑；
- 每个 token 通过**多传输循环**（multi-transmission loop）在横向连接中反复传播直到收敛，
  然后从膜电压线性读出到词表（16,384 BPE 条目）；
- 两种实现共享同一组权重：**并行 SSM 扫描**（训练用，GPU 高效）与
  **事件驱动 RSNN**（部署用，CPU/神经形态硬件可用）；
- 在 BabyLM 2026 三条赛道（Strict-100M、Strict-Small、Multilingual）上提交。

**与本项目的关键差异**：

| | URCHIN | **本项目（ConnectomeLM）** |
|---|---|---|
| 神经元数 | **128** | **164,587** |
| 连接组来源 | **学习的** Dale 律约束拓扑（随机初始化） | **真实果蝇连接组**（2556 万条突触，冻结为 buffer） |
| 递质符号 | Dale 律（80:20 E/I 比） | **真实**递质标注（106,791 兴奋 / 57,255 抑制） |
| 横向连接 | 全连接（128×128） | 稀疏（平均入度 ~155） |
| 内部记忆 | 无（每 token 独立收敛到固定点） | **跨 token 递归**（膜状态在 token 间持续） |
| 可训练参数 | 4.23M（99.2% 是嵌入/读出层） | 97.4M（释放增益可训练，接触数冻结） |
| 数据 | BabyLM（发育合理的小规模） | 领域语料（领域特定） |
| 注意力 | 无 | 无 |
| 硬件 | 未指定（JAX/PyTorch） | RX 7900 XTX（ROCm） |

**对本项目的意义**：
1. **领域验证**：URCHIN 是第一篇"脉冲 + Dale 律 + 横向连接组 + 语言建模"的已发表工作，
   说明这个方向本身有学术社区在关注。但它**没有使用真实生物连接组**——
   它的"connectome"是一个可学习的随机拓扑加上 Dale 约束，
   与本项目使用果蝇 MaleCNS v1.0 的 2556 万条突触完全不同。
2. **可比性有限**：128 vs 164,587 神经元、随机拓扑 vs 真实连接组、
   每 token 独立收敛 vs 跨 token 递归——架构差异太大，不能直接比较数字。
3. **技术参考**：其并行 SSM 扫描 vs 事件驱动 RSNN 的双模式设计有参考价值，
   但本项目的 `csr-scaled` 内核走的是不同路线（稀疏矩阵常量化）。
4. **诚实性**：不能因为 URCHIN 有 Dale 律就说"本项目不孤独"——
   它们解决的是非常不同的问题。但也不能因为 URCHIN 没用真实连接组就说"本项目更优"——
   它的目标是 BabyLM 效率，不是连接组验证。

### 1.6 BANC 连接组：第一个脑-神经索统一的果蝇全中枢神经系统连接组（本轮新增，**摘要已读**）

**BANC**（Brain-And-Nerve-Cord connectome），
*Distributed control circuits across a brain-and-cord connectome*，*Nature*（2026），
DOI **`10.1038/s41586-026-10735-w`**（PubMed 42259917）。
**独立核实过**（由维护者在 2026-09-19 复核）：Nature 文章页、PubMed、Wilson Lab（HMS）
与 drugowitschlab.org 的新闻稿都能找到同一标题；中文报道的标题也一致
（"果蝇全脑-体连接组图谱问世：行为控制并非大脑独揽大权"），与下面"运动控制高度分布式"的结论相符。
**注意**：原记录写的卷期页码（656: 957–970）**未能独立确认**，这里改记 DOI；
"第一作者 Bates、与 Rachel I. Wilson 合作"来自检索片段，未逐一核对作者列表。

**它做了什么**：
- 把之前**分别发表**的果蝇脑连接组（FlyWire, ~140,000 神经元）与
  腹神经索连接组（VNC, ~20,000 神经元）**统一成一个完整的突触级连接组**；
- 覆盖**1,300 个下行神经元**与**1,800 个上行神经元**（之前只有少数被"桥接"分析重建）；
- 提出"影响网络"（influence network）度量，评估任意两个细胞间的功能连接强度；
- 核心发现：**运动控制是高度分布式的**——每条腿的运动主要由该腿局部回路控制，
  各局部回路再通过上行/下行神经元协调；脑区（学习/导航）起**监督**作用但不是**必要**条件。

**与本项目的关系**：
- **数据层面**：本项目使用的 MaleCNS v1.0 是 2024 年的脑连接组；
  BANC 是**更新、更完整**的数据集（脑 + 神经索），可能成为下一代基准。
  **但本项目不直接使用 BANC**——它的格式、规模、和神经元标注体系不同，
  迁移需要重新做数据管道。
- **科学层面**：BANC 的"分布式控制"发现与本项目的观察有呼应——
  本项目训练中脑利用率仅 ~44%、突触使用率 ~45%，暗示大部分连接组
  在语言建模任务中没有被激活，这与"运动控制是分布式的、脑起监督作用"的发现方向一致。
- **下一条路**：如果长期训练完成后仍想验证"连接组在感觉/运动任务上更有效"的假说
  （§1.3 的归纳），BANC 的数据可能比 MaleCNS v1.0 更适合——因为它统一了脑与身体的连接。

### 1.4 三个负结果可能有一个**理论解释**

到此为止，三项独立工作的阴性结果（本项目 §7.6、FLM、FlyToLLM）都还是**经验事实**：
试了，没优势。本轮找到了一篇**理论**工作，它给出了一个可能的**原因**：

**Beiran & Litwin-Kumar（2025）**，*Prediction of neural activity in connectome-constrained
recurrent networks*，Nature Neuroscience，DOI `10.1038/s41593-025-02080-4`
（预印本 `bioRxiv 10.1101/2024.02.22.581667`）。摘要原文（两个独立来源一致）：

> "We found that a connectome often does not substantially constrain the dynamics of
> recurrent networks."

即：**连接组往往并不能实质性地约束递归网络的动力学**——同一个连接组配上不同的
神经元参数，可以产生很不一样的动力学，所以"接线对了"这件事本身携带的信息比直觉上少。
该工作进一步指出，**从一小群神经元的记录**（也就是"观察哪些神经元"）可以消除这种退化。

**为什么这条对本项目特别重要**：它把 §1 与 §1.1 的三项负结果从"三次各自独立的失败"
变成了"一个理论所预期的事情"。而且它直接支持 §1.1 第 5 条的经验发现：
**决定性的变量不是接线，而是"从哪里进、从哪里看"**——也就是端口选择。

**诚实的边界**：这是别人的理论，**不是**对本项目的证明。它不排除"用上真实递质符号、
真实 LIF、真实接触数"之后接线会变得有信息量——恰恰相反，它指出
**只有当神经元参数与接线一起被约束时**接线才起约束作用，而本项目正是
把接触数与符号**冻结**在权重里（§3.3.3）。这是一个**可检验的分歧**，
写在 `research/next_experiments.md` 实验 A 里。另外：该文全文未读（见 §6 条目 3e）。

**由第 5 条得出的、本项目可直接做的一个实验**：本项目的
`drive_from_tokens` 是**致密投影到全部 164,587 个神经元**
（`wholebrain.py` 的 `up`/`down` → `norm_drive` → `drive_scale`），
既不是"真实感觉端口"也不是"等量随机端口"。FlyToLLM 的结果说明
**入口的选择是一个比接线更大的杠杆**，而且它是一个**便宜**的实验：
只改入口的稀疏性/位置，不动连接组。这条已列入 §6。

---

### 1.3 第四个独立项目，而且它测出了**正面结果**（但任务不同）：FlyGM（本轮新增）

**FlyGM — Fly-connectomic Graph Model**（Jin, Zhu, Zhang & Sui, 2026，
arXiv `2602.17997`，v3 发表于 2026-06-14）

**它怎么做的**：把成年果蝇**全脑连接组**直接实例化为一个**图结构的神经控制器**，
用深度强化学习控制一个模拟的果蝇生物力学身体的运动（locomotion）。
架构是 GNN（图神经网络），不是语言模型。

**核心发现**：
- 在多种运动任务上实现**稳定性能**
- 与图和非图基线相比，**样本效率更高**（即用更少的交互就能学到控制策略）
- 作者的结论："a biologically informed way towards effective control policy design
  by translating whole-brain wiring principles into actionable architectural priors"

**结构分析**（来自配套 GitHub 仓库 `Yanjin-ai/connectome`，用 FlyWire v783 连接组）：

| 指标 | 值 | 对控制的含义 |
|---|---|---|
| 前馈权重比例 | 10.9% | 只有 ~11% 的突触权重从感觉流向运动 |
| 反馈权重比例 | 6.4% | 显著的反向环路结构 |
| **侧向/递归比例** | **82.8%** | **主导性的层内递归——内置的时间记忆** |
| 全局到达中心性 | 0.014 | 接近零的层级性；大脑**不是**前馈管道 |
| 功能社区（≥50 神经元） | 12 | 可识别的模块化结构 |

**对本项目的意义（重要但有边界）**：

1. **正面信号**：这是第一个用全脑连接组作为结构先验并报告**正面结果**的独立工作
   （FLM 和 FlyToLLM 都是负结果）。但任务完全不同——运动控制 vs 语言建模。
   运动控制的奖励信号密集且低维（关节角度、速度），而语言建模的信号稀疏且高维
   （下一个 token 的交叉熵）。所以"连接组对运动控制有用"不能直接推出
   "连接组对语言建模有用"。
2. **82.8% 侧向/递归**这个数字值得记住：它意味着连接组的**递归性**远强于前馈性。
   本项目的 `ConnectomeLM` 恰好完全依赖递归（没有注意力、没有前馈管道），
   从这个角度看，本项目的架构选择与连接组的统计特性是一致的。
   但"一致"不等于"因此更好"——需要实验证据。
3. **方法差异**：FlyGM 用 GNN（消息传递），本项目用稀疏矩阵扫描（CSR）。
   两者在数学上是等价的（稀疏矩阵乘 = 图上的消息传递），但优化方式不同
   （GNN 用 PyTorch 的 autograd，本项目用自定义的 CSR 扫描内核）。
4. **连接组来源不同**：FlyGM 用 FlyWire v783（134,013 神经元 / 2.7M 突触），
   本项目用 MaleCNS v1.0（164,587 神经元 / 25.5M 突触）。两份连接组的规模差一个数量级。

**读到的程度**：arXiv 摘要**已读**；GitHub README **已读**（12,247 字符）；
论文全文 PDF **未读**（付费墙 / 未获取）。结论来自摘要与 README 的自述，
**不是**对论文方法与结果的独立核实。

**引用核实**：arXiv `2602.17997` 由独立搜索（BAAI hub、alphaXiv、chatpaper）
确认存在，标题、作者、日期一致。DOI `10.48550/arXiv.2602.17997`。

### 1.7 BriLLM：非 Transformer 脑启发语言模型（本轮新增）

**BriLLM — Brain-inspired Large Language Model**（Zhao, Wu, Yang, Zou & Hong，
上海交通大学，arXiv `2503.11299`，v8 2025-09-08，cs.CL）

**它怎么做的**：提出 "Signal Fully-connected Flowing (SiFu)" 机制——
一个有向图上的信号流模型，每个 token 是图中的一个节点，节点间通过
全连接矩阵传递"能量张量"，沿"最小阻力路径"流动。
声称是"非 Transformer、非 GPT、非传统"架构。
当前版本：中/英文各一个，4000 token 词表，32 维节点，32 token 序列预测，
~1B/2B 参数。

**声称的结果**："replicate GPT-1-level generative capabilities while demonstrating
stable perplexity reduction"——即与 GPT-1 相当的语言建模能力。

**与本项目的关键区别**：

| | BriLLM | **本项目（ConnectomeLM）** |
|---|---|---|
| 图结构 | **全连接有向图**（每个 token 节点与所有其他节点相连） | **真实生物连接组**（稀疏、164,587 神经元 / 25.5M 突触） |
| 图是否来自生物 | **否**（token 节点 + 全连接） | **是**（MaleCNS v1.0） |
| 信号机制 | 能量张量 + 最小阻力路径 + GeLU | LIF 膜动力学 + SuperSpike 代理梯度 |
| 位置编码 | 正弦余弦 PE（原文公式含 PE_i） | **无**（连接组自身的递归提供时序） |
| Dale 定律 | 无 | 有（106,791 兴奋 / 57,255 抑制） |
| 词表大小 | 4,000 | ~32,000 |
| 规模 | 1-2B 参数 | 97M 参数 |

**对本项目的意义**：
BriLLM 证明了"非 Transformer 的图信号流语言模型"这个方向在学术界有关注，
但它的图结构是全连接的（即完全放弃拓扑信息），与本项目使用真实稀疏连接组
的做法在根本上不同。"非 Transformer"不等于"用了生物连接组"——
BriLLM 是前者但不是后者。

**读到的程度**：arXiv 摘要**已读**；HTML 全文（ar5iv）**已读**（含完整公式与架构图）；
代码仓库 `brillm05/BriLLM0.5` **未读**。结论来自论文自述，
GPT-1 相当的声称**未独立核实**（没有可比较的 benchmark 数据）。

**引用核实**：arXiv `2503.11299` 由 arXiv 搜索直接确认存在，标题、作者、日期一致。
v8（2025-09-08）为最新版本。OpenReview `eR8raBLZW7` 同时存在。

---

### 1.8 FLYNN：果蝇连接组拓扑的机器人导航（本轮新增）

**FLYNN — Fly-connectome Neural Network for Robot Navigation**
（Wang & Chen, arXiv `2607.00025`，v2 2026-07-13，cs.RO）

**它怎么做的**：用果蝇连接组拓扑构建一个 RNN，在 MuJoCo 中做基于视觉的导航。
架构直接从突触分辨率的大脑连接组推导而来。

**核心发现**：
- 性能与同参数量的手工设计网络**相当**
- **对 OOD 数据和感觉丧失的抗性显著更强**（无需额外训练）
- 完全失去视觉后仍能工作，而手工网络即使专门用 camera dropout 训练也大面积失败
- PCA 分析表明 FLYNN 的内部状态具有**高度表征模块化**

**与本项目的关键区别**：

| | FLYNN | **本项目** |
|---|---|---|
| 任务 | 机器人导航（MuJoCo） | 语言建模（中文） |
| 神经元模型 | 未明确说明（RNN） | LIF + SuperSpike |
| 连接组来源 | 果蝇连接组（具体版本未说明） | MaleCNS v1.0 |
| 训练方式 | 反向传播 | BPTT（同） |
| 核心贡献 | 拓扑带来的**鲁棒性** | 拓扑作为**唯一**序列建模机制 |

**对本项目的意义**：
FLYNN 是又一个用果蝇连接组拓扑作为架构先验并报告**正面结果**的工作。
它的发现（鲁棒性、模块化表征）与本项目 §7.6 的阴性结果（接线 vs 随机图）
不矛盾——FLYNN 测的是**鲁棒性**而不是**语言建模质量**。
连接组拓扑可能在鲁棒性上有优势但在损失函数上无差异，这是一个值得记住的区分。

**读到的程度**：arXiv 摘要**已读**；论文全文 **未读**（只有 HTML 摘要可访问）。
结论来自摘要自述。

**引用核实**：arXiv `2607.00025` 由 arXiv 搜索直接确认存在，标题、作者、日期一致。

---

### 1.9 Eon Systems：果蝇全脑仿真驱动虚拟身体（本轮新增）

**Eon Systems PBC — Embodied Brain Emulation**
（Scott Harris, Aarav Sinha, Viktor Toth 等，2026-03-09/10，eon.systems）

**它怎么做的**：把多个已有组件整合成一个完整闭环：
1. **脑**：基于 Shiu et al. 的 LIF 模型，~140,000 神经元 / ~50M 突触
   （来自 FlyWire 连接组），用推断的神经递质标识确定突触符号
2. **视觉**：Lappalainen et al. 的 connectome-constrained 视觉通路模型
   （64 种视觉细胞类型）
3. **身体**：NeuroMechFly v2（MuJoCo 物理引擎），87 个独立关节
4. **控制接口**：少量下行神经元（DNa01/DNa02 用于转向，oDN1 用于前进速度等）
   作为脑模型与生物力学之间的接口

**已实现的行为**：梳理（antennal grooming）、觅食（feeding）、
趋化导航（foraging）、视觉逃逸（looming response）。

**技术细节**：脑-身体同步步长 15 ms；感觉输入映射到特定感觉神经元；
下行神经元输出转换为低维运动指令。

**与本项目的关键区别**：

| | Eon Systems | **本项目** |
|---|---|---|
| 连接组 | FlyWire（~140K 神经元 / ~50M 突触） | MaleCNS v1.0（164,587 / 25.5M） |
| 任务 | 具身行为仿真（运动、感觉） | 语言建模 |
| 输出接口 | **特定下行神经元**（~少量） | **全体 164,587 神经元**（致密投影） |
| 训练 | **无训练**（连接组约束 + 手工控制器） | BPTT 端到端训练 |

**对本项目的意义**：
1. **下行神经元作为输出接口**：Eon 用少量特定下行神经元作为脑与身体的接口，
   与本项目 §7.13 的端口实验直接相关。§7.13 发现**全体神经元读出优于解剖选端口**，
   但那是语言建模任务；具身控制可能有不同结论。
2. **连接组结构可以恢复感觉运动结构**（Shiu et al. 的发现），这是"连接组承重"的
   正面证据——但同样是**非语言任务**。
3. **无训练 vs 端到端训练**：Eon 的方法证明连接组本身编码了大量行为结构，
   但本项目的目标是让模型**从文本中学习**，这需要训练。
4. 这是目前**最完整的果蝇全脑具身仿真**，值得关注后续发展。

**读到的程度**：eon.systems 技术博客**全文已读**（19,107 字符，截断前已读大部分）。
结论来自博客自述，**不是对底层论文的独立核实**。

**引用核实**：eon.systems 网站由搜索直接访问确认存在。
底层论文引用（Shiu et al. 2024, Lappalainen et al. 2024）**未独立核实**。

---

### 1.10 `train-your-fly` 与它的配套论文：**同一个问题上的第三个独立结果，方向与 FLM/FlyToLLM 相反**（本轮核实，README 全文已读）

上一轮（报告 §7.21）把 `eudald-seeslab/train-your-fly` 列为"新发现，尚未读到 README 全文，下一轮应核实"。
**本轮已核实**，而且它比预期的更有价值：它不只是个工具箱，它是一个**正面结论**。

**已读到的程度**：三份第一方材料**全文已读**——
① `train-your-fly` 的 README（`open-websearch__fetchGithubReadme` 直接抓取，约 12 KB）；
② 配套仓库 `eudald-seeslab/connectome` 的 README（约 15.6 KB）；
③ 数据存档 Zenodo `10.5281/zenodo.21549559`（直接抓取，2026-07-25 发布，v1.0.0，CC BY 4.0，7.5 GB，
含 `connections_biological.csv.gz`、四个 randomized 图、Fig. 2/3 的源数据表与 SHA-256）。
**论文本身未读**（arXiv/期刊版未检索到，出版方标注 "In prep."）。

**它做什么**：把 FlyWire v783 校对过的全脑连接组接到一个**按解剖重建的复眼模型**上，
包成 PyTorch Geometric 模型，训练做**图像分类**——颜色辨别、形状识别、数量辨别（近似数感，
面积已控制）。流程是：512×512 图像按复眼小眼的 Voronoi 区域取平均 → 感光细胞按光谱敏感性读通道
→ 在连接组上做 3 步消息传递 → Kenyon 细胞群体活动 → 线性分类器。

**动态方程与可训练参数**（README 原文）：

$$x_i^{(k)} = \gamma\Big(\sum_{j \to i} x_j^{(k-1)}\, e_{ji}\, \omega_{ji} - \xi_i\Big), \qquad \omega_{ji} = \tanh(\theta_{ji}) \in [-1, 1]$$

- **神经元在步与步之间不留状态**（没有膜电位、没有递归），只有 3 步前馈传播；
- 学的是**每个突触一个标量增益** $\theta$（可选再加每神经元一个阈值 $\xi$）加线性读出，
  **接线图不动**（"The wiring diagram never changes"）。

**方向性结论（本轮最有价值的一条，原文引用）**：它把生物图与四个随机化系综比较，
四个系综保留的结构越来越多（`unconstrained` / `connection-pruned` / `synapse-bin` / `neuron-bin`）：

> "At matched wiring cost, the biological network is consistently the most accurate. Rewirings that
> ignore spatial constraints surpass it, but only by inflating the wiring budget or by favouring
> long-range connections that let activity reach the Kenyon cells in fewer steps."

以及形状任务上的排序（README 原文）："the unconstrained and connection-pruned ensembles lead,
followed by the biological graph and the two distance-binned ensembles."

**⚠️ 措辞纪律：不能把它写成"真实接线更好"，也不能拿它推翻本项目 §7.6**：

- 这是本项目核心问题上的**第三个独立结果**，而且方向**不同于** FLM（§1）与 FlyToLLM（§1.1）
  ——那两个是"**测出更差**"。所以"所有工作都测不出接线优势"这句话本轮起**不成立**。
- 但**两者问的不是同一个问题**，不可直接比：(a) 任务是视觉分类，不是语言建模；
  (b) 它的"接线优势"判据是**匹配布线成本 / 平均突触长度**，不是困惑度；
  (c) 它的连接组是 FlyWire v783（139,255 神经元），本项目是 MaleCNS v1.0（164,587）。
- **照实记录一个内部张力**：README 的系综表里 `connection-pruned` 的总布线 = 生物，
  但平均突触长度标为 "above biological"，而它仍然**领先生物图**；同时那句话又说
  "matched wiring cost 下生物图最准"。`matched wiring cost` 到底指**总布线**还是**长度**，
  README 没有说清。**不解释、不选边**，等论文本身可读再判。

**对本项目实验设计的直接影响**：它验证了"**用等量/等约束的随机图做对照**"是这类工作的标准做法
（与本项目 §7.6 的 shuffled 对照、实验 A 的 `random_sensory` 等量随机端口是同一套逻辑），
并且它把"约束强度"做成了**四个梯度**而不是一个。本项目若再测接线，应照此设计**多档约束**
而不是单一 shuffled 对照。

---

### 1.11 NeuroCraft Fly：同一个 MaleCNS 图被第三方独立复现，并自报一个**负面**数字（本轮核实，README 全文已读）

**怎么找到的**：核实 §7.21 那条"待核实"的中文报道（"谷歌 MaleCNS v1.0 + GPT-6 在 Minecraft 里做全脑仿真"）时找到的。

- 仓库：GitHub `evnsnclr/neurocraft-fly-public`，标题
  "NeuroCraft Fly by Evan Sinclair Smith: an interactive fruit-fly connectome model in Minecraft"，
  155 stars / 12 forks / 3 commits，**README 全文已读**。
- **与本项目的图交叉验证（本轮新增，值得记）**：它自述本机 demo 跑的是
  **"a retained MaleCNS graph with 166,700 neurons and 25,582,938 directed edges"**；
  本项目 `data/wholebrain.npz` 是 **164,587 神经元 / 25,563,096 突触**。
  **边数相差 0.077%**；神经元数差 2,113（本项目应是只取了有标注/有坐标的子集）。
  两个互不相干的项目从同一份 MaleCNS v1.0 里数出几乎相同的边数，
  说明**本项目用的整脑图就是公开的那份 MaleCNS v1.0，没有私有改动**——
  这是一条独立的数据管道健全性证据，之前没有过。
- **它自报的一个负面数字（README 原文）**：

  > "A trained readout had **95.66×** higher raw error and **403.70×** higher projected error than a
  > matched direct controller on its navigation task."

  即**在同一份连接组图上**，从连接组训练出的读出在导航任务上比匹配的直接控制器差**两个数量级**
  （它自己限定："These concern their original configurations, not biological validation or a general
  control advantage"）。它还写："Network comparisons are still being qualified; **a response surviving
  shuffled weights is a result to report**"——即它也把"随机权重下仍然成立"当作必须报告的阴性对照。
- **媒体把它讲成了"全脑仿真"，它自己否认**（README 原文）："Sensory mappings and dynamics are **modeled**,
  and **hand-chosen readouts** select and modulate **scripted body programs**. This makes the connection
  visible and testable; **it does not establish recovered fly physiology or natural behavior**."
  也就是说"Minecraft 里的果蝇"**不是**脑仿真，是脚本化身体程序 + 手工选择的读出。
- **对本项目**：这是"连接组驱动读出"在**第三个任务**（具身导航）上的负面证据，
  方向上支持本项目 §7.6 的阴性结论与硬约束 7 的措辞纪律（"没测出差异"≠"真实接线更好"）。
  引用时必须带上它自己的限定语。

---

### 1.12 MaleCNS v1.0 的**首要来源**核实 + 一条对本项目记录的更正（本轮新增）

上一轮（§7.21）把中文报道"谷歌 MaleCNS v1.0 + GPT-6 实现 Minecraft 全脑仿真"标为**未核实**，
并推测"GPT-6 尚未正式发布，该报道可能是对 Google DeepMind 果蝇脑模型工作的不准确转述"。
**本轮独立核实后，这个判断是错的**：两半**都基本属实**（只有"全脑仿真"这个说法是媒体的夸张，
见 §1.11）。照 §7.3 的规矩，这要写进文档，不能悄悄改。

**① MaleCNS v1.0（首要来源，全文已读）**

| 来源 | 读到程度 | 关键事实 |
|---|---|---|
| `janelia.org/project-team/flyem/male-cns-connectome`（HHMI Janelia FlyEM 项目页） | **全文已读** | "the first finished connectome of an entire male *Drosophila* central nervous system"，覆盖中央脑 + 视叶 + 腹神经索；**262 个性别特异 + 114 个性别二型细胞类型 = 中央脑的 4.8%**；协作方 = Janelia FlyEM + 剑桥 Drosophila Connectomics Group / MRC LMB + **Google Connectomics**；CC-BY。**新闻时序**：2025-10-03 v0.9 → 2026-**06-08 v1.0 released** → 2026-**09-03 official publication** |
| `blog.google/.../male-fruit-fly-brain-map/`（Google 官方博客，2026-09-03） | **全文已读** | 第一方：Google Research（Michał Januszewski、Viren Jain），"over **166,000** neurons and nearly 12,000 distinct cell types" |

**更正一个日期**：中文聚合站普遍写"2026-09-03 发布 MaleCNS v1.0"，
而 Janelia 的项目页写 **v1.0 发布于 2026-06-08**、**2026-09-03 是正式论文发表**。
聚合站把"论文发表"当成了"数据发布"。

**② GPT-6 Astra（首要来源，全文已读）**

| 来源 | 读到程度 | 关键事实 |
|---|---|---|
| `azure.microsoft.com/.../gpt-6-astra-frontier-intelligence-for-work-now-generally-available-in-microsoft-foundry/`（微软 Azure 官方博客，2026-09-03） | **全文已读** | "**GPT-6 Astra**, OpenAI's newest frontier model, is now **generally available for all customers** in Microsoft Foundry"；作者 Steve Sweetman、Naomi Moneypenny |
| `aws.amazon.com/about-aws/whats-new/2026/09/openai-gpt-6-astra-on-amazon-bedrock/`（AWS 官方公告，2026-09-08） | 标题+摘要 | "general availability of GPT-6 Astra from OpenAI on Amazon Bedrock" |
| 9to5mac（2026-09-04） | 标题+摘要 | 第三方技术媒体，同日报道 |

**两个第一方云厂商公告独立确认 GPT-6 Astra 于 2026-09-03 正式发布（GA）**，
所以"GPT-6 尚未正式发布"这条上一轮的判断**已作废**。
**注意边界**：这只证明"GPT-6 Astra 存在且已 GA"，**不**证明新闻里"两天做出 Minecraft 果蝇"
的任何技术细节；那部分另见 §1.11（NeuroCraft Fly 仓库确实存在，但它自己否认那是脑仿真）。
另：`openai.com/news/` 对本机 fetch 返回 **403**，所以**没有**读到 OpenAI 自己的公告页，
这一条靠 Azure/AWS 两侧第一方公告补上。

---

## 2. 蘑菇体（mushroom body）回路与学习规则

检索到的来源（**全部只读到标题与摘要片段，未读全文**）：

| 来源 | URL | 读到程度 |
|---|---|---|
| Dopaminergic mushroom body neurons in *Drosophila*: flexibility and heterogeneity（Neurosci Biobehav Rev, 2022） | https://www.sciencedirect.com/science/article/pii/S0149763422000598 | 标题+摘要片段 |
| Dopamine-Dependent Plasticity Is Heterogeneously Expressed by Presynaptic...（eNeuro, 2023） | https://www.eneuro.org/content/10/10/ENEURO.0275-23.2023 | 标题+摘要片段 |
| Coincident postsynaptic activity gates presynaptic dopamine release（PMC5262376） | https://pmc.ncbi.nlm.nih.gov/articles/PMC5262376/ | 标题+摘要片段 |
| Visualization of learning-induced synaptic plasticity in MBON γ-lobes（Sci Rep, 2022） | https://www.nature.com/articles/s41598-022-14413-5 | 标题 |
| Convergent multi-modular architecture for adaptive learning（iScience, 2025） | https://www.cell.com/iscience/fulltext/S2589-0042(25)02060-7 | 标题+摘要片段 |
| 果蝇蘑菇体多巴胺能神经元回路及功能研究进展（昆虫学报, 2025） | https://www.insect.org.cn/CN/10.16380/j.kcxb.2025.12.012 | 标题+摘要片段 |

**从片段中可以确证的一点**：DAN（多巴胺能神经元）对蘑菇体可塑性的作用在单个突触层面是
**高度异质**的（eNeuro 那篇的标题即为此），而不是一个统一的全局学习率。这对本项目有直接影响：
本项目目前对**每个神经元**只有一个释放增益标量，而真实蘑菇体的可塑性是按突触、按 DAN 分区区分的。

**诚实的边界**：这些来源我只读到标题和摘要片段。本项目混合模型里的蘑菇体子层
（PN→KC→MBON，KC 稀疏度 5%、APL 抑制）**来自本项目更早一轮的证据整理**，
不是本轮新读出来的。纯连接组模型（最终交付的那个）**根本不用蘑菇体子层**，它只用全脑回路。

---

## 3. 不需要 BPTT 的生物学可信学习规则

**这一节我真正读了原文**：Mazurek, Caputa, Argasiński, Wielgosz,
*Three-Factor Learning in Spiking Neural Networks: An Overview of Methods and Trends from a
Machine Learning Perspective*，arXiv:2504.05341v2 [cs.NE]，2025-04-25，
CC BY-NC-SA 4.0，`https://arxiv.org/html/2504.05341`。
抓取 84,559 字符，读了摘要、目录、第 1 节与第 2 节开头。

从读到内容中可以确证：

- 三因子规则 = STDP × **第三个调制因子**（多巴胺、血清素、乙酰胆碱等全局信号）。
  引入第三因子的动机是：**STDP 本身不足以解释复杂学习行为**（原文："Research in neuroscience
  has shown that STDP is insufficient to fully explain complex learning behaviors"）。
- 审稿人把 **"scalability considerations"** 单列为一节（§3.6），并在"局限与挑战"（§4）、
  "未来方向"（§5）中反复回到这一点。也就是说：**三因子学习能否扩展到大规模网络，是该领域
  自己承认的未解决问题。**

其他来源（**只读到标题**）：

| 来源 | URL | 读到程度 |
|---|---|---|
| Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons*（Nat Commun, 2020）—— e-prop | https://www.nature.com/articles/s41467-020-17236-y | 标题+摘要片段；另有 bioRxiv PDF 链接 https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf |
| *Eligibility traces provide a data-inspired alternative to BPTT*（OpenReview） | https://openreview.net/pdf?id=SkxJ4QKIIS | 标题 |
| Reward-Modulated STDP（Lava / Intel 文档） | https://lava-nc.org/lava/notebooks/in_depth/three_factor_learning/tutorial01_Reward_Modulated_STDP.html | 标题 |
| Three-factor learning 综述（同上一篇的 ScienceDirect 版） | https://www.sciencedirect.com/science/article/pii/S2666389925002624 | 标题 |

### 对本项目的结论：**不建议**把 BPTT 换成三因子规则

理由三条，都可检验：

1. **代价**：本项目现在拿到的梯度是**精确**的，而且因为把稀疏矩阵常量化
   （§5.2，`csr-scaled`），整个"前向 + 反向"一步只要 **27.6 ms**。
   反向里已经**没有**逐边梯度项了——`raw_gain` 的梯度是一次 16.5 万维的归约。
   三因子规则要用"资格迹（eligibility trace）"换掉这条通路，而资格迹是**每个神经元一个额外状态**，
   在 164,587 神经元的规模上是一笔固定的额外开销与额外的超参（迹的时间常数、第三因子形状）。
2. **证据**：该综述自己把大规模可扩展性列为未解决问题；检索到的综述与教程的演示规模都在
   小网络上（Lava 教程、OpenReview 的 STDP 数据集）。
3. **风险方向**：换掉学习规则**只会让结果变差或不变**，不会带来可测的质量提升——
   而本项目已经明确了下一步的收益来源是**算力**（§7.9.1：40 分钟只见到语料的 2%）。

**因此**：三因子学习被记为"**已调研、明确不采用**"，理由是代价与证据，而不是"没试过"。
如果以后要试，最小可检验的形式是：**保留 BPTT，但把释放增益的学习率乘上一个 DAN 式的全局调制标量**
（即三因子里只借"调制"这一条，不借"资格迹"），代价接近零，可以用固定 24 批验证集判断是否有用。

---

## 4. 果蝇 LIF 参数的实测值 —— **没找到干净可引用的表**

检索到的都是**方法学**论文，不是参数表（**全部只读到标题**）：

| 来源 | URL | 读到程度 |
|---|---|---|
| Neuronal synchronization in *Drosophila*（bioRxiv, 2024 / iScience, 2025）—— LNv 全细胞膜片钳 | https://www.biorxiv.org/content/10.1101/2024.09.19.613913v1.full.pdf | 标题+摘要片段 |
| Patch-Clamping *Drosophila* Brain Neurons（CONICET, 2023） | https://ri.conicet.gov.ar/bitstream/handle/11336/215592/CONICET_Digital_Nro.eed758c4-5740-4217-93b1-fa723e8f1d91_B.pdf | 标题 |
| Preparation of *Drosophila* Central Neurons for in situ Patch Clamp（PMC3490319） | https://pmc.ncbi.nlm.nih.gov/articles/PMC3490319/ | 标题 |
| Cornell 课程讲义：*Drosophila* 膜特性（2011） | https://courses.cit.cornell.edu/bionb4910/labresources/DROS%20NEURO_12.pdf | 标题 |

**必须如实说明的一件事**：本项目把膜时间常数取为 `tau_m = 2.0`，**这不是从实测值来的**。
它的理由是工程性的、并且写在了代码注释里：`brain_iters=1`、每个 token 只扫一次，
如果用一个接近真实（约 20 ms）的时间常数，神经元在这一次扫描里根本到不了阈值。
换句话说，**本项目在"生物时间尺度"这一点上是不忠实的，而且是明知故不忠实**。

这是"可靠的生物学基础"这个要求下**最薄弱的一环**，应当写进报告的局限一节，
而不是含糊过去。可检验的改进方向：把 `iters` 提到 2–4 并同时调 `tau_m`（让膜在多次扫描内积分），
代价是吞吐按扫描次数线性下降。

---

## 5. 脑机接口（BCI）文献中真正可迁移的部分

**诚实声明**：本节我**没有成功读到任何一篇全文**。最相关的一篇抓取失败：

| 来源 | URL | 读到程度 |
|---|---|---|
| *Neural manifolds for the control of movement*（PMC6122849） | https://pmc.ncbi.nlm.nih.gov/articles/PMC6122849/ | **抓取失败**（SSL: certificate verify failed, self-signed certificate in certificate chain）；只有检索结果的标题与摘要片段 |
| Churchland lab 论文列表 | https://churchland.zuckermaninstitute.columbia.edu/content/publications | 标题片段（其中可见 Churchland, Yu, Ryu, Santhanam, Shenoy 2006） |
| 神经流形与动力学（中文笔记） | https://jeffliulab.github.io/neuro-sci-notes/en/06_Brain_Computer_Interface/ | 标题+摘要片段 |

> 这个抓取失败本身是本地 fetch 的一个已知限制（部分站点用自签证书链，需要显式信任），
> 已记入 §7。

### 可以说的（基于流形/因子分析这一支的公开共识，且与本项目已有实现直接对应）

BCI 的群体解码文献反复得到同一个结论：**神经群体的活动落在一个远低于神经元数目的低维流形上**，
而对这个流形的**线性读出**在解码上已经接近最优。本项目已经做的是**完全同构**的事：

- 读出是 `read_latent`：`read_down: 164,587 → rank(256)`，再接 `head: 256 → vocab`。这就是一个线性群体解码器。
- 实测的参与度指标（驱动参与度 6.45/7、读出参与度 5.83/7，见报告 §8）说明**有效维度很低**。

**由此得到一个真正可检验的 BCI 派生实验**（而不是空谈）：

> **扫描 `brain_rank`（读出流形的维数）**：128 / 256 / 512。
> 如果流形的固有维数已经低于 256，增大 rank 不会有用（读出已经饱和）——这正是 BCI 解码文献里
> "解码性能随维度增加而饱和"的那条曲线。如果固有维数更高，增大 rank 应当降低验证困惑度。
> 代价：`brain_rank` 只影响 `up`/`read_down` 两个线性层（256 相对 164,587 很小），
> **吞吐影响可忽略**，是一个便宜的实验。

**明确**不**可迁移的部分**（避免把 BCI 的说法硬搬过来）：
侵入式电极的标定与漂移、被试个体的会话间漂移、闭环控制的实时延迟预算（本项目按秒计，不按毫秒计）、
以及"用同一被试的数据训练再解码同一被试"这类设定——本项目没有被试，也没有闭环。

---

## 6. 可以直接用到本项目的地方

按"是否便宜"与"是否有可测结果"排序。度量一律是**固定 24 批验证集上的困惑度**
（`scripts/eval_checkpoint.py`，见报告 §7.5）。

| # | 建议 | 依据 | 预期效果 | 风险 |
|---|---|---|---|---|
| 1 | **已完成**：把稀疏矩阵常量化（`csr-scaled`） | §5.2 的逐段计时 | 一步 3.276 → 1.502 s，实测 2.18×；同时省 6.3 GiB 显存 | 已用两条独立测试锁住恒等性 |
| 2 | **加大批量**（32 → 64/128） | 每次扫描的固定成本是读 2556 万条边；批量只加宽稠密算子 | 每 token 成本下降，GPU 利用率上升 | 改变优化动力学，需同步调 lr 或看曲线 |
| 3 | **扫描 `brain_rank`（读出流形维数）** | §5 的 BCI 流形/线性解码结论 | 要么确认读出已饱和（阴性但便宜），要么直接降困惑度 | 低 |
| 3b | **改文本入口：致密全脑 vs 稀疏子集** | §1.1 第 5 条：FlyToLLM 实测**入口位置的效应大于接线本身**，且真实感觉端口在解剖上处于外周、局部抑制之后，反而是**劣势**。本项目现在是致密投影到**全部**神经元，既非"真实端口"也非"等量随机端口" | **已实测，阴性**（报告 §7.13）：真实感觉入口 **5.6194** 比致密入口 **5.3785** 差 0.241（seed 散布的 6.1 倍）；等量随机端口 **5.5953** 与真实感觉端口差 **0.024**，仅 0.61 倍 seed 散布 → **不可分辨**。即 FlyToLLM 那条"随机优于真实端口"在**本项目设定下没有复现** | 已做完，**现状（致密）胜出，不改模型**。未验证：这是 500 步筛选，跑到 9000 步后排序是否改变未知 |
| 3c | 入口位置要按**解剖**选，而不是随机 | 同上；本项目已有 MaleCNS 的细胞类型信息，可以真正选出感觉/下行神经元，而不是随机子集 | 若 3b 显示入口重要，这一步决定该把入口放在哪 | **已完成并已测**（§7.13）：掩码由 `scripts/build_port_masks.py` 建好（感觉 15,760 / 出口 2,239 / 等量随机对照），模型侧已接入并有 5 项测试守住"无掩码时逐位等同"。**结论：按解剖选端口反而更差**，所以这一步的答案是"不要按解剖选" |
| 3d | 致密 vs 稀疏输入的控制论证据（本轮新增） | Betzel et al. (2026), *Commun. Biol.* 9:501：空间弥散输入比局域点输入所需控制能量**低两个数量级**。但该结论是宏观脑区级，不是单神经元级；判据是控制能量，不是语言建模损失。**不能替代**本项目实验 A 的实测，因为 FlyToLLM 的经验负结果仍是约束。 | 为"致密投影到全部神经元"（本项目现状）提供理论支持，但不消除实测必要性 | 低；仅作为背景引用，不影响实验设计 |
| 3e | **连接组约束下输入/输出节点选择的理论后果**（本轮新增，**引用已独立核实**） | Beiran & Litwin-Kumar, *Prediction of neural activity in connectome-constrained recurrent networks*, **Nature Neuroscience** (2025), DOI `10.1038/s41593-025-02080-4`（预印本 biorxiv `10.1101/2024.02.22.581667`）：连接组约束的递归网络中，**连接组本身通常不能充分约束网络动力学**（退化/degeneracy），但从**一小群神经元的记录可以消除这种退化**——即**观察哪些神经元**才是有信息量的。另有 conn2res 工具箱（Suárez et al., *Nature Communications* 15, 2024, DOI `10.1038/s41467-024-44900-4`）把 "input nodes" 与 "readout nodes" 作为**可配置参数**，说明这个领域已经把端口选择当作独立设计变量。| **为实验 A 提供理论背景**（见下面的 §1.2）。 | 低；仅作为背景引用。**全文未读**：PMC 版本被本机 fetch 的自签证书链挡住，bioRxiv 返回 429，Nature 版本在付费墙后。结论来自摘要与两个独立来源一致的原文片段。 |
| 3f | **递归网络中复杂神经元的缩放定律**（本轮新增，**引用已独立核实**） | Spieler, Martius & Levina, *Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons*, arXiv `2605.12049`（2026-05-12，alphaXiv 独立确认存在）。他们用 ELM（Expressive Leaky Memory）神经元——功能上类似皮层神经元的多时间尺度积分器——构建递归网络，系统地扫描参数预算在"神经元数量 N"、"单神经元复杂度 k_e"、"单神经元连接度 k_c"三者之间的分配。核心发现：**(1)** 性能沿三个轴各自单调提升；**(2)** 固定预算下存在非平凡最优分配；**(3)** 更大的预算 favor 既更多又更复杂的神经元；**(4)** 信息论模型把两端的收益递减归因于"单神经元信噪比饱和"和"跨神经元冗余"。**对本项目的意义**：ConnectomeLM 的 LIF 神经元是"复杂单位"（膜动力学、阈值、SuperSpike 梯度），连接组固定了 k_c。该论文的结论暗示"简单单位默认"（ML 的标准做法）不一定最优——复杂单位在大预算下可能更好。但**直接可比性有限**：(a) 该论文的 ELM 神经元有可调参数控制复杂度，本项目的 LIF 只有固定的 tau_m 和阈值；(b) 该论文的连接是全连接或随机图，本项目的连接是固定的解剖连接组；(c) 该论文测的是标准基准（SHD-Adding、Enwik8），不是领域语料。 | 为本项目的架构选择（复杂 LIF 神经元 vs 简单 tanh 单元）提供理论支持，但不能替代实测。**摘要已读**，全文未读。 |
| 4 | 更长的训练（把 40 分钟换成小时级） | §7.9.1：算力是唯一还没被榨干的杠杆 | 线性受益 | 低；但要注意会话中断需要能续训（已开 `resume: auto`） |
| 5 | **不要**声称"真实接线更好" | §1 的 FLM 负结果 + §1.1 的 FlyToLLM 负结果 + 本项目 §7.6 | ——（这是诚实性要求，不是性能建议） | 若不写，报告会过度声称。注意措辞：本项目是"**没测出差异**"，两者是"**测出更差**"，不能混着说 |
| 6 | **不要**把 BPTT 换成三因子/资格迹 | §3 | —— | 大改且无证据 |
| 7 | 如实标注 `tau_m = 2.0` 无实测依据 | §4 | —— | 不写就是含糊 |
| 8 | 可选：DAN 式全局调制标量乘在释放增益的学习率上 | §2、§3（只借"调制"不借"资格迹"） | 未知，可能为零 | 便宜，但需要一次完整训练才能判定 |
| 9 | 可选：把释放增益从"每神经元一个"细化为"按突触类型分组" | §2（DAN 可塑性在突触层面高度异质） | 未知 | 增加参数量；会削弱"整个模型只有连接组参数"的简洁性 |
| 10 | 若要用 PMC/PubMed 全文，fetch 需要能接受自签证书链 | §5、§7 | 让文献调研真正读到全文而不是摘要 | 低，但要改 fetch 的信任策略 |
| 11 | **BANC 连接组**：若长期训练完成后仍想验证"连接组在感觉/运动任务上更有效"，BANC（§1.6，*Nature* 2026）提供了更完整的脑+神经索数据，可能比 MaleCNS v1.0 更适合 | §1.3、§1.6 | 统一的运动控制回路数据可能更适合验证"任务相关性"假说 | 中；需要新的数据管道，且 BANC 的标注体系与 MaleCNS 不同 |
| 12 | **URCHIN 的并行-串行双模式**（§1.5）：如果未来要部署到神经形态硬件，其"同一组权重、两种运行模式"的设计值得参考 | §1.5 | 训练与部署的零转换成本 | 低；仅作为架构参考，不影响当前实验 |
| 13 | 若再测接线，用**多档约束强度**的随机图对照，而不是单一 shuffled | §1.10：`train-your-fly` 的四个系综按"保留多少结构"排成 `unconstrained → connection-pruned → synapse-bin → neuron-bin`，并把"平均突触长度"与"总布线"分开报 | 能把"接线是否承重"的答案从是/否变成**沿约束强度的曲线**；本项目现有的单一 shuffled 对照只能给一个点 | 低（生成随机图是纯 CPU 活）；但要先把"约束"定义清楚，且必须等预算齐平 |
| 14 | **不要**再写"所有相关工作都测不出接线优势" | §1.10 与 §1.11：`train-your-fly` 在视觉任务上**测出了**生物图优势（虽然是在匹配布线成本下）；NeuroCraft 则在导航任务上报了一个负面数字 | ——（诚实性要求，不是性能建议） | 不写就是**过度归纳**。正确的写法是逐条列出方向与任务：FLM/FlyToLLM = 测出更差；`train-your-fly` = 视觉分类、匹配布线成本下生物图最准；本项目 §7.6 = 语言建模、没测出差异 |

---

## 7. 搜了但没有找到的东西（保证不编造）

1. **没有找到该连接组神经元 LIF 参数的实测表**。检索到的是膜片钳**方法学**论文；
   没有找到可直接引用的 `tau_m` / `v_th` / `v_rest` 数值表。因此 §4 的结论是"没有依据"，
   而不是"依据是某个值"。
2. **没有找到任何"使用真实生物连接组的纯语言模型"（不带预训练 transformer）的已发表工作**。
   最接近的有两个：
   - **FLM**（§1）用的是**冻结的预训练 1.2B LLM + 27.8 万参数 adapter**，图是真实的但语言能力来自预训练 LLM。
   - **URCHIN**（§1.5）是纯脉冲 + Dale 律 + 横向连接组的语言模型，但它用的是**学习的随机拓扑**
     而不是真实生物连接组。因此，"使用真实果蝇连接组、不带预训练 LLM、从头学习语言"这个设定
     **仍然只有本项目一家**——但也意味着**没有可比的基线**，报告里不能拿别人的困惑度来对比。
3. **PMC / PubMed Central 全文抓取失败**（`SSL: CERTIFICATE_VERIFY_FAILED`，
   自签证书链），因此 BCI 一节（§5）**没有全文依据**，只有检索片段。
4. **没有找到对"连接组作为序列混合机制"的正面结果**。找到的是：
   FLM 的 direct-input 对照**略好**（§1），以及 README 里"relabeling 不是
   topology 优于任意接线的证据"这句话。本项目 §7.6 与之一致。
5. **SearXNG 本地实例没有跑起来**：Docker Desktop 在这台机器上未能启动
   （daemon pipe 不存在，等待约 5 分钟后仍未就绪），因此本地搜索引擎这条后端
   目前是**未启用**状态，搜索实际走的是 playwright 驱动本机 Chrome 的那条路。
   代码与配置文件已就位（`tools/searxng/settings.yml`、`tools/localsearch/server.py`），
   Docker 可用时无需改代码即可启用。

---

## 7.3 引用核实记录（维护者亲自复核，2026-09-19）

自动循环每一轮都会往这份文档里加文献。**加进来的每一条引用，维护者都独立复核过它真实存在**，
方法是在 arXiv / Nature / PubMed 上按标题或 DOI 直接抓取，而不是相信摘要里的自述。
下表是复核结果，**如实记录**（包括没核实成功的）：

| 引用 | 标识 | 复核结果 |
|---|---|---|
| FLM（Fly Language Model） | GitHub `FLModel/flm`、`nftechie/flm` | ✅ 仓库 README 全文已读 |
| FlyToLLM | GitHub `ArtyomITA/flytollm` | ✅ 仓库 README 全文已读 |
| Beiran & Litwin-Kumar 2025 | `10.1038/s41593-025-02080-4`；bioRxiv `2024.02.22.581667` | ✅ 存在；**全文未读**（PMC 被本机自签证书链挡住，bioRxiv 429，Nature 付费墙） |
| conn2res（Suárez et al. 2024） | `10.1038/s41467-024-44900-4` | ✅ 存在；摘要已读 |
| Betzel et al. 2026 | `10.1038/s42003-026-09560-8`，*Commun. Biol.* **9**:501 | ✅ **已复核（2026-09-19 §7.5）**：出版方文章页正文已读（标题/13 位作者/卷期文章号/摘要/引言）。**同时更正了一处引用措辞**：原 §3d 写的"能量低两个数量级"是错的，见 §7.5 |
| FlyGM（Jin et al. 2026） | arXiv `2602.17997` | ✅ 直接抓取 arXiv 确认；摘要已读 |
| Liew et al. 2025（视叶朝向图） | arXiv `2609.01330`，NeurIPS 2025 | ✅ **直接抓取 arXiv 确认**，含完整作者列表（Jia-Nuo Liew, Shenghan Lin, Bowen Chen, Wei Zhang, Xiaowei Zhu, Wei Zhang, Xiaolin Hu），2026-09-01 提交；摘要已读 |
| Spieler, Martius & Levina 2026 | arXiv `2605.12049` | ✅ **已复核（2026-09-19 §7.5）**：arXiv 摘要页全文已读，作者 Aaron Spieler / Georg Martius / Anna Levina，2026-05-12 提交 |
| URCHIN（Po-Han Chiang 2026） | arXiv `2609.13899` | ✅ **直接抓取 arXiv 确认**：标题 *URCHIN: A Horizontal Spiking Language Model for Data-Constrained Pretraining*，作者 Po-Han Chiang，2026-09-12 提交，q-bio.NC；摘要已读 |
| BANC（脑-神经索统一连接组） | `10.1038/s41586-026-10735-w` | ✅ 由 Nature / PubMed / 两个实验室站点确认存在；摘要与片段已读，**全文未读** |
| BriLLM（脑启发非 Transformer LLM） | arXiv `2503.11299` | ✅ 检索到 arXiv PDF 与 OpenReview 页面；摘要与片段已读 |
| FLYNN（连接组拓扑用于机器人导航） | arXiv `2607.00025` | ✅ 检索到 arXiv HTML 与第三方索引，标题一致；摘要与片段已读 |
| Eon Systems（果蝇全脑具身仿真） | `eon.systems`；其 Nature 论文 2024-10 | ✅ **直接抓取 eon.systems 首页确认**：网站自述"139,255 个生物神经元、5000 万连接"、"用**仅凭连接与神经递质身份**在生物果蝇神经响应上达到 **91%** 一致"，并指向 2024-10 的 Nature 论文（**该论文本身未读**）。<br>**记录一次我自己的错误**：第一次核实失败是因为我查了 `eonsystems.com`（域名不存在），而正确的域名是 `eon.systems` ——**先确认要查的标识本身写对了，再判定"查不到"** |
| BriLLM（Zhao et al. 2025） | arXiv `2503.11299`，OpenReview `eR8raBLZW7` | ✅ arXiv + ar5iv HTML 全文已读；GPT-1 相当的声称**未独立核实** |
| FLYNN（Wang & Chen 2026） | arXiv `2607.00025` | ✅ arXiv 摘要已读；论文全文**未读** |
| Eon Systems 具身脑仿真 | eon.systems 技术博客 | ✅ 博客全文已读；底层论文引用**未独立核实** |
| `train-your-fly`（工具箱） | GitHub `eudald-seeslab/train-your-fly` | ✅ **README 全文已读**（本轮，§1.10） |
| `train-your-fly` 配套论文 | *Structure alone supports efficient visual computation in the Drosophila visual system*（Correig-Fraga, Guimerà, Sales-Pardo, In prep.） | ⚠️ **论文本身未读**（未检索到可读版本，出版方自标 "In prep."）；**其数据存档与配套仓库已核实**——见下两行 |
| 上述论文的数据存档 | Zenodo DOI `10.5281/zenodo.21549559` | ✅ **直接抓取确认**：2026-07-25 发布，v1.0.0，CC BY 4.0，7.5 GB，含生物图 + 四个 randomized 图 + Fig.2/3 源数据 + SHA-256 |
| `eudald-seeslab/connectome`（配套实验仓库） | GitHub `eudald-seeslab/connectome` | ✅ **README 全文已读**（本轮，§1.10）；方向性结论引自此 README |
| NeuroCraft Fly（Minecraft 果蝇） | GitHub `evnsnclr/neurocraft-fly-public` | ✅ **README 全文已读**（本轮，§1.11）。**注意**：媒体说的"全脑仿真"被它自己否认；软件尚未发布（仓库自称 landing page） |
| MaleCNS v1.0（本项目所用连接组的来源） | Janelia FlyEM 项目页 `janelia.org/project-team/flyem/male-cns-connectome`；Google 官方博客 `blog.google/.../male-fruit-fly-brain-map/` | ✅ **两处第一方来源全文已读**（本轮，§1.12）；论文本体（2026-09-03 正式发表）**未读** |
| GPT-6 Astra（上一轮误判为"尚未发布"） | 微软 Azure 官方博客 2026-09-03；AWS 官方公告 2026-09-08；9to5mac 2026-09-04 | ✅ **Azure 博客全文已读**、AWS/9to5mac 标题+摘要（本轮，§1.12）。`openai.com/news/` 对本机返回 **403**，OpenAI 自家公告**未读** |

**上表两条曾经的 ⚠️（Betzel、Spieler）已于 2026-09-19 复核完毕，见 §7.5。**
其余 ⚠️ 行（`train-your-fly` 的论文本体）仍**只作为背景**，不支撑本项目任何结论；
若要引用其具体数字，**必须先复核**。

### 7.4 本轮（自动循环，2026-09-19 10:2x–10:5x）的核实记录

| 结论 | 判据 |
|---|---|
| `train-your-fly` 真实存在且是**正面结果** | README 全文已读（约 12 KB）+ 配套仓库 README 全文已读 + Zenodo 存档直接抓取 |
| 上一轮"GPT-6 尚未正式发布"的判断**错误** | Azure 与 AWS 两处**第一方**云厂商公告独立确认 GPT-6 Astra 于 2026-09-03 GA（Azure 博客全文已读） |
| 上一轮"中文报道可能是不准确转述"的判断**部分错误** | MaleCNS v1.0 由 Janelia 项目页与 Google 官方博客确认（两处全文已读）；"Minecraft 果蝇"对应的 NeuroCraft Fly 仓库真实存在（README 全文已读）。**只有**"全脑仿真"这个词是媒体的夸大，NeuroCraft README 自己否认 |
| MaleCNS v1.0 的**数据发布**日期是 2026-06-08，**论文发表**是 2026-09-03 | Janelia FlyEM 项目页的新闻时序；中文聚合站把两者混为一谈 |
| 本项目的整脑图 = 公开的 MaleCNS v1.0 | NeuroCraft 报 166,700 神经元 / 25,582,938 边，本项目 164,587 / 25,563,096，**边数差 0.077%** |
| 没能读到的 | `openai.com` 全部页面（403）；`train-your-fly` 的论文本体（未检索到可读版本）；NeuroCraft 的软件本体（作者尚未发布） |

## 8. 附：为什么要有 `tools/localsearch/`

原来的搜索路径对**一次高度具体的查询**返回了泛泛的 "Drosophila" 百科页面，
同时仍然报告成功。这类失败是**静默**的：看起来像"搜索没找到相关内容"，
而不是"搜索其实没跑对"——这对研究工具是最糟的一种性质。

新的本地搜索服务器做三件事：

1. 先问**本机上的 SearXNG**（不需要 API key、不需要账号、不把第三方服务放进链路），
   拿不到就退回**用 playwright 驱动本机已安装的 Chrome**——后者是在这台机器上实测可用的那条路；
2. 对**每一条结果**按查询词打分，无关时明确返回 `reliable: false` 与 `term_coverage`，
   让糟糕的搜索**不可能**被误当成"没搜到"；
3. **只用标准库**，不会因为依赖变动而坏；诊断信息走 stderr（stdout 上多打一行就会破坏协议流）。

自检：`python tools/localsearch/selftest.py`（走真实 stdio 协议、含一条中文查询用于验证编码）。

---

## 7.5 本轮（自动循环，2026-09-19 11:2x–11:3x）复核两条 ⚠️ 引用 + 一处引用措辞更正

检索走 `local-search` MCP，**两条查询的 `reliable` 都是 `true`**（`term_coverage` 0.778 / 1.0）。
失败的后端只有本机 SearXNG（`WinError 10061`，未启用），实际走的是 playwright 驱动本机 Chrome 那条路。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| **Betzel et al. 2026 存在，期刊、卷、文章号都对**：*Controlling the human connectome with spatially diffuse input signals*，*Communications Biology* **9**, Article number **501** (2026)，DOI `10.1038/s42003-026-09560-8`，2026-02-28 发表，Open Access；作者 **Richard Betzel**, Maria Grazia Puxeddu, Caio Seguin, Vincent Bazinet, Andrea Luppi, Alina Podschun, S. Parker Singleton, Joshua Faskowitz, Vibin Parakkattu, Bratislav Misic, Sebastian Markett, Amy Kuceyeski, Linden Parkes | **出版方文章页已读**（nature.com 文章页 HTML 抓取 176 KB：标题、13 位作者、卷期与文章号、摘要、引言）。PMC（`PMC13066548`）与 PubMed（`41764296`）在本机被自签证书链挡住（`CERTIFICATE_VERIFY_FAILED`），**未读**；DOAJ 页返回 403，**未读** |
| **⚠️ 更正（本项目自己的引用写错了）**：§3d 原来写的是"空间弥散输入比局域点输入所需**控制能量**低**两个数量级**" | **原文摘要的"两个数量级"指的是"输入数量"**：*"We identify near-optimal control strategies that **reduce the number of inputs**, in some cases **by two orders of magnitude**."* 对能量只说 *"substantially reducing the energy required for brain state transitions"*。引言里"by orders of magnitude"的能量说法属于**另一件事**：利用输入信号之间的相关性（可压缩性）**逐次转换**再降能量。**正确写法**：近最优策略能把**输入数量**降到 1/100；**能量**是"大幅降低"，数量级的说法属于可压缩性那条策略。（该论文是宏观脑区级 + 控制能量判据，与本项目单神经元级 + 语言建模损失**不可直接比**，这一条原判断不变） |
| **Spieler, Martius & Levina 2026（arXiv `2605.12049`）存在，作者与日期都对**：*Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons*，作者 **Aaron Spieler, Georg Martius, Anna Levina**，2026-05-12 提交（cs.LG），25 页 / 21 图 / 3 表，"Submitted for peer review" | **arXiv 摘要页全文已读**（`arxiv.org/abs/2605.12049`：标题、作者列表、提交日期、abstract 原文、分类）。**全文仍未读**（`arxiv.org/html/2605.12049v1` 可访问，下一轮若要用它的具体数字应先读全文） |
| §3f 对该论文结论的概括**与摘要一致**：三条轴（单元数 `N`、单元复杂度 `k_e`、单元连接度 `k_c`）各自单调变好；固定预算下存在非平凡最优分配；更大预算 favor 又多又复杂的单元；信息论模型把两端收益递减归因于"单神经元信噪比饱和"与"跨神经元冗余"；基准是 SHD-Adding 与 Enwik8 字符级语言建模 | 同上（逐句与 abstract 比对）。**"对本项目不可直接比"的三条理由（ELM 复杂度可调 / 连接非解剖图 / 基准不同）也成立** |

**本轮没有改写 §3d、§3f 的两行正文表格**（避免与主代理的并发编辑撞车）：
准确措辞以本节的更正行为准。

---

## 7.6 本轮（自动循环，2026-09-19 12:2x–12:4x）核 PyTorch 官方《Reproducibility》页：低阶位数值差异的公认来源

这一节为报告 **§7.35** 提供外部依据。**本轮没有新的脑/连接组文献**；要用的是"框架层面的非确定性来源"这条一手文档，
因为 §7.32 的形状（同配置重跑逐位相同、跨 `num_workers` 系统性不同）需要一个"低阶位差异从哪里来"的公认来源。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| PyTorch 官方《Reproducibility》页存在，**本轮读到正文** | `https://docs.pytorch.org/docs/main/notes/randomness.html`，由 `local-search` 的 playwright 后端抓取；该查询 `reliable: true` / `term_coverage: 1.0`（同一轮另一条查询 `reliable: false`、覆盖 0.11，**已按规则弃用**）。抓取返回 **23,330 字符**，本轮读的是其中 **19.5 KB**（fetch `max_chars=20000`，页尾截断）。页面自标 Created / Last Updated **2026-05-14**，属 main（开发版）文档；本机装的是 **torch 2.9.1+rocm7.10.0a20251120**（版本号不同），所以下面每条都用本机源码复核过——见报告 §7.35 (4) |
| 文档原话：**不同 SDPA 后端之间不保证逐位一致** | *"Bitwise matching numerics across different SDPA backends are not guaranteed, even for the same inputs and dtype. Each backend performs floating-point accumulation in a different order, and because floating-point addition is not associative, the results will differ between backends."*（同页 "Low-precision dtypes and numerical reproducibility"） |
| 文档表格：**FLASH / EFFICIENT / CUDNN 三种 SDPA 后端的 backward 默认非确定** | 同页 "CUDA Scaled Dot Product Attention" 表：FLASH = *"The backward pass uses non-deterministic atomic operations by default"*；EFFICIENT = *"The backward pass may split work across keys (num_splits_key > 1) for performance, which is non-deterministic"*；CUDNN = 未接入确定性实现。三条都指 **backward**（forward 列写确定性）；本项目 `src/flybrain/model.py:218` 用的就是 `F.scaled_dot_product_attention(..., is_causal=True)`，在 ROCm 上走这类 fused 后端 |
| 文档提醒：**有些算子内部会用随机数** | *"Some PyTorch operations may use random numbers internally. torch.svd_lowrank() does this, for instance."*（同页 "PyTorch random number generator"）。本项目训练路径**没有** `svd_lowrank`/`pca_lowrank`；`torch.multinomial` 只在生成/预览路径（`connectome_lm.py:152`、`generate.py:154`、`train.py:428`，而 `sample_interval: 0`） |
| **不要过度外推**：这一页能支持"低阶位差异有公认的一手来源"，**不能**支持"本项目的 `num_workers` 差异机制已定位" | 实测反证见报告 §7.35：同配置（nw=2）两次独立运行 **13/13 个记录步逐位相同**，nw=2 与 nw=0 的差异**由配置唯一决定、两次都复现同一组数值**。文档说的 non-determinism 通常表现为**运行间不可复现**，与此处"可复现的跨配置差异"不是同一现象 → 只能列为**候选放大器** |

---

## 7.7 本轮（自动循环，2026-09-19 13:2x–13:4x）复核 `train-your-fly` 配套论文：**仍无可读版本**（阴性）

§7.3 表里剩下的最后一条 ⚠️ 是 `train-your-fly` 的配套论文本体（*Structure alone supports efficient visual computation
in the Drosophila visual system*，Correig-Fraga / Guimerà / Sales-Pardo，出版方自标 "In prep."）。本轮再核一次：
**结论仍是"论文本体读不到"**，但拿到几条新的一手细节。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| **该论文仍没有可检索到的出版或预印版本** | 精确标题查询（带引号）`reliable: true`，但**只返回 1 条**：Zenodo 数据存档 `10.5281/zenodo.21549559`，其标题是 "**Data and source data for** 'Structure alone supports efficient visual computation…'"——即标题**只出现在数据存档的标题里**，没有论文页/预印页 |
| 同轮搜索里被捎带出来的 bioRxiv 链接**不是这篇论文** | 另一条查询（`reliable: true`，覆盖 0.818）给出 `biorxiv.org/content/10.64898/2026.07.27.740875v1`；按该 DOI 再查，标题是 *Hidden genetic diversity in 320 nearly-complete East Asian…* → **与果蝇无关**，是搜索引擎噪声。**不要**把它当成"论文已上线" |
| （本机限制）bioRxiv 正文抓不到 | `local-search` fetch 该 bioRxiv 页返回 **HTTP 429**（与 §7.3 记的 Beiran 论文同因）；上面那个标题是从**搜索结果摘要**读到的，**不是**读全文 |
| Zenodo 存档页**全文已读**（`https://zenodo.org/records/21549559`：2026-07-25 发布、v1.0.0、CC BY 4.0、7.5 GB、86 次浏览 / 202 次下载），比 §7.3 记的多出： | 存档创建者 = **Correig-Fraga, Eudald**（Innovamat）、**Guimerà, Roger**（Universitat Rovira i Virgili; ICREA）、**Sales-Pardo, Marta**（同上）。描述写明：生物图**来自 FlyWire whole-brain v783**、另含**四个 randomized 图**、Fig.2/3 数值源表、README / 数据字典 / manifest / **SHA-256**；Notes 给出精确代码 commit（`connectome 93476a27…`、`train-your-fly 733a8bdb…`、`cogstim e61ebf1d…`）。**"Related works" 里没有任何 "Is supplement to" 的论文 DOI**（只有 derived-from / documented-by / requires），这也是"论文查不到"的正面证据 |
| **对本项目最有用的一条新信息** | 这个"结构本身够用"的结果跑在 **FlyWire whole-brain v783** 上，**不是**本项目用的 **MaleCNS v1.0** → 与 §1.10 的表述一致：它是**第三个独立结果、方向相反**，且是**另一张图**上的结果。引用时不要写成"在同一个连接组上" |
| **不要做的事** | 存档的 "Authors/Creators" 是**数据存档的作者**，不等于论文的完整作者列表；本轮**没有**读到论文摘要或正文，所以它的 Fig.2/3 具体数字**不能**被本项目引用。文献 §1.10 目前引的是**仓库 README**（已读），这一点不变 |

---

## 7.8 本轮（自动循环，2026-09-19 14:3x–14:5x）核两条"固定环路 + 低秩接口"引用，并记下一次措辞陷阱

本轮要为 `brain_rank`（连接组**驱动/读出**的秩）这条主线找一个**框架级**的旁证：本项目是"16.5 万神经元的
固定环路 + 低秩接口"，文献里对应的原型就是 reservoir computing / FORCE 那一支。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| **Sussillo & Abbott 2009 存在**：*Generating Coherent Patterns of Activity from Chaotic Neural Networks*，*Neuron* **63**(4):544–557，2009-08-27（FORCE 学习） | **四处独立索引确认**：ScienceDirect pii `S0896627309005479`、PMC `PMC2756108`、cell.com `S0896-6273(09)00547-9`、PubMed `19709635`；标题与摘要首句（*"We develop a procedure called FORCE learning for modifying synaptic strengths either external to or within a model neural network…"*）一致。**摘要页与正文未读**：`cell.com` fetch 返回 **HTTP 403**，PMC/PubMed 在本机被自签证书链挡住（原因同 §7.5） |
| 它为什么相关、以及**不能**拿来做什么 | 相关：FORCE 就是"在**固定的**混沌 RNN 上训练**低秩**反馈/读出"，与本项目接口同形。不能：它判的是**复现给定轨迹**，本项目判的是**语言建模损失**——**不能用它支持任何关于 `brain_rank` 的结论** |
| **LoRA-RC: Reservoir Computing with Low-Rank Adaptation** 存在：arXiv `2609.12327`，2026-09-11 提交，作者 **Wenbin Wan**，注释 **accepted at IEEE LCSS** | **arXiv abs 页全文已读**（标题 / 作者 / 提交日期 / 分类 / abstract 全文 / comment）。其出发点："*Reservoir computing (RC) trains only a linear readout over a fixed recurrent layer*"；结论：只调读出在系统漂移下**不够**，要用**低秩修正改循环矩阵**；Lorenz 漂移基准上漂移后误差比固定 RC 降 **56%**、比只调读出降 **51%**，消融显示去掉投影使该误差膨胀 **40 倍以上** |
| 对本项目的用法（**界限写清**） | 只提供一条**框架级**旁证：**"固定基底 + 低秩接口"是被承认的设计，而低秩接口单靠自己在漂移下会不够**。它**不**支持本项目的 rank 扫描结论（设置不同：在线漂移适应 vs 离线语言建模；Lorenz vs token 语料）。若将来 rank2048 仍不饱和，它指出的对应做法是**改循环矩阵本身**——对本项目即"解冻连接组的一部分"，那正好是目前**测不了**的方向 |
| **⚠️ 一次措辞陷阱（记下来，别下轮再踩）** | 查询 `low-rank readout capacity saturation recurrent neural network language model rank scaling` 返回 **`reliable: true`、`term_coverage: 1.0`**，但 8 条结果**全是 LoRA 适配器**（知乎/CSDN/arXiv `2106.09685`）——这个引擎把 "low-rank" 直接映射成 "LoRA"。**`term_coverage` 高不等于答对了问题**；必须看结果的语义。换成 `Sussillo Abbott 2009 …` 与 `fixed chaotic recurrent network trained low-rank feedback output weights reservoir capacity` 之后才拿到上面两条真正相关的 |

---

## 7.11 本轮（自动循环，2026-09-19 15:3x–15:4x）把本项目**基底来源**那篇论文的身份钉死：MaleCNS = *Sexual dimorphism in the complete Drosophila male central nervous system*（Cell, 2026-09-03）

> 编号说明：本节的插入锚点落在 §7.8 末尾，而主代理的 §7.9/§7.10 当时已经写在后面，所以本节**物理位置在 §7.9 之前**；
> 编号取当前最大号之后（7.11）以保证唯一，内容与 §7.9/§7.10 不重叠。

§7.3 里 MaleCNS v1.0 这一行此前只有"Janelia 项目页 + Google 博客"两条**第一方新闻来源**，**论文本体未读**。
本项目**整张连接组图**都来自它，所以这一行的身份（标题/期刊/日期/标识符）必须准确。本轮把它钉死了。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| 论文标题：**_Sexual dimorphism in the complete Drosophila male central nervous system_**（不是 §7.3 之前只写的"MaleCNS v1.0 论文"） | **五处独立索引互相一致**：Cell `S0092-8674(26)00942-6`、ScienceDirect `S0092867426009426`、PubMed `42691995`、bioRxiv 预印本 `10.1101/2025.10.09.680999`（2025-10-09）、Janelia FlyEM 项目页 |
| 发表信息：***Cell*，2026-09-03**；预印本 2025-10-09 | 上面五处的日期一致；与 §7.4 记的"数据 2026-06-08 发布、论文 2026-09-03 发表"不冲突 |
| 摘要片段："We present the connectome of the entire Drosophila male central nervous system. This contains **166,700 neurons**…" | ScienceDirect / Cell / bioRxiv 三处的摘要首句一致。**166,700 神经元**与 §7.4 记的"NeuroCraft 报 166,700 / 25,582,938 边，本项目 164,587 / 25,563,096"对得上（本项目是同一张图的一个导出） |
| **同期刊还有一篇姊妹论文**：*The organization of visual pathways in the Drosophila brain*（Cell `S0092-8674(26)00941-4` / ScienceDirect `S0092867426009414`，2026-09-03） | 两条独立索引（cell.com、sciencedirect）标题与日期一致；**摘要未读** |
| 没读到的 | **正文与摘要页都没读到**：cell.com fetch **403**、ScienceDirect **403**、bioRxiv **429**、PubMed 在本机被自签证书链挡（同 §7.5）。所以本行只能标"标题 + 摘要首句（多处索引确认）"，**不能**引用它的具体数字 |

**顺带记住的操作事实**：本项目用的 `data/wholebrain.npz`（164,587 神经元 / 25,563,096 突触）是这张图为导出，
**不是** FlyWire whole-brain（v783 是 `train-your-fly` 那条线用的，见 §7.7）。两篇文献经常被混着引用，别混。

---

## 7.9 本轮（主代理，2026-09-19 下午）训练效率方向检索：稀疏访存负载不均

> **编号更正**：本节原写作 §7.8，与自动循环同一时段写的另一节 §7.8（Sussillo & Abbott 2009 / LoRA-RC，
> 在文件里位于本节之前）**撞号**。为避免破坏循环那节的既有引用，**本节改为 §7.9**，
> 引用本节的地方（`research/next_experiments.md` 状态横幅、`research/training_efficiency.md` §6）已同步更新。

触发：用户要求"搜索训练效率更高的生物脑神经结构训练方法"并优化本项目（速度太慢、质量太低）。
本轮检索走 `local-search` MCP，**三条查询的 `reliable` 全部为 `true`**
（`term_coverage` 0.889 / 0.700 / 0.900；失败后端只有本机未启用的 SearXNG）。
**读到程度：全部只读到标题与搜索摘要片段，本轮没有 fetch 任何一篇全文。** 下方一律如实标注。

**结论先行**：文献方向与本项目自己的测量一致——瓶颈是**稀疏访存的负载不均**（→ 分箱 / 剪枝），
**不是算术量**（→ 算力没用）。本项目的实测成本模型与优化菜单见
[`research/training_efficiency.md`](./training_efficiency.md)。

| 文献 | URL | 读到程度 | 与本项目的关系 |
|---|---|---|---|
| HC-SpMM: Accelerating Sparse Matrix-Matrix Multiplication for Hybrid GPU Cores | https://arxiv.org/html/2412.08902v1 | 标题+摘要片段 | SpMM 在长短行混合时的调度 |
| Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication | https://arxiv.org/pdf/2501.09251 | 标题+摘要片段 | 通用 SpMM 的负载划分与数据亲和性 |
| Optimizing sparse-dense matrix–matrix multiplication for DCUs（CCF Trans. HPC 2025） | https://link.springer.com/article/10.1007/s42514-025-00254-x | 标题+摘要片段 | **摘要明确写"粗粒度两级分箱（coarse-grained two-level binning）"处理负载不均**——正是本项目那个 5.1 ms 固定项要用的手段 |
| VCSR-MM: A Bandwidth-Optimized SpMM Design for Modern GPUs（IEEE） | https://ieeexplore.ieee.org/abstract/document/11527293 | 标题+摘要片段 | 面向带宽的 SpMM 布局 |
| Connectome-constrained networks predict neural activity across the fly brain（Nature 2024） | https://www.nature.com/articles/s41586-024-07939-3 | 标题+摘要片段 | 连接组约束网络、预测全脑活动；与本项目"冻结接线、只训少数自由度"同一思路 |
| Whole-Brain Connectomic Graph Model (FlyGM) | https://arxiv.org/html/2602.17997 | 标题+摘要片段 | 全脑连接组图模型；§1 已复核存在，此处仅补"强调高效" |
| FLYNN: Robust Neural Network for Robot Navigation using fly connectome | https://arxiv.org/pdf/2607.00025 | 标题+摘要片段 | 训练果蝇连接组做**导航**（任务型），与本项目"连接组对进化过的任务有用"一致 |
| Adaptive and lightweight surrogate gradients (AdaLi)（Front. Neurosci. 2026） | https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1795946/full | 标题+摘要片段 | 轻量替代梯度降低反传成本。**适用性有限**：本项目连接组 LM 用 `brain_spiking: false`（模拟读出），不走替代梯度 |
| Masked Surrogate Gradient（arXiv 2406.19645） | https://arxiv.org/pdf/2406.19645 | 标题+摘要片段 | 同上，SNN 直接训练 |

**没有找到的东西**（保证不编造）：

1. **没有找到任何"减少连接组扫描次数"的文献**。检索到的 SpMM 文献都在优化**单次**稀疏矩阵乘的调度，
   而本项目的成本结构是"每步扫几次 × 每次多贵"——后半段有文献，前半段（扫几次）是**本项目自己的架构决定**，
   即 `brain_chunks`，没有外部依据可引。
2. **没有找到"生物连接组 + 语言建模 + 训练效率"三者同时出现的论文**。这一条与 §7 第 2 条一致：
   该设定仍然只有本项目一家。
3. 本轮**没有**读任何全文，因此上表的"关系"列只用于**指明方向**，不支撑任何具体数字。
   若要引用其中任何一个数字，必须先读全文（按 §7.3 的规矩）。

---

## 7.10 本轮（主代理，2026-09-19 下午）生物可行的学习规则：资格迹 + 神经调质

触发：用户要求"用正确的生物电刺激训练苍蝇大脑，而不是沿用传统 LLM 训练方式"。
诊断与分阶段方案见 [`research/biological_training.md`](./biological_training.md)。
本轮检索走 `local-search`，**两条查询 `reliable` 均为 `true`**（`term_coverage` 1.0 / 0.9）。
**读到程度：全部只读到标题与搜索摘要，本轮没有 fetch 任何全文。**

| 文献 | URL | 关系 |
|---|---|---|
| A solution to the learning dilemma for recurrent networks of spiking neurons（e-prop, Bellec 等, Nature Comms 2020） | https://www.nature.com/articles/s41467-020-17236-y | 循环脉冲网络在线、局部学习的标准答案：资格迹 + 学习信号，无需 BPTT |
| Event-driven eligibility propagation in large sparse networks | https://www.alphaxiv.org/overview/2511.21674v1 | **形状最接近本项目**（大而稀疏；本项目 2,556 万突触）的事件驱动 e-prop |
| Can Biologically Plausible Temporal Credit Assignment…（e-prop） | https://ar5iv.labs.arxiv.org/html/2506.06904 | 直接研究"生物可行规则能否匹配 BPTT" |
| Three-factor learning in spiking neural networks: an overview（Cell Patterns 2025） | https://pmc.ncbi.nlm.nih.gov/articles/PMC12745983/ | 三因子（资格迹 × 神经调质）综述，含奖励调制 STDP |
| Reward-Modulated STDP（Lava 教程） | https://lava-nc.org/lava/notebooks/in_depth/three_factor_learning/tutorial01_Reward_Modulated_STDP.html | R-STDP 的可运行参考 |
| Modulated STDP-based learning in deep convolutional SNNs（Neurocomputing 2025） | https://www.sciencedirect.com/science/article/pii/S0925231224019416 | 奖励调制 STDP 在较深网络上的实测 |

**对本项目的可接入点**（记录，尚属推断）：连接组权重是冻结的，可学参数是**每神经元一个 `release_gain`（164,587 个）**，
所以资格迹只需 16 万个浮点，**不必为 2,556 万条边各存一条**；广播的调制信号可用逐 token 的损失改善量充当。

**没有找到**：任何"生物电刺激标定"（把群体驱动到阈值附近的响应带）与"连接组语言建模"结合的论文。
本项目 §2.2 那个"只有 13.55% 神经元落在阈值 ±0.5 响应带"的测量，**没有外部文献可直接对照**。

---

## 7.12 本轮（自动循环，2026-09-19 16:3x–16:4x）把 e-prop 那条引用从"只有标题"升级到"读了摘要+引言"

主代理新开的"用生物电刺激训练"方向（`research/biological_training.md`，其检索记录在本文档 §7.10）把
**e-prop** 列为"循环脉冲网络在线局部学习的标准答案"，但当时**只读到标题与搜索摘要**。本轮把它补成可引用的程度。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| **论文存在，且出版信息完整**：*A solution to the learning dilemma for recurrent networks of spiking neurons*，***Nature Communications* 11, Article 3625（2020-07-17）**，**Open Access**，DOI `10.1038/s41467-020-17236-y` | **出版方页面已读**（`nature.com/articles/s41467-020-17236-y` 抓取 97.8 KB）：标题、7 位作者（**Guillaume Bellec**, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, **Wolfgang Maass**）、卷/文章号/日期、**摘要全文**、引言的 Fig.1 段落、访问与引用计数（79k Accesses / 599 Citations / 142 Altmetric） |
| 摘要（逐句核对）主张：把神经科学的两项实验发现与一个数学结果组合起来，得到**生物可行的在线**梯度下降学习（含深度强化学习），称为 e-prop；**性能接近 BPTT**；并指向节能脉冲硬件上的片上学习 | 摘要原文：*"This learning method--called e-prop--approaches the performance of backpropagation through time (BPTT) …"*。引言另给出两个限定：BPTT 需要存下所有神经元的中间状态再做离线反向（"very unlikely that BPTT is used by the brain"），而 e-prop 的资格迹是**前向**算的（Fig.1d） |
| **对本项目的界限（必须写清）** | e-prop 讲的是**脉冲（spiking）循环网络**的在线局部学习；本项目现在的整脑模块**不是脉冲的**——`brain_spiking: false`、状态是 rate 型、训练用 **BPTT**（§3.3、§7.4）。所以它是**目标设计**（§7.10 提出的"资格迹 + 调制信号"路线的原始出处），**不是本项目的现状**；引用时不要写成"本项目已实现 e-prop" |
| 顺带核实（同一方向、仍在标题级） | §7.10 表里其余条目（`2511.21674` 事件驱动 e-prop、`2506.06904` 生物可行规则 vs BPTT、Cell Patterns 2025 三因子综述、Lava R-STDP 教程、Neurocomputing 2025 调制 STDP）**本轮没有复核**，仍按主代理标注的"标题+搜索摘要"对待 |
| 没读到的 | 论文正文（方法/结果的具体数字）**未读**：本轮读的是出版方页面上的摘要与引言；正文 PDF（`preview-www.nature.com/.../s41467-020-17236-y.pdf`）**未抓** |

---

## 7.13 本轮（自动循环，2026-09-19 17:3x–17:4x）复核"形状最接近本项目"那条：arXiv `2511.21674` 读了摘要

主代理在 `research/biological_training.md`（检索记录在本文档 §7.10）把这条标为"**形状最接近本项目**（大而稀疏）"，
但当时**只读到标题与搜索摘要**。本轮把它补到**摘要级**。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| 论文存在，标题与编号一致：*Event-driven eligibility propagation in large sparse networks: **efficiency shaped by biological realism***，**arXiv `2511.21674`**（cs.NE），**2025-11-26 提交** | **arXiv abs 页全文已读**（标题、8 位作者、提交日期、分类、abstract 全文、DOI `10.48550/arXiv.2511.21674`）。作者：**Agnes Korcsak-Gorzo**, Jesús A. Espinoza Valverde, Jonas Stapmanns, Hans Ekkehard Plesser, David Dahmen, Matthias Bolten, Sacha J. van Albada, **Markus Diesmann** |
| 摘要要点：把 e-prop 的**时间驱动**更新改成**事件驱动**，并接进大规模脉冲网络仿真平台；任务举例 **neuromorphic MNIST**；强调连续动力学/连续权重、**严格局部性**、**稀疏连接**；声称"可扩展到**百万级神经元**而不牺牲学习性能" | 同上（abstract 逐句）。**关键词是"生物约束塑造效率"，不是"连接组语言建模"** |
| **对本项目的界限（必须写清）** | ①它训的是**脉冲网络**（本项目的整脑是 rate 型、`brain_spiking: false`）；②它的"百万神经元"是**脉冲仿真**里的结果，不是在语言建模损失上（本项目 164,587 神经元、词表 16,384）；③它的基准是 neuromorphic MNIST，本项目的判据是固定 24 批的 val_loss。⇒ **"形状最接近"只在接口形态（大、稀疏、固定连接 + 可学局部量）上成立**，不要当成同类结果引用 |
| 没读到的 | **正文/PDF 未读**：本轮读的是 arXiv abs 页（摘要级）。§7.10 表里其余条目（`2506.06904`、Cell Patterns 2025 三因子综述、Lava R-STDP 教程、Neurocomputing 2025 调制 STDP）**仍未复核** |

---

## 7.14 本轮（自动循环，2026-09-19 18:2x–18:4x）为"脉冲整脑"新方向补两条能用的引用：**firing-rate collapse** 与"**稀疏放电不等于坏**"

本轮观察到（报告 §7.41）：`brain_spiking: True` 的三个臂里，**被掩码的两个臂放电率 3.8–6.5%、96% 神经元沉默，损失比未掩码臂差 ~2.2**。
主代理的 `research/biological_training.md` 方向需要两条能对得上的文献，于是检索了两条**一手**来源（查询 `reliable: true`、覆盖 1.0）。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| **"放电率塌缩"是一个有名字、有专门论文的问题**：*Spiking Network Initialisation and Firing Rate Collapse*，**arXiv `2305.08879`**（2023-05-13，cs.NE），作者 **Nicolas Perez-Nieves, Dan F. M. Goodman** | **arXiv abs 页全文已读**（标题/作者/提交日期/分类/abstract 全文/DOI `10.48550/arXiv.2305.08879`）。摘要原话点明：ANN 的初始化方案对 SNN"**often inadequate and require manual tuning**"，并专门提出"**the firing rate collapse problem**"并给出若干解法（随机游走/Wiener 过程、扩散与散粒噪声近似） |
| **但"放电少"本身不等于坏**：*High-performance deep spiking neural networks with **0.3 spikes per neuron***，***Nature Communications* 15, Article 6793**（2024-08-09，**Open Access**），作者 Ana Stanojevic, Stanisław Woźniak, **Guillaume Bellec**, Giovanni Cherubini, Angeliki Pantazi, **Wulfram Gerstner** | **出版方页面已读**（摘要全文 + 引言；`nature.com/articles/s41467-024-51110-5`，抓取 90 KB）：他们用 **<0.3 spikes/neuron** 达到与 ANN **相同**的性能，并把问题归到一类"**vanishing-or-exploding gradient**"（阈值处膜电位斜率的选择），而不是"放电率低"本身；49k Accesses / 75 Citations |
| **对本项目的用法与界限** | 用法：①本项目 `brain_activity` 3.8%、`silent_fraction` 96% 的现象**有文献名字直接对应**（firing rate collapse）；②**不能**由此推出"放电率必须高"——Stanojevic 等人正好说明低放电可以高性能，前提是初始化/梯度映射对。⇒ 把 `brain_firing_penalty: 0.0` + 脉冲 + 掩码的组合当作**"可能是初始化/梯度问题"**来查，比当作"生物不真实"更有依据。**界限**：两条都是**前馈图像分类**的 SNN（MNIST/Fashion-MNIST/CIFAR、time-to-first-spike 或 surrogate gradient），本项目是**递归连接组**（164,587 神经元、32 sweeps、语言建模），**没有**可直接搬的结论或数字 |
| 没读到的 | 两条的**正文**都未读（都是摘要级；PDF 未抓） |

---

## 7.15 本轮（自动循环，2026-09-19 19:3x–19:4x）**找到了与本项目核心问题同形的已发表工作**：冻结连接组当 rate operator，拆开"统计 vs 精确接线"

本轮为 §7.42 的解剖学结论找对照，检索命中一篇**直接的近邻工作**（`reliable: true`，覆盖 0.615——**缺词说明引擎没检索到我想找的"shuffled 对照"那条线，但这条结果本身与查询高度相关**，所以按语义采纳并逐条核对）。

| 结论 | 判据（读到的是全文 / 摘要 / 只有标题） |
|---|---|
| 论文：*A frozen rate operator from the complete larval connectome: **degree and weight govern the gross response, exact wiring governs input routing and mushroom-body modes***，**arXiv `2606.17745`**（v1 2026-06-16，**v2 2026-07-02**），作者 **Stavros Therianos**（单作者预印本，21 页 6 图 + supplement） | **arXiv abs 页全文已读**（标题/作者/提交与修订日期/分类 q-bio.NC/DOI `10.48550/arXiv.2606.17745`/abstract 全文/comment 里的 v2 修订说明） |
| 做法：把**完整幼虫连接组**（强连通核心 **2,825 神经元**）当作**冻结的 leaky-tanh rate operator**，**不拟合任何单神经元参数**，并与**度与权重匹配的重连集合**等对照比较 | 摘要原文；他们强调"All properties are the operator's, attributable to wiring and weights alone, **not a measurement of larval activity**" |
| 结论（两半，正好是本项目那条问题的两半）：**① 粗略动力学的确是统计决定的**——operator 非正规、近似线性，**增益与有效维度**与重连集合只差百分之几；**② 两个空间分辨的性质打破这个规律**——稀疏传入驱动下真实接线把活动**限制在核心的 1/5**（重连集合 2/3、随机图全部），且**蘑菇体把主要驱动模式集中**得远超集合（过了尺寸匹配、奇异子空间、family-wise 对照）。⇒ "input routing"与"主导驱动模式"**写在精确接线里** | 同上（abstract 逐句） |
| **为什么这条对本项目重要** | 它与本项目的设计同形（**冻结连接组 + rate 动力学 + 重连对照**），而且把"真实接线 vs 随机接线"**拆成了两个可分别测的量**：**粗略行为**（本项目测的是 loss/ppl，属于"gross"）与**输入路由/主导模式**（本项目刚在 §7.42 用端口掩码去碰的正是这个）。它还给出了一个**不需要训练就能测**的判据：**稀疏传入驱动下活动被限制在多少比例的核心**（真实 vs 度/权匹配重连）——本项目的 `brain_activity`/`silent_fraction`/参与度正是同一类量 |
| **界限与不可外推处（必须写清）** | ① 它是**幼虫**（2,825 神经元核心）、本项目是**成蝇 MaleCNS**（164,587）；② 它是**冻结、无学习**的线性-近线性 operator，本项目要**训 BPTT 低秩接口**；③ **单作者预印本、未同行评审**；④ 它没有报告任务损失，不能用来解释本项目的 loss 差异；⑤ 它的"真实接线更重要"与本项目 §7.42 在 450 步上的"真实端口反而更差"**不构成矛盾**（设置、尺度、判据都不同），**不要**互相引用成支持/反对 |
| 没读到的 | **正文未读**（只有 abs 页）；正文里的重连集合构造细节、统计量定义都没有核对 |

**由此给本项目的一条具体建议（写进 `next_experiments.md`）**：在做更贵的训练实验之前，先做一次**零训练的"路由"测量**——
用同一份 `data/wholebrain.npz`（真实）与一份**度/权匹配的重连版**，在**稀疏传入驱动**下测
①被激活神经元占比、②`drive/readout` 参与度、③与随机图的对比。这正是 §2.2/§7.4 那条"13.55% 落在响应带、脉冲图案逐批相同"的测量
被**正规化**后的版本，成本是几次前向，不需要训练。
