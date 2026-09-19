# 自动循环（AutoLoop）：让本项目的"北极星要求"自己一轮轮跑下去

这里实现的是：**每次问答彻底结束后，自动把本项目的标准要求再发一遍，不断不停。**
业界把这种"Stop 钩子拦住收工、把提示词塞回去让 agent 接着干"的做法叫 Ralph Loop。

它由**两个引擎**组成，缺一不可：

| 引擎 | 触发时机 | 上限 | 现在是否生效 |
|---|---|---|---|
| **Stop 钩子** `%GROK_HOME%\hooks\flybrain-autoloop.json` | 每次 turn 要正常结束时 | grok 在一个 turn 内**最多接受 8 次**续跑，用满即强制结束 | 需要重载钩子（见下） |
| **调度器** `scheduler_create`（1 小时一次，durable） | 墙钟定时 | 7 天后自动过期 | 已生效 |

**为什么要两个**：钩子只能在**单个 turn 内**续跑 8 次，用满之后这个 turn 会真正结束，
接下来没有任何东西会再发提示词——循环就断了。调度器就是补这个位：
它到点发一次新提示词，开一个新 turn，钩子的 8 次配额随之重置。
反过来，只有调度器的话，两次触发之间要干等一个小时，也不叫"不停"。

---

## 怎么用

```powershell
# 看现在的状态：钩子在不在、开关是开还是关、已经自动跑了多少轮
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Status

# 停（最可靠的一键停止：放一个哨兵文件，钩子看到就放行）
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Disable

# 再开
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -Enable

# 把轮次计数清零
powershell -ExecutionPolicy Bypass -File scripts\autoloop_ctl.ps1 -ResetCounter
```

### 只装了一半：Stop 钩子需要重载才会生效

grok **在会话启动时读取钩子**。刚写进磁盘的钩子文件在当前会话里不会自动加载，
两个办法二选一：

- 在 TUI 里运行 `/hooks`，在 Hooks 标签页按 **`r`**（中途重载，会重新读盘上所有钩子）
  —— 推荐，不用重启；
- 或者重启会话。

在此之前，只有调度器那一半在工作（它一小时一次）。

### 彻底停掉（两半都要停）

`-Disable` 只关掉钩子那一半。调度器活在 grok 会话里、不在磁盘上，要单独删：

- 在 TUI 里跑 `/loop` 查看并删除，或者
- 让 agent 调 `scheduler_delete`，任务 ID 是 `01a0aef5-9e95-77c1-a7f5-ff5a149fd48a`。

---

## 为什么钩子装在全局目录，而不是项目里的 `.grok/hooks/`

grok 的钩子来源里，**全局目录 `%GROK_HOME%\hooks\` 永远受信任**，而项目目录
`<项目>/.grok/hooks/` 需要先 `/hooks-trust` 授权，否则**被静默跳过**。

实测本机的 `trusted_folders.toml` 里只有 `C:\Users\heyiy\Desktop\免费api`，
本项目**不在**信任列表里。也就是说：把钩子放进项目里，它会一声不响地永远不触发
——这比不装还糟，因为你会以为它在跑。

所以钩子装在全局目录，并让**脚本自己按路径限定作用域**：
只有 `workspaceRoot` 正好等于本项目时才注入，其它项目一律放行。
本机 `GROK_HOME` 是 `C:\Users\heyiy\Desktop\a6\.grok`（**不是** `~/.grok`）——
装错目录同样会静默失效。

---

## 脚本会把提示词原样塞回去吗

`autoloop/prompt.txt` 里除了**逐字保留**你的原始要求，还加了三条防呆：

1. **先判断进度再动手**：先读 `reports/FINAL_REPORT.md`，只推进"未完成或未验证"的部分。
   不加这条的话，每一轮都会从头重来——重新下载连接组、重新生成教师语料、重跑已完成的实验，
   把时间全烧在重复劳动上。
2. **长任务必须带墙钟上限**（`max_minutes`），且不允许空等轮询（你之前明确提过这一点）。
3. **只准报实测数字**，没测过的必须写"未验证"。

`{{ROUND}}` 会被替换成真实轮次，所以每一轮 agent 都知道这是第几次自动触发。

---

## 设计上的取舍与已知风险

- **跑训练/基准前必须确认显卡是空的**（本项目已经栽过三次在"测量方式本身"上）。
  实测：`flybrain.serve` 占着卡时，真实接线训练从 **218 tok/s 掉到 51 tok/s**，
  于是同样 40 分钟墙钟只跑出 150 步而不是 492 步——看起来像"某条臂慢 3.3 倍"，
  其实纯粹是进程争抢。所以：**不要一边挂着网页服务一边跑训练**；
  跨运行比较任何墙钟/吞吐数字之前，先 `Get-Process python` 确认。
  这条已经写进 `prompt.txt`，每一轮都会读到。
- **最短间隔 60 秒**，且调度器**按墙钟触发，不等上一轮结束**。
  本项目一轮动辄几十分钟（一次有上限的训练 + 验证），间隔太短会让提示词排队、
  然后连着空跑好几轮。所以取 1 小时。想改就在 `/loop` 里删掉重建。
- **调度器加了一道"让位"闸门**（`scripts/autoloop_busy_check.ps1`）。
  上面那条"按墙钟触发、不等上一轮结束"曾经造成过一次真实事故：
  调度器在一轮还在跑的时候又起了一轮，两条臂同时在一张卡上，
  于是正在跑的训练从 356 tok/s 掉到 51 tok/s，按墙钟停下的那一轮只跑完成比例的步数
  ——本项目因此得出过一次"打乱接线慢 3.3 倍"的结论，随后被撤回
  （见 `reports/FINAL_REPORT.md` §7.6）。钩子那半边**本来就有**这个保护
  （`backgroundTasks` 非空就放行，见 `test_work_already_in_flight_is_not_stacked_on`），
  调度器那半边没有，所以补上：调度器每轮开工前先跑一次这个检查，
  `BUSY`（退出码 3）就当场结束本轮、不碰显卡，`FREE`（退出码 0）才开工。
  判定用两个信号：本项目还有 python 作业在跑，或者 `reports/` `configs/` `scripts/`
  `checkpoints/` 里有文件是最近 5 分钟内写过的（后者会自己衰减，不会把循环卡死）。
  **失败方向同样是"放行"**：进程列表读不到、目录不存在、参数异常，一律报 `FREE`——
  一个永远说"忙"的检查会让循环彻底停摆，那比偶尔重叠更糟。
- **调度器 7 天后过期**，这是 grok 的硬限制。要真正"永不停止"，
  得改用 Windows 计划任务在会话外拉起 `grok` 无头模式（见用户手册
  `14-headless-mode.md`），那超出本会话能配置的范围。
- **钩子会拦住这个项目里的每一次正常收工**，包括你只是想随口问一句的时候。
  想正常对话就先 `-Disable`。
- **失败方向是"放行"**：钩子脚本任何异常、JSON 解析失败、超时、文件缺失，
  都走 `exit 0` 放行，不会卡住你的会话。每一条放行原因都写进 `inject.log`，
  所以"为什么没触发"永远可以从日志里查，不用猜。

---

## 文件清单

| 路径 | 作用 |
|---|---|
| `autoloop/prompt.txt` | 每轮注入的提示词（唯一来源，直接编辑这里） |
| `autoloop/state.json` | 轮次计数与最近一次注入时间 |
| `autoloop/inject.log` | 每次触发/放行及原因 |
| `autoloop/DISABLE` | 存在即停用（哨兵文件） |
| `scripts/autoloop_stop_hook.ps1` | 钩子本体 |
| `scripts/autoloop_ctl.ps1` | 状态 / 开 / 关 / 清零 |
| `scripts/autoloop_busy_check.ps1` | 调度器开工前的"让位"检查：`FREE`=可以开工，`BUSY`=已有另一轮在跑 |
| `tests/test_autoloop.py` | 17 项测试：每条放行分支 + 中文提示词完整性 + 无 BOM + 让位检查的四个判定 |

## 踩过的两个编码坑（已修，并有测试守住）

1. **Windows PowerShell 5.1 把没有 BOM 的 `.ps1` 当 ANSI 读**，
   所以脚本里**不能出现中文常量**——它还没运行就已经乱码了。
   提示词因此从不内联，一律从 `prompt.txt` 用显式 UTF-8 读入。
2. **`Set-Content -Encoding UTF8` 会写入 BOM**（PS 5.1），
   于是任何按普通 UTF-8 读该文件的程序都会在 JSON 解析时炸。
   状态文件和日志改用无 BOM 编码器写入；决策输出则整体转成**纯 ASCII**
   （`\uXXXX` 转义），这样宿主机的任何代码页都改不动它。
   这条是测试先发现的：`test_a_completed_turn_is_kept_working_with_the_prompt_intact`
   一开始就红在 BOM 上。
