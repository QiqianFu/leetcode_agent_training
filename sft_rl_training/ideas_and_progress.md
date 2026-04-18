# LeetCode Agent SFT+RL Training — Ideas & Progress

## 项目目标

用 Qwen3-1.7B/3B 替换 leetcode_agent 的 DeepSeek API backbone，通过 SFT+RL 训练使其在 agent 辅助刷题任务上接近 DeepSeek 的表现。

- **任务性质**: tool routing + 对话理解（不是让模型自己解题，而是辅助用户刷题）
- **核心能力**: 根据用户意图正确选择并调用 17 个 tools 中的某一个
- **训练框架**: Agent-R1 (基于 veRL 的 step-level MDP PPO 训练)
- **基座模型**: Qwen3-1.7B（Qwen3 最小的是 1.7B，没有 1.5B）

---

## 硬件方案

| 阶段 | 硬件 | 用途 |
|------|------|------|
| 代码验证 | 2x A40 (学校) | 用 Qwen3-1.5B 跑通 SFT 和 RL 全流程，确认代码无 bug |
| 正式训练 | 2x A100-80G (AutoDL) | 用 Qwen3-3B 跑完整训练，产出最终模型 |

---

## 整体策略

### Phase 1: SFT — 教格式，不教策略

**目标**: 让模型学会按正确格式输出 tool call，能稳定生成合法的 Hermes XML 格式调用。

**为什么需要 SFT (Agent-R1 的 GSM8K 例子不需要)**:
- Agent-R1 的 GSM8K 任务只有 1 个 tool (`calc_gsm8k_reward`)，且 Qwen Instruct 原生支持 Hermes tool call 格式
- 我们的任务有 17 个自定义 tools，参数各异，Qwen 1.5B 没见过这些 schema
- SFT 的作用是 cold start，让 RL 有一个合理的探索起点

**数据生成**:
- 用 DeepSeek API 生成 ReAct 轨迹 (problem -> think -> tool_call -> result -> ...)
- 关键：数据要多样化，不能只有"最优路径"，要包含:
  - 直接调用的轨迹（模型"知道"答案时）
  - 先 search/check 再调用的轨迹（模型"不确定"时）
  - 出错后恢复的轨迹（模型犯错后用其他 tool 修正）
- 数据量：500-2000 条轨迹
- Loss 设计：只在 assistant turns 上计算 loss，tool output / user turns mask 掉

**工具**:
- 可用 transformers Trainer / LLaMA-Factory / swift 等标准 SFT 框架
- 不需要 Agent-R1（它没有 SFT 模块）

### Phase 2: RL — 教策略，用 outcome reward

**目标**: 让模型学会"什么情况下选什么 tool"，探索出适合自身能力水平的最优策略。

**为什么 RL 而不是更多 SFT**:
- SFT = 模仿 DeepSeek 的行为，但 DeepSeek 的"最优路径"不适合 1.5B 模型
  - DeepSeek 知道 LeetCode 146 是 LRU Cache，直接 start_problem
  - 1.5B 不知道，硬模仿会导致幻觉
- RL = 让模型在自己的能力范围内探索最优策略
  - 1.5B 会自己发现"不确定时先 check_problem 是最稳的"
  - 这种策略不会出现在 DeepSeek 的轨迹里，只有 RL 能学到

**Reward 设计 — Outcome-based，不做轨迹匹配**:

核心原则：只评判最终状态是否达成，不管中间走了几步。

| 场景 | 期望最终状态 | reward |
|------|-------------|--------|
| 用户说"做 146 题" | `start_problem(146)` 被成功调用 | +1.0 |
| 用户问"有没有类似的题" | `find_similar_problems` 返回有效结果 | +1.0 |
| 用户做完了，需要总结 | `analyze_and_memorize` 被调用 | +1.0 |
| 未达成目标 | — | 0.0 |

可选的效率 bonus（谨慎使用）：
- 1 步完成: +0.2
- 2-3 步: +0.0
- 超过 5 步未完成: -0.3

**训练框架**: Agent-R1
- 利用其 step-level MDP 和 AgentEnvLoop 抽象
- 需要实现自定义的 `LeetCodeToolEnv`，注册我们的 17 个 tools
- 算法：GRPO（和 Agent-R1 example 一致）

### Phase 3: 评估

**评估集**: 50-100 个 held-out 的用户意图场景（不在训练集中出现的题目/交互模式）

**指标**:
- Tool 选择准确率：是否调了正确的 tool
- 参数准确率：tool 参数是否正确
- 效率：平均几步完成任务
- 格式正确率：输出是否是合法的 tool call
- 对比基线：DeepSeek API / SFT-only / SFT+RL

---

## 关键设计决策记录

### 2026-04-12: 为什么 outcome reward 优于 trajectory matching

**问题**: 用 DeepSeek 生成的轨迹作为 RL 的 ground truth 行不行？

**结论**: 不行。不同能力的模型应该有不同的最优策略。

**例子**: 用户说"做 146 题"
- DeepSeek（知识丰富）: 直接 `start_problem(146)` → 1 步完成
- Qwen 1.5B（知识有限）: `check_problem(146)` → 看到是 LRU Cache → `start_problem(146)` → 2 步完成
- 两条路径都是"对的"，reward 都应该是正的
- 如果用 trajectory matching，第二条路径会被惩罚，这是错误的

**决策**: RL reward 只看最终状态是否达成，不比较中间路径。

---

## 进展记录

### 2026-04-12: 模型选型修正 & SFT 流程验证

**模型修正**: Qwen3 没有 1.5B，官方最小的是 **Qwen3-1.7B** (1.72B params)。后续全部改用 Qwen3-1.7B。
- 模型已下载到 `/shared/rsaas/qiqianf2/hf_models/models--Qwen--Qwen3-1.7B/`
- 不需要 HuggingFace gating，直接可下载

**环境搭建**:
- 克隆 conda env `qwen` → `qwen_RL`，避免影响原环境稳定性（后续还要装 veRL）
- 在 `qwen_RL` 中安装了 peft 0.18.1
- 基础包: torch 2.5.1+cu121, transformers 4.57.6, datasets 4.8.4

**SFT 训练流程验证** (用合成模板数据，非真实轨迹):
- 训练脚本: `scripts/train_sft.py` — transformers Trainer + LoRA (r=64, alpha=128)
  - 只在 assistant turns 上计算 loss，user/tool/system turns mask 掉
  - Qwen3 chat template 原生支持 tool calling（Hermes XML 格式）
- Condor 提交: `scripts/sft.sub` + `scripts/run_sft.sh`
- 踩坑记录:
  1. flash_attn 未安装 → 改用 eager attention
  2. `nvidia-smi -L` 列出所有 GPU 但 condor 只分配 2 个 → 改用 `CUDA_VISIBLE_DEVICES` 检测
  3. LoRA + gradient_checkpointing 需要 `model.enable_input_require_grads()`
  4. DDP + gradient_checkpointing 的 "mark variable ready twice" → 加 `--ddp_find_unused_parameters False`
  5. batch_size=4 在 A40 45GB 上 OOM → 改为 batch_size=2 + grad_accum=8
- 结果: 训练 20min 完成，train_loss 2.204→0.0023，eval_loss 0.0003，流程完全跑通（详见 experiment_log.md Exp-001）

**SFT 数据生成方案讨论**:
- 合成模板数据只能验证流程，不能用于正式训练
- **决定用方案 A**: 直接跑 leetcode_agent 的真实 agent loop，用 DeepSeek 作为 backbone 录制对话轨迹
  - 优势: 轨迹质量高，包含真实的 tool 调用和返回结果
  - 需要: 构造多样化用户输入，模拟各种刷题场景
- DeepSeek API 已验证可用 (deepseek-chat, key 来自 Qwen3-VL 项目)

**RL 框架确认**: Agent-R1 基于 **veRL** (字节跳动)，不是 TRL。核心依赖: verl==0.7.0 + Ray + vLLM。

### 2026-04-12: leetcode_agent 代码理解 & SFT 数据生成方案

#### leetcode_agent 核心交互流程

**99% 的典型 session**:
1. 用户说想做题 → agent 调 `pick_problem` 或 `search_problem`（涉及 CodeTop/LeetCode API + 终端 arrow_select UI）
2. 用户选中题目 → agent 调 `start_problem`（创建 `{category}/{id}_{title}.py` + `.memories/{id}_{title}.md`，AI 分类题目到 12 个目录之一）
3. agent 自动调 `find_similar_problems`（sub-agent 在已做题目的 L3 记忆中找相似题）
4. 用户做题过程中对话（提示、讲解等，不涉及 tool）
5. 用户做完 → agent 调 `read_solution` 读代码 → 给反馈 → 调 `analyze_and_memorize` 写总结到 L3

**1% 的其他场景**:
- 复习旧题: `check_problem` → `read_memory` → 对话
- 查找文件: `find_problem_file` / `search_workspace_files` / `list_category_problems`
- 追加参考解法: `find_problem_file` → `append_solution`
- 用户偏好: `update_user_memory`（sub-agent 更新 L2）
- 今日计划/高频题: `get_daily_plan` / `get_hot_problems`
- 知识搜索: `web_search`

#### 17 个 tools 按使用频率分层

| 层级 | Tools | 说明 |
|------|-------|------|
| **核心链路** (每次做题必走) | `pick_problem`/`search_problem` → `start_problem` → `find_similar_problems` → `read_solution` → `analyze_and_memorize` | 做题全流程 |
| **常用辅助** | `check_problem`, `read_memory`, `write_memory`, `append_solution`, `find_problem_file` | 复习、查找、追加 |
| **偶尔使用** | `search_workspace_files`, `list_category_problems`, `get_daily_plan`, `get_hot_problems`, `web_search`, `update_user_memory` | 浏览、计划、搜索 |

#### 数据生成的关键技术挑战

1. **交互式 UI 需要改造**: `pick_problem` 和 `search_problem` 内部用 `arrow_select()` 做终端箭头键选择，headless 模式下必须绕过或 mock
2. **Sub-agent 调用**: `find_similar_problems`、`analyze_and_memorize`、`update_user_memory` 内部会再调 DeepSeek，复用主 agent 的 message history 做 KV cache 优化
3. **文件系统依赖**: 需要准备模拟工作区（已有一些题目文件 + 记忆文件），模拟"用户已经用了一段时间"的状态
4. **多轮交互**: 一条完整轨迹是 3-5 轮 user↔agent 交互，后续轮次的用户输入需要 DeepSeek-User 根据 agent 回复动态生成

#### SFT 数据生成方案（方案 A：真实 agent loop 录制）

**架构**: DeepSeek self-play —— 一个 DeepSeek 扮演用户，一个 DeepSeek 作为 agent backbone，tools 真实执行

```
步骤 1: 场景剧本生成（1 次 DeepSeek 调用）
  - 输入: 17 个 tool 描述 + 场景类型要求
  - 输出: ~500 条场景描述，确保覆盖所有 tool

步骤 2: 对每个场景跑 self-play loop
  for each 场景:
    a. DeepSeek-User 根据场景描述生成第一句话
    b. leetcode_agent.chat(第一句话) → 录制 messages
    c. 如果需要多轮:
       DeepSeek-User 看到 agent 回复，生成用户下一句
       leetcode_agent.chat(下一句) → 继续录制
    d. 重复到场景完成 (max ~5 轮)
    e. dump 完整 {messages, tools} 到 JSONL
```

**需要的改造**:
- leetcode_agent 的 headless runner（绕过 CLI REPL 和 arrow_select UI）
- 模拟工作区（已有题目 + 记忆文件，用户后续提供）
- DeepSeek-User prompt 设计（扮演不同类型的用户）

**成本**: ~500 条轨迹，每条 3000-5000 tokens，预计 ¥10-20，可以接受

### 2026-04-13: Headless agent 调试 & 轨迹生成 prototype

**Headless agent 实现**: `scripts/generate_trajectories.py`
- 绕过了三个 headless 阻塞点:
  1. `flush_stdin()`: `select.select + stdin.read` 在 pipe 环境下死循环 → patch 为 no-op
  2. `arrow_select()`: prompt_toolkit 终端交互 → patch 为自动随机选择
  3. `_call_model_once()`: Rich Live streaming 在非 TTY 下阻塞 → 改用 non-streaming API
- 模拟工作区: `sft_rl_training/workspace/`（25 道题 + 11 个记忆文件 + DB）

**Prototype 测试**: 成功生成 2 条完整轨迹
- 每条 14 messages，5 个 tool calls，约 1.5 分钟/条
- 覆盖完整链路: pick/search → start → find_similar → (用户做题) → read_solution → analyze_and_memorize
- 输出: `/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/test_2.jsonl`

**500 条轨迹生成计划**:

400 条结构化轨迹（严格 prompt 控制每一步）:

| 步骤 | 内容 | 分布 |
|------|------|------|
| Step 1: 用户开题 | 用 prompt 生成第一句话 | 1/3 "来一道题"（pick_problem）<br>1/3 "我要第xxx题"（start_problem）<br>1/3 "我想做xx方向的题"（search_problem） |
| Step 2: 用户做题 | DeepSeek-User 模拟写代码到文件 | 1/3 做完且正确<br>1/3 做完但有 bug<br>1/3 做一半放弃 |
| Step 3: 用户结束 | 用 prompt 让 DeepSeek 随机生成一句结束语 | 如"我做完了"、"大概就这样了"、"ok我只能做这么多了"、"你看看" |

100 条自由轨迹（待补充，思路如下）:

#### 思路 1: 从任务侧做多样性，而非步骤侧

400 条结构化轨迹用的是固定 pipeline（开题→做题→结束→总结），在每步里加随机分支。更好的做法是反过来——先用 LLM 大量生成不同的 **task description**，每个 task 自然会走出不同的 tool 调用路径。

不是"1/3 概率来道题、1/3 指定题号、1/3 指定方向"，而是生成几百个不同的用户画像 + 场景：
- "一个准备面试的人，连续做了 3 道 hard 都没过，心态有点崩"
- "一个只有 15 分钟的人，想快速复习昨天做错的那道"
- "一个刚学完 BFS 的新手，想找道 easy 练练手，但选了题之后发现太难想换"

每个画像自然会覆盖不同的 tool 组合。"复习昨天的题"自然走 `check_problem` + `read_memory`，"想换题"自然走两次 `pick/search`，不需要 hard code 这些分支。

#### 思路 2: Curriculum / 覆盖率驱动的采样

跑一批数据出来之后，统计每个 tool 被调用了多少次，然后针对性地补。不是一次性生成 500 条完事，而是迭代的：
1. 第一轮跑 200 条，发现 `web_search` 出现 0 次，`get_daily_plan` 出现 2 次
2. 针对这些低频 tool 设计场景再跑 100 条
3. 再统计，再补

已有 `fail_cases` 分流机制，加一个 tool coverage 的统计和自动补采就是自然延伸。

#### 思路 3: 环境扰动 / rejection 场景

当前方案最缺的一块。真实使用中很多轨迹不是 happy path——tool 调用失败、API 超时、用户给了不合法输入、文件不存在。这些场景对 RL 阶段特别重要（模型需要学会从错误中恢复），但当前的真实环境太"顺利"了。需要人为注入错误场景。

---

## TODO

- [x] Phase 0: 环境搭建
  - [x] 克隆 conda env qwen → qwen_RL
  - [x] 下载 Qwen3-1.7B 到 cache path
  - [x] 验证 Qwen3 chat template 支持 tool calling
  - [x] 验证 DeepSeek API 可用
  - [ ] 在 A40 上验证 veRL + Agent-R1 能正常启动
- [ ] Phase 1: SFT
  - [x] SFT 训练代码 (LoRA + loss masking)
  - [x] Condor 提交脚本
  - [x] 合成数据验证训练流程跑通 (Exp-001)
  - [x] Headless agent 实现 + prototype 验证 (2 条轨迹成功)
  - [x] 用 DeepSeek + leetcode_agent 真实 loop 生成 511 条轨迹 (479 结构化 + 32 自由)
  - [x] 用真实轨迹数据正式 SFT 训练 (Exp-002)
  - [x] 验证模型能输出正确格式的 tool call (Exp-005 实机测试)
  - [x] 补充 clarify（反问）数据 — Exp-006：66 条，通过 DeepSeek self-play 造 "模糊开场 → 反问 → 具体需求" 的多轮轨迹
  - [x] 补充 no-tool-call 数据 — Exp-007：117 条，Qwen3-1.7B self-distillation，6 类（闲聊/状态分享/意图模糊/做题中讨论/抱怨/偏离主题）
  - [x] 基于合并数据 (653 条) 重训 SFT — Exp-009：5 epochs on c22, ep3 (checkpoint-60) 为 eval 最优，eval_loss 0.532 (vs Exp-002 的 0.577)
  - [x] 构建 behavior benchmark + 评分脚本 — Exp-008：41 条分层抽样，`run_benchmark.py` rule-based 按桶打分
  - [x] 三方对比评估 — Exp-010：Base 70.7% / Exp-002 82.9% / **Exp-009 95.1%**；Exp-002 的 should_clarify 0/5 被 Exp-009 修到 5/5
- [ ] Phase 2: RL
  - [x] ~~实现 LeetCodeToolEnv (继承 Agent-R1 的 AgentEnv)~~ → 改用 TRL `environment_factory`，已实现 `LeetCodeEnvironment`（Exp-004, Exp-011 debug）
  - [x] Pipeline 层 debug + dry-run 跑通（Exp-011：job 9979 成功训 2 steps，tool call / sub-agent / model 保存全通）
  - [ ] Rubric-based reward function（用户在造）
  - [ ] RL 训练 prompt 数据集（用户在造，需含 clarify + no_tool 模式）
  - [ ] 正式 GRPO 训练
  - [ ] （later）记忆相关 3 个 tool 单独 single-turn RL 训练
- [ ] Phase 3: 评估
  - [ ] 构建 eval 场景集
  - [ ] 对比 DeepSeek / SFT-only / SFT+RL
