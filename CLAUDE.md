# CLAUDE.md

## 项目概述

LC-Agent: 用 SFT+RL 训练 Qwen3-1.7B 替换 leetcode_agent 的 DeepSeek API backbone。

## 工作规范

### 实验记录

每次跑实验（训练、评估、数据生成等）都必须在 `sft_rl_training/experiment_log.md` 中记录，包括：
- 日期、实验目的
- 代码路径、结果路径
- 具体配置（模型、超参、硬件）
- 结果和分析

### 实验输出路径

所有实验结果输出到 `/shared/rsaas/qiqianf2/lc_agent_experiments/`，不要放在 home 目录下。

### 模型缓存路径

HuggingFace 模型下载到 `/shared/rsaas/qiqianf2/hf_models/`。

### 环境

- 训练用 conda env `qwen_RL`（不要动 `qwen` 原环境）
- 任务通过 condor 提交，优先选 A40 机器（vision-c21/c22/c23）

### Condor 提交

参考 `sft_rl_training/scripts/sft.sub`，注意：
- 用 `CUDA_VISIBLE_DEVICES` 检测 GPU 数量，不要用 `nvidia-smi -L`
- Requirements 限制到有 A40 的机器
