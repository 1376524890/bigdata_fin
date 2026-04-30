# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

本项目研究股票市场短期不稳定性的动态结构与宏观变量的增量解释作用，基于沪深300指数（2015-2026年样本）。项目构建了"HAR基准模型—HARX宏观扩展—稳健性检验"的分析框架，最终产出LaTeX论文。

**当前版本（HAR/HARX）**：基于HAR回归的短期不稳定性基准模型，所有代码和结果集中在 `har/` 目录。

**早期版本（MIDAS，已归档）**：MIDAS混频模型两阶段实证分析，代码已归档至 `archive/midas_old_scripts/`。

## 主要命令

### 运行 HAR/HARX 核心实验（当前版本）

```bash
cd har/scripts

# 完整 HARX 实验（最完整版，包含所有模型和诊断）
python harx_instability_experiment.py

# HARX 论文模型（精简版，直接生成论文所需结果）
python 07_stage1_harx_paper.py

# 收尾稳健性检验（非重叠窗口 + 宏观单变量进入）
python harx_final_checks.py

# 替代模型探索（可选）
python 06_stage1_alternative_models.py    # Ridge/ElasticNet/PCR 对比
python 06_stage1_alt_targets.py           # 替代目标变量探索
python 07_stage1_restructured_models.py   # 重构模型探索
```

### 编译论文

```bash
cd latex_paper
xelatex main.tex
```

## 架构结构

### HAR/HARX 实验（har/）

```
har/
├── scripts/                              # HAR/HARX 核心实验脚本
│   ├── harx_instability_experiment.py     # 最完整实验：HAR基准 + HARX扩展 + HAC推断 + 分样本稳定性 + 综合诊断
│   ├── 07_stage1_harx_paper.py           # 论文版精简模型：主/辅助目标变量双轨回归
│   ├── harx_final_checks.py              # 收尾检验：非重叠窗口稳健性 + 宏观变量逐个进入
│   ├── 06_stage1_alternative_models.py   # Ridge/ElasticNet/PCR 替代模型对比
│   ├── 06_stage1_alt_targets.py          # 多种目标变量探索（absret/logrv/signbalance/upratio 家族）
│   ├── 07_stage1_restructured_models.py  # 重构模型（直接OLS替代MIDAS加权）
│   └── fix_vif.py                        # VIF多重共线性诊断修复工具
└── results/                              # 实验结果输出
    ├── harx_instability_full/            # 完整HARX实验输出（核心结果）
    ├── harx_final_checks/                # 收尾稳健性检验输出
    ├── stage1_harx_paper/                # 论文版模型输出
    ├── stage1_alt_targets/               # 替代目标变量探索输出
    ├── stage1_restructured_experiment/   # 重构模型输出
    ├── stage1_alternative_experiment/    # 替代模型对比输出
    └── full_data_with_predictions.csv    # 含预测值的完整数据
```

### 关键目录

- `har/` - **HAR/HARX实验（当前版本，所有新版内容）**
- `real_data/` - 原始数据存储，按类别分子目录（指数、波动率、情绪、宏观、汇率等）
- `real_data_complete.csv` - 整合后的完整分析数据集
- `latex_paper/` - 论文LaTeX源文件（chapter1-6.tex, main.tex, references.bib, figures/）
- `archive/` - 历史版本归档
  - `midas_old_scripts/` - 旧版MIDAS基础流程脚本（01-05）+ 旧版MIDAS脚本（v1-v4）
  - `midas_papers/` - 旧版MIDAS论文PDF
  - `midas_reports/` - 旧版MIDAS Markdown报告
  - `midas_results/` - 旧版MIDAS实验输出和图表
  - `midas_text_outputs/` - 旧版文本输出
  - `har_intermediate/` - HAR中间版本论文
  - `else/` - 早期研究笔记

## 依赖

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels akshare yfinance
```
