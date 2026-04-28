# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

本项目研究股票市场短期不稳定性的动态结构与宏观变量的增量解释作用，基于沪深300指数（2015-2026年样本）。项目构建了"HAR基准模型—HARX宏观扩展—稳健性检验"的分析框架，最终产出LaTeX论文。

**当前版本（HAR/HARX）**：基于HAR回归的短期不稳定性基准模型，核心实验脚本为 `harx_instability_experiment.py`、`harx_final_checks.py` 和 `07_stage1_harx_paper.py`。

**早期版本（MIDAS，已归档）**：MIDAS混频模型两阶段实证分析，代码已归档至 `archive/midas_old_scripts/`。

## 主要命令

### 运行基础数据流程（阶段1-5）

```bash
cd experiment_pipeline

# 运行完整基础流程
python main.py

# 运行指定阶段
python main.py --stage 4
python main.py --from 3

# 查看阶段列表
python main.py --list
```

### 运行 HAR/HARX 核心实验（当前版本）

```bash
cd experiment_pipeline

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

### 实验流程（experiment_pipeline/）

**基础流程（01-05）**：
1. **数据获取** (`01_data_fetch.py`) - 从akshare/yfinance获取指数、宏观、情绪数据
2. **数据整合** (`02_data_integration.py`) - 合并数据，计算收益率、异常收益、Amihud指标
3. **数据分析** (`03_data_analysis.py`) - 描述统计、相关性分析、训练/测试集划分
4. **MIDAS实验** (`04_midas_experiment.py`) - 原始MIDAS两阶段回归分析
5. **生成图表** (`05_generate_plots.py`) - 生成论文所需的PNG图表

**HAR/HARX 核心实验**（当前版本）：
- `harx_instability_experiment.py` - **最完整实验**：HAR基准 + HARX扩展 + HAC推断 + 分样本稳定性 + 综合诊断
- `07_stage1_harx_paper.py` - 论文版精简模型：主/辅助目标变量双轨回归
- `harx_final_checks.py` - 收尾检验：非重叠窗口稳健性 + 宏观变量逐个进入

**探索性脚本**：
- `06_stage1_alternative_models.py` - Ridge/ElasticNet/PCR 替代模型对比
- `06_stage1_alt_targets.py` - 多种目标变量探索（absret/logrv/signbalance/upratio 家族）
- `07_stage1_restructured_models.py` - 重构模型（直接OLS替代MIDAS加权）
- `fix_vif.py` - VIF多重共线性诊断修复工具

### 关键目录

- `real_data/` - 原始数据存储，按类别分子目录（指数、波动率、情绪、宏观、汇率等）
- `real_data_complete.csv` - 整合后的完整分析数据集
- `experiment_results/` - 实验结果输出
  - `harx_instability_full/` - 完整HARX实验输出（**最新核心结果**）
  - `harx_final_checks/` - 收尾稳健性检验输出
  - `figures/` - 图表输出
  - `full_data_with_predictions.csv` - 含预测值的完整数据
- `latex_paper/` - 论文LaTeX源文件（chapter1-6.tex, main.tex, references.bib, figures/）
- `archive/` - 历史版本归档
  - `midas_old_scripts/` - 旧版MIDAS脚本（v1-v4多版本迭代）
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

## 运行顺序要求

阶段2-5依赖阶段1的数据获取，需按顺序执行或确保`real_data/`目录已有数据文件。