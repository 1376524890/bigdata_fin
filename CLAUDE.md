# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是一个基于沪深300指数的MIDAS混频模型两阶段实证分析研究项目。项目构建了"宏观基本面—理论预期收益率—实际市场表现—预期偏离—偏离成因识别"的分析框架。

## 主要命令

### 运行实验流程

```bash
# 进入实验流程目录
cd experiment_pipeline

# 运行完整流程（阶段1-5）
python main.py

# 运行指定阶段
python main.py --stage 4    # 只运行MIDAS实验（核心）
python main.py --from 3     # 从阶段3开始运行

# 单独运行某个阶段的脚本
python 01_data_fetch.py
python 02_data_integration.py
python 03_data_analysis.py
python 04_midas_experiment.py
python 05_generate_plots.py

# 查看阶段列表说明
python main.py --list
```

### 编译论文

```bash
cd latex_paper
xelatex main.tex
```

## 架构结构

### 实验流程（experiment_pipeline/）

5个阶段的Python脚本，按顺序执行：

1. **数据获取** (`01_data_fetch.py`) - 从akshare/yfinance获取指数、宏观、情绪数据
2. **数据整合** (`02_data_integration.py`) - 合并数据，计算收益率、异常收益、Amihud指标
3. **数据分析** (`03_data_analysis.py`) - 描述统计、相关性分析、训练/测试集划分
4. **MIDAS实验** (`04_midas_experiment.py`) - 核心实验，两阶段回归分析
5. **生成图表** (`05_generate_plots.py`) - 生成论文所需的PNG图表

### 关键目录

- `real_data/` - 原始数据存储，按类别分子目录（指数、波动率、情绪、宏观、汇率等）
- `experiment_results/` - 实验结果输出（CSV文件和figures/图表）
- `latex_paper/` - 论文LaTeX源文件（chapter1-6.tex, main.tex）
- `real_data_complete.csv` - 整合后的完整分析数据集

### 核心模型变量

**第一阶段MIDAS回归**：
- 月度宏观变量：cpi, ppi, m2_growth, epu, usd_cny
- 预测目标：未来5日/60日累计收益率 (R_5d, R_60d)

**第二阶段分组嵌套回归**：
- 被解释变量：异常收益绝对值 (AbsAR_5d/AbsAR_60d)
- 模型I-IV分层加入：sentiment_zscore, ivix, north_flow, margin_balance, amihud, momentum_20d, intraday_range, epu, fx_vol

## 依赖

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels akshare yfinance
```

## 运行顺序要求

阶段2-5依赖阶段1的数据获取，需按顺序执行或确保`real_data/`目录已有数据文件。