# HAR/HARX 短期不稳定性实验

本目录包含基于HAR回归的股票市场短期不稳定性研究的所有代码和结果。

## 目录结构

```
har/
├── README.md                             # 本说明文件
├── scripts/                              # HAR/HARX 核心实验脚本
│   ├── harx_instability_experiment.py     # 完整HARX实验（最完整版）
│   ├── 07_stage1_harx_paper.py           # 论文版精简模型
│   ├── harx_final_checks.py              # 收尾稳健性检验
│   ├── 06_stage1_alternative_models.py   # Ridge/ElasticNet/PCR 替代模型
│   ├── 06_stage1_alt_targets.py          # 替代目标变量探索
│   ├── 07_stage1_restructured_models.py  # 重构模型探索
│   └── fix_vif.py                        # VIF多重共线性诊断
└── results/                              # 实验结果输出
    ├── harx_instability_full/            # 完整HARX实验输出（核心结果）
    ├── harx_final_checks/                # 收尾稳健性检验输出
    ├── stage1_harx_paper/                # 论文版模型输出
    ├── stage1_alt_targets/               # 替代目标变量探索输出
    ├── stage1_restructured_experiment/   # 重构模型输出
    ├── stage1_alternative_experiment/    # 替代模型对比输出
    └── full_data_with_predictions.csv    # 含预测值的完整数据
```

## 运行命令

```bash
cd /home/marktom/bigdata-fin/har/scripts

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

## 数据依赖

所有脚本读取项目根目录的 `real_data_complete.csv`，输出写入 `har/results/` 对应子目录。

## 论文

论文LaTeX源文件位于 `latex_paper/`，编译命令：

```bash
cd /home/marktom/bigdata-fin/latex_paper
xelatex main.tex
```
