#!/usr/bin/env python3
"""
主运行脚本
按顺序执行所有实验阶段
用法:
    python main.py              # 运行所有阶段
    python main.py --stage 4    # 只运行第4阶段
    python main.py --from 3     # 从第3阶段开始运行
"""

import argparse
import sys
import os

# 将当前目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_stage(stage_num):
    """运行指定阶段"""
    if stage_num == 1:
        print("\n" + "="*70)
        print("运行阶段1: 数据获取")
        print("="*70)
        import importlib
        stage1 = importlib.import_module('01_data_fetch')
        stage1.main()

    elif stage_num == 2:
        print("\n" + "="*70)
        print("运行阶段2: 数据整合")
        print("="*70)
        import importlib
        stage2 = importlib.import_module('02_data_integration')
        stage2.main()

    elif stage_num == 3:
        print("\n" + "="*70)
        print("运行阶段3: 数据分析")
        print("="*70)
        import importlib
        stage3 = importlib.import_module('03_data_analysis')
        stage3.main()

    elif stage_num == 4:
        print("\n" + "="*70)
        print("运行阶段4: MIDAS实验")
        print("="*70)
        import importlib
        stage4 = importlib.import_module('04_midas_experiment')
        stage4.main()

    elif stage_num == 5:
        print("\n" + "="*70)
        print("运行阶段5: 生成图表")
        print("="*70)
        import importlib
        stage5 = importlib.import_module('05_generate_plots')
        stage5.main()

    else:
        print(f"错误: 未知阶段 {stage_num}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='运行MIDAS实验流程')
    parser.add_argument('--stage', type=int, help='只运行指定阶段 (1-5)')
    parser.add_argument('--from', type=int, dest='from_stage', help='从指定阶段开始运行 (1-5)')
    parser.add_argument('--list', action='store_true', help='列出所有阶段')

    args = parser.parse_args()

    if args.list:
        print("""
实验流程阶段:
  阶段1: 数据获取 (01_data_fetch.py)
         - 获取指数数据 (沪深300、上证指数等)
         - 获取宏观数据 (GDP、CPI、PPI、M2、EPU等)
         - 获取市场情绪数据 (iVIX、北向资金、融资融券)
         - 生成情绪指标

  阶段2: 数据整合 (02_data_integration.py)
         - 合并各类数据
         - 计算衍生指标 (收益率、异常收益、Amihud等)
         - 前向填充宏观数据
         - 生成完整数据集

  阶段3: 数据分析 (03_data_analysis.py)
         - 构造未来收益率 (5日、60日)
         - 描述性统计
         - 缺失值分析
         - 相关性分析

  阶段4: MIDAS实验 (04_midas_experiment.py) - 核心
         - 第一阶段: MIDAS回归预测收益率
           * 单变量MIDAS模型
           * VIF检验
           * 多变量MIDAS模型
         - 第二阶段: 分组嵌套回归
           * 模型I: 情绪与风险感知
           * 模型II: 资金与杠杆
           * 模型III: 流动性指标
           * LASSO筛选
           * 联合显著性检验

  阶段5: 生成图表 (05_generate_plots.py)
         - 真实值vs预测值对比图
         - 第一阶段残差诊断图
         - 第二阶段残差诊断图
         - LASSO交叉验证图
         - LASSO系数路径图
         - 分组嵌套回归边际解释力图

用法示例:
  python main.py              # 运行所有阶段
  python main.py --stage 4    # 只运行第4阶段 (MIDAS实验)
  python main.py --from 3     # 从第3阶段开始运行
  python main.py --list       # 列出所有阶段
        """)
        return

    if args.stage:
        # 只运行指定阶段
        if 1 <= args.stage <= 5:
            run_stage(args.stage)
        else:
            print("错误: 阶段必须在1-5之间")
            sys.exit(1)

    elif args.from_stage:
        # 从指定阶段开始运行
        if 1 <= args.from_stage <= 5:
            for stage in range(args.from_stage, 6):
                run_stage(stage)
        else:
            print("错误: 阶段必须在1-5之间")
            sys.exit(1)

    else:
        # 运行所有阶段
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║              MIDAS模型两阶段实证分析 - 完整流程                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        for stage in range(1, 6):
            run_stage(stage)

        print("\n" + "="*70)
        print("所有阶段运行完成！")
        print("="*70)
        print("""
输出文件:
  - 数据文件: /home/marktom/bigdata-fin/real_data_complete.csv
  - 实验结果: /home/marktom/bigdata-fin/experiment_results/
  - 图表文件: /home/marktom/bigdata-fin/experiment_results/figures/
        """)


if __name__ == '__main__':
    main()
