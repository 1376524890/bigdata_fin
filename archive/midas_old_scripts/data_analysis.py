#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据描述统计与分析脚本
用于生成论文第二部分所需的描述统计、缺失值分析、相关性矩阵等
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# 读取主数据文件
df = pd.read_csv('/home/marktom/bigdata-fin/real_data_complete.csv')

# 转换日期格式
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("=" * 80)
print("数据概览")
print("=" * 80)
print(f"样本时间区间: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
print(f"总观测数: {len(df)}")
print(f"时间跨度: {(df['date'].max() - df['date'].min()).days} 天")
print(f"\n数据列: {list(df.columns)}")

# 构建未来5日和60日收益率
print("\n" + "=" * 80)
print("构造未来累计收益率")
print("=" * 80)

# 计算日对数收益率
df['daily_return'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))

# 计算未来5日累计收益率
df['future_return_5d'] = df['daily_return'].shift(-1).rolling(window=5).sum()
for i in range(1, 6):
    df['future_return_5d'] = df['future_return_5d'] + df['daily_return'].shift(-i)
df['future_return_5d'] = df['daily_return'].shift(-1).rolling(window=5).sum().shift(-4)

# 重新计算未来累计收益率（使用对数收益率求和）
df['future_return_5d'] = np.log(df['hs300_close'].shift(-5) / df['hs300_close'])
df['future_return_60d'] = np.log(df['hs300_close'].shift(-60) / df['hs300_close'])

# 去除无法计算未来收益的尾部观测
df_analysis = df[df['future_return_5d'].notna() & df['future_return_60d'].notna()].copy()

print(f"用于分析的样本量 (去除尾部缺失): {len(df_analysis)}")
print(f"未来5日收益率 - 均值: {df_analysis['future_return_5d'].mean():.6f}, 标准差: {df_analysis['future_return_5d'].std():.6f}")
print(f"未来60日收益率 - 均值: {df_analysis['future_return_60d'].mean():.6f}, 标准差: {df_analysis['future_return_60d'].std():.6f}")

# 定义变量分组
print("\n" + "=" * 80)
print("变量分组定义")
print("=" * 80)

# 被解释变量基础字段
dependent_vars = {
    'hs300_close': '沪深300收盘价',
    'daily_return': '日对数收益率',
    'future_return_5d': '未来5日累计收益率',
    'future_return_60d': '未来60日累计收益率'
}

# 宏观变量 (月度)
macro_vars = {
    'cpi': 'CPI同比增速',
    'ppi': 'PPI同比增速',
    'm2_growth': 'M2同比增速',
    'epu': '经济政策不确定性指数',
    'usd_cny': '美元兑人民币汇率',
    'gdp_growth': 'GDP同比增速(季度)'
}

# 市场状态变量 (日度)
market_vars = {
    'sentiment_zscore': '情绪标准分',
    'ivix': '隐含波动率指数',
    'north_flow': '北向资金净流入',
    'margin_balance': '融资融券余额',
    'amihud': 'Amihud非流动性指标',
    'momentum_20d': '20日动量',
    'intraday_range': '日内振幅'
}

print("\n【被解释变量基础字段】")
for k, v in dependent_vars.items():
    if k in df_analysis.columns:
        print(f"  {k}: {v}")

print("\n【宏观变量 (月度)】")
for k, v in macro_vars.items():
    if k in df_analysis.columns:
        print(f"  {k}: {v}")

print("\n【市场状态变量 (日度)】")
for k, v in market_vars.items():
    if k in df_analysis.columns:
        print(f"  {k}: {v}")

# 描述性统计
print("\n" + "=" * 80)
print("描述性统计")
print("=" * 80)

# 选择核心变量进行描述统计
analysis_vars = ['future_return_5d', 'future_return_60d', 'daily_return',
                 'cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny',
                 'sentiment_zscore', 'ivix', 'north_flow', 'margin_balance',
                 'amihud', 'momentum_20d', 'intraday_range']

available_vars = [v for v in analysis_vars if v in df_analysis.columns]
desc_stats = df_analysis[available_vars].describe().T
desc_stats['median'] = df_analysis[available_vars].median()
desc_stats['skewness'] = df_analysis[available_vars].skew()
desc_stats['kurtosis'] = df_analysis[available_vars].kurtosis()

# 重新排列列顺序
desc_stats = desc_stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skewness', 'kurtosis']]

print("\n核心变量描述性统计:")
print(desc_stats.round(6).to_string())

# 缺失值分析
print("\n" + "=" * 80)
print("缺失值统计")
print("=" * 80)

all_vars = ['date', 'hs300_close', 'daily_return', 'future_return_5d', 'future_return_60d'] + \
           list(macro_vars.keys()) + list(market_vars.keys())
available_all = [v for v in all_vars if v in df_analysis.columns]

missing_stats = pd.DataFrame({
    '变量名': available_all,
    '缺失数': [df_analysis[v].isna().sum() for v in available_all],
    '缺失率(%)': [df_analysis[v].isna().sum() / len(df_analysis) * 100 for v in available_all],
    '有效观测数': [df_analysis[v].notna().sum() for v in available_all]
})

print(missing_stats.to_string(index=False))

# 相关性分析
print("\n" + "=" * 80)
print("相关性矩阵")
print("=" * 80)

# 宏观变量之间的相关性
print("\n【宏观变量间相关性】")
macro_available = [v for v in macro_vars.keys() if v in df_analysis.columns]
if len(macro_available) > 1:
    macro_corr = df_analysis[macro_available].corr()
    print(macro_corr.round(4).to_string())

# 市场状态变量之间的相关性
print("\n【市场状态变量间相关性】")
market_available = [v for v in market_vars.keys() if v in df_analysis.columns]
if len(market_available) > 1:
    market_corr = df_analysis[market_available].corr()
    print(market_corr.round(4).to_string())

# 宏观变量与未来收益的相关性
print("\n【宏观变量与未来收益相关性】")
predictive_vars = ['future_return_5d', 'future_return_60d'] + macro_available
pred_corr = df_analysis[predictive_vars].corr()[['future_return_5d', 'future_return_60d']]
print(pred_corr.round(4).to_string())

# 缩尾处理前的极端值分析
print("\n" + "=" * 80)
print("极端值分析 (缩尾处理前)")
print("=" * 80)

extreme_vars = ['north_flow', 'margin_balance', 'ivix', 'amihud', 'intraday_range']
extreme_available = [v for v in extreme_vars if v in df_analysis.columns]

for var in extreme_available:
    p1 = df_analysis[var].quantile(0.01)
    p99 = df_analysis[var].quantile(0.99)
    below_p1 = (df_analysis[var] < p1).sum()
    above_p99 = (df_analysis[var] > p99).sum()
    print(f"{var}: 1%分位数={p1:.4f}, 99%分位数={p99:.4f}, 低于1%={below_p1}个, 高于99%={above_p99}个")

# 样本时间分布
print("\n" + "=" * 80)
print("样本时间分布")
print("=" * 80)

df_analysis['year'] = df_analysis['date'].dt.year
year_counts = df_analysis['year'].value_counts().sort_index()
print("\n各年份观测数:")
print(year_counts.to_string())

# 训练集/测试集划分 (60%训练, 40%测试)
print("\n" + "=" * 80)
print("训练集/测试集划分")
print("=" * 80)

train_cutoff = int(len(df_analysis) * 0.6)
train_df = df_analysis.iloc[:train_cutoff]
test_df = df_analysis.iloc[train_cutoff:]

print(f"总样本量: {len(df_analysis)}")
print(f"训练集: {len(train_df)} ({len(train_df)/len(df_analysis)*100:.1f}%)")
print(f"  时间区间: {train_df['date'].min().strftime('%Y-%m-%d')} 至 {train_df['date'].max().strftime('%Y-%m-%d')}")
print(f"测试集: {len(test_df)} ({len(test_df)/len(df_analysis)*100:.1f}%)")
print(f"  时间区间: {test_df['date'].min().strftime('%Y-%m-%d')} 至 {test_df['date'].max().strftime('%Y-%m-%d')}")

# 保存详细结果到文件
print("\n" + "=" * 80)
print("保存分析结果")
print("=" * 80)

# 保存描述统计
desc_stats.to_csv('/home/marktom/bigdata-fin/desc_stats.csv')
print("描述统计已保存至: desc_stats.csv")

# 保存缺失值统计
missing_stats.to_csv('/home/marktom/bigdata-fin/missing_stats.csv', index=False)
print("缺失值统计已保存至: missing_stats.csv")

# 保存相关性矩阵
if len(macro_available) > 1:
    macro_corr.to_csv('/home/marktom/bigdata-fin/macro_correlation.csv')
    print("宏观变量相关性矩阵已保存至: macro_correlation.csv")

if len(market_available) > 1:
    market_corr.to_csv('/home/marktom/bigdata-fin/market_correlation.csv')
    print("市场状态变量相关性矩阵已保存至: market_correlation.csv")

pred_corr.to_csv('/home/marktom/bigdata-fin/predictive_correlation.csv')
print("预测相关性矩阵已保存至: predictive_correlation.csv")

# 生成汇总报告
summary_report = f"""
数据摘要报告
============

样本时间区间: {df_analysis['date'].min().strftime('%Y年%m月%d日')} 至 {df_analysis['date'].max().strftime('%Y年%m月%d日')}
总观测数: {len(df_analysis)}

关键变量统计:
- 沪深300收盘价: 均值={df_analysis['hs300_close'].mean():.2f}, 标准差={df_analysis['hs300_close'].std():.2f}
- 日对数收益率: 均值={df_analysis['daily_return'].mean():.6f}, 标准差={df_analysis['daily_return'].std():.6f}
- 未来5日累计收益率: 均值={df_analysis['future_return_5d'].mean():.6f}, 标准差={df_analysis['future_return_5d'].std():.6f}
- 未来60日累计收益率: 均值={df_analysis['future_return_60d'].mean():.6f}, 标准差={df_analysis['future_return_60d'].std():.6f}

宏观变量:
- CPI: {df_analysis['cpi'].mean():.2f}±{df_analysis['cpi'].std():.2f}
- PPI: {df_analysis['ppi'].mean():.2f}±{df_analysis['ppi'].std():.2f}
- M2增速: {df_analysis['m2_growth'].mean():.2f}±{df_analysis['m2_growth'].std():.2f}
- EPU指数: {df_analysis['epu'].mean():.2f}±{df_analysis['epu'].std():.2f}

市场状态变量:
- 情绪标准分: {df_analysis['sentiment_zscore'].mean():.4f}±{df_analysis['sentiment_zscore'].std():.4f}
- iVIX: {df_analysis['ivix'].mean():.2f}±{df_analysis['ivix'].std():.2f}
- Amihud非流动性: {df_analysis['amihud'].mean():.6f}±{df_analysis['amihud'].std():.6f}
"""

with open('/home/marktom/bigdata-fin/data_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("\n数据摘要报告已保存至: data_summary_report.txt")
print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
