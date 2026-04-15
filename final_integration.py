#!/usr/bin/env python3
"""
最终数据整合脚本 - 合并所有数据
"""

import pandas as pd
import numpy as np
import os
import warnings
import re

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
FINAL_DIR = '/home/marktom/bigdata-fin'

print("=" * 60)
print("最终数据整合脚本")
print("=" * 60)

# ============================================================================
# 1. 加载基础数据
# ============================================================================
print("\n【1】加载基础数据...")

# 沪深300
hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
hs300['date'] = pd.to_datetime(hs300['date'])
hs300 = hs300.rename(columns={'close': 'hs300_close', 'volume': 'hs300_volume'})

# 上证指数
sh = pd.read_csv(f'{OUTPUT_DIR}/sh_index_daily.csv')
sh['date'] = pd.to_datetime(sh['date'])
sh = sh.rename(columns={'close': 'sh_close', 'volume': 'sh_volume'})

# 以沪深300为基准
daily_data = hs300[['date', 'hs300_close', 'hs300_volume']].copy()
daily_data = daily_data.merge(sh[['date', 'sh_close']], on='date', how='left')

print(f"   基础数据: {len(daily_data)}条")

# ============================================================================
# 2. 合并市场情绪数据
# ============================================================================
print("\n【2】合并市场情绪数据...")

# 加载情绪数据
sentiment = pd.read_csv(f'{OUTPUT_DIR}/market_sentiment_history.csv')
sentiment['date'] = pd.to_datetime(sentiment['date'])

# 选择关键情绪指标
sentiment_cols = ['date', 'return', 'volatility_20d', 'volatility_60d',
                  'momentum_5d', 'momentum_10d', 'momentum_20d',
                  'volume_ratio_5d', 'volume_ratio_20d', 'intraday_range',
                  'sentiment_zscore', 'sentiment_level']

sentiment_selected = sentiment[sentiment_cols].copy()

# 合并
daily_data = daily_data.merge(sentiment_selected, on='date', how='left')
print(f"   合并情绪数据后: {len(daily_data)}条")

# ============================================================================
# 3. 合并中国波指
# ============================================================================
print("\n【3】合并中国波指...")

ivix = pd.read_csv(f'{OUTPUT_DIR}/ivix_50etf.csv')
ivix['date'] = pd.to_datetime(ivix['date'])
ivix = ivix.rename(columns={'close': 'ivix'})

daily_data = daily_data.merge(ivix[['date', 'ivix']], on='date', how='left')
print(f"   合并ivix后: {len(daily_data)}条")

# ============================================================================
# 4. 合并北向资金
# ============================================================================
print("\n【4】合并北向资金...")

north = pd.read_csv(f'{OUTPUT_DIR}/north_money_hist.csv')
north['date'] = pd.to_datetime(north['日期'])
north['north_flow'] = north['当日成交净买额']

daily_data = daily_data.merge(north[['date', 'north_flow']], on='date', how='left')
print(f"   合并北向资金后: {len(daily_data)}条")

# ============================================================================
# 5. 合并融资融券
# ============================================================================
print("\n【5】合并融资融券...")

margin = pd.read_csv(f'{OUTPUT_DIR}/margin_account.csv')
margin['date'] = pd.to_datetime(margin['日期'])
margin['margin_balance'] = margin['融资余额']

daily_data = daily_data.merge(margin[['date', 'margin_balance']], on='date', how='left')
print(f"   合并融资融券后: {len(daily_data)}条")

# ============================================================================
# 6. 合并汇率
# ============================================================================
print("\n【6】合并汇率...")

boc_rate = pd.read_csv(f'{OUTPUT_DIR}/boc_rate.csv')
boc_rate['date'] = pd.to_datetime(boc_rate['日期'])
boc_rate['usd_cny'] = boc_rate['美元']

daily_data = daily_data.merge(boc_rate[['date', 'usd_cny']], on='date', how='left')
print(f"   合并汇率后: {len(daily_data)}条")

# ============================================================================
# 7. 合并宏观数据
# ============================================================================
print("\n【7】合并宏观数据...")

# GDP
gdp = pd.read_csv(f'{OUTPUT_DIR}/gdp.csv')
def parse_quarter(s):
    match = re.search(r'(\d{4})年第(\d)', str(s))
    if match:
        year, q = int(match.group(1)), int(match.group(2))
        return pd.Timestamp(f"{year}-{(q-1)*3+1:02d}-01")
    return None
gdp['date'] = gdp['季度'].apply(parse_quarter)
gdp['gdp_growth'] = pd.to_numeric(gdp['国内生产总值-同比增长'], errors='coerce')
gdp_data = gdp[['date', 'gdp_growth']].dropna()

# CPI
cpi = pd.read_csv(f'{OUTPUT_DIR}/cpi.csv')
cpi['date'] = pd.to_datetime(cpi['日期'])
cpi['cpi'] = pd.to_numeric(cpi['今值'], errors='coerce')
cpi_data = cpi[['date', 'cpi']].dropna()

# PPI
ppi = pd.read_csv(f'{OUTPUT_DIR}/ppi.csv')
ppi['date'] = pd.to_datetime(ppi['日期'])
ppi['ppi'] = pd.to_numeric(ppi['今值'], errors='coerce')
ppi_data = ppi[['date', 'ppi']].dropna()

# M2
m2 = pd.read_csv(f'{OUTPUT_DIR}/m2.csv')
m2['date'] = pd.to_datetime(m2['日期'])
m2['m2_growth'] = pd.to_numeric(m2['今值'], errors='coerce')
m2_data = m2[['date', 'm2_growth']].dropna()

# EPU
epu = pd.read_csv(f'{OUTPUT_DIR}/epu_index.csv')
epu['date'] = pd.to_datetime(epu['year'].astype(str) + '-' + epu['month'].astype(str) + '-01')
epu['epu'] = epu['China_Policy_Index']
epu_data = epu[['date', 'epu']].dropna()

# 前向填充宏观数据
daily_data = daily_data.set_index('date')

def fill_macro(daily_df, macro_df, col_name):
    result = pd.Series(index=daily_df.index, dtype=float)
    for date in daily_df.index:
        # date是Timestamp类型
        available = macro_df[macro_df['date'] <= pd.Timestamp(date)]
        if len(available) > 0:
            result[date] = available[col_name].iloc[-1]
    return result

gdp_data_reset = gdp_data.reset_index(drop=True)
cpi_data_reset = cpi_data.reset_index(drop=True)
ppi_data_reset = ppi_data.reset_index(drop=True)
m2_data_reset = m2_data.reset_index(drop=True)
epu_data_reset = epu_data.reset_index(drop=True)

daily_data['gdp_growth'] = fill_macro(daily_data.reset_index(), gdp_data_reset, 'gdp_growth').values
daily_data['cpi'] = fill_macro(daily_data.reset_index(), cpi_data_reset, 'cpi').values
daily_data['ppi'] = fill_macro(daily_data.reset_index(), ppi_data_reset, 'ppi').values
daily_data['m2_growth'] = fill_macro(daily_data.reset_index(), m2_data_reset, 'm2_growth').values
daily_data['epu'] = fill_macro(daily_data.reset_index(), epu_data_reset, 'epu').values

daily_data = daily_data.reset_index()
print(f"   合并宏观数据后: {len(daily_data)}条")

# ============================================================================
# 8. 计算异常收益率
# ============================================================================
print("\n【8】计算异常收益率...")

# 市场收益率
daily_data['market_return'] = daily_data['hs300_close'].pct_change()

# 异常收益率（相对于上证指数）
daily_data['sh_return'] = daily_data['sh_close'].pct_change()
daily_data['abnormal_return'] = daily_data['market_return'] - daily_data['sh_return']
daily_data['abs_ar'] = daily_data['abnormal_return'].abs()

# Amihud非流动性指标
daily_data['amihud'] = daily_data['market_return'].abs() / (daily_data['hs300_volume'] / 1e10)

print(f"   计算完成")

# ============================================================================
# 9. 清理并保存
# ============================================================================
print("\n【9】清理并保存数据...")

# 过滤日期范围
final_data = daily_data[daily_data['date'] >= '2015-07-01'].copy()

# 填充缺失值
numeric_cols = final_data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    final_data[col] = final_data[col].ffill().bfill()

# 删除关键变量缺失的行
key_cols = ['abs_ar', 'ivix', 'market_return', 'sentiment_zscore']
final_data = final_data.dropna(subset=key_cols)

print(f"\n最终数据集: {len(final_data)}条")
print(f"日期范围: {final_data['date'].min()} ~ {final_data['date'].max()}")

# 保存
output_cols = ['date', 'market_return', 'abnormal_return', 'abs_ar',
               'hs300_close', 'hs300_volume', 'ivix',
               'north_flow', 'margin_balance', 'usd_cny',
               'gdp_growth', 'cpi', 'ppi', 'm2_growth', 'epu',
               'volatility_20d', 'volatility_60d',
               'momentum_5d', 'momentum_10d', 'momentum_20d',
               'volume_ratio_5d', 'volume_ratio_20d',
               'intraday_range', 'sentiment_zscore', 'sentiment_level',
               'amihud']

final_output = final_data[output_cols].copy()
final_output.to_csv(f'{FINAL_DIR}/real_data_complete.csv', index=False)
print(f"\n已保存: {FINAL_DIR}/real_data_complete.csv")

# ============================================================================
# 10. 描述统计
# ============================================================================
print("\n【10】数据描述统计...")

stats_cols = ['abs_ar', 'ivix', 'sentiment_zscore', 'cpi', 'm2_growth', 'epu',
              'volatility_20d', 'north_flow', 'margin_balance']

print("\n变量统计:")
for col in stats_cols:
    if col in final_output.columns:
        s = final_output[col].describe()
        print(f"\n{col}:")
        print(f"   均值: {s['mean']:.4f}, 标准差: {s['std']:.4f}")
        print(f"   最小值: {s['min']:.4f}, 最大值: {s['max']:.4f}")

# 情绪分布
print("\n情绪等级分布:")
sentiment_dist = final_output['sentiment_level'].value_counts()
for level, count in sentiment_dist.items():
    print(f"   {level}: {count}条 ({count/len(final_output)*100:.1f}%)")

print("\n" + "=" * 60)
print("数据整合完成！")
print("=" * 60)

print("\n【最终数据文件】")
print(f"  - real_data_complete.csv: 完整数据集，{len(final_output)}条，{len(output_cols)}个变量")
print(f"  - 包含: 市场数据 + 情绪指标 + 宏观数据")