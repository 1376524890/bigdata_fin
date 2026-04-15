#!/usr/bin/env python3
"""
数据整合脚本 - 简化版
"""

import pandas as pd
import numpy as np
import warnings
import re

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
FINAL_DIR = '/home/marktom/bigdata-fin'

print("=" * 60)
print("数据整合脚本（简化版）")
print("=" * 60)

# ============================================================================
# 1. 加载并合并数据
# ============================================================================
print("\n【1】加载并合并数据...")

# 沪深300日度数据
hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
hs300['date'] = pd.to_datetime(hs300['date'])
hs300 = hs300.rename(columns={'close': 'hs300_close', 'volume': 'hs300_volume'})
print(f"   沪深300: {len(hs300)}条")

# 上证指数
sh = pd.read_csv(f'{OUTPUT_DIR}/sh_index_daily.csv')
sh['date'] = pd.to_datetime(sh['date'])
sh = sh.rename(columns={'close': 'sh_close', 'volume': 'sh_volume'})
print(f"   上证指数: {len(sh)}条")

# 以沪深300为基准
daily_data = hs300[['date', 'hs300_close', 'hs300_volume']].copy()

# 合并上证指数
daily_data = daily_data.merge(sh[['date', 'sh_close']], on='date', how='left')

# 中国波指
ivix = pd.read_csv(f'{OUTPUT_DIR}/ivix_50etf.csv')
ivix['date'] = pd.to_datetime(ivix['date'])
ivix = ivix.rename(columns={'close': 'ivix'})
daily_data = daily_data.merge(ivix[['date', 'ivix']], on='date', how='left')
print(f"   中国波指: {len(ivix)}条")

# 北向资金
north = pd.read_csv(f'{OUTPUT_DIR}/north_money_hist.csv')
north['date'] = pd.to_datetime(north['日期'])
north['north_flow'] = north['当日成交净买额']
daily_data = daily_data.merge(north[['date', 'north_flow']], on='date', how='left')
print(f"   北向资金: {len(north)}条")

# 融资融券
margin = pd.read_csv(f'{OUTPUT_DIR}/margin_account.csv')
margin['date'] = pd.to_datetime(margin['日期'])
margin['margin_balance'] = margin['融资余额']
daily_data = daily_data.merge(margin[['date', 'margin_balance']], on='date', how='left')
print(f"   融资融券: {len(margin)}条")

# 央行汇率
boc_rate = pd.read_csv(f'{OUTPUT_DIR}/boc_rate.csv')
boc_rate['date'] = pd.to_datetime(boc_rate['日期'])
boc_rate['usd_cny'] = boc_rate['美元']
daily_data = daily_data.merge(boc_rate[['date', 'usd_cny']], on='date', how='left')
print(f"   央行汇率: {len(boc_rate)}条")

# ============================================================================
# 2. 计算衍生指标
# ============================================================================
print("\n【2】计算衍生指标...")

# 市场收益率
daily_data['market_return'] = daily_data['hs300_close'].pct_change()

# 异常收益率（相对于上证指数）
daily_data['sh_return'] = daily_data['sh_close'].pct_change()
daily_data['abnormal_return'] = daily_data['market_return'] - daily_data['sh_return']
daily_data['abs_ar'] = daily_data['abnormal_return'].abs()

# 波动率
daily_data['volatility_20d'] = daily_data['market_return'].rolling(20).std()

# 换手率比率
daily_data['turnover_ratio'] = daily_data['hs300_volume'] / daily_data['hs300_volume'].rolling(20).mean()

# Amihud非流动性指标
daily_data['amihud'] = daily_data['market_return'].abs() / (daily_data['hs300_volume'] / 1e10)

# ============================================================================
# 3. 处理宏观数据
# ============================================================================
print("\n【3】处理宏观数据...")

# GDP - 季度数据
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
gdp_data = gdp_data.set_index('date')
print(f"   GDP: {len(gdp_data)}条")

# CPI - 月度数据
cpi = pd.read_csv(f'{OUTPUT_DIR}/cpi.csv')
cpi['date'] = pd.to_datetime(cpi['日期'])
cpi['cpi'] = pd.to_numeric(cpi['今值'], errors='coerce')
cpi_data = cpi[['date', 'cpi']].dropna()
cpi_data = cpi_data.set_index('date')
print(f"   CPI: {len(cpi_data)}条")

# PPI - 月度数据
ppi = pd.read_csv(f'{OUTPUT_DIR}/ppi.csv')
ppi['date'] = pd.to_datetime(ppi['日期'])
ppi['ppi'] = pd.to_numeric(ppi['今值'], errors='coerce')
ppi_data = ppi[['date', 'ppi']].dropna()
ppi_data = ppi_data.set_index('date')
print(f"   PPI: {len(ppi_data)}条")

# M2 - 月度数据
m2 = pd.read_csv(f'{OUTPUT_DIR}/m2.csv')
m2['date'] = pd.to_datetime(m2['日期'])
m2['m2_growth'] = pd.to_numeric(m2['今值'], errors='coerce')
m2_data = m2[['date', 'm2_growth']].dropna()
m2_data = m2_data.set_index('date')
print(f"   M2: {len(m2_data)}条")

# EPU - 月度数据
epu = pd.read_csv(f'{OUTPUT_DIR}/epu_index.csv')
epu['date'] = pd.to_datetime(epu['year'].astype(str) + '-' + epu['month'].astype(str) + '-01')
epu['epu'] = epu['China_Policy_Index']
epu_data = epu[['date', 'epu']].dropna()
epu_data = epu_data.set_index('date')
print(f"   EPU: {len(epu_data)}条")

# ============================================================================
# 4. 合并宏观数据（前向填充）
# ============================================================================
print("\n【4】合并宏观数据...")

# 创建日期索引
daily_data = daily_data.set_index('date')

# 前向填充宏观数据
# 找到每个日期最近的宏观数据
def fill_macro_data(daily_df, macro_df, col_name):
    """将宏观数据填充到日度数据"""
    result = pd.Series(index=daily_df.index, dtype=float)
    for date in daily_df.index:
        # 找到小于等于该日期的最近值
        available = macro_df[macro_df.index <= date]
        if len(available) > 0:
            result[date] = available[col_name].iloc[-1]
    return result

daily_data['gdp_growth'] = fill_macro_data(daily_data, gdp_data, 'gdp_growth')
daily_data['cpi'] = fill_macro_data(daily_data, cpi_data, 'cpi')
daily_data['ppi'] = fill_macro_data(daily_data, ppi_data, 'ppi')
daily_data['m2_growth'] = fill_macro_data(daily_data, m2_data, 'm2_growth')
daily_data['epu'] = fill_macro_data(daily_data, epu_data, 'epu')

# 重置索引
daily_data = daily_data.reset_index()

print(f"   合并后: {len(daily_data)}条")

# ============================================================================
# 5. 清理数据
# ============================================================================
print("\n【5】清理数据...")

# 过滤日期范围
final_data = daily_data[daily_data['date'] >= '2015-07-01'].copy()
print(f"   2015-07-01后: {len(final_data)}条")

# 关键变量列表
key_vars = ['abs_ar', 'ivix', 'north_flow', 'margin_balance', 'gdp_growth', 'cpi', 'm2_growth', 'epu', 'volatility_20d']

# 检查缺失情况
print("\n   缺失值统计:")
for var in key_vars:
    if var in final_data.columns:
        missing = final_data[var].isna().sum()
        print(f"   {var}: {missing} ({missing/len(final_data)*100:.1f}%)")

# 填充缺失值
for col in final_data.columns:
    if col != 'date':
        final_data[col] = final_data[col].ffill().bfill()

# 最终清理
final_data = final_data.dropna(subset=['abs_ar', 'ivix', 'market_return'])

print(f"\n   最终数据: {len(final_data)}条")
print(f"   日期范围: {final_data['date'].min()} ~ {final_data['date'].max()}")

# ============================================================================
# 6. 保存数据
# ============================================================================
print("\n【6】保存数据...")

# 选择输出列
output_cols = ['date', 'market_return', 'abnormal_return', 'abs_ar',
               'hs300_close', 'hs300_volume', 'ivix',
               'north_flow', 'margin_balance', 'usd_cny',
               'gdp_growth', 'cpi', 'ppi', 'm2_growth', 'epu',
               'volatility_20d', 'turnover_ratio', 'amihud']

output_data = final_data[output_cols].copy()
output_data.to_csv(f'{FINAL_DIR}/real_data_for_analysis.csv', index=False)
print(f"   已保存: {FINAL_DIR}/real_data_for_analysis.csv")

# ============================================================================
# 7. 描述统计
# ============================================================================
print("\n【7】数据描述统计...")

print("\n变量统计:")
stats_vars = ['abs_ar', 'ivix', 'gdp_growth', 'cpi', 'm2_growth', 'epu', 'volatility_20d', 'north_flow', 'margin_balance']
for col in stats_vars:
    if col in output_data.columns:
        s = output_data[col].describe()
        print(f"\n{col}:")
        print(f"   均值: {s['mean']:.4f}, 标准差: {s['std']:.4f}")
        print(f"   最小值: {s['min']:.4f}, 最大值: {s['max']:.4f}")

# ============================================================================
# 8. 数据真实性验证
# ============================================================================
print("\n【8】数据真实性验证...")

# 2015年股灾
crash = output_data[(output_data['date'] >= '2015-06-15') & (output_data['date'] <= '2015-07-15')]
if len(crash) > 0:
    print(f"\n2015年股灾期间（6.15-7.15）:")
    print(f"   平均市场收益率: {crash['market_return'].mean()*100:.2f}%")
    print(f"   平均波动率: {crash['volatility_20d'].mean()*100:.2f}%")

# 2020年疫情
covid = output_data[(output_data['date'] >= '2020-02-01') & (output_data['date'] <= '2020-03-31')]
if len(covid) > 0:
    print(f"\n2020年疫情期间（2-3月）:")
    print(f"   平均市场收益率: {covid['market_return'].mean()*100:.2f}%")
    print(f"   平均波动率: {covid['volatility_20d'].mean()*100:.2f}%")

print("\n" + "=" * 60)
print("数据整合完成！")
print("=" * 60)