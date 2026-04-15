#!/usr/bin/env python3
"""
数据分类整理脚本
将原始数据按类别整理到不同子目录
"""

import os
import shutil
import pandas as pd
from datetime import datetime

BASE_DIR = '/home/marktom/bigdata-fin'
DATA_DIR = os.path.join(BASE_DIR, 'real_data')

# 创建分类目录
categories = {
    '01_指数数据': ['hs300_daily.csv', 'sh_index_daily.csv', 'sz_index_daily.csv',
                   'cyb_index_daily.csv', 'hs300_baostock.csv', 'hs300_with_indicators.csv',
                   'sh_with_indicators.csv'],
    '02_波动率数据': ['ivix_50etf.csv', 'ivix_300etf.csv'],
    '03_市场情绪': ['market_sentiment_history.csv', 'historical_sentiment.csv',
                    'north_money_hist.csv', 'margin_account.csv', 'margin_szse.csv',
                    'investor_account_stats.csv', 'north_flow_summary.csv', 'north_hold_stats.csv'],
    '04_宏观指标': ['gdp.csv', 'cpi.csv', 'ppi.csv', 'm2.csv', 'social_financing.csv',
                   'industrial_production_yoy.csv', 'lpr.csv', 'reserve_requirement_ratio.csv',
                   'epu_index.csv', 'new_financial_credit.csv'],
    '05_汇率外汇': ['boc_rate.csv', 'fx_reserves.csv', 'gold_fx.csv', 'bond_yield.csv',
                   'bond_yield_cn.csv', 'bond_yield_curve.csv'],
    '06_资金流向': ['fund_etf_300.csv', 'thshy_concept_flow.csv', 'industry_list.csv'],
    '07_实时数据': ['sina_realtime_index.csv', 'sina_index_data.csv', 'sina_market_data.csv',
                   'today_sentiment_report.csv', 'zt_pool.csv', 'dt_pool.csv', 'ipo_info.csv'],
    '08_其他数据': []  # 剩余文件
}

print("=" * 60)
print("数据分类整理")
print("=" * 60)

# 创建目录
for cat in categories.keys():
    cat_dir = os.path.join(DATA_DIR, cat)
    os.makedirs(cat_dir, exist_ok=True)
    print(f"创建目录: {cat}")

# 移动文件
moved_files = set()
for cat, files in categories.items():
    if not files:
        continue
    cat_dir = os.path.join(DATA_DIR, cat)
    for f in files:
        src = os.path.join(DATA_DIR, f)
        if os.path.exists(src):
            dst = os.path.join(cat_dir, f)
            shutil.move(src, dst)
            moved_files.add(f)
            print(f"  移动: {f} -> {cat}/")

# 移动剩余文件到"其他数据"
other_dir = os.path.join(DATA_DIR, '08_其他数据')
for f in os.listdir(DATA_DIR):
    if f.endswith('.csv'):
        src = os.path.join(DATA_DIR, f)
        if os.path.isfile(src) and f not in moved_files:
            dst = os.path.join(other_dir, f)
            shutil.move(src, dst)
            print(f"  移动: {f} -> 08_其他数据/")

print("\n" + "=" * 60)
print("整理完成!")
print("=" * 60)

# 统计各目录文件数
print("\n目录结构:")
for cat in sorted(os.listdir(DATA_DIR)):
    cat_path = os.path.join(DATA_DIR, cat)
    if os.path.isdir(cat_path):
        files = [f for f in os.listdir(cat_path) if f.endswith('.csv')]
        print(f"  {cat}: {len(files)}个文件")