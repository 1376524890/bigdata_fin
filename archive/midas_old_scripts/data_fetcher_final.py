#!/usr/bin/env python3
"""
最终数据获取脚本 - 解决日期问题 + 国际数据
"""

import pandas as pd
import numpy as np
import requests
import time
import warnings
import os
from bs4 import BeautifulSoup
import json
import re

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'

print("=" * 60)
print("最终数据获取脚本")
print("=" * 60)

import akshare as ak

# ============================================================================
# 1. 直接爬取沪深300估值数据（绕过日期格式问题）
# ============================================================================
print("\n【1】直接爬取沪深300估值数据...")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

# 乐咕乐股沪深300 PE
try:
    url = "https://legulegu.com/sdata/stock/000300"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 尝试从页面提取数据
        print("   乐咕乐股页面获取成功")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 2. 使用baostock获取估值数据
# ============================================================================
print("\n【2】尝试baostock获取估值数据...")

try:
    import baostock as bs
    lg = bs.login()
    print(f"   baostock登录: {lg.error_msg}")

    # 获取沪深300 PE数据
    rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,code,open,high,low,close,preclose,volume,amount,turn,pbMRQ,peTTM",
        start_date='2015-01-01', end_date='2024-12-31',
        frequency="d", adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if len(data_list) > 0:
        df = pd.DataFrame(data_list, columns=rs.fields)
        df.to_csv(f'{OUTPUT_DIR}/hs300_baostock.csv', index=False)
        print(f"   成功: {len(df)}条")
        print(f"   包含PE/PB数据: {df.columns.tolist()}")
    bs.logout()
except ImportError:
    print("   baostock未安装，尝试安装...")
    try:
        import subprocess
        subprocess.run(['pip', 'install', 'baostock', '-q'], check=True)
        import baostock as bs
        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,code,open,high,low,close,volume,amount,pbMRQ,peTTM",
            start_date='2015-01-01', end_date='2024-12-31',
            frequency="d", adjustflag="3"
        )
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if len(data_list) > 0:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df.to_csv(f'{OUTPUT_DIR}/hs300_baostock.csv', index=False)
            print(f"   成功安装并获取: {len(df)}条")
        bs.logout()
    except Exception as e:
        print(f"   安装失败: {e}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 3. 使用yfinance获取国际数据（间隔延迟）
# ============================================================================
print("\n【3】获取国际市场数据（延迟请求）...")

try:
    import yfinance as yf

    international_data = {}

    # 标普500
    time.sleep(5)
    print("   标普500...")
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/sp500.csv', index=False)
            print(f"   成功: {len(hist)}条")
            international_data['sp500_return'] = hist
    except Exception as e:
        print(f"   失败: {e}")

    # VIX
    time.sleep(5)
    print("   VIX...")
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/vix.csv', index=False)
            print(f"   成功: {len(hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 美债10年
    time.sleep(5)
    print("   美债10年期...")
    try:
        tnk = yf.Ticker("^TNX")
        hist = tnk.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/us10y.csv', index=False)
            print(f"   成功: {len(hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 美元指数
    time.sleep(5)
    print("   美元指数...")
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/dxy.csv', index=False)
            print(f"   成功: {len(hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 黄金
    time.sleep(5)
    print("   黄金...")
    try:
        gold = yf.Ticker("GC=F")
        hist = gold.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/gold.csv', index=False)
            print(f"   成功: {len(hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 原油
    time.sleep(5)
    print("   原油...")
    try:
        oil = yf.Ticker("BZ=F")
        hist = oil.history(start="2020-01-01", end="2024-12-31")
        if len(hist) > 0:
            hist = hist.reset_index()
            hist.to_csv(f'{OUTPUT_DIR}/oil.csv', index=False)
            print(f"   成功: {len(hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

except Exception as e:
    print(f"   yfinance失败: {e}")

# ============================================================================
# 4. 从中国人民银行获取国债收益率
# ============================================================================
print("\n【4】从央行获取国债收益率...")

try:
    # 尝试从央行网站获取
    url = "http://www.chinabond.com.cn/d2s100/cbIndex/YieldCurveIndex.html"
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"   中债登网站响应: {resp.status_code}")
except Exception as e:
    print(f"   失败: {e}")

# 使用AKShare国债数据接口
try:
    bond_spot = ak.bond_zh_spot()
    bond_spot.to_csv(f'{OUTPUT_DIR}/bond_spot.csv', index=False)
    print(f"   债券即期数据: {len(bond_spot)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 5. 从理杏仁爬取估值数据
# ============================================================================
print("\n【5】尝试理杏仁估值数据...")

try:
    # 理杏仁需要登录，尝试其他方式
    # 使用AKShare的指数PE接口，跳过日期解析
    pe_data = []
    for page in range(1, 5):
        url = f"https://www.lixinger.com/analytics/stock/000300/detail/pe-history"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            # 解析数据
        except:
            break
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 6. 计算指标数据
# ============================================================================
print("\n【6】计算衍生指标...")

# 从已有数据计算
try:
    # 沪深300数据计算收益率和波动率
    hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
    hs300['date'] = pd.to_datetime(hs300['date'])
    hs300['return'] = hs300['close'].pct_change()
    hs300['volatility_20d'] = hs300['return'].rolling(20).std()
    hs300['amihud'] = hs300['return'].abs() / (hs300['volume'] / 1e8)  # Amihud非流动性指标
    hs300['range'] = (hs300['high'] - hs300['low']) / hs300['close']  # 日内振幅

    hs300.to_csv(f'{OUTPUT_DIR}/hs300_with_indicators.csv', index=False)
    print(f"   计算沪深300衍生指标: {len(hs300)}条")

    # 上证指数计算
    sh = pd.read_csv(f'{OUTPUT_DIR}/sh_index_daily.csv')
    sh['date'] = pd.to_datetime(sh['date'])
    sh['return'] = sh['close'].pct_change()
    sh['volatility_20d'] = sh['return'].rolling(20).std()
    sh.to_csv(f'{OUTPUT_DIR}/sh_with_indicators.csv', index=False)
    print(f"   计算上证指数衍生指标: {len(sh)}条")

except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 统计结果
# ============================================================================
print("\n【统计】最终汇总...")

files = os.listdir(OUTPUT_DIR)
csv_files = [f for f in files if f.endswith('.csv')]

valid_files = []
for f in csv_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    try:
        df = pd.read_csv(filepath)
        if len(df) > 0:
            valid_files.append((f, len(df), len(df.columns)))
    except:
        pass

print(f"\n总有效数据文件数: {len(valid_files)}")
print("\n按类别分类:")
categories = {
    '指数数据': ['hs300', 'sh_index', 'sz_index', 'cyb_index', 'sp500', 'vix'],
    '宏观数据': ['gdp', 'cpi', 'ppi', 'm2', 'social_financing', 'industrial', 'lpr', 'unemployment', 'reserve', 'epu'],
    '市场情绪': ['margin', 'north_money', 'investor', 'ivix', 'zt_pool', 'dt_pool'],
    '估值数据': ['pe', 'pb', 'baostock'],
    '国际数据': ['us10y', 'dxy', 'gold', 'oil', 'fx'],
    '其他': []
}

for cat, keywords in categories.items():
    cat_files = [f for f, r, c in valid_files if any(k in f.lower() for k in keywords)]
    if cat_files:
        print(f"\n  {cat}:")
        for f in cat_files:
            rows = next(r for ff, r, c in valid_files if ff == f)
            print(f"    - {f}: {rows}行")