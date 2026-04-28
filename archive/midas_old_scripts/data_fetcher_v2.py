#!/usr/bin/env python3
"""
补充数据获取脚本 - 使用替代数据源
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import warnings
from datetime import datetime, timedelta
import os
import re

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("补充数据获取脚本（替代数据源）")
print("=" * 60)

# ============================================================================
# 1. AKShare替代API获取更多数据
# ============================================================================
print("\n【1】使用AKShare替代接口...")

try:
    import akshare as ak

    # 查看可用的融资融券接口
    print("\n   搜索融资融券相关接口...")
    margin_funcs = [f for f in dir(ak) if 'margin' in f.lower()]
    print(f"   找到: {margin_funcs}")

    # 1.1 尝试融资融券汇总
    print("\n   [1.1] 获取融资融券汇总数据...")
    try:
        # 使用正确的方法名
        margin_all = ak.stock_margin_underlying_info_sz_sh()
        margin_all.to_csv(f'{OUTPUT_DIR}/margin_all.csv', index=False)
        print(f"   成功: {len(margin_all)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.2 尝试获取融资融券详情
    try:
        margin_detail = ak.stock_margin_detail_sz_sh(date='20240115')
        margin_detail.to_csv(f'{OUTPUT_DIR}/margin_detail.csv', index=False)
        print(f"   成功: {len(margin_detail)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.3 北向资金 - 使用正确接口
    print("\n   [1.3] 获取北向资金数据...")
    hsgt_funcs = [f for f in dir(ak) if 'hsgt' in f.lower()]
    print(f"   找到北向资金接口: {hsgt_funcs}")

    try:
        north = ak.stock_hsgt_hist_em(symbol="北向资金")
        north.to_csv(f'{OUTPUT_DIR}/north_money_hist.csv', index=False)
        print(f"   成功: {len(north)}条")
    except Exception as e:
        print(f"   失败: {e}")

    try:
        north_daily = ak.stock_hsgt_north_net_flow_in_em(symbol="北向")
        north_daily.to_csv(f'{OUTPUT_DIR}/north_money_daily.csv', index=False)
        print(f"   成功: {len(north_daily)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.4 行业指数
    print("\n   [1.4] 获取行业指数...")
    board_funcs = [f for f in dir(ak) if 'board' in f.lower() and 'industry' in f.lower()]
    print(f"   找到行业接口: {board_funcs}")

    try:
        industry_index = ak.stock_board_industry_index_em(symbol="小金属")
        industry_index.to_csv(f'{OUTPUT_DIR}/industry_index_sample.csv', index=False)
        print(f"   成功: {len(industry_index)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 获取行业列表
    try:
        industry_list = ak.stock_board_industry_name_em()
        industry_list.to_csv(f'{OUTPUT_DIR}/industry_list.csv', index=False)
        print(f"   成功获取行业列表: {len(industry_list)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.5 涨跌停数据
    print("\n   [1.5] 获取涨跌停数据...")
    zt_funcs = [f for f in dir(ak) if 'zt' in f.lower() or 'dt' in f.lower()]
    print(f"   找到涨跌停接口: {zt_funcs}")

    try:
        zt_pool = ak.stock_zt_pool_em(date='20240115')
        zt_pool.to_csv(f'{OUTPUT_DIR}/zt_pool.csv', index=False)
        print(f"   成功涨停池: {len(zt_pool)}条")
    except Exception as e:
        print(f"   失败: {e}")

    try:
        dt_pool = ak.stock_dt_pool_em(date='20240115')
        dt_pool.to_csv(f'{OUTPUT_DIR}/dt_pool.csv', index=False)
        print(f"   成功跌停池: {len(dt_pool)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.6 市场情绪指标
    print("\n   [1.6] 获取市场情绪数据...")
    try:
        # 获取A股全市场指标
        market_stats = ak.stock_a_code_change()
        market_stats.to_csv(f'{OUTPUT_DIR}/market_code_change.csv', index=False)
        print(f"   成功: {len(market_stats)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.7 新增投资者
    print("\n   [1.7] 获取新增投资者...")
    investor_funcs = [f for f in dir(ak) if 'investor' in f.lower() or 'account' in f.lower()]
    print(f"   找到投资者接口: {investor_funcs}")

    try:
        investors = ak.stock_em_account_info()
        investors.to_csv(f'{OUTPUT_DIR}/investor_account.csv', index=False)
        print(f"   成功: {len(investors)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.8 汇率数据
    print("\n   [1.8] 获取汇率数据...")
    currency_funcs = [f for f in dir(ak) if 'currency' in f.lower() or 'fx' in f.lower()]
    print(f"   找到汇率接口: {currency_funcs}")

    try:
        # 美元人民币汇率
        usd_cny = ak.fx_spot_quote(symbol="USDCNY")
        usd_cny.to_csv(f'{OUTPUT_DIR}/usd_cny.csv', index=False)
        print(f"   成功: {len(usd_cny)}条")
    except Exception as e:
        print(f"   失败: {e}")

    try:
        # 外汇历史
        fx_hist = ak.currency_hist(symbol="USDCNY", start_date="20150101", end_date="20241231")
        fx_hist.to_csv(f'{OUTPUT_DIR}/usd_cny_hist.csv', index=False)
        print(f"   成功: {len(fx_hist)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.9 工业增加值 - 搜索替代接口
    print("\n   [1.9] 获取工业增加值...")
    industrial_funcs = [f for f in dir(ak) if 'industrial' in f.lower()]
    print(f"   找到工业接口: {industrial_funcs}")

    try:
        industrial = ak.macro_china_industrial_cargo_transport()
        industrial.to_csv(f'{OUTPUT_DIR}/industrial_cargo.csv', index=False)
        print(f"   成功: {len(industrial)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.10 国债收益率曲线
    print("\n   [1.10] 获取国债收益率曲线...")
    bond_funcs = [f for f in dir(ak) if 'bond' in f.lower() and 'yield' in f.lower()]
    print(f"   找到国债接口: {bond_funcs}")

    try:
        bond_yield_curve = ak.bond_china_yield_curve(start_date="20150101", end_date="20241231")
        bond_yield_curve.to_csv(f'{OUTPUT_DIR}/bond_yield_curve.csv', index=False)
        print(f"   成功: {len(bond_yield_curve)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.11 信用债利差
    print("\n   [1.11] 获取信用利差...")
    credit_funcs = [f for f in dir(ak) if 'credit' in f.lower()]
    print(f"   找到信用接口: {credit_funcs}")

    try:
        credit_spread = ak.bond_china_credit_spread(start_date="20150101", end_date="20241231")
        credit_spread.to_csv(f'{OUTPUT_DIR}/credit_spread.csv', index=False)
        print(f"   成功: {len(credit_spread)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.12 沪深300估值数据
    print("\n   [1.12] 获取沪深300估值...")
    pe_funcs = [f for f in dir(ak) if 'pe' in f.lower() or 'pb' in f.lower()]
    print(f"   找到估值接口: {pe_funcs}")

    try:
        valuation_hs300 = ak.stock_a_pe_and_pb_indicator(symbol="hs300")
        valuation_hs300.to_csv(f'{OUTPUT_DIR}/hs300_pe_pb_indicator.csv', index=False)
        print(f"   成功: {len(valuation_hs300)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.13 期权数据
    print("\n   [1.13] 获取期权数据...")
    option_funcs = [f for f in dir(ak) if 'option' in f.lower()]
    print(f"   找到期权接口: {option_funcs}")

    try:
        option_current = ak.option_current_em()
        option_current.to_csv(f'{OUTPUT_DIR}/option_current.csv', index=False)
        print(f"   成功: {len(option_current)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.14 基金数据
    print("\n   [1.14] 获取基金数据...")
    fund_funcs = [f for f in dir(ak) if 'fund' in f.lower() and 'issue' in f.lower()]
    print(f"   找到基金接口: {fund_funcs}")

    try:
        fund_etf = ak.fund_etf_fund_info_em(fund="510300")
        fund_etf.to_csv(f'{OUTPUT_DIR}/fund_etf_300.csv', index=False)
        print(f"   成功: {len(fund_etf)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.15 失业率
    print("\n   [1.15] 获取失业率...")
    unemployment_funcs = [f for f in dir(ak) if 'unemployment' in f.lower() or 'employ' in f.lower()]
    print(f"   找到失业率接口: {unemployment_funcs}")

    try:
        # 城镇调查失业率
        unemployment = ak.macro_china_unemployment()
        unemployment.to_csv(f'{OUTPUT_DIR}/unemployment.csv', index=False)
        print(f"   成功: {len(unemployment)}条")
    except Exception as e:
        print(f"   失败: {e}")

except Exception as e:
    print(f"AKShare替代接口失败: {e}")

# ============================================================================
# 2. 东方财富网直接爬取
# ============================================================================
print("\n【2】东方财富网数据爬取...")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://data.eastmoney.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}

# 2.1 市场成交数据
print("\n   [2.1] 东方财富市场成交数据...")
try:
    url = "http://push2.eastmoney.com/api/qt/clist/get"
    params = {
        'fid': 'f3',
        'po': '1',
        'pz': '5000',
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'fs': 'm:1,m:2,m:3',
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f11,f62,f128,f136,f115,f152'
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if data.get('data') and data['data'].get('diff'):
            df = pd.DataFrame(data['data']['diff'])
            df.to_csv(f'{OUTPUT_DIR}/eastmoney_market_stats.csv', index=False)
            print(f"   成功: {len(df)}条")
except Exception as e:
    print(f"   失败: {e}")

# 2.2 北向资金详细数据
print("\n   [2.2] 北向资金详细数据...")
try:
    url = "http://push2.eastmoney.com/api/qt/kamt.kamt/get"
    params = {
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56',
        'kmt': '1',
        'ut': 'b2884a393a59ad64002292a3e90d46a5'
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if data.get('data'):
            records = data['data'].get('kamt', [])
            if records:
                df = pd.DataFrame(records)
                df.to_csv(f'{OUTPUT_DIR}/north_money_em.csv', index=False)
                print(f"   成功: {len(df)}条")
except Exception as e:
    print(f"   失败: {e}")

# 2.3 沪深300成分股及市值
print("\n   [2.3] 沪深300成分股...")
try:
    url = "http://push2.eastmoney.com/api/qt/clist/get"
    params = {
        'fid': 'f62',
        'po': '1',
        'pz': '500',
        'pn': '1',
        'np': '1',
        'fs': 'b:BK0505',
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f11,f62,f128,f136,f115,f152'
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if data.get('data') and data['data'].get('diff'):
            df = pd.DataFrame(data['data']['diff'])
            df.to_csv(f'{OUTPUT_DIR}/hs300_components.csv', index=False)
            print(f"   成功: {len(df)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 3. yfinance重新尝试（带延迟）
# ============================================================================
print("\n【3】yfinance国际数据（带延迟）...")

import yfinance as yf

international_tickers = {
    'SP500': '^GSPC',
    'VIX': '^VIX',
    'US10Y': '^TNX',
    'GOLD': 'GC=F',
    'OIL': 'BZ=F',
}

for name, ticker in international_tickers.items():
    time.sleep(3)  # 添加延迟避免速率限制
    print(f"\n   [{name}] 获取{ticker}...")
    try:
        data = yf.download(ticker, start='2020-01-01', end='2024-12-31', progress=False)
        if len(data) > 0:
            data = data.reset_index()
            data.columns = [col.lower() if col != 'Date' else 'date' for col in data.columns]
            data.to_csv(f'{OUTPUT_DIR}/{name.lower()}_intl.csv', index=False)
            print(f"   成功: {len(data)}条")
        else:
            print(f"   无数据")
    except Exception as e:
        print(f"   失败: {e}")

# ============================================================================
# 4. 统计结果
# ============================================================================
print("\n【4】汇总数据获取结果...")

files = os.listdir(OUTPUT_DIR)
csv_files = [f for f in files if f.endswith('.csv')]

# 过滤空文件
valid_files = []
for f in csv_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    try:
        df = pd.read_csv(filepath)
        if len(df) > 0:
            valid_files.append((f, len(df), len(df.columns)))
    except:
        pass

print(f"\n有效数据文件数: {len(valid_files)}")
print("\n文件列表:")
for f, rows, cols in sorted(valid_files):
    print(f"  - {f}: {rows}行, {cols}列")