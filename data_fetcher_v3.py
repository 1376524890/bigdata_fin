#!/usr/bin/env python3
"""
第三版数据获取脚本 - 使用正确的AKShare接口
"""

import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("第三版数据获取脚本（正确API接口）")
print("=" * 60)

import akshare as ak

# ============================================================================
# 1. 融资融券数据
# ============================================================================
print("\n【1】获取融资融券数据...")

try:
    # 沪市融资融券
    print("   沪市融资融券...")
    margin_sh = ak.stock_margin_sse(date='20240115')
    margin_sh.to_csv(f'{OUTPUT_DIR}/margin_sse.csv', index=False)
    print(f"   成功: {len(margin_sh)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 深市融资融券
    print("   深市融资融券...")
    margin_sz = ak.stock_margin_szse(date='20240115')
    margin_sz.to_csv(f'{OUTPUT_DIR}/margin_szse.csv', index=False)
    print(f"   成功: {len(margin_sz)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 融资融券账户统计
    print("   融资融券账户统计...")
    margin_account = ak.stock_margin_account_info()
    margin_account.to_csv(f'{OUTPUT_DIR}/margin_account.csv', index=False)
    print(f"   成功: {len(margin_account)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 2. 中国波指（隐含波动率）
# ============================================================================
print("\n【2】获取中国波指数据...")

try:
    # 50ETF期权波动率指数
    print("   50ETF期权波动率指数...")
    ivix_50 = ak.index_option_50etf_qvix()
    ivix_50['date'] = pd.to_datetime(ivix_50['date'])
    ivix_50.to_csv(f'{OUTPUT_DIR}/ivix_50etf.csv', index=False)
    print(f"   成功: {len(ivix_50)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 300ETF期权波动率指数
    print("   300ETF期权波动率指数...")
    ivix_300 = ak.index_option_300etf_qvix()
    ivix_300['date'] = pd.to_datetime(ivix_300['date'])
    ivix_300.to_csv(f'{OUTPUT_DIR}/ivix_300etf.csv', index=False)
    print(f"   成功: {len(ivix_300)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 3. 市场估值数据
# ============================================================================
print("\n【3】获取市场估值数据...")

try:
    # 全市场PB
    print("   全市场PB...")
    all_pb = ak.stock_a_all_pb()
    all_pb.to_csv(f'{OUTPUT_DIR}/market_pb.csv', index=False)
    print(f"   成功: {len(all_pb)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 指数PB
    print("   指数PB...")
    index_pb = ak.stock_index_pb_lg(symbol="沪深300")
    index_pb.to_csv(f'{OUTPUT_DIR}/hs300_pb.csv', index=False)
    print(f"   成功: {len(index_pb)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 指数PE
    print("   指数PE...")
    index_pe = ak.stock_index_pe_lg(symbol="沪深300")
    index_pe.to_csv(f'{OUTPUT_DIR}/hs300_pe.csv', index=False)
    print(f"   成功: {len(index_pe)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 市场PE
    print("   市场PE...")
    market_pe = ak.stock_market_pe_lg()
    market_pe.to_csv(f'{OUTPUT_DIR}/market_pe.csv', index=False)
    print(f"   成功: {len(market_pe)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 4. 失业率数据
# ============================================================================
print("\n【4】获取失业率数据...")

try:
    # 城镇失业率
    print("   城镇调查失业率...")
    unemployment = ak.macro_china_urban_unemployment()
    unemployment.to_csv(f'{OUTPUT_DIR}/urban_unemployment.csv', index=False)
    print(f"   成功: {len(unemployment)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 5. 工业增加值
# ============================================================================
print("\n【5】获取工业增加值数据...")

try:
    # 工业增加值同比
    print("   工业增加值同比...")
    industrial = ak.macro_china_industrial_production_yoy()
    industrial.to_csv(f'{OUTPUT_DIR}/industrial_production_yoy.csv', index=False)
    print(f"   成功: {len(industrial)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 6. 国债收益率
# ============================================================================
print("\n【6】获取国债收益率数据...")

try:
    # 中国国债收益率
    print("   中国国债收益率...")
    bond_yield = ak.bond_china_yield(start_date='20150101', end_date='20241231')
    bond_yield.to_csv(f'{OUTPUT_DIR}/bond_yield_cn.csv', index=False)
    print(f"   成功: {len(bond_yield)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 7. 汇率数据
# ============================================================================
print("\n【7】获取汇率数据...")

try:
    # 人民币汇率历史
    print("   人民币汇率历史...")
    cny_hist = ak.currency_history(symbol="USDCNY", start_date="20150101", end_date="20241231")
    cny_hist.to_csv(f'{OUTPUT_DIR}/usd_cny_history.csv', index=False)
    print(f"   成功: {len(cny_hist)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 央行汇率
    print("   央行汇率中间价...")
    boc_rate = ak.currency_boc_safe()
    boc_rate.to_csv(f'{OUTPUT_DIR}/boc_rate.csv', index=False)
    print(f"   成功: {len(boc_rate)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 8. 新增投资者数据
# ============================================================================
print("\n【8】获取投资者账户数据...")

try:
    # 投资者账户统计
    print("   投资者账户统计...")
    investor_stats = ak.stock_account_statistics_em()
    investor_stats.to_csv(f'{OUTPUT_DIR}/investor_account_stats.csv', index=False)
    print(f"   成功: {len(investor_stats)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 9. 涨跌停数据
# ============================================================================
print("\n【9】获取涨跌停数据...")

try:
    # 涨停板
    print("   涨停板...")
    zt_pool = ak.stock_zt_pool_em(date='20240115')
    zt_pool.to_csv(f'{OUTPUT_DIR}/zt_pool.csv', index=False)
    print(f"   成功: {len(zt_pool)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 跌停板 - 使用其他方式获取
    print("   获取跌停板数据...")
    dt_pool = ak.stock_gsrl_gsdt_em(date='20240115')
    dt_pool.to_csv(f'{OUTPUT_DIR}/dt_pool.csv', index=False)
    print(f"   成功: {len(dt_pool)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 10. 行业数据
# ============================================================================
print("\n【10】获取行业数据...")

try:
    # 行业板块行情
    print("   行业板块行情...")
    industry_spot = ak.stock_board_industry_spot_em()
    industry_spot.to_csv(f'{OUTPUT_DIR}/industry_spot.csv', index=False)
    print(f"   成功: {len(industry_spot)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 11. 公开市场操作
# ============================================================================
print("\n【11】获取公开市场操作数据...")

try:
    # 央行公开市场操作
    print("   公开市场操作...")
    omo_funcs = [f for f in dir(ak) if 'open_market' in f.lower() or 'omo' in f.lower()]
    print(f"   找到接口: {omo_funcs}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 12. 存款准备金率
# ============================================================================
print("\n【12】获取存款准备金率数据...")

try:
    rr_funcs = [f for f in dir(ak) if 'reserve' in f.lower() or 'rr' in f.lower()]
    print(f"   找到准备金接口: {rr_funcs}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 13. EPU指数
# ============================================================================
print("\n【13】获取EPU政策不确定性指数...")

try:
    epu_funcs = [f for f in dir(ak) if 'epu' in f.lower() or 'uncertainty' in f.lower()]
    print(f"   找到EPU接口: {epu_funcs}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 统计结果
# ============================================================================
print("\n【统计】汇总数据获取结果...")

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

print(f"\n有效数据文件数: {len(valid_files)}")
print("\n文件列表:")
for f, rows, cols in sorted(valid_files):
    print(f"  - {f}: {rows}行, {cols}列")