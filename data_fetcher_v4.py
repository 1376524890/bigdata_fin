#!/usr/bin/env python3
"""
第四版数据获取脚本 - 存款准备金率、EPU等
"""

import pandas as pd
import numpy as np
import time
import warnings
import os
import akshare as ak

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'

print("=" * 60)
print("第四版数据获取脚本")
print("=" * 60)

# ============================================================================
# 1. 存款准备金率
# ============================================================================
print("\n【1】获取存款准备金率数据...")

try:
    rr = ak.macro_china_reserve_requirement_ratio()
    rr.to_csv(f'{OUTPUT_DIR}/reserve_requirement_ratio.csv', index=False)
    print(f"   成功: {len(rr)}条")
    print(f"   数据列: {rr.columns.tolist()}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 2. EPU政策不确定性指数
# ============================================================================
print("\n【2】获取EPU指数...")

try:
    epu = ak.article_epu_index()
    epu.to_csv(f'{OUTPUT_DIR}/epu_index.csv', index=False)
    print(f"   成功: {len(epu)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 3. 沪深300估值数据（直接爬取）
# ============================================================================
print("\n【3】获取沪深300估值数据（乐咕乐股）...")

try:
    # 使用正确的接口
    hs300_pe = ak.stock_index_pe_lg(symbol="沪深300")
    # 处理日期格式问题
    if 'date' in hs300_pe.columns:
        hs300_pe['date'] = pd.to_datetime(hs300_pe['date'], errors='coerce')
    hs300_pe.to_csv(f'{OUTPUT_DIR}/hs300_pe_lg.csv', index=False)
    print(f"   成功: {len(hs300_pe)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    hs300_pb = ak.stock_index_pb_lg(symbol="沪深300")
    if 'date' in hs300_pb.columns:
        hs300_pb['date'] = pd.to_datetime(hs300_pb['date'], errors='coerce')
    hs300_pb.to_csv(f'{OUTPUT_DIR}/hs300_pb_lg.csv', index=False)
    print(f"   成功: {len(hs300_pb)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 创业板估值
    cyb_pe = ak.stock_index_pe_lg(symbol="创业板指")
    cyb_pe.to_csv(f'{OUTPUT_DIR}/cyb_pe_lg.csv', index=False)
    print(f"   成功: {len(cyb_pe)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 4. 国债收益率曲线
# ============================================================================
print("\n【4】获取国债收益率曲线...")

try:
    bond_yield = ak.bond_china_yield(start_date="20150101", end_date="20241231")
    bond_yield.to_csv(f'{OUTPUT_DIR}/bond_yield_curve.csv', index=False)
    print(f"   成功: {len(bond_yield)}条")
    if len(bond_yield) > 0:
        print(f"   数据列: {bond_yield.columns.tolist()}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 5. 汇率数据（使用正确接口）
# ============================================================================
print("\n【5】获取汇率数据...")

try:
    # 使用currency_history正确参数
    cny = ak.currency_history(symbols="USDCNY", start_date="20150101", end_date="20241231")
    cny.to_csv(f'{OUTPUT_DIR}/usd_cny_hist.csv', index=False)
    print(f"   成功: {len(cny)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 最新汇率
    fx_latest = ak.currency_latest(symbols="USDCNY")
    fx_latest.to_csv(f'{OUTPUT_DIR}/fx_latest.csv', index=False)
    print(f"   成功: {len(fx_latest)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 6. 外汇储备
# ============================================================================
print("\n【6】获取外汇储备数据...")

try:
    fx_reserves = ak.macro_china_fx_reserves_yearly()
    fx_reserves.to_csv(f'{OUTPUT_DIR}/fx_reserves.csv', index=False)
    print(f"   成功: {len(fx_reserves)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 7. 信用利差相关
# ============================================================================
print("\n【7】获取信用相关数据...")

try:
    # 新增金融信贷
    new_credit = ak.macro_china_new_financial_credit()
    new_credit.to_csv(f'{OUTPUT_DIR}/new_financial_credit.csv', index=False)
    print(f"   成功: {len(new_credit)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 8. 黄金外汇
# ============================================================================
print("\n【8】获取黄金外汇数据...")

try:
    gold_fx = ak.macro_china_fx_gold()
    gold_fx.to_csv(f'{OUTPUT_DIR}/gold_fx.csv', index=False)
    print(f"   成功: {len(gold_fx)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 9. 北向资金详细
# ============================================================================
print("\n【9】获取北向资金详细数据...")

try:
    # 北向资金流入
    north_flow = ak.stock_hsgt_fund_flow_summary_em()
    north_flow.to_csv(f'{OUTPUT_DIR}/north_flow_summary.csv', index=False)
    print(f"   成功: {len(north_flow)}条")
except Exception as e:
    print(f"   失败: {e}")

try:
    # 北向资金持股统计
    north_hold = ak.stock_hsgt_stock_statistics_em()
    north_hold.to_csv(f'{OUTPUT_DIR}/north_hold_stats.csv', index=False)
    print(f"   成功: {len(north_hold)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 10. 行业板块历史数据
# ============================================================================
print("\n【10】获取行业板块历史数据...")

try:
    # 获取几个主要行业的历史数据
    industries = ['小金属', '白酒', '银行', '半导体', '房地产']
    for ind in industries:
        try:
            hist = ak.stock_board_industry_hist_em(symbol=ind, start_date="20150101", end_date="20241231", adjust="")
            hist.to_csv(f'{OUTPUT_DIR}/industry_{ind}_hist.csv', index=False)
            print(f"   {ind}: {len(hist)}条")
        except Exception as e:
            print(f"   {ind}: 失败 - {e}")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 11. 市场成交统计
# ============================================================================
print("\n【11】获取市场成交统计...")

try:
    market_stats_funcs = [f for f in dir(ak) if 'market' in f.lower() and 'stat' in f.lower()]
    print(f"   找到接口: {market_stats_funcs}")
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