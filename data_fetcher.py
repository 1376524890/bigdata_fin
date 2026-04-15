#!/usr/bin/env python3
"""
真实数据获取脚本 - 异常收益率与市场不稳定性研究
使用多种数据源：AKShare, yfinance, 公开网站爬取
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

# 输出目录
OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("真实数据获取脚本")
print("=" * 60)

# ============================================================================
# 1. AKShare 中国A股市场数据
# ============================================================================
print("\n【1】通过AKShare获取A股市场数据...")

try:
    import akshare as ak
    print("   AKShare版本:", ak.__version__)

    # 1.1 沪深300指数数据
    print("\n   [1.1] 获取沪深300指数...")
    try:
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        hs300 = hs300.rename(columns={
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300['market_return'] = hs300['close'].pct_change()
        hs300.to_csv(f'{OUTPUT_DIR}/hs300_daily.csv', index=False)
        print(f"   成功获取沪深300数据: {len(hs300)}条记录, 日期范围: {hs300['date'].min()} ~ {hs300['date'].max()}")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.2 上证指数
    print("\n   [1.2] 获取上证指数...")
    try:
        sh_index = ak.stock_zh_index_daily(symbol="sh000001")
        sh_index['date'] = pd.to_datetime(sh_index['date'])
        sh_index.to_csv(f'{OUTPUT_DIR}/sh_index_daily.csv', index=False)
        print(f"   成功: {len(sh_index)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.3 创业板指数
    print("\n   [1.3] 获取创业板指数...")
    try:
        cyb_index = ak.stock_zh_index_daily(symbol="sz399006")
        cyb_index['date'] = pd.to_datetime(cyb_index['date'])
        cyb_index.to_csv(f'{OUTPUT_DIR}/cyb_index_daily.csv', index=False)
        print(f"   成功: {len(cyb_index)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.4 深证成指
    print("\n   [1.4] 获取深证成指...")
    try:
        sz_index = ak.stock_zh_index_daily(symbol="sz399001")
        sz_index['date'] = pd.to_datetime(sz_index['date'])
        sz_index.to_csv(f'{OUTPUT_DIR}/sz_index_daily.csv', index=False)
        print(f"   成功: {len(sz_index)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.5 融资融券数据
    print("\n   [1.5] 获取融资融券数据...")
    try:
        # 沪市融资融券
        margin_sh = ak.margin_detail_szsh(date="20240101", symbol="沪市")
        margin_sh.to_csv(f'{OUTPUT_DIR}/margin_sh.csv', index=False)
        print(f"   沪市融资融券: {len(margin_sh)}条")
    except Exception as e:
        print(f"   沪市融资融券失败: {e}")

    try:
        # 深市融资融券
        margin_sz = ak.margin_detail_szsh(date="20240101", symbol="深市")
        margin_sz.to_csv(f'{OUTPUT_DIR}/margin_sz.csv', index=False)
        print(f"   深市融资融券: {len(margin_sz)}条")
    except Exception as e:
        print(f"   深市融资融券失败: {e}")

    # 1.6 北向资金
    print("\n   [1.6] 获取北向资金数据...")
    try:
        north_money = ak.stock_hsgt_north_net_flow_in_em()
        north_money['date'] = pd.to_datetime(north_money['date'])
        north_money.to_csv(f'{OUTPUT_DIR}/north_money.csv', index=False)
        print(f"   成功: {len(north_money)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.7 A股换手率
    print("\n   [1.7] 获取市场换手率...")
    try:
        turnover = ak.stock_a_lg_indicator(category="换手率")
        turnover['date'] = pd.to_datetime(turnover['date'])
        turnover.to_csv(f'{OUTPUT_DIR}/market_turnover.csv', index=False)
        print(f"   成功: {len(turnover)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.8 涨跌停统计
    print("\n   [1.8] 获取涨跌停统计...")
    try:
        zt_dt = ak.stock_zt_dt_em()
        zt_dt.to_csv(f'{OUTPUT_DIR}/zt_dt_stats.csv', index=False)
        print(f"   成功: {len(zt_dt)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.9 市场成交额
    print("\n   [1.9] 获取市场成交额...")
    try:
        market_amount = ak.stock_a_lg_indicator(category="成交额")
        market_amount['date'] = pd.to_datetime(market_amount['date'])
        market_amount.to_csv(f'{OUTPUT_DIR}/market_amount.csv', index=False)
        print(f"   成功: {len(market_amount)}条记录")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.10 50ETF期权数据（用于隐含波动率）
    print("\n   [1.10] 获取50ETF期权数据...")
    try:
        option_50etf = ak.option_sina_sse_list(symbol="商品期权", exchange="null", type="null")
        print(f"   获取期权列表: {len(option_50etf)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.11 行业数据
    print("\n   [1.11] 获取行业指数...")
    try:
        industry = ak.stock_board_industry_index_em()
        industry.to_csv(f'{OUTPUT_DIR}/industry_index.csv', index=False)
        print(f"   成功: {len(industry)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 1.12 新股发行（IPO）
    print("\n   [1.12] 获取IPO数据...")
    try:
        ipo = ak.stock_ipo_info()
        ipo.to_csv(f'{OUTPUT_DIR}/ipo_info.csv', index=False)
        print(f"   成功: {len(ipo)}条")
    except Exception as e:
        print(f"   失败: {e}")

except Exception as e:
    print(f"AKShare初始化失败: {e}")

# ============================================================================
# 2. AKShare 宏观经济数据
# ============================================================================
print("\n【2】通过AKShare获取宏观经济数据...")

try:
    import akshare as ak

    # 2.1 GDP数据
    print("\n   [2.1] 获取GDP数据...")
    try:
        gdp = ak.macro_china_gdp()
        gdp.to_csv(f'{OUTPUT_DIR}/gdp.csv', index=False)
        print(f"   成功: {len(gdp)}条记录")
        print(f"   数据列: {gdp.columns.tolist()}")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.2 CPI数据
    print("\n   [2.2] 获取CPI数据...")
    try:
        cpi = ak.macro_china_cpi_yearly()
        cpi.to_csv(f'{OUTPUT_DIR}/cpi.csv', index=False)
        print(f"   成功: {len(cpi)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.3 PPI数据
    print("\n   [2.3] 获取PPI数据...")
    try:
        ppi = ak.macro_china_ppi_yearly()
        ppi.to_csv(f'{OUTPUT_DIR}/ppi.csv', index=False)
        print(f"   成功: {len(ppi)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.4 M2数据
    print("\n   [2.4] 获取M2货币供应量...")
    try:
        m2 = ak.macro_china_m2_yearly()
        m2.to_csv(f'{OUTPUT_DIR}/m2.csv', index=False)
        print(f"   成功: {len(m2)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.5 社会融资规模
    print("\n   [2.5] 获取社会融资规模...")
    try:
        sr = ak.macro_china_shrzgm()
        sr.to_csv(f'{OUTPUT_DIR}/social_financing.csv', index=False)
        print(f"   成功: {len(sr)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.6 工业增加值
    print("\n   [2.6] 获取工业增加值...")
    try:
        industrial = ak.macro_china_industrial_production()
        industrial.to_csv(f'{OUTPUT_DIR}/industrial_production.csv', index=False)
        print(f"   成功: {len(industrial)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.7 失业率
    print("\n   [2.7] 获取失业率...")
    try:
        unemployment = ak.macro_china_unemployment_rate()
        unemployment.to_csv(f'{OUTPUT_DIR}/unemployment.csv', index=False)
        print(f"   成功: {len(unemployment)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.8 利率数据 - LPR
    print("\n   [2.8] 获取LPR利率...")
    try:
        lpr = ak.macro_china_lpr()
        lpr.to_csv(f'{OUTPUT_DIR}/lpr.csv', index=False)
        print(f"   成功: {len(lpr)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.9 国债收益率
    print("\n   [2.9] 获取国债收益率...")
    try:
        bond_yield = ak.bond_china_yield(start_date="20150101", end_date="20241231")
        bond_yield.to_csv(f'{OUTPUT_DIR}/bond_yield.csv', index=False)
        print(f"   成功: {len(bond_yield)}条")
    except Exception as e:
        print(f"   失败: {e}")

    # 2.10 居民收入
    print("\n   [2.10] 获取居民收入...")
    try:
        income = ak.macro_china_household_income()
        income.to_csv(f'{OUTPUT_DIR}/household_income.csv', index=False)
        print(f"   成功: {len(income)}条")
    except Exception as e:
        print(f"   失败: {e}")

except Exception as e:
    print(f"宏观经济数据获取失败: {e}")

# ============================================================================
# 3. yfinance 国际市场数据
# ============================================================================
print("\n【3】通过yfinance获取国际市场数据...")

try:
    import yfinance as yf

    # 国际市场股票代码
    international_tickers = {
        'SP500': '^GSPC',          # 标普500
        'NASDAQ': '^NDX',          # 纳斯达克100
        'VIX': '^VIX',             # VIX波动率指数
        'DXY': 'DX-Y.NYB',         # 美元指数
        'US10Y': '^TNX',           # 美国10年期国债收益率
        'GOLD': 'GC=F',            # 黄金期货
        'OIL': 'BZ=F',             # 布伦特原油
        'COPPER': 'HG=F',          # 铜
        'HSI': '^HSI',             # 恒生指数
    }

    for name, ticker in international_tickers.items():
        print(f"\n   [{name}] 获取{ticker}...")
        try:
            data = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)
            if len(data) > 0:
                data = data.reset_index()
                data.columns = [col.lower() if col != 'Date' else 'date' for col in data.columns]
                data.to_csv(f'{OUTPUT_DIR}/{name.lower()}.csv', index=False)
                print(f"   成功: {len(data)}条记录")
            else:
                print(f"   无数据")
        except Exception as e:
            print(f"   失败: {e}")

except Exception as e:
    print(f"yfinance获取失败: {e}")

# ============================================================================
# 4. 网站爬取补充数据
# ============================================================================
print("\n【4】爬取补充数据...")

# 4.1 东方财富 - 市场PE/PB数据
print("\n   [4.1] 东方财富估值数据...")
try:
    import akshare as ak
    # 沪深300估值
    valuation = ak.stock_a_pe_and_pb(symbol="000300")
    valuation['date'] = pd.to_datetime(valuation['date'])
    valuation.to_csv(f'{OUTPUT_DIR}/hs300_pe_pb.csv', index=False)
    print(f"   成功: {len(valuation)}条")
except Exception as e:
    print(f"   失败: {e}")

# 4.2 中国波指（iVIX）
print("\n   [4.2] 中国波指...")
try:
    import akshare as ak
    ivix = ak.index_option_50etf_qvix()
    ivix['date'] = pd.to_datetime(ivix['date'])
    ivix.to_csv(f'{OUTPUT_DIR}/ivix.csv', index=False)
    print(f"   成功: {len(ivix)}条")
except Exception as e:
    print(f"   失败: {e}")

# 4.3 公开市场操作
print("\n   [4.3] 公开市场操作...")
try:
    import akshare as ak
    omo = ak.macro_china_open_market_operation()
    omo.to_csv(f'{OUTPUT_DIR}/open_market_operation.csv', index=False)
    print(f"   成功: {len(omo)}条")
except Exception as e:
    print(f"   失败: {e}")

# 4.4 汇率数据
print("\n   [4.4] 人民币汇率...")
try:
    import akshare as ak
    # 离岸人民币
    cny_offshore = ak.currency_cny_offshore()
    cny_offshore.to_csv(f'{OUTPUT_DIR}/cny_offshore.csv', index=False)
    print(f"   成功: {len(cny_offshore)}条")
except Exception as e:
    print(f"   失败: {e}")

# 4.5 新增投资者数量
print("\n   [4.5] 新增投资者...")
try:
    import akshare as ak
    investors = ak.stock_em_account()
    investors.to_csv(f'{OUTPUT_DIR}/new_investors.csv', index=False)
    print(f"   成功: {len(investors)}条")
except Exception as e:
    print(f"   失败: {e}")

# 4.6 基金发行
print("\n   [4.6] 基金发行...")
try:
    import akshare as ak
    fund_issue = ak.fund_em_open_fund_daily()
    fund_issue.to_csv(f'{OUTPUT_DIR}/fund_issue.csv', index=False)
    print(f"   成功: {len(fund_issue)}条")
except Exception as e:
    print(f"   失败: {e}")

# ============================================================================
# 5. 数据汇总
# ============================================================================
print("\n【5】汇总数据获取结果...")

# 统计获取的文件
files = os.listdir(OUTPUT_DIR)
csv_files = [f for f in files if f.endswith('.csv')]

print(f"\n成功获取的文件数: {len(csv_files)}")
print("\n文件列表:")
for f in sorted(csv_files):
    filepath = os.path.join(OUTPUT_DIR, f)
    df = pd.read_csv(filepath)
    print(f"  - {f}: {len(df)}行, {len(df.columns)}列")

print("\n" + "=" * 60)
print("数据获取完成！")
print("=" * 60)