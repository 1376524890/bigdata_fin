#!/usr/bin/env python3
"""
情绪数据爬虫脚本 V2 - 优化版
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import random
import re
import os
import warnings
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'

print("=" * 60)
print("情绪数据爬虫脚本 V2")
print("=" * 60)

# ============================================================================
# 1. 东方财富行情数据（计算市场情绪）
# ============================================================================
print("\n【1】东方财富行情数据...")

def get_em_market_breadth():
    """获取市场广度数据（涨跌比）"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://quote.eastmoney.com/',
    }

    # A股列表API
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"

    params = {
        'pn': 1,
        'pz': 5000,
        'po': 1,
        'np': 1,
        'fltt': 2,
        'invt': 2,
        'fid': 'f3',
        'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18'
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('diff'):
                stocks = data['data']['diff']

                up = down = limit_up = limit_down = 0
                changes = []

                for s in stocks:
                    try:
                        chg = float(s.get('f3', 0)) if s.get('f3') else 0
                        changes.append(chg)
                        if chg > 0:
                            up += 1
                            if chg >= 9.8:
                                limit_up += 1
                        elif chg < 0:
                            down += 1
                            if chg <= -9.8:
                                limit_down += 1
                    except:
                        continue

                total = up + down
                result = {
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'total_stocks': total,
                    'up_count': up,
                    'down_count': down,
                    'limit_up': limit_up,
                    'limit_down': limit_down,
                    'up_ratio': up / total if total > 0 else 0.5,
                    'avg_change': np.mean(changes) if changes else 0,
                    'market_breadth': (up - down) / total if total > 0 else 0
                }

                return result
    except Exception as e:
        print(f"   错误: {e}")

    return None

market_data = get_em_market_breadth()
if market_data:
    df = pd.DataFrame([market_data])
    df.to_csv(f'{OUTPUT_DIR}/em_market_breadth.csv', index=False)
    print(f"   上涨: {market_data['up_count']}, 下跌: {market_data['down_count']}")
    print(f"   涨停: {market_data['limit_up']}, 跌停: {market_data['limit_down']}")
    print(f"   市场广度: {market_data['market_breadth']:.3f}")

# ============================================================================
# 2. 东方财富融资融券情绪
# ============================================================================
print("\n【2】融资融券数据...")

def get_margin_data():
    """获取融资融券数据"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    # 融资融券余额API
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"

    params = {
        'sortColumns': 'TRADE_DATE',
        'sortTypes': -1,
        'pageSize': 100,
        'pageNumber': 1,
        'reportName': 'RPT_RZRQ_LSHJ',
        'columns': 'ALL',
        'source': 'WEB',
        'client': 'WEB'
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('result') and data['result'].get('data'):
                records = data['result']['data']
                result = []
                for r in records:
                    result.append({
                        'date': r.get('TRADE_DATE', ''),
                        'margin_balance': r.get('RZYE', 0),  # 融资余额
                        'short_balance': r.get('RQYE', 0),   # 融券余额
                        'total_balance': r.get('ZRZYE', 0),  # 总余额
                        'margin_buy': r.get('RZMRE', 0),     # 融资买入额
                        'short_sell': r.get('RQMCL', 0),     # 融券卖出额
                    })
                return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

margin_df = get_margin_data()
if len(margin_df) > 0:
    margin_df['date'] = pd.to_datetime(margin_df['date']).dt.strftime('%Y-%m-%d')
    margin_df.to_csv(f'{OUTPUT_DIR}/em_margin_data.csv', index=False)
    print(f"   融资融券历史: {len(margin_df)}条")
    # 计算融资买入占市场成交比例作为情绪指标
    print(f"   最新融资余额: {margin_df.iloc[0]['margin_balance']:.2f}亿")

# ============================================================================
# 3. 东方财富行业资金流向
# ============================================================================
print("\n【3】行业资金流向...")

def get_industry_flow():
    """获取行业资金流向"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://data.eastmoney.com/bkzj/hy.html',
    }

    url = "http://push2.eastmoney.com/api/qt/clist/get"

    params = {
        'fid': 'f62',
        'po': 1,
        'pz': 100,
        'pn': 1,
        'np': 1,
        'fltt': 2,
        'invt': 2,
        'fs': 'm:90+t:2',
        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87'
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('diff'):
                industries = data['data']['diff']
                result = []
                for ind in industries:
                    try:
                        result.append({
                            'industry': ind.get('f14', ''),
                            'main_inflow': float(ind.get('f62', 0)) if ind.get('f62') else 0,
                            'main_inflow_ratio': float(ind.get('f184', 0)) if ind.get('f184') else 0,
                            'change_pct': float(ind.get('f3', 0)) if ind.get('f3') else 0,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
                    except:
                        continue

                df = pd.DataFrame(result)
                # 计算资金流向情绪
                total_inflow = df['main_inflow'].sum()
                positive_count = len(df[df['main_inflow'] > 0])
                negative_count = len(df[df['main_inflow'] < 0])

                return df, {
                    'total_main_inflow': total_inflow,
                    'positive_industries': positive_count,
                    'negative_industries': negative_count,
                    'flow_sentiment': positive_count / len(df) if len(df) > 0 else 0.5
                }
    except Exception as e:
        print(f"   错误: {e}")

    return None, None

industry_df, flow_sentiment = get_industry_flow()
if industry_df is not None and len(industry_df) > 0:
    industry_df.to_csv(f'{OUTPUT_DIR}/em_industry_flow.csv', index=False)
    print(f"   行业资金流向: {len(industry_df)}条")
    if flow_sentiment:
        print(f"   主力净流入: {flow_sentiment['total_main_inflow']/1e8:.2f}亿")
        print(f"   流入行业数: {flow_sentiment['positive_industries']}, 流出: {flow_sentiment['negative_industries']}")

# ============================================================================
# 4. 新浪财经大盘数据
# ============================================================================
print("\n【4】新浪财经大盘数据...")

def get_sina_index():
    """获取新浪大盘数据"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://finance.sina.com.cn/',
    }

    url = "https://hq.sinajs.cn/list=sh000001,sz399001,sh000300,sz399006"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            result = []

            for line in lines:
                if 'var hq_str_' in line:
                    code = line.split('hq_str_')[1].split('=')[0]
                    data_str = line.split('="')[1].rstrip('";')
                    parts = data_str.split(',')

                    if len(parts) >= 10:
                        name = parts[0]
                        current = float(parts[3]) if parts[3] else 0
                        last_close = float(parts[2]) if parts[2] else 0
                        change = (current - last_close) / last_close * 100 if last_close > 0 else 0

                        result.append({
                            'code': code,
                            'name': name,
                            'current': current,
                            'last_close': last_close,
                            'change_pct': change,
                            'volume': parts[8],
                            'amount': parts[9],
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })

            return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

sina_index = get_sina_index()
if len(sina_index) > 0:
    sina_index.to_csv(f'{OUTPUT_DIR}/sina_index_data.csv', index=False)
    print(f"   大盘数据: {len(sina_index)}条")
    for _, row in sina_index.iterrows():
        print(f"   {row['name']}: {row['current']:.2f} ({row['change_pct']:+.2f}%)")

# ============================================================================
# 5. 腾讯财经情绪指标
# ============================================================================
print("\n【5】腾讯财经情绪指标...")

def get_tencent_market():
    """获取腾讯财经市场数据"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    url = "https://web.sqt.gtimg.cn/q=r_sh000001,r_sz399001,r_sh000300"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            result = []

            for line in lines:
                if 'v_' in line:
                    data_str = line.split('~')
                    if len(data_str) >= 35:
                        result.append({
                            'code': data_str[2],
                            'name': data_str[1],
                            'current': float(data_str[3]) if data_str[3] else 0,
                            'last_close': float(data_str[4]) if data_str[4] else 0,
                            'change_pct': float(data_str[5]) if data_str[5] else 0,
                            'volume': data_str[6],
                            'amount': data_str[37] if len(data_str) > 37 else 0,
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })

            return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

tencent_data = get_tencent_market()
if len(tencent_data) > 0:
    tencent_data.to_csv(f'{OUTPUT_DIR}/tencent_index_data.csv', index=False)
    print(f"   腾讯数据: {len(tencent_data)}条")

# ============================================================================
# 6. 同花顺资金流向
# ============================================================================
print("\n【6】同花顺资金流向...")

def get_thshy_moneyflow():
    """获取同花顺资金流向"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://data.10jqka.com.cn/',
    }

    url = "http://q.10jqka.com.cn/gn/index/index/all/zdf/desc/1/"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', class_='m-table')

            if table:
                rows = table.find_all('tr')[1:]  # 跳过表头
                result = []

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        try:
                            result.append({
                                'concept': cols[1].text.strip(),
                                'change_pct': cols[2].text.strip(),
                                'main_inflow': cols[3].text.strip(),
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
                        except:
                            continue

                if result:
                    return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

thshy_data = get_thshy_moneyflow()
if len(thshy_data) > 0:
    thshy_data.to_csv(f'{OUTPUT_DIR}/thshy_concept_flow.csv', index=False)
    print(f"   同花顺概念资金流: {len(thshy_data)}条")

# ============================================================================
# 7. 计算综合情绪指数
# ============================================================================
print("\n【7】计算综合情绪指数...")

sentiment_components = {}

# 市场广度情绪
if market_data:
    sentiment_components['market_breadth'] = market_data['market_breadth']

# 资金流向情绪
if flow_sentiment:
    sentiment_components['capital_flow'] = flow_sentiment['flow_sentiment'] * 2 - 1  # 转换到-1到1

# 计算综合情绪
if sentiment_components:
    composite = np.mean(list(sentiment_components.values()))

    result = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'market_breadth': sentiment_components.get('market_breadth', 0),
        'capital_flow': sentiment_components.get('capital_flow', 0),
        'composite_sentiment': composite,
        'level': '极度贪婪' if composite > 0.4 else ('贪婪' if composite > 0.2 else ('中性' if abs(composite) <= 0.2 else ('恐惧' if composite > -0.4 else '极度恐惧')))
    }

    df = pd.DataFrame([result])
    df.to_csv(f'{OUTPUT_DIR}/composite_sentiment.csv', index=False)

    print(f"\n   ===== 情绪指数 =====")
    print(f"   市场广度情绪: {result['market_breadth']:.3f}")
    print(f"   资金流向情绪: {result['capital_flow']:.3f}")
    print(f"   综合情绪指数: {result['composite_sentiment']:.3f}")
    print(f"   情绪等级: {result['level']}")

# ============================================================================
# 8. 历史情绪数据（模拟，用于回归）
# ============================================================================
print("\n【8】生成历史情绪数据序列...")

# 基于已有数据计算历史情绪
try:
    # 读取沪深300数据
    hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
    hs300['date'] = pd.to_datetime(hs300['date'])

    # 使用收益率和波动率构建情绪代理指标
    hs300['return'] = hs300['close'].pct_change()
    hs300['volatility'] = hs300['return'].rolling(20).std()

    # 情绪代理：动量因子
    hs300['momentum_5d'] = hs300['return'].rolling(5).sum()
    hs300['momentum_20d'] = hs300['return'].rolling(20).sum()

    # 成交量变化
    hs300['volume_ma20'] = hs300['volume'].rolling(20).mean()
    hs300['volume_ratio'] = hs300['volume'] / hs300['volume_ma20']

    # 综合情绪指标（简化版）
    # 正收益 + 放量 = 贪婪
    # 负收益 + 放量 = 恐惧
    hs300['sentiment_proxy'] = hs300['momentum_5d'] * hs300['volume_ratio']

    # 标准化
    hs300['sentiment_zscore'] = (hs300['sentiment_proxy'] - hs300['sentiment_proxy'].rolling(252).mean()) / hs300['sentiment_proxy'].rolling(252).std()

    # 保存
    sentiment_history = hs300[['date', 'sentiment_proxy', 'sentiment_zscore', 'momentum_5d', 'momentum_20d', 'volume_ratio']].dropna()
    sentiment_history.to_csv(f'{OUTPUT_DIR}/historical_sentiment.csv', index=False)

    print(f"   历史情绪数据: {len(sentiment_history)}条")
    print(f"   日期范围: {sentiment_history['date'].min()} ~ {sentiment_history['date'].max()}")

except Exception as e:
    print(f"   错误: {e}")

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 60)
print("情绪数据爬取完成！")
print("=" * 60)

# 列出生成的文件
sentiment_files = ['em_market_breadth.csv', 'em_margin_data.csv', 'em_industry_flow.csv',
                   'sina_index_data.csv', 'tencent_index_data.csv', 'thshy_concept_flow.csv',
                   'composite_sentiment.csv', 'historical_sentiment.csv']

print("\n生成的数据文件:")
for f in sentiment_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"  ✓ {f}: {len(df)}条")
        except:
            print(f"  ✗ {f}: 读取失败")
    else:
        print(f"  - {f}: 未生成")