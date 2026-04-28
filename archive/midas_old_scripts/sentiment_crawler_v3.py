#!/usr/bin/env python3
"""
情绪数据爬虫脚本 V3 - 最终版
整合多个数据源，生成完整的情绪指标
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
print("情绪数据爬虫脚本 V3（最终版）")
print("=" * 60)

# ============================================================================
# 1. 新浪财经大盘数据（稳定）
# ============================================================================
print("\n【1】新浪财经大盘数据...")

def get_sina_index():
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
                        high = float(parts[4]) if parts[4] else 0
                        low = float(parts[5]) if parts[5] else 0
                        volume = int(parts[8]) if parts[8] else 0
                        amount = float(parts[9]) if parts[9] else 0
                        change = (current - last_close) / last_close * 100 if last_close > 0 else 0
                        amplitude = (high - low) / last_close * 100 if last_close > 0 else 0

                        result.append({
                            'code': code,
                            'name': name,
                            'current': current,
                            'last_close': last_close,
                            'change_pct': change,
                            'amplitude': amplitude,
                            'volume': volume,
                            'amount': amount,
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })

            return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")
    return pd.DataFrame()

sina_index = get_sina_index()
if len(sina_index) > 0:
    sina_index.to_csv(f'{OUTPUT_DIR}/sina_realtime_index.csv', index=False)
    print(f"   成功获取: {len(sina_index)}条")
    for _, row in sina_index.iterrows():
        print(f"   {row['name']}: {row['current']:.2f} ({row['change_pct']:+.2f}%)")

# ============================================================================
# 2. 东方财富涨停跌停统计
# ============================================================================
print("\n【2】东方财富涨跌停数据...")

def get_em_zt_dt():
    """获取涨停跌停数据"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    results = {'date': datetime.now().strftime('%Y-%m-%d')}

    # 涨停
    try:
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        params = {
            'fid': 'f3',
            'po': 1,
            'pz': 500,
            'pn': 1,
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
            'fields': 'f1,f2,f3,f4,f5,f6,f12,f14'
        }

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('diff'):
                stocks = data['data']['diff']
                zt_count = 0
                dt_count = 0

                for s in stocks:
                    try:
                        chg = float(s.get('f3', 0)) if s.get('f3') else 0
                        if chg >= 9.8:
                            zt_count += 1
                        elif chg <= -9.8:
                            dt_count += 1
                    except:
                        continue

                results['limit_up'] = zt_count
                results['limit_down'] = dt_count
                print(f"   涨停: {zt_count}, 跌停: {dt_count}")
    except Exception as e:
        print(f"   错误: {e}")
        results['limit_up'] = 0
        results['limit_down'] = 0

    return results

zt_dt_data = get_em_zt_dt()

# ============================================================================
# 3. 金十数据市场快讯情绪
# ============================================================================
print("\n【3】金十数据市场快讯...")

def get_jin10_news():
    """获取金十数据快讯"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.jin10.com/',
    }

    url = "https://flash-api.jin10.com/get_flash_list"

    params = {
        'channel': '-8200',  # A股频道
        'vip': 1,
        'max_time': int(time.time())
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                news_list = []
                for item in data[:50]:  # 取最新50条
                    news_list.append({
                        'time': item.get('time', ''),
                        'content': item.get('content', ''),
                        'type': item.get('type', ''),
                        'date': datetime.now().strftime('%Y-%m-%d')
                    })
                return pd.DataFrame(news_list)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

jin10_news = get_jin10_news()
if len(jin10_news) > 0:
    jin10_news.to_csv(f'{OUTPUT_DIR}/jin10_news.csv', index=False)
    print(f"   快讯数量: {len(jin10_news)}条")

# ============================================================================
# 4. 财联社电报
# ============================================================================
print("\n【4】财联社电报...")

def get_cls_telegraph():
    """获取财联社电报"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.cls.cn/',
    }

    url = "https://www.cls.cn/nodeapi/telegraphs"

    params = {
        'app': 'CailianpressWeb',
        'os': 'web',
        'sv': '8.4.6',
        'rn': 30,
        'last_time': ''
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('roll_data'):
                news_list = []
                for item in data['data']['roll_data'][:50]:
                    news_list.append({
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'ctime': item.get('ctime', ''),
                        'date': datetime.now().strftime('%Y-%m-%d')
                    })
                return pd.DataFrame(news_list)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

cls_news = get_cls_telegraph()
if len(cls_news) > 0:
    cls_news.to_csv(f'{OUTPUT_DIR}/cls_telegraph.csv', index=False)
    print(f"   电报数量: {len(cls_news)}条")

# ============================================================================
# 5. 同花顺概念板块资金流
# ============================================================================
print("\n【5】同花顺概念资金流...")

def get_ths_concept_flow():
    """获取同花顺概念板块资金流向"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://data.10jqka.com.cn/funds/gnzjl/',
    }

    url = "http://data.10jqka.com.cn/funds/gnzjl/data/hsboard/data/gnzjl-board-data.json"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data'):
                result = []
                for item in data['data']:
                    result.append({
                        'concept': item.get('name', ''),
                        'change_pct': item.get('hszdf', ''),
                        'main_inflow': item.get('zljlr', ''),
                        'main_inflow_ratio': item.get('zljzb', ''),
                        'date': datetime.now().strftime('%Y-%m-%d')
                    })
                return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

ths_flow = get_ths_concept_flow()
if len(ths_flow) > 0:
    ths_flow.to_csv(f'{OUTPUT_DIR}/ths_concept_flow.csv', index=False)
    print(f"   概念板块: {len(ths_flow)}条")

# ============================================================================
# 6. 计算历史情绪序列
# ============================================================================
print("\n【6】计算历史情绪序列...")

try:
    # 读取沪深300数据
    hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
    hs300['date'] = pd.to_datetime(hs300['date'])

    # 计算各种情绪代理指标
    hs300['return'] = hs300['close'].pct_change()
    hs300['volatility_20d'] = hs300['return'].rolling(20).std()
    hs300['volatility_60d'] = hs300['return'].rolling(60).std()

    # 动量
    hs300['momentum_5d'] = hs300['return'].rolling(5).sum()
    hs300['momentum_10d'] = hs300['return'].rolling(10).sum()
    hs300['momentum_20d'] = hs300['return'].rolling(20).sum()

    # 成交量比率
    hs300['volume_ma5'] = hs300['volume'].rolling(5).mean()
    hs300['volume_ma20'] = hs300['volume'].rolling(20).mean()
    hs300['volume_ratio_5d'] = hs300['volume'] / hs300['volume_ma5']
    hs300['volume_ratio_20d'] = hs300['volume'] / hs300['volume_ma20']

    # 日内振幅
    hs300['intraday_range'] = (hs300['high'] - hs300['low']) / hs300['close']

    # 综合情绪指标（标准化）
    # 方法：收益率动量 × 成交量放大
    hs300['sentiment_raw'] = hs300['momentum_5d'] * np.log(hs300['volume_ratio_5d'].clip(0.5, 3))

    # 滚动标准化
    window = 252
    hs300['sentiment_mean'] = hs300['sentiment_raw'].rolling(window).mean()
    hs300['sentiment_std'] = hs300['sentiment_raw'].rolling(window).std()
    hs300['sentiment_zscore'] = (hs300['sentiment_raw'] - hs300['sentiment_mean']) / hs300['sentiment_std']

    # 情绪等级
    def classify_sentiment(z):
        if pd.isna(z):
            return '未知'
        elif z > 2:
            return '极度贪婪'
        elif z > 1:
            return '贪婪'
        elif z > -1:
            return '中性'
        elif z > -2:
            return '恐惧'
        else:
            return '极度恐惧'

    hs300['sentiment_level'] = hs300['sentiment_zscore'].apply(classify_sentiment)

    # 选择输出列
    sentiment_cols = ['date', 'close', 'return', 'volatility_20d', 'volatility_60d',
                      'momentum_5d', 'momentum_10d', 'momentum_20d',
                      'volume_ratio_5d', 'volume_ratio_20d', 'intraday_range',
                      'sentiment_raw', 'sentiment_zscore', 'sentiment_level']

    sentiment_df = hs300[sentiment_cols].copy()
    sentiment_df = sentiment_df.dropna()

    sentiment_df.to_csv(f'{OUTPUT_DIR}/market_sentiment_history.csv', index=False)
    print(f"   历史情绪数据: {len(sentiment_df)}条")
    print(f"   日期范围: {sentiment_df['date'].min()} ~ {sentiment_df['date'].max()}")

    # 统计情绪分布
    sentiment_dist = sentiment_df['sentiment_level'].value_counts()
    print(f"\n   情绪分布:")
    for level, count in sentiment_dist.items():
        print(f"   {level}: {count}条 ({count/len(sentiment_df)*100:.1f}%)")

    # 最新情绪
    latest = sentiment_df.iloc[-1]
    print(f"\n   最新情绪指标:")
    print(f"   日期: {latest['date']}")
    print(f"   情绪Z值: {latest['sentiment_zscore']:.3f}")
    print(f"   情绪等级: {latest['sentiment_level']}")

except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 7. 生成今日情绪报告
# ============================================================================
print("\n【7】今日情绪报告...")

report = {
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'market_status': '交易中' if 9 <= datetime.now().hour < 15 else '已收盘'
}

# 添加大盘数据
if len(sina_index) > 0:
    for _, row in sina_index.iterrows():
        report[f"{row['code']}_price"] = row['current']
        report[f"{row['code']}_change"] = row['change_pct']

# 添加涨跌停数据
if zt_dt_data:
    report['limit_up'] = zt_dt_data.get('limit_up', 0)
    report['limit_down'] = zt_dt_data.get('limit_down', 0)

# 添加情绪指标
try:
    if 'sentiment_df' in dir() and len(sentiment_df) > 0:
        latest = sentiment_df.iloc[-1]
        report['sentiment_zscore'] = latest['sentiment_zscore']
        report['sentiment_level'] = latest['sentiment_level']
except:
    pass

# 保存报告
report_df = pd.DataFrame([report])
report_df.to_csv(f'{OUTPUT_DIR}/today_sentiment_report.csv', index=False)
print(f"   今日情绪报告已保存")

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 60)
print("情绪数据爬取完成！")
print("=" * 60)

# 列出所有情绪相关文件
print("\n生成的情绪数据文件:")
files = ['sina_realtime_index.csv', 'jin10_news.csv', 'cls_telegraph.csv',
         'ths_concept_flow.csv', 'market_sentiment_history.csv', 'today_sentiment_report.csv']

for f in files:
    filepath = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"  ✓ {f}: {len(df)}条")
        except:
            pass

print("\n【说明】")
print("  - market_sentiment_history.csv: 包含历史情绪序列，可直接用于回归分析")
print("  - sentiment_zscore: 标准化情绪指标（正值=贪婪，负值=恐惧）")
print("  - sentiment_level: 情绪等级分类")