#!/usr/bin/env python3
"""
阶段1：数据获取
整合所有数据获取功能，包括指数数据、宏观数据、市场情绪数据
"""

import pandas as pd
import numpy as np
import os
import warnings
import akshare as ak
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_index_data():
    """获取指数数据"""
    print("\n【1】获取指数数据...")

    indices = {
        'sh000300': 'hs300_daily.csv',      # 沪深300
        'sh000001': 'sh_index_daily.csv',   # 上证指数
        'sz399001': 'sz_index_daily.csv',   # 深证成指
        'sz399006': 'cyb_index_daily.csv',  # 创业板指
    }

    for symbol, filename in indices.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df['date'] = pd.to_datetime(df['date'])
            df.to_csv(f'{OUTPUT_DIR}/{filename}', index=False)
            print(f"   {symbol}: {len(df)}条记录")
        except Exception as e:
            print(f"   {symbol}失败: {e}")


def fetch_macro_data():
    """获取宏观经济数据"""
    print("\n【2】获取宏观数据...")

    macro_funcs = [
        ('gdp.csv', ak.macro_china_gdp, 'GDP'),
        ('cpi.csv', ak.macro_china_cpi_yearly, 'CPI'),
        ('ppi.csv', ak.macro_china_ppi_yearly, 'PPI'),
        ('m2.csv', ak.macro_china_m2_yearly, 'M2'),
        ('social_financing.csv', ak.macro_china_shrzgm, '社融'),
        ('lpr.csv', ak.macro_china_lpr, 'LPR'),
        ('epu_index.csv', ak.article_epu_index, 'EPU'),
    ]

    for filename, func, name in macro_funcs:
        try:
            df = func()
            df.to_csv(f'{OUTPUT_DIR}/{filename}', index=False)
            print(f"   {name}: {len(df)}条记录")
        except Exception as e:
            print(f"   {name}失败: {e}")


def fetch_market_sentiment_data():
    """获取市场情绪数据"""
    print("\n【3】获取市场情绪数据...")

    sentiment_funcs = [
        ('ivix_50etf.csv', ak.index_option_50etf_qvix, '50ETF波指'),
        ('ivix_300etf.csv', ak.index_option_300etf_qvix, '300ETF波指'),
        ('north_money_hist.csv', ak.stock_hsgt_hist_em, '北向资金'),
        ('margin_account.csv', ak.stock_margin_account_info, '融资融券'),
        ('investor_account_stats.csv', ak.stock_account_statistics_em, '投资者账户'),
    ]

    for filename, func, name in sentiment_funcs:
        try:
            df = func()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            df.to_csv(f'{OUTPUT_DIR}/{filename}', index=False)
            print(f"   {name}: {len(df)}条记录")
        except Exception as e:
            print(f"   {name}失败: {e}")


def fetch_fx_data():
    """获取汇率数据"""
    print("\n【4】获取汇率数据...")

    try:
        boc_rate = ak.currency_boc_safe()
        boc_rate.to_csv(f'{OUTPUT_DIR}/boc_rate.csv', index=False)
        print(f"   央行汇率: {len(boc_rate)}条记录")
    except Exception as e:
        print(f"   汇率失败: {e}")


def generate_sentiment_index():
    """生成情绪指标（基于已有数据计算）"""
    print("\n【5】生成情绪指标...")

    try:
        # 读取沪深300数据
        hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300 = hs300.sort_values('date')

        # 计算情绪代理指标
        hs300['return'] = hs300['close'].pct_change()
        hs300['volatility_20d'] = hs300['return'].rolling(20).std()
        hs300['volatility_60d'] = hs300['return'].rolling(60).std()
        hs300['momentum_5d'] = hs300['return'].rolling(5).sum()
        hs300['momentum_10d'] = hs300['return'].rolling(10).sum()
        hs300['momentum_20d'] = hs300['return'].rolling(20).sum()
        hs300['volume_ma20'] = hs300['volume'].rolling(20).mean()
        hs300['volume_ratio_20d'] = hs300['volume'] / hs300['volume_ma20']
        hs300['intraday_range'] = (hs300['high'] - hs300['low']) / hs300['close']

        # 综合情绪指标
        hs300['sentiment_raw'] = hs300['momentum_5d'] * np.log(hs300['volume_ratio_20d'].clip(0.5, 3))
        window = 252
        hs300['sentiment_mean'] = hs300['sentiment_raw'].rolling(window).mean()
        hs300['sentiment_std'] = hs300['sentiment_raw'].rolling(window).std()
        hs300['sentiment_zscore'] = (hs300['sentiment_raw'] - hs300['sentiment_mean']) / hs300['sentiment_std']

        # 情绪等级分类
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

        # 保存
        sentiment_cols = ['date', 'close', 'return', 'volatility_20d', 'volatility_60d',
                         'momentum_5d', 'momentum_10d', 'momentum_20d',
                         'volume_ratio_20d', 'intraday_range',
                         'sentiment_zscore', 'sentiment_level']
        sentiment_df = hs300[sentiment_cols].dropna()
        sentiment_df.to_csv(f'{OUTPUT_DIR}/market_sentiment_history.csv', index=False)
        print(f"   情绪指标: {len(sentiment_df)}条记录")

    except Exception as e:
        print(f"   情绪指标生成失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("阶段1：数据获取")
    print("=" * 60)

    fetch_index_data()
    fetch_macro_data()
    fetch_market_sentiment_data()
    fetch_fx_data()
    generate_sentiment_index()

    print("\n" + "=" * 60)
    print("数据获取完成！")
    print(f"数据保存位置: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
