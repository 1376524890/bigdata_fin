#!/usr/bin/env python3
"""
论文图表生成脚本
用于生成论文所需的描述性统计表和相关性矩阵
"""

import pandas as pd
import numpy as np
from scipy import stats

# 设置中文显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试找到中文字体
try:
    # 尝试使用系统自带的中文字体
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # 文泉驿正黑
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto CJK
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            font_found = True
            break
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    pass

plt.rcParams['axes.unicode_minus'] = False

import os

def load_hs300_data():
    """加载沪深300数据"""
    df = pd.read_csv('/home/marktom/bigdata-fin/real_data/01_指数数据/hs300_daily.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def load_sentiment_data():
    """加载市场情绪数据"""
    df = pd.read_csv('/home/marktom/bigdata-fin/real_data/03_市场情绪/market_sentiment_history.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def load_north_money():
    """加载北向资金数据"""
    df = pd.read_csv('/home/marktom/bigdata-fin/real_data/03_市场情绪/north_money_hist.csv')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    return df

def load_cpi_data():
    """加载CPI数据"""
    df = pd.read_csv('/home/marktom/bigdata-fin/real_data/04_宏观指标/cpi.csv')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    return df

def generate_descriptive_stats():
    """生成描述性统计表数据"""
    # 加载数据
    hs300 = load_hs300_data()
    sentiment = load_sentiment_data()
    north = load_north_money()

    # 筛选样本期 2015-2025
    start_date = '2015-07-01'
    end_date = '2025-12-31'

    hs300 = hs300[(hs300['date'] >= start_date) & (hs300['date'] <= end_date)]
    sentiment = sentiment[(sentiment['date'] >= start_date) & (sentiment['date'] <= end_date)]
    north = north[(north['日期'] >= start_date) & (north['日期'] <= end_date)]

    stats_data = []

    # 沪深300收盘价
    close_stats = hs300['close'].describe()
    stats_data.append({
        '变量': '沪深300收盘价',
        '观测数': len(hs300),
        '均值': hs300['close'].mean(),
        '标准差': hs300['close'].std(),
        '最小值': hs300['close'].min(),
        '最大值': hs300['close'].max(),
        '偏度': hs300['close'].skew(),
        '峰度': hs300['close'].kurtosis()
    })

    # 日收益率
    returns = hs300['market_return'].dropna()
    stats_data.append({
        '变量': '日对数收益率',
        '观测数': len(returns),
        '均值': returns.mean() * 100,  # 转换为百分比
        '标准差': returns.std() * 100,
        '最小值': returns.min() * 100,
        '最大值': returns.max() * 100,
        '偏度': returns.skew(),
        '峰度': returns.kurtosis()
    })

    # 计算未来5日和60日累计收益率
    hs300_copy = hs300.copy()
    hs300_copy['future_5d'] = hs300_copy['close'].shift(-5) / hs300_copy['close'] - 1
    hs300_copy['future_60d'] = hs300_copy['close'].shift(-60) / hs300_copy['close'] - 1

    for col, name in [('future_5d', '未来5日累计收益率'), ('future_60d', '未来60日累计收益率')]:
        data = hs300_copy[col].dropna() * 100
        stats_data.append({
            '变量': name,
            '观测数': len(data),
            '均值': data.mean(),
            '标准差': data.std(),
            '最小值': data.min(),
            '最大值': data.max(),
            '偏度': data.skew(),
            '峰度': data.kurtosis()
        })

    # 情绪指标
    if not sentiment.empty:
        for col, name in [('sentiment_zscore', '情绪标准分'), ('intraday_range', '日内振幅')]:
            if col in sentiment.columns:
                data = sentiment[col].dropna()
                if col == 'intraday_range':
                    data = data * 100  # 转换为百分比
                stats_data.append({
                    '变量': name,
                    '观测数': len(data),
                    '均值': data.mean(),
                    '标准差': data.std(),
                    '最小值': data.min(),
                    '最大值': data.max(),
                    '偏度': data.skew(),
                    '峰度': data.kurtosis()
                })

    # 北向资金
    if not north.empty:
        data = north['当日成交净买额'].dropna()
        stats_data.append({
            '变量': '北向资金净流入（亿元）',
            '观测数': len(data),
            '均值': data.mean(),
            '标准差': data.std(),
            '最小值': data.min(),
            '最大值': data.max(),
            '偏度': data.skew(),
            '峰度': data.kurtosis()
        })

    df_stats = pd.DataFrame(stats_data)
    return df_stats

def generate_correlation_tables():
    """生成相关性矩阵"""
    # 这里使用论文中提供的数据
    macro_corr = pd.DataFrame({
        'CPI': [1.000, 0.168, 0.126, 0.001, -0.157],
        'PPI': [0.168, 1.000, -0.279, -0.167, -0.421],
        'M2增速': [0.126, -0.279, 1.000, -0.352, -0.264],
        'EPU': [0.001, -0.167, -0.352, 1.000, 0.676],
        '美元兑人民币汇率': [-0.157, -0.421, -0.264, 0.676, 1.000]
    }, index=['CPI', 'PPI', 'M2增速', 'EPU', '美元兑人民币汇率'])

    market_corr = pd.DataFrame({
        '情绪标准分': [1.000, 0.013, 0.122, -0.017, 0.102, 0.157, 0.043],
        'iVIX': [0.013, 1.000, -0.014, -0.111, 0.247, -0.178, 0.661],
        '北向资金': [0.122, -0.014, 1.000, -0.352, 0.082, 0.074, -0.004],
        '融资融券余额': [-0.017, -0.111, -0.352, 1.000, -0.209, 0.007, -0.021],
        'Amihud': [0.102, 0.247, 0.082, -0.209, 1.000, -0.232, 0.484],
        '20日动量': [0.157, -0.178, 0.074, 0.007, -0.232, 1.000, -0.230],
        '日内振幅': [0.043, 0.661, -0.004, -0.021, 0.484, -0.230, 1.000]
    }, index=['情绪标准分', 'iVIX', '北向资金', '融资融券余额', 'Amihud', '20日动量', '日内振幅'])

    return macro_corr, market_corr

def save_latex_tables():
    """保存LaTeX表格代码"""
    output_dir = '/home/marktom/bigdata-fin/latex_paper'
    os.makedirs(output_dir, exist_ok=True)

    # 描述性统计表
    stats_df = generate_descriptive_stats()
    latex_code = stats_df.to_latex(index=False, float_format='%.2f')
    with open(f'{output_dir}/desc_stats.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print("描述性统计表已保存")

    # 相关性表
    macro_corr, market_corr = generate_correlation_tables()

    macro_latex = macro_corr.to_latex(float_format='%.3f')
    with open(f'{output_dir}/macro_corr.tex', 'w', encoding='utf-8') as f:
        f.write(macro_latex)
    print("宏观变量相关性表已保存")

    market_latex = market_corr.to_latex(float_format='%.3f')
    with open(f'{output_dir}/market_corr.tex', 'w', encoding='utf-8') as f:
        f.write(market_latex)
    print("市场状态变量相关性表已保存")

if __name__ == '__main__':
    print("开始生成论文统计数据...")
    save_latex_tables()
    print("完成！")
