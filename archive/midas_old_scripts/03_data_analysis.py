#!/usr/bin/env python3
"""
阶段3：数据分析
生成描述统计、相关性分析、缺失值分析等
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = '/home/marktom/bigdata-fin/real_data_complete.csv'
OUTPUT_DIR = '/home/marktom/bigdata-fin'


def load_data():
    """加载数据"""
    print("\n【1】加载数据...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"样本区间: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"总观测数: {len(df)}")
    return df


def construct_future_returns(df):
    """构造未来收益率"""
    print("\n【2】构造未来累计收益率...")

    # 计算日对数收益率
    df['daily_return'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))

    # 计算未来收益率
    df['future_return_5d'] = np.log(df['hs300_close'].shift(-5) / df['hs300_close'])
    df['future_return_60d'] = np.log(df['hs300_close'].shift(-60) / df['hs300_close'])

    # 去除尾部缺失
    df_analysis = df[df['future_return_5d'].notna() & df['future_return_60d'].notna()].copy()

    print(f"用于分析的样本量: {len(df_analysis)}")
    print(f"未来5日收益率 - 均值: {df_analysis['future_return_5d'].mean():.6f}, 标准差: {df_analysis['future_return_5d'].std():.6f}")
    print(f"未来60日收益率 - 均值: {df_analysis['future_return_60d'].mean():.6f}, 标准差: {df_analysis['future_return_60d'].std():.6f}")

    return df_analysis


def descriptive_statistics(df):
    """描述性统计"""
    print("\n【3】描述性统计...")

    # 核心变量
    core_vars = ['future_return_5d', 'future_return_60d', 'daily_return',
                 'cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny',
                 'sentiment_zscore', 'ivix', 'north_flow', 'margin_balance',
                 'amihud', 'momentum_20d', 'intraday_range']

    available_vars = [v for v in core_vars if v in df.columns]
    desc_stats = df[available_vars].describe().T
    desc_stats['median'] = df[available_vars].median()
    desc_stats['skewness'] = df[available_vars].skew()
    desc_stats['kurtosis'] = df[available_vars].kurtosis()

    # 保存
    desc_stats.to_csv(f'{OUTPUT_DIR}/desc_stats.csv')
    print(f"   描述统计已保存: {OUTPUT_DIR}/desc_stats.csv")

    print("\n核心变量统计:")
    print(desc_stats.round(6).to_string())

    return desc_stats


def missing_value_analysis(df):
    """缺失值分析"""
    print("\n【4】缺失值统计...")

    all_vars = ['date', 'hs300_close', 'daily_return', 'future_return_5d', 'future_return_60d',
                'cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny',
                'sentiment_zscore', 'ivix', 'north_flow', 'margin_balance', 'amihud', 'intraday_range']

    available_vars = [v for v in all_vars if v in df.columns]

    missing_stats = pd.DataFrame({
        '变量名': available_vars,
        '缺失数': [df[v].isna().sum() for v in available_vars],
        '缺失率(%)': [df[v].isna().sum() / len(df) * 100 for v in available_vars],
        '有效观测数': [df[v].notna().sum() for v in available_vars]
    })

    missing_stats.to_csv(f'{OUTPUT_DIR}/missing_stats.csv', index=False)
    print(f"   缺失值统计已保存: {OUTPUT_DIR}/missing_stats.csv")
    print(missing_stats.to_string(index=False))

    return missing_stats


def correlation_analysis(df):
    """相关性分析"""
    print("\n【5】相关性分析...")

    # 宏观变量间相关性
    macro_vars = ['cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny']
    macro_available = [v for v in macro_vars if v in df.columns]
    if len(macro_available) > 1:
        print("\n【宏观变量间相关性】")
        macro_corr = df[macro_available].corr()
        macro_corr.to_csv(f'{OUTPUT_DIR}/macro_correlation.csv')
        print(macro_corr.round(4).to_string())

    # 市场状态变量间相关性
    market_vars = ['sentiment_zscore', 'ivix', 'north_flow', 'margin_balance', 'amihud', 'intraday_range']
    market_available = [v for v in market_vars if v in df.columns]
    if len(market_available) > 1:
        print("\n【市场状态变量间相关性】")
        market_corr = df[market_available].corr()
        market_corr.to_csv(f'{OUTPUT_DIR}/market_correlation.csv')
        print(market_corr.round(4).to_string())

    # 与未来收益相关性
    print("\n【宏观变量与未来收益相关性】")
    predictive_vars = ['future_return_5d', 'future_return_60d'] + macro_available
    pred_corr = df[predictive_vars].corr()[['future_return_5d', 'future_return_60d']]
    pred_corr.to_csv(f'{OUTPUT_DIR}/predictive_correlation.csv')
    print(pred_corr.round(4).to_string())


def train_test_split(df):
    """训练集/测试集划分"""
    print("\n【6】训练集/测试集划分...")

    train_ratio = 0.6
    train_cutoff = int(len(df) * train_ratio)
    train_df = df.iloc[:train_cutoff]
    test_df = df.iloc[train_cutoff:]

    print(f"总样本量: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  时间区间: {train_df['date'].min().strftime('%Y-%m-%d')} 至 {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  时间区间: {test_df['date'].min().strftime('%Y-%m-%d')} 至 {test_df['date'].max().strftime('%Y-%m-%d')}")

    return train_df, test_df


def main():
    """主函数"""
    print("=" * 60)
    print("阶段3：数据分析")
    print("=" * 60)

    df = load_data()
    df = construct_future_returns(df)
    descriptive_statistics(df)
    missing_value_analysis(df)
    correlation_analysis(df)
    train_test_split(df)

    print("\n" + "=" * 60)
    print("数据分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
