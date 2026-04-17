#!/usr/bin/env python3
"""
阶段2：数据整合
整合各类数据，计算衍生指标，生成分析数据集
"""

import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'
FINAL_DIR = '/home/marktom/bigdata-fin'


def load_base_data():
    """加载基础数据"""
    print("\n【1】加载基础数据...")

    # 沪深300
    hs300 = pd.read_csv(f'{OUTPUT_DIR}/hs300_daily.csv')
    hs300['date'] = pd.to_datetime(hs300['date'])
    hs300 = hs300.rename(columns={'close': 'hs300_close', 'volume': 'hs300_volume'})

    # 上证指数
    sh = pd.read_csv(f'{OUTPUT_DIR}/sh_index_daily.csv')
    sh['date'] = pd.to_datetime(sh['date'])
    sh = sh.rename(columns={'close': 'sh_close', 'volume': 'sh_volume'})

    # 合并基础数据
    daily_data = hs300[['date', 'hs300_close', 'hs300_volume', 'open', 'high', 'low']].copy()
    daily_data = daily_data.merge(sh[['date', 'sh_close']], on='date', how='left')

    print(f"   基础数据: {len(daily_data)}条")
    return daily_data


def merge_sentiment_data(daily_data):
    """合并情绪数据"""
    print("\n【2】合并情绪数据...")

    # 情绪指标
    sentiment = pd.read_csv(f'{OUTPUT_DIR}/market_sentiment_history.csv')
    sentiment['date'] = pd.to_datetime(sentiment['date'])
    sentiment_cols = ['date', 'return', 'volatility_20d', 'volatility_60d',
                     'momentum_5d', 'momentum_10d', 'momentum_20d',
                     'volume_ratio_20d', 'intraday_range',
                     'sentiment_zscore', 'sentiment_level']
    sentiment_selected = sentiment[[c for c in sentiment_cols if c in sentiment.columns]].copy()
    daily_data = daily_data.merge(sentiment_selected, on='date', how='left')

    # iVIX
    try:
        ivix = pd.read_csv(f'{OUTPUT_DIR}/ivix_50etf.csv')
        ivix['date'] = pd.to_datetime(ivix['date'])
        ivix = ivix.rename(columns={'close': 'ivix'})
        daily_data = daily_data.merge(ivix[['date', 'ivix']], on='date', how='left')
    except:
        pass

    # 北向资金
    try:
        north = pd.read_csv(f'{OUTPUT_DIR}/north_money_hist.csv')
        north['date'] = pd.to_datetime(north['日期'])
        north['north_flow'] = north['当日成交净买额']
        daily_data = daily_data.merge(north[['date', 'north_flow']], on='date', how='left')
    except:
        pass

    # 融资融券
    try:
        margin = pd.read_csv(f'{OUTPUT_DIR}/margin_account.csv')
        margin['date'] = pd.to_datetime(margin['日期'])
        margin['margin_balance'] = margin['融资余额']
        daily_data = daily_data.merge(margin[['date', 'margin_balance']], on='date', how='left')
    except:
        pass

    # 汇率
    try:
        boc_rate = pd.read_csv(f'{OUTPUT_DIR}/boc_rate.csv')
        boc_rate['date'] = pd.to_datetime(boc_rate['日期'])
        boc_rate['usd_cny'] = boc_rate['美元']
        daily_data = daily_data.merge(boc_rate[['date', 'usd_cny']], on='date', how='left')
    except:
        pass

    print(f"   合并后: {len(daily_data)}条")
    return daily_data


def merge_macro_data(daily_data):
    """合并宏观数据（前向填充）"""
    print("\n【3】合并宏观数据...")

    # GDP - 季度数据
    try:
        gdp = pd.read_csv(f'{OUTPUT_DIR}/gdp.csv')
        def parse_quarter(s):
            match = re.search(r'(\d{4})年第(\d)', str(s))
            if match:
                year, q = int(match.group(1)), int(match.group(2))
                return pd.Timestamp(f"{year}-{(q-1)*3+1:02d}-01")
            return None
        gdp['date'] = gdp['季度'].apply(parse_quarter)
        gdp['gdp_growth'] = pd.to_numeric(gdp['国内生产总值-同比增长'], errors='coerce')
        gdp_data = gdp[['date', 'gdp_growth']].dropna()
    except:
        gdp_data = pd.DataFrame(columns=['date', 'gdp_growth'])

    # CPI
    try:
        cpi = pd.read_csv(f'{OUTPUT_DIR}/cpi.csv')
        cpi['date'] = pd.to_datetime(cpi['日期'])
        cpi['cpi'] = pd.to_numeric(cpi['今值'], errors='coerce')
        cpi_data = cpi[['date', 'cpi']].dropna()
    except:
        cpi_data = pd.DataFrame(columns=['date', 'cpi'])

    # PPI
    try:
        ppi = pd.read_csv(f'{OUTPUT_DIR}/ppi.csv')
        ppi['date'] = pd.to_datetime(ppi['日期'])
        ppi['ppi'] = pd.to_numeric(ppi['今值'], errors='coerce')
        ppi_data = ppi[['date', 'ppi']].dropna()
    except:
        ppi_data = pd.DataFrame(columns=['date', 'ppi'])

    # M2
    try:
        m2 = pd.read_csv(f'{OUTPUT_DIR}/m2.csv')
        m2['date'] = pd.to_datetime(m2['日期'])
        m2['m2_growth'] = pd.to_numeric(m2['今值'], errors='coerce')
        m2_data = m2[['date', 'm2_growth']].dropna()
    except:
        m2_data = pd.DataFrame(columns=['date', 'm2_growth'])

    # EPU
    try:
        epu = pd.read_csv(f'{OUTPUT_DIR}/epu_index.csv')
        epu['date'] = pd.to_datetime(epu['year'].astype(str) + '-' + epu['month'].astype(str) + '-01')
        epu['epu'] = epu['China_Policy_Index']
        epu_data = epu[['date', 'epu']].dropna()
    except:
        epu_data = pd.DataFrame(columns=['date', 'epu'])

    # 前向填充函数
    def fill_macro(daily_df, macro_df, col_name):
        result = pd.Series(index=daily_df.index, dtype=float)
        for idx, date in enumerate(daily_df.index):
            available = macro_df[macro_df['date'] <= date]
            if len(available) > 0:
                result.iloc[idx] = available[col_name].iloc[-1]
        return result

    daily_data = daily_data.set_index('date')

    if not gdp_data.empty:
        daily_data['gdp_growth'] = fill_macro(daily_data.reset_index(), gdp_data.reset_index(drop=True), 'gdp_growth').values
    if not cpi_data.empty:
        daily_data['cpi'] = fill_macro(daily_data.reset_index(), cpi_data.reset_index(drop=True), 'cpi').values
    if not ppi_data.empty:
        daily_data['ppi'] = fill_macro(daily_data.reset_index(), ppi_data.reset_index(drop=True), 'ppi').values
    if not m2_data.empty:
        daily_data['m2_growth'] = fill_macro(daily_data.reset_index(), m2_data.reset_index(drop=True), 'm2_growth').values
    if not epu_data.empty:
        daily_data['epu'] = fill_macro(daily_data.reset_index(), epu_data.reset_index(drop=True), 'epu').values

    daily_data = daily_data.reset_index()
    print(f"   合并宏观数据后: {len(daily_data)}条")
    return daily_data


def calculate_derived_indicators(daily_data):
    """计算衍生指标"""
    print("\n【4】计算衍生指标...")

    # 市场收益率
    daily_data['market_return'] = daily_data['hs300_close'].pct_change()

    # 异常收益率（相对于上证指数）
    daily_data['sh_return'] = daily_data['sh_close'].pct_change()
    daily_data['abnormal_return'] = daily_data['market_return'] - daily_data['sh_return']
    daily_data['abs_ar'] = daily_data['abnormal_return'].abs()

    # Amihud非流动性指标
    daily_data['amihud'] = daily_data['market_return'].abs() / (daily_data['hs300_volume'] / 1e10)

    print("   衍生指标计算完成")
    return daily_data


def clean_and_save(daily_data):
    """清理数据并保存"""
    print("\n【5】清理并保存数据...")

    # 过滤日期范围
    final_data = daily_data[daily_data['date'] >= '2015-07-01'].copy()

    # 填充缺失值
    numeric_cols = final_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        final_data[col] = final_data[col].ffill().bfill()

    # 删除关键变量缺失的行
    key_cols = ['abs_ar', 'ivix', 'market_return', 'sentiment_zscore']
    final_data = final_data.dropna(subset=[c for c in key_cols if c in final_data.columns])

    print(f"\n最终数据集: {len(final_data)}条")
    print(f"日期范围: {final_data['date'].min()} ~ {final_data['date'].max()}")

    # 保存
    output_path = f'{FINAL_DIR}/real_data_complete.csv'
    final_data.to_csv(output_path, index=False)
    print(f"\n已保存: {output_path}")

    return final_data


def main():
    """主函数"""
    print("=" * 60)
    print("阶段2：数据整合")
    print("=" * 60)

    daily_data = load_base_data()
    daily_data = merge_sentiment_data(daily_data)
    daily_data = merge_macro_data(daily_data)
    daily_data = calculate_derived_indicators(daily_data)
    final_data = clean_and_save(daily_data)

    print("\n" + "=" * 60)
    print("数据整合完成！")
    print("=" * 60)

    return final_data


if __name__ == '__main__':
    main()
