#!/usr/bin/env python3
"""
修复VIF计算并更新结果
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTPUT_DIR = '/home/marktom/bigdata-fin/har/results/harx_instability_full'
DATA_FILE = '/home/marktom/bigdata-fin/real_data_complete.csv'

def compute_vif_fixed(X, feature_names):
    """计算VIF - 修复版"""
    # 确保X是float类型
    X = np.asarray(X, dtype=np.float64)
    X_const = sm.add_constant(X)
    vif_results = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X_const, i + 1)
            vif_results.append({
                'variable': name,
                'vif': float(vif)
            })
        except Exception as e:
            print(f"  VIF计算失败 {name}: {e}")
            vif_results.append({
                'variable': name,
                'vif': np.nan
            })
    return pd.DataFrame(vif_results)

# 加载预测数据
pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'harx_instability_test_predictions.csv'))

# 加载原始数据
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 样本划分
n_total = len(df)
n_train = int(n_total * 0.6)
df_train = df.iloc[:n_train]

# 重新计算VIF
import os

har_features1 = ['past_absret_5', 'past_absret_20', 'past_absret_60']
macro_features = ['epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']
harx_features1 = har_features1 + macro_features
harx_lite_features = har_features1 + ['fx_ret1_m1', 'ppi_yoy_m1']

# 构造变量
EPS = 1e-12

df['r_t'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))
df['past_absret_5'] = df['r_t'].rolling(5).apply(lambda x: np.mean(np.abs(x)), raw=True)
df['past_absret_20'] = df['r_t'].rolling(20).apply(lambda x: np.mean(np.abs(x)), raw=True)
df['past_absret_60'] = df['r_t'].rolling(60).apply(lambda x: np.mean(np.abs(x)), raw=True)

# 宏观变量 - 简化处理，直接使用现有数据中的月度值
# 使用上一月数据
df['epu_log_m1'] = np.log(df['epu'] + EPS).shift(20)  # 滚动月度滞后
df['fx_ret1_m1'] = (np.log(df['usd_cny']) - np.log(df['usd_cny'].shift(20))).shift(20)
df['ppi_yoy_m1'] = df['ppi'].shift(20)
df['m2_delta1_m1'] = (df['m2_growth'] - df['m2_growth'].shift(20)).shift(20)

# 过滤有效数据
cols_to_check = harx_features1
df_clean = df.dropna(subset=cols_to_check).copy()

n_train = int(len(df_clean) * 0.6)
df_train_clean = df_clean.iloc[:n_train]

# 计算VIF
print("计算 HARX_OLS VIF...")
X_harx = df_train_clean[harx_features1].values
vif_harx = compute_vif_fixed(X_harx, harx_features1)
vif_harx['model'] = 'HARX_OLS'
vif_harx['target'] = 'future_absret_5'

print("计算 HARX_lite VIF...")
X_lite = df_train_clean[harx_lite_features].values
vif_lite = compute_vif_fixed(X_lite, harx_lite_features)
vif_lite['model'] = 'HARX_lite_OLS'
vif_lite['target'] = 'future_absret_5'

# 合并并保存
vif_combined = pd.concat([vif_harx, vif_lite])
vif_combined.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_vif.csv'), index=False)

print("\nVIF结果:")
print(vif_combined.to_string())

print(f"\nVIF已更新保存至: {OUTPUT_DIR}/harx_instability_vif.csv")