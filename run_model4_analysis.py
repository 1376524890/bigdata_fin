#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行扩展模型IV回归并生成结果表和图表
扩展模型IV：在模型III基础上加入EPU和FX波动代理变量
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

print("=" * 60)
print("运行扩展模型IV回归分析")
print("=" * 60)

# 加载实验数据
print("\n加载实验数据...")
df_full = pd.read_csv('/home/marktom/bigdata-fin/experiment_results/full_data_with_predictions.csv')
df_full['date'] = pd.to_datetime(df_full['date'])

# 分割训练集和测试集
train_ratio = 0.6
train_size = int(len(df_full) * train_ratio)
df_train = df_full.iloc[:train_size]
df_test = df_full.iloc[train_size:]

print(f"训练集: {len(df_train)}, 测试集: {len(df_test)}")

# 构造汇率波动代理变量 fx_vol
# 使用美元兑人民币汇率的滚动波动率（20日窗口）
print("\n构造汇率波动代理变量 fx_vol...")
df_full['usd_cny_log_return'] = np.log(df_full['usd_cny'] / df_full['usd_cny'].shift(1))
rolling_window = 20
df_full['fx_vol'] = df_full['usd_cny_log_return'].rolling(window=rolling_window).std()
df_full['fx_vol'] = df_full['fx_vol'].fillna(df_full['fx_vol'].median())

# 更新训练集和测试集
df_train = df_full.iloc[:train_size]
df_test = df_full.iloc[train_size:]

# 标准化新增变量
scaler_epu = StandardScaler()
scaler_fx_vol = StandardScaler()

df_full['epu_z'] = 0.0
df_full['fx_vol_z'] = 0.0

train_epu = df_full['epu'].iloc[:train_size].values.reshape(-1, 1)
test_epu = df_full['epu'].iloc[train_size:].values.reshape(-1, 1)
scaler_epu.fit(train_epu)
df_full['epu_z'] = np.concatenate([scaler_epu.transform(train_epu).flatten(),
                                   scaler_epu.transform(test_epu).flatten()])

train_fx_vol = df_full['fx_vol'].iloc[:train_size].values.reshape(-1, 1)
test_fx_vol = df_full['fx_vol'].iloc[train_size:].values.reshape(-1, 1)
scaler_fx_vol.fit(train_fx_vol)
df_full['fx_vol_z'] = np.concatenate([scaler_fx_vol.transform(train_fx_vol).flatten(),
                                       scaler_fx_vol.transform(test_fx_vol).flatten()])

# 更新训练集
df_train = df_full.iloc[:train_size]
df_test = df_full.iloc[train_size:]

print(f"EPU训练集均值: {scaler_epu.mean_[0]:.4f}, 标准差: {scaler_epu.scale_[0]:.4f}")
print(f"FX_vol训练集均值: {scaler_fx_vol.mean_[0]:.6f}, 标准差: {scaler_fx_vol.scale_[0]:.6f}")

# 定义变量集合
# 模型I-III变量
base_vars = ['sentiment_zscore', 'ivix_z']
fund_vars = ['north_flow_z', 'margin_balance_z']
liquid_vars = ['amihud_z', 'momentum_20d', 'intraday_range_z']
# 扩展变量
ext_vars = ['epu_z', 'fx_vol_z']

# 模型IV变量（完整模型 + EPU + FX_vol）
model4_vars = base_vars + fund_vars + liquid_vars + ext_vars

print(f"\n模型IV变量: {model4_vars}")

# ===============================================
# 运行扩展模型IV回归
# ===============================================
print("\n" + "=" * 60)
print("运行扩展模型IV回归（加入EPU和FX波动）")
print("=" * 60)

def run_hac_regression(X, y, lags):
    """使用HAC稳健标准误运行回归"""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model

results = {}

for window in ['5d', '60d']:
    y_train = df_train[f'AbsAR_{window}'].values

    # 确保所有变量在训练集中存在
    available_vars = [v for v in model4_vars if v in df_train.columns]
    X_train = df_train[available_vars].values

    # HAC滞后阶数
    lags = 4 if window == '5d' else 59

    print(f"\n--- {window}窗口模型IV ---")
    print(f"可用变量: {available_vars}")

    model = run_hac_regression(X_train, y_train, lags)

    results[window] = {
        'model': model,
        'vars': available_vars,
        'adj_r2': model.rsquared_adj,
        'r2': model.rsquared,
        'n_obs': int(model.nobs)
    }

    print(f"调整后R2: {model.rsquared_adj:.4f}")
    print(f"样本内R2: {model.rsquared:.4f}")
    print(f"观测数: {int(model.nobs)}")

    # 输出系数表
    print("\n系数估计:")
    for i, var in enumerate(available_vars):
        coef = model.params[i+1]  # 跳过常数项
        t_stat = model.tvalues[i+1]
        p_val = model.pvalues[i+1]
        sig = ''
        if p_val < 0.01:
            sig = '***'
        elif p_val < 0.05:
            sig = '**'
        elif p_val < 0.10:
            sig = '*'
        print(f"  {var}: coef={coef:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")

# ===============================================
# 计算模型IV相对于模型III的增量解释力
# ===============================================
print("\n" + "=" * 60)
print("模型III vs 模型IV 增量解释力检验")
print("=" * 60)

# 首先运行模型III获取基准R2
model3_vars = base_vars + fund_vars + liquid_vars
model3_vars = [v for v in model3_vars if v in df_train.columns]

for window in ['5d', '60d']:
    y_train = df_train[f'AbsAR_{window}'].values
    lags = 4 if window == '5d' else 59

    # 模型III
    X3_train = df_train[model3_vars].values
    model3 = run_hac_regression(X3_train, y_train, lags)

    # 模型IV
    model4_vars_avail = [v for v in model4_vars if v in df_train.columns]
    X4_train = df_train[model4_vars_avail].values
    model4 = run_hac_regression(X4_train, y_train, lags)

    r2_diff = model4.rsquared_adj - model3.rsquared_adj

    # 联合检验：检验新增变量(epu_z, fx_vol_z)的联合显著性
    # 使用F检验
    n = len(y_train)
    k3 = len(model3_vars) + 1  # 加常数项
    k4 = len(model4_vars_avail) + 1

    # SSR差异检验
    ssr3 = model3.ssr
    ssr4 = model4.ssr
    q = len(ext_vars)  # 新增变量个数

    f_stat = ((ssr3 - ssr4) / q) / (ssr4 / (n - k4))
    f_pval = 1 - stats.f.cdf(f_stat, q, n - k4)

    print(f"\n{window}窗口:")
    print(f"  模型III调整后R2: {model3.rsquared_adj:.4f}")
    print(f"  模型IV调整后R2: {model4.rsquared_adj:.4f}")
    print(f"  增量解释力: {r2_diff:.4f}")
    print(f"  新增变量联合F检验: F={f_stat:.2f}, p={f_pval:.4f}")

    sig = ''
    if f_pval < 0.01:
        sig = '***'
    elif f_pval < 0.05:
        sig = '**'
    elif f_pval < 0.10:
        sig = '*'
    print(f"  显著性: {sig}")

# ===============================================
# 运行包含所有变量的LASSO（包括EPU和FX_vol）
# ===============================================
print("\n" + "=" * 60)
print("包含扩展变量的LASSO筛选")
print("=" * 60)

all_vars = model4_vars
all_vars_avail = [v for v in all_vars if v in df_train.columns]

lasso_results = {}

for window in ['5d', '60d']:
    y_train = df_train[f'AbsAR_{window}'].values
    X_train = df_train[all_vars_avail].values

    # 标准化（已标准化，但再次确保）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y_train)

    # 识别保留变量
    retained_vars = []
    for i, var in enumerate(all_vars_avail):
        if abs(lasso.coef_[i]) > 1e-6:
            retained_vars.append(var)

    print(f"\n{window}窗口:")
    print(f"  最优lambda: {lasso.alpha_:.6f}")
    print(f"  LASSO保留变量: {retained_vars}")
    print(f"  各变量系数:")
    for i, var in enumerate(all_vars_avail):
        print(f"    {var}: {lasso.coef_[i]:.6f}")

    lasso_results[window] = {
        'alpha': lasso.alpha_,
        'retained': retained_vars,
        'coef': lasso.coef_
    }

# ===============================================
# 生成模型IV结果表
# ===============================================
print("\n" + "=" * 60)
print("生成模型IV结果汇总表")
print("=" * 60)

# 创建汇总表
summary_data = []

for window in ['5d', '60d']:
    model = results[window]['model']
    vars_list = results[window]['vars']

    for i, var in enumerate(vars_list):
        coef = model.params[i+1]
        t_stat = model.tvalues[i+1]
        p_val = model.pvalues[i+1]
        sig = ''
        if p_val < 0.01:
            sig = '***'
        elif p_val < 0.05:
            sig = '**'
        elif p_val < 0.10:
            sig = '*'

        summary_data.append({
            'window': window,
            'var': var,
            'coef': coef,
            't_stat': t_stat,
            'p_val': p_val,
            'sig': sig
        })

summary_df = pd.DataFrame(summary_data)
print("\n模型IV系数估计汇总:")
print(summary_df.to_string())

# 保存结果
output_dir = '/home/marktom/bigdata-fin/experiment_results'
summary_df.to_csv(f'{output_dir}/model4_results.csv', index=False)

# 保存模型IV整体统计
model4_stats = pd.DataFrame({
    'window': ['5d', '60d'],
    'adj_r2': [results['5d']['adj_r2'], results['60d']['adj_r2']],
    'r2': [results['5d']['r2'], results['60d']['r2']],
    'n_obs': [results['5d']['n_obs'], results['60d']['n_obs']]
})
model4_stats.to_csv(f'{output_dir}/model4_stats.csv', index=False)

# 保存LASSO结果
lasso_summary = pd.DataFrame({
    'window': ['5d', '60d'],
    'alpha': [lasso_results['5d']['alpha'], lasso_results['60d']['alpha']],
    'retained_vars': [','.join(lasso_results['5d']['retained']), ','.join(lasso_results['60d']['retained'])]
})
lasso_summary.to_csv(f'{output_dir}/lasso_extended_results.csv', index=False)

print("\n结果已保存:")
print(f"  - {output_dir}/model4_results.csv")
print(f"  - {output_dir}/model4_stats.csv")
print(f"  - {output_dir}/lasso_extended_results.csv")

# ===============================================
# 生成图表
# ===============================================
fig_dir = '/home/marktom/bigdata-fin/latex_paper/figures'
os.makedirs(fig_dir, exist_ok=True)

# 图13: 扩展模型IV解释力变化图
print("\n生成图13: 扩展模型IV解释力变化图...")

# 需要重新计算模型I-III的R2以便比较
r2_comparison = {}

for window in ['5d', '60d']:
    y_train = df_train[f'AbsAR_{window}'].values
    lags = 4 if window == '5d' else 59

    # 模型I
    X1 = df_train[['sentiment_zscore', 'ivix_z']].values
    model1 = run_hac_regression(X1, y_train, lags)

    # 模型II
    X2 = df_train[['sentiment_zscore', 'ivix_z', 'north_flow_z', 'margin_balance_z']].values
    model2 = run_hac_regression(X2, y_train, lags)

    # 模型III
    X3 = df_train[model3_vars].values
    model3 = run_hac_regression(X3, y_train, lags)

    # 模型IV
    model4_vars_avail = [v for v in model4_vars if v in df_train.columns]
    X4 = df_train[model4_vars_avail].values
    model4 = run_hac_regression(X4, y_train, lags)

    # LASSO综合模型（使用保留变量重新回归）
    retained = lasso_results[window]['retained']
    if len(retained) > 0:
        X_lasso = df_train[retained].values
        model_lasso = run_hac_regression(X_lasso, y_train, lags)
        r2_lasso = model_lasso.rsquared_adj
    else:
        r2_lasso = 0

    r2_comparison[window] = {
        'model1': model1.rsquared_adj,
        'model2': model2.rsquared_adj,
        'model3': model3.rsquared_adj,
        'model4': model4.rsquared_adj,
        'lasso': r2_lasso
    }

# 绘制图13
fig, ax = plt.subplots(figsize=(12, 6))

models = ['模型I\n(情绪)', '模型II\n(+资金)', '模型III\n(+流动性)', '模型IV\n(+EPU+FX)', '综合模型\n(LASSO)']
x_pos = np.arange(len(models))

r2_5d = [r2_comparison['5d']['model1'], r2_comparison['5d']['model2'],
         r2_comparison['5d']['model3'], r2_comparison['5d']['model4'],
         r2_comparison['5d']['lasso']]
r2_60d = [r2_comparison['60d']['model1'], r2_comparison['60d']['model2'],
          r2_comparison['60d']['model3'], r2_comparison['60d']['model4'],
          r2_comparison['60d']['lasso']]

bars_5d = ax.bar(x_pos - 0.2, r2_5d, 0.4, label='5日窗口', color='blue', alpha=0.7)
bars_60d = ax.bar(x_pos + 0.2, r2_60d, 0.4, label='60日窗口', color='orange', alpha=0.7)

# 标注数值
for i, bar in enumerate(bars_5d):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{r2_5d[i]:.4f}', ha='center', va='bottom', fontsize=9)

for i, bar in enumerate(bars_60d):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{r2_60d[i]:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('模型', fontproperties=font_prop)
ax.set_ylabel('调整后 R2', fontproperties=font_prop)
ax.set_title('分组嵌套回归与扩展模型IV的解释力变化', fontproperties=font_prop, fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
ax.legend(prop=font_prop)
ax.grid(True, alpha=0.3, axis='y')

# 图注
note_text = f'注: 模型IV在模型III基础上加入EPU与FX波动代理变量\n5日窗口模型IV调整后R2={r2_comparison["5d"]["model4"]:.4f}, 60日窗口={r2_comparison["60d"]["model4"]:.4f}'
ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontproperties=font_prop,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{fig_dir}/model4_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig_dir}/model4_r2_comparison.png")

# 更新图12（原边际解释力变化图，加入模型IV）
print("\n更新图12: 分组嵌套回归边际解释力变化图...")

fig, ax = plt.subplots(figsize=(12, 6))

models_12 = ['模型I\n(情绪)', '模型II\n(+资金)', '模型III\n(+流动性)', '模型IV\n(+EPU+FX)', '综合模型\n(LASSO)']
x_pos = np.arange(len(models_12))

bars_5d = ax.bar(x_pos - 0.2, r2_5d, 0.4, label='5日窗口', color='blue', alpha=0.7)
bars_60d = ax.bar(x_pos + 0.2, r2_60d, 0.4, label='60日窗口', color='orange', alpha=0.7)

for i, bar in enumerate(bars_5d):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{r2_5d[i]:.4f}', ha='center', va='bottom', fontsize=9)

for i, bar in enumerate(bars_60d):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{r2_60d[i]:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('模型', fontproperties=font_prop)
ax.set_ylabel('调整后 R2', fontproperties=font_prop)
ax.set_title('分组嵌套回归的边际解释力变化（含扩展模型IV）', fontproperties=font_prop, fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_12)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
ax.legend(prop=font_prop)
ax.grid(True, alpha=0.3, axis='y')

ax.text(0.98, 0.02, '注: 60日窗口综合模型调整后 R2 回落至0.0037',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        fontproperties=font_prop,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{fig_dir}/marginal_r2_extended.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig_dir}/marginal_r2_extended.png")

# 图14: 扩展LASSO系数路径图（包含EPU和FX_vol）
print("\n生成扩展LASSO系数路径图...")

alphas_path = np.logspace(-4, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

var_names_cn = {
    'sentiment_zscore': 'sentiment',
    'ivix_z': 'ivix',
    'north_flow_z': 'north_flow',
    'margin_balance_z': 'margin_balance',
    'amihud_z': 'amihud',
    'momentum_20d': 'momentum_20d',
    'intraday_range_z': 'intraday_range',
    'epu_z': 'epu',
    'fx_vol_z': 'fx_vol'
}

for window_idx, window in enumerate(['5d', '60d']):
    y_train = df_train[f'AbsAR_{window}'].values
    X_train = df_train[all_vars_avail].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    coefs = []
    for a in alphas_path:
        l = Lasso(alpha=a, max_iter=2000)
        l.fit(X_scaled, y_train)
        coefs.append(l.coef_)
    coefs = np.array(coefs)

    for j, var in enumerate(all_vars_avail):
        display_name = var_names_cn.get(var, var.replace('_z', '').replace('_', ' '))
        axes[window_idx].plot(np.log(alphas_path), coefs[:, j], label=display_name)

    axes[window_idx].set_title(f'{window}窗口: LASSO系数路径', fontproperties=font_prop)
    axes[window_idx].set_xlabel('log(lambda)', fontproperties=font_prop)
    axes[window_idx].set_ylabel('系数', fontproperties=font_prop)
    axes[window_idx].legend(fontsize=8, loc='best', prop=font_prop)

plt.tight_layout()
plt.savefig(f'{fig_dir}/lasso_path_extended.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig_dir}/lasso_path_extended.png")

print("\n" + "=" * 60)
print("扩展模型IV分析完成！")
print("=" * 60)
print(f"\n结果文件位置: {output_dir}")
print(f"图表文件位置: {fig_dir}")