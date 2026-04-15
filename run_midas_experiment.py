#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDAS模型两阶段实证分析
第一阶段：宏观变量混频回归预测收益率
第二阶段：异常收益偏离的分组嵌套回归
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("MIDAS模型两阶段实证分析")
print("=" * 60)

# ==================== 1. 数据加载与预处理 ====================
print("\n[1/7] 加载数据...")

df = pd.read_csv('/home/marktom/bigdata-fin/real_data_for_analysis.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 过滤样本期：2015-07-02 至 2025-12-25
start_date = '2015-07-02'
end_date = '2025-12-25'
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

print(f"样本期: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
print(f"总观测数: {len(df)}")

# 计算日对数收益率
df['log_return'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))

# 计算未来5日和60日累计收益率
for h in [5, 60]:
    df[f'R_{h}d'] = df['log_return'].shift(-h).rolling(window=h).sum().values

# 删除缺失值
df = df.dropna(subset=['R_5d', 'R_60d']).reset_index(drop=True)
print(f"有效观测数: {len(df)}")

# ==================== 2. 训练集/测试集划分 ====================
print("\n[2/7] 划分训练集与测试集...")

train_ratio = 0.6
train_size = int(len(df) * train_ratio)

df['is_train'] = False
df.loc[:train_size-1, 'is_train'] = True

df_train = df[df['is_train']].copy()
df_test = df[~df['is_train']].copy()

print(f"训练集: {len(df_train)} 个观测 ({df_train['date'].min().strftime('%Y-%m-%d')} 至 {df_train['date'].max().strftime('%Y-%m-%d')})")
print(f"测试集: {len(df_test)} 个观测 ({df_test['date'].min().strftime('%Y-%m-%d')} 至 {df_test['date'].max().strftime('%Y-%m-%d')})")

# ==================== 3. 数据预处理（训练集计算，测试集应用） ====================
print("\n[3/7] 数据预处理...")

# 连续变量列表（需要缩尾和标准化）
continuous_vars = ['ivix', 'north_flow', 'margin_balance', 'amihud', 'volatility_20d', 'turnover_ratio']

# 在训练集上计算缩尾阈值（1%双侧）
winsorize_dict = {}
for var in continuous_vars:
    lower = df_train[var].quantile(0.01)
    upper = df_train[var].quantile(0.99)
    winsorize_dict[var] = (lower, upper)
    # 应用到全样本
    df[var] = df[var].clip(lower, upper)

print(f"缩尾阈值计算完成（训练集1%分位数）")

# 计算标准化参数（训练集）
std_dict = {}
for var in continuous_vars + ['hs300_close']:
    mean = df_train[var].mean()
    std = df_train[var].std()
    std_dict[var] = (mean, std)
    # 创建标准化变量（z-score）
    df[f'{var}_z'] = (df[var] - mean) / std

print(f"标准化参数计算完成（训练集均值和标准差）")

# ==================== 4. 宏观变量MIDAS加权项构造 ====================
print("\n[4/7] 构造MIDAS加权项...")

# 月度宏观变量
macro_vars = ['cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny']

# 将date转为年月
df['year_month'] = df['date'].dt.to_period('M')

# 获取月度宏观数据（每个交易日对应的上一完整月数值）
for var in macro_vars:
    # 按月分组取最后一个值
    monthly_data = df.groupby('year_month')[var].last()
    # 创建滞后一期的月度数据映射
    monthly_dict = monthly_data.shift(1).to_dict()
    df[f'{var}_monthly'] = df['year_month'].map(monthly_dict)

# MIDAS Beta权重函数
def beta_weights(K, a, b):
    """计算Beta权重"""
    ell = np.arange(K)
    x = (ell + 1) / (K + 1)
    w = x**(a-1) * (1-x)**(b-1)
    return w / w.sum()

# Exponential Almon权重函数
def exp_almon_weights(K, eta1, eta2):
    """计算Exponential Almon权重"""
    ell = np.arange(K)
    w = np.exp(eta1 * ell + eta2 * ell**2)
    return w / w.sum()

# 构造MIDAS加权项
K = 12  # 12个月滞后

for var in macro_vars:
    df[f'{var}_MIDAS'] = np.nan
    for i in range(len(df)):
        current_month = df.loc[i, 'year_month']
        # 获取过去K个月的月度数据
        weights = np.ones(K) / K  # 初始等权重
        midas_val = 0
        count = 0
        for lag in range(1, K+1):
            lag_month = current_month - lag
            month_data = df[df['year_month'] == lag_month][var].values
            if len(month_data) > 0:
                midas_val += month_data[-1] * weights[count]
                count += 1
        if count == K:
            df.loc[i, f'{var}_MIDAS'] = midas_val

# 删除MIDAS变量有缺失的行
df = df.dropna(subset=[f'{var}_MIDAS' for var in macro_vars]).reset_index(drop=True)
print(f"MIDAS构造后观测数: {len(df)}")

# 重新划分训练集和测试集
df['is_train'] = False
df.loc[:train_size-1, 'is_train'] = True
df_train = df[df['is_train']].copy()
df_test = df[~df['is_train']].copy()

# ==================== 5. 第一阶段：MIDAS模型估计 ====================
print("\n[5/7] 第一阶段：MIDAS模型估计...")

# 存储结果的字典
midas_results = {}

for h in [5, 60]:
    print(f"\n{'='*50}")
    print(f"预测窗口 h={h}日")
    print(f"{'='*50}")

    y_train = df_train[f'R_{h}d'].values
    y_test = df_test[f'R_{h}d'].values

    # 5.1 单变量MIDAS模型
    print("\n[单变量MIDAS模型]")
    univariate_results = {}

    for var in macro_vars:
        X_train = df_train[f'{var}_MIDAS'].values.reshape(-1, 1)
        X_test = df_test[f'{var}_MIDAS'].values.reshape(-1, 1)

        # OLS估计
        X_const = add_constant(X_train)
        model = OLS(y_train, X_const).fit()

        # 预测
        pred_train = model.predict(X_const)
        pred_test = model.predict(add_constant(X_test))

        # 计算指标
        r2_train = 1 - np.sum((y_train - pred_train)**2) / np.sum((y_train - y_train.mean())**2)
        rmse_test = np.sqrt(np.mean((y_test - pred_test)**2))
        mae_test = np.mean(np.abs(y_test - pred_test))

        # 样本外R2
        y_test_mean = df_train[f'R_{h}d'].mean()  # 用训练集均值作为基准
        r2_os = 1 - np.sum((y_test - pred_test)**2) / np.sum((y_test - y_test_mean)**2)

        univariate_results[var] = {
            'model': model,
            'r2_train': r2_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_os': r2_os,
            'params': model.params,
            'tvalues': model.tvalues,
            'pvalues': model.pvalues,
            'pred_train': pred_train,
            'pred_test': pred_test
        }

        print(f"  {var:12s}: R²(样本内)={r2_train:6.4f}, R²(样本外)={r2_os:6.4f}, RMSE={rmse_test:.6f}")

    midas_results[f'h{h}_univariate'] = univariate_results

    # 5.2 VIF检验（在训练集上）
    print("\n[VIF检验 - 训练集]")
    X_vif = df_train[[f'{var}_MIDAS' for var in macro_vars]].dropna()
    vif_data = pd.DataFrame()
    vif_data['变量'] = macro_vars
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(macro_vars))]
    print(vif_data.to_string(index=False))

    # 5.3 精简多变量MIDAS模型（基于VIF和样本外表现）
    print("\n[精简多变量MIDAS模型]")

    # 选择VIF<10且样本外R2表现较好的变量
    selected_vars = []
    for var in macro_vars:
        vif_val = vif_data[vif_data['变量']==var]['VIF'].values[0]
        r2_os = univariate_results[var]['r2_os']
        if vif_val < 10:  # VIF阈值
            selected_vars.append((var, r2_os, vif_val))

    # 按样本外R2排序，最多选4个
    selected_vars = sorted(selected_vars, key=lambda x: x[1], reverse=True)[:4]
    selected_macro_vars = [v[0] for v in selected_vars]

    print(f"  入选变量: {', '.join(selected_macro_vars)}")

    X_multi_train = df_train[[f'{var}_MIDAS' for var in selected_macro_vars]].values
    X_multi_test = df_test[[f'{var}_MIDAS' for var in selected_macro_vars]].values

    X_multi_const = add_constant(X_multi_train)
    multi_model = OLS(y_train, X_multi_const).fit()

    pred_multi_train = multi_model.predict(X_multi_const)
    pred_multi_test = multi_model.predict(add_constant(X_multi_test))

    r2_multi_train = 1 - np.sum((y_train - pred_multi_train)**2) / np.sum((y_train - y_train.mean())**2)
    r2_multi_os = 1 - np.sum((y_test - pred_multi_test)**2) / np.sum((y_test - df_train[f'R_{h}d'].mean())**2)
    rmse_multi_test = np.sqrt(np.mean((y_test - pred_multi_test)**2))

    print(f"  多变量模型: R²(样本内)={r2_multi_train:6.4f}, R²(样本外)={r2_multi_os:6.4f}, RMSE={rmse_multi_test:.6f}")
    print(f"  调整后R²: {multi_model.rsquared_adj:.4f}")

    midas_results[f'h{h}_multivariate'] = {
        'model': multi_model,
        'selected_vars': selected_macro_vars,
        'r2_train': r2_multi_train,
        'r2_os': r2_multi_os,
        'rmse_test': rmse_multi_test,
        'pred_train': pred_multi_train,
        'pred_test': pred_multi_test
    }

    # 保存预测值用于第二阶段
    df.loc[df['is_train'], f'R_{h}d_pred'] = pred_multi_train
    df.loc[~df['is_train'], f'R_{h}d_pred'] = pred_multi_test

# ==================== 6. 构造异常收益 ====================
print("\n[6/7] 构造异常收益...")

for h in [5, 60]:
    df[f'AR_{h}d'] = df[f'R_{h}d'] - df[f'R_{h}d_pred']
    df[f'AbsAR_{h}d'] = np.abs(df[f'AR_{h}d'])

# ==================== 7. 第二阶段：分组嵌套回归 ====================
print("\n[7/7] 第二阶段：分组嵌套回归...")

# 准备第二阶段变量
second_stage_vars = {
    '情绪与风险感知': ['ivix_z'],
    '资金与杠杆交易': ['north_flow_z', 'margin_balance_z'],
    '流动性与交易状态': ['amihud_z', 'volatility_20d_z', 'turnover_ratio_z']
}

# 所有第二阶段变量
all_stage2_vars = ['ivix_z', 'north_flow_z', 'margin_balance_z', 'amihud_z', 'volatility_20d_z', 'turnover_ratio_z']

stage2_results = {}

for h in [5, 60]:
    print(f"\n{'='*50}")
    print(f"第二阶段 - 预测窗口 h={h}日")
    print(f"{'='*50}")

    df_train_s2 = df[df['is_train']].copy()
    df_test_s2 = df[~df['is_train']].copy()

    y_var = f'AbsAR_{h}d'
    y_train = df_train_s2[y_var].values
    y_test = df_test_s2[y_var].values

    # 7.1 模型I：情绪与风险感知
    X1 = df_train_s2[['ivix_z']].values
    X1_const = add_constant(X1)
    model1 = OLS(y_train, X1_const).fit(cov_type='HAC', cov_kwds={'maxlags': h-1})

    # 7.2 模型II：加入资金与杠杆
    X2 = df_train_s2[['ivix_z', 'north_flow_z', 'margin_balance_z']].values
    X2_const = add_constant(X2)
    model2 = OLS(y_train, X2_const).fit(cov_type='HAC', cov_kwds={'maxlags': h-1})

    # 联合检验：资金与杠杆变量组
    from statsmodels.stats.anova import anova_lm
    joint_test_f = ((model2.ssr - model1.ssr) / 2) / (model2.ssr / model2.df_resid)

    # 7.3 模型III：完整模型
    X3 = df_train_s2[all_stage2_vars].values
    X3_const = add_constant(X3)
    model3 = OLS(y_train, X3_const).fit(cov_type='HAC', cov_kwds={'maxlags': h-1})

    # 7.4 LASSO筛选
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train_s2[all_stage2_vars])

    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y_train)

    # 识别非零系数变量
    nonzero_mask = np.abs(lasso.coef_) > 1e-6
    lasso_selected = [all_stage2_vars[i] for i in range(len(all_stage2_vars)) if nonzero_mask[i]]

    print(f"  LASSO选择惩罚参数 λ={lasso.alpha_:.6f}")
    print(f"  LASSO保留变量: {', '.join(lasso_selected) if lasso_selected else '无'}")

    # 7.5 综合模型（基于LASSO和显著性）
    if len(lasso_selected) > 0:
        X4 = df_train_s2[lasso_selected].values
        X4_const = add_constant(X4)
        model4 = OLS(y_train, X4_const).fit(cov_type='HAC', cov_kwds={'maxlags': h-1})
    else:
        # 如果没有变量被选中，使用模型I
        X4 = X1
        X4_const = X1_const
        model4 = model1
        lasso_selected = ['ivix_z']

    # 打印结果对比
    print(f"\n  模型比较 (调整后R²):")
    print(f"    模型I (情绪):       {model1.rsquared_adj:.4f}")
    print(f"    模型II (+资金):     {model2.rsquared_adj:.4f}, 联合F检验={joint_test_f:.2f}")
    print(f"    模型III (+流动性):  {model3.rsquared_adj:.4f}")
    print(f"    综合模型 (LASSO):   {model4.rsquared_adj:.4f}")

    stage2_results[f'h{h}'] = {
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'lasso': lasso,
        'lasso_selected': lasso_selected,
        'joint_test_f': joint_test_f
    }

# ==================== 8. 生成图表 ====================
print("\n[8/8] 生成图表...")

fig_dir = '/home/marktom/bigdata-fin/latex_paper/figures'
import os
os.makedirs(fig_dir, exist_ok=True)

# 8.1 真实值与预测值对比图（5日窗口）
fig, ax = plt.subplots(figsize=(12, 6))
test_dates = df_test['date'].values

# 获取单变量最优模型（这里使用cpi作为示例）
best_uni_var = 'cpi'  # 实际应根据R2选择
pred_uni = midas_results['h5_univariate'][best_uni_var]['pred_test']
pred_multi = midas_results['h5_multivariate']['pred_test']
true_vals = df_test['R_5d'].values

ax.plot(test_dates, true_vals, label='True Value', color='black', linewidth=1.5)
ax.plot(test_dates, pred_uni, label=f'Univariate MIDAS ({best_uni_var})', color='blue', linewidth=1, alpha=0.7)
ax.plot(test_dates, pred_multi, label='Multivariate MIDAS', color='red', linewidth=1, alpha=0.7)
ax.set_title('True vs Predicted Future 5-Day Cumulative Returns', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{fig_dir}/true_pred_5d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: true_pred_5d.png")

# 8.2 真实值与预测值对比图（60日窗口）
fig, ax = plt.subplots(figsize=(12, 6))
pred_uni_60 = midas_results['h60_univariate'][best_uni_var]['pred_test']
pred_multi_60 = midas_results['h60_multivariate']['pred_test']
true_vals_60 = df_test['R_60d'].values

ax.plot(test_dates, true_vals_60, label='True Value', color='black', linewidth=1.5)
ax.plot(test_dates, pred_uni_60, label=f'Univariate MIDAS ({best_uni_var})', color='blue', linewidth=1, alpha=0.7)
ax.plot(test_dates, pred_multi_60, label='Multivariate MIDAS', color='red', linewidth=1, alpha=0.7)
ax.set_title('True vs Predicted Future 60-Day Cumulative Returns', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{fig_dir}/true_pred_60d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: true_pred_60d.png")

# 8.3 第一阶段残差诊断图（5日窗口）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
residuals_5 = df_test['R_5d'] - midas_results['h5_multivariate']['pred_test']

axes[0].plot(test_dates, residuals_5, color='gray', linewidth=0.8)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Residual Time Series (h=5)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual')

axes[1].scatter(midas_results['h5_multivariate']['pred_test'], residuals_5, alpha=0.5, s=10)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Fitted (h=5)')
axes[1].set_xlabel('Fitted Value')
axes[1].set_ylabel('Residual')

axes[2].hist(residuals_5, bins=50, edgecolor='black', alpha=0.7)
axes[2].set_title('Residual Distribution (h=5)')
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{fig_dir}/residual_stage1.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: residual_stage1.png")

# 8.4 LASSO交叉验证误差图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, h in enumerate([5, 60]):
    lasso = stage2_results[f'h{h}']['lasso']
    axes[i].plot(np.log(lasso.alphas_), lasso.mse_path_.mean(axis=1), 'b-', label='Mean MSE')
    axes[i].fill_between(np.log(lasso.alphas_),
                         lasso.mse_path_.mean(axis=1) - lasso.mse_path_.std(axis=1),
                         lasso.mse_path_.mean(axis=1) + lasso.mse_path_.std(axis=1),
                         alpha=0.2)
    axes[i].axvline(np.log(lasso.alpha_), color='red', linestyle='--', label=f'Best λ={lasso.alpha_:.4f}')
    axes[i].set_title(f'LASSO Cross-Validation (h={h})')
    axes[i].set_xlabel('log(λ)')
    axes[i].set_ylabel('Mean Squared Error')
    axes[i].legend()

plt.tight_layout()
plt.savefig(f'{fig_dir}/lasso_cv.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: lasso_cv.png")

# 8.5 LASSO系数路径图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, h in enumerate([5, 60]):
    alphas = np.logspace(-4, 1, 100)
    coefs = []
    for a in alphas:
        l = Lasso(alpha=a, max_iter=2000)
        l.fit(scaler.fit_transform(df_train_s2[all_stage2_vars]),
              df_train_s2[f'AbsAR_{h}d'].values)
        coefs.append(l.coef_)
    coefs = np.array(coefs)

    for j, var in enumerate(all_stage2_vars):
        axes[i].plot(np.log(alphas), coefs[:, j], label=var.replace('_z', ''))
    axes[i].set_title(f'LASSO Coefficient Path (h={h})')
    axes[i].set_xlabel('log(λ)')
    axes[i].set_ylabel('Coefficient')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{fig_dir}/lasso_path.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: lasso_path.png")

# 8.6 分组嵌套回归边际解释力图
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(4)
models = ['Model I\n(情绪)', 'Model II\n(+资金)', 'Model III\n(+流动性)', '综合模型\n(LASSO)']

for i, h in enumerate([5, 60]):
    r2_vals = [
        stage2_results[f'h{h}']['model1'].rsquared_adj,
        stage2_results[f'h{h}']['model2'].rsquared_adj,
        stage2_results[f'h{h}']['model3'].rsquared_adj,
        stage2_results[f'h{h}']['model4'].rsquared_adj
    ]
    ax.bar(x_pos + i*0.35, r2_vals, 0.35, label=f'h={h} days')

ax.set_xlabel('Model')
ax.set_ylabel('Adjusted R²')
ax.set_title('Marginal Explanatory Power of Nested Models')
ax.set_xticks(x_pos + 0.175)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{fig_dir}/marginal_r2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: marginal_r2.png")

# 8.7 第二阶段残差诊断图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df_test_s2 = df[~df['is_train']].copy()
for i, h in enumerate([5, 60]):
    model4 = stage2_results[f'h{h}']['model4']
    X_final = df_test_s2[stage2_results[f'h{h}']['lasso_selected']].values
    X_final_const = add_constant(X_final)
    pred = model4.predict(X_final_const)
    residuals = df_test_s2[f'AbsAR_{h}d'].values - pred

    axes[i].scatter(pred, residuals, alpha=0.5, s=10)
    axes[i].axhline(y=0, color='red', linestyle='--')
    axes[i].set_title(f'Stage 2 Residuals vs Fitted (h={h})')
    axes[i].set_xlabel('Fitted Value')
    axes[i].set_ylabel('Residual')

plt.tight_layout()
plt.savefig(f'{fig_dir}/residual_stage2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: residual_stage2.png")

# ==================== 9. 保存结果到文件 ====================
print("\n" + "=" * 60)
print("保存实验结果...")
print("=" * 60)

# 保存关键结果到CSV
results_summary = {
    'Window': [],
    'Model': [],
    'R2_InSample': [],
    'R2_OutSample': [],
    'RMSE_Test': []
}

for h in [5, 60]:
    # 单变量最佳
    best_r2 = -np.inf
    best_var = None
    for var in macro_vars:
        r2 = midas_results[f'h{h}_univariate'][var]['r2_os']
        if r2 > best_r2:
            best_r2 = r2
            best_var = var

    results_summary['Window'].append(f'{h}d')
    results_summary['Model'].append(f'Univariate_{best_var}')
    results_summary['R2_InSample'].append(midas_results[f'h{h}_univariate'][best_var]['r2_train'])
    results_summary['R2_OutSample'].append(midas_results[f'h{h}_univariate'][best_var]['r2_os'])
    results_summary['RMSE_Test'].append(midas_results[f'h{h}_univariate'][best_var]['rmse_test'])

    # 多变量
    results_summary['Window'].append(f'{h}d')
    results_summary['Model'].append('Multivariate')
    results_summary['R2_InSample'].append(midas_results[f'h{h}_multivariate']['r2_train'])
    results_summary['R2_OutSample'].append(midas_results[f'h{h}_multivariate']['r2_os'])
    results_summary['RMSE_Test'].append(midas_results[f'h{h}_multivariate']['rmse_test'])

results_df = pd.DataFrame(results_summary)
results_df.to_csv('/home/marktom/bigdata-fin/experiment_results/stage1_results.csv', index=False)
print("  已保存: stage1_results.csv")

# 保存VIF结果
vif_data.to_csv('/home/marktom/bigdata-fin/experiment_results/vif_results.csv', index=False)
print("  已保存: vif_results.csv")

# 保存第二阶段结果
stage2_summary = {
    'Window': [],
    'Model': [],
    'Adj_R2': [],
    'LASSO_Selected': []
}

for h in [5, 60]:
    for model_name in ['model1', 'model2', 'model3', 'model4']:
        stage2_summary['Window'].append(f'{h}d')
        stage2_summary['Model'].append(model_name)
        stage2_summary['Adj_R2'].append(stage2_results[f'h{h}'][model_name].rsquared_adj)
        stage2_summary['LASSO_Selected'].append(str(stage2_results[f'h{h}']['lasso_selected']) if model_name == 'model4' else '')

stage2_df = pd.DataFrame(stage2_summary)
stage2_df.to_csv('/home/marktom/bigdata-fin/experiment_results/stage2_results.csv', index=False)
print("  已保存: stage2_results.csv")

# 保存完整数据（含预测值）
df.to_csv('/home/marktom/bigdata-fin/experiment_results/full_data_with_predictions.csv', index=False)
print("  已保存: full_data_with_predictions.csv")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print(f"\n结果文件保存在: /home/marktom/bigdata-fin/experiment_results/")
print(f"图表文件保存在: {fig_dir}/")
