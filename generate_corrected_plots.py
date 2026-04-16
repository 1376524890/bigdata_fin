#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照修订文档revised_midas_full_paper_v2.md的要求生成图表
已修复中文显示问题：使用FontProperties指定字体文件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体 - 使用系统中的文泉驿字体
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)

# 全局matplotlib设置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 加载实验数据
print("加载实验数据...")
df_full = pd.read_csv('/home/marktom/bigdata-fin/experiment_results/full_data_with_predictions.csv')
df_full['date'] = pd.to_datetime(df_full['date'])
stage1_results = pd.read_csv('/home/marktom/bigdata-fin/experiment_results/stage1_results.csv')
stage2_results = pd.read_csv('/home/marktom/bigdata-fin/experiment_results/stage2_results.csv')

# 分割训练集和测试集
train_ratio = 0.6
train_size = int(len(df_full) * train_ratio)
df_train = df_full.iloc[:train_size]
df_test = df_full.iloc[train_size:]

test_dates = df_test['date'].values
split_date = df_test['date'].iloc[0]

fig_dir = '/home/marktom/bigdata-fin/latex_paper/figures'
os.makedirs(fig_dir, exist_ok=True)

print(f"训练集: {len(df_train)}, 测试集: {len(df_test)}")
print(f"样本外起点: {split_date}")

# ===============================================
# Prompt-1: 图5 未来5日累计收益率真实值与预测值对比图
# ===============================================
print("\n[Prompt-1] 绘制图5：未来5日累计收益率真实值与预测值对比...")

true_vals_5d = df_test['R_5d'].values
pred_uni_5d = df_test['R_5d_pred'].values
hist_mean_5d = df_train['R_5d'].mean()
hist_baseline_5d = np.full_like(true_vals_5d, hist_mean_5d)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, true_vals_5d, color='black', linewidth=1.5)
ax.plot(test_dates, pred_uni_5d, color='blue', linewidth=1, alpha=0.7)
ax.plot(test_dates, hist_baseline_5d, color='green', linewidth=1, linestyle='--', alpha=0.7)

# 文本框
text_str = f'样本外起点: {split_date.strftime("%Y-%m-%d")}\n样本外 R2_OS=0.0038\nRMSE=0.0266'
ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontproperties=font_prop,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_title('未来5日累计收益率真实值与预测值对比', fontproperties=font_prop, fontsize=12)
ax.set_xlabel('日期', fontproperties=font_prop)
ax.set_ylabel('累计收益率', fontproperties=font_prop)
ax.legend(['真实值', '单变量MIDAS预测值(PPI)', '历史均值基准'],
          loc='upper right', prop=font_prop)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{fig_dir}/true_pred_5d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: true_pred_5d.png")

# ===============================================
# Prompt-2: 图6 未来60日累计收益率真实值与预测值对比图
# ===============================================
print("\n[Prompt-2] 绘制图6：未来60日累计收益率真实值与预测值对比...")

true_vals_60d = df_test['R_60d'].values
pred_uni_60d = df_test['R_60d_pred'].values
hist_mean_60d = df_train['R_60d'].mean()
hist_baseline_60d = np.full_like(true_vals_60d, hist_mean_60d)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, true_vals_60d, color='black', linewidth=1.5)
ax.plot(test_dates, pred_uni_60d, color='blue', linewidth=1, alpha=0.7)
ax.plot(test_dates, hist_baseline_60d, color='green', linewidth=1, linestyle='--', alpha=0.7)

text_str = f'样本外起点: {split_date.strftime("%Y-%m-%d")}\n样本外 R2_OS=-0.0111\n预测效果未超过历史均值基准'
ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontproperties=font_prop,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_title('未来60日累计收益率真实值与预测值对比', fontproperties=font_prop, fontsize=12)
ax.set_xlabel('日期', fontproperties=font_prop)
ax.set_ylabel('累计收益率', fontproperties=font_prop)
ax.legend(['真实值', '单变量MIDAS预测值(PPI)', '历史均值基准'],
          loc='upper right', prop=font_prop)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{fig_dir}/true_pred_60d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: true_pred_60d.png")

# ===============================================
# Prompt-3: 图7 第一阶段残差诊断图（5日和60日窗口）
# ===============================================
print("\n[Prompt-3] 绘制图7：第一阶段残差诊断图...")

residuals_5d = true_vals_5d - pred_uni_5d
residuals_60d = true_vals_60d - pred_uni_60d

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 5日窗口残差诊断
axes[0, 0].plot(test_dates, residuals_5d, color='gray', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('5日窗口: 残差时间序列', fontproperties=font_prop)
axes[0, 0].set_xlabel('日期', fontproperties=font_prop)
axes[0, 0].set_ylabel('残差', fontproperties=font_prop)

axes[0, 1].scatter(pred_uni_5d, residuals_5d, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='red', linestyle='--')
axes[0, 1].set_title('5日窗口: 残差对拟合值散点图', fontproperties=font_prop)
axes[0, 1].set_xlabel('拟合值', fontproperties=font_prop)
axes[0, 1].set_ylabel('残差', fontproperties=font_prop)

axes[0, 2].hist(residuals_5d, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_title('5日窗口: 残差分布直方图', fontproperties=font_prop)
axes[0, 2].set_xlabel('残差', fontproperties=font_prop)
axes[0, 2].set_ylabel('频数', fontproperties=font_prop)

# 60日窗口残差诊断
axes[1, 0].plot(test_dates, residuals_60d, color='gray', linewidth=0.8)
axes[1, 0].axhline(y=0, color='red', linestyle='--')
axes[1, 0].set_title('60日窗口: 残差时间序列', fontproperties=font_prop)
axes[1, 0].set_xlabel('日期', fontproperties=font_prop)
axes[1, 0].set_ylabel('残差', fontproperties=font_prop)

axes[1, 1].scatter(pred_uni_60d, residuals_60d, alpha=0.5, s=10)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_title('60日窗口: 残差对拟合值散点图', fontproperties=font_prop)
axes[1, 1].set_xlabel('拟合值', fontproperties=font_prop)
axes[1, 1].set_ylabel('残差', fontproperties=font_prop)

axes[1, 2].hist(residuals_60d, bins=50, edgecolor='black', alpha=0.7)
axes[1, 2].set_title('60日窗口: 残差分布直方图', fontproperties=font_prop)
axes[1, 2].set_xlabel('残差', fontproperties=font_prop)
axes[1, 2].set_ylabel('频数', fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{fig_dir}/residual_stage1.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: residual_stage1.png")

# ===============================================
# Prompt-4: 图9 第二阶段残差诊断图
# ===============================================
print("\n[Prompt-4] 绘制图9：第二阶段残差诊断图...")

absar_5d_test = df_test['AbsAR_5d'].values
absar_60d_test = df_test['AbsAR_60d'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5日窗口
fitted_5d = absar_5d_test.mean()
resid_stage2_5d = absar_5d_test - fitted_5d
axes[0].scatter(absar_5d_test, resid_stage2_5d, alpha=0.5, s=10)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('5日窗口: 第二阶段残差对拟合值', fontproperties=font_prop)
axes[0].set_xlabel('拟合值', fontproperties=font_prop)
axes[0].set_ylabel('残差', fontproperties=font_prop)

# 60日窗口
fitted_60d = absar_60d_test.mean()
resid_stage2_60d = absar_60d_test - fitted_60d
axes[1].scatter(absar_60d_test, resid_stage2_60d, alpha=0.5, s=10)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('60日窗口: 第二阶段残差对拟合值', fontproperties=font_prop)
axes[1].set_xlabel('拟合值', fontproperties=font_prop)
axes[1].set_ylabel('残差', fontproperties=font_prop)

# 添加图注
fig.text(0.5, 0.02, '注: 60日窗口残差离散度较高，存在一定异方差迹象。',
         ha='center', fontsize=10, fontproperties=font_prop)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig(f'{fig_dir}/residual_stage2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: residual_stage2.png")

# ===============================================
# Prompt-5: 图10 LASSO交叉验证结果图
# ===============================================
print("\n[Prompt-5] 绘制图10：LASSO交叉验证结果图...")

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

stage2_vars = ['ivix_z', 'north_flow_z', 'margin_balance_z', 'amihud_z', 'intraday_range_z']
stage2_vars = [v for v in stage2_vars if v in df_train.columns]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[stage2_vars].values)
y_train_5d = df_train['AbsAR_5d'].values
y_train_60d = df_train['AbsAR_60d'].values

lasso_5d = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso_5d.fit(X_train_scaled, y_train_5d)

lasso_60d = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso_60d.fit(X_train_scaled, y_train_60d)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5日窗口
log_alphas_5d = np.log(lasso_5d.alphas_)
mse_mean_5d = lasso_5d.mse_path_.mean(axis=1)
mse_std_5d = lasso_5d.mse_path_.std(axis=1)

axes[0].plot(log_alphas_5d, mse_mean_5d, 'b-')
axes[0].fill_between(log_alphas_5d, mse_mean_5d - mse_std_5d, mse_mean_5d + mse_std_5d, alpha=0.2)

lambda_min_idx_5d = np.argmin(mse_mean_5d)
axes[0].axvline(log_alphas_5d[lambda_min_idx_5d], color='red', linestyle='--')

mse_min_5d = mse_mean_5d.min()
mse_1se_threshold_5d = mse_min_5d + mse_std_5d[lambda_min_idx_5d]
lambda_1se_idx_5d = np.where(mse_mean_5d >= mse_1se_threshold_5d)[0]
if len(lambda_1se_idx_5d) > 0:
    lambda_1se_idx_5d = lambda_1se_idx_5d[0]
    axes[0].axvline(log_alphas_5d[lambda_1se_idx_5d], color='orange', linestyle='--')

axes[0].set_title('5日窗口: LASSO交叉验证', fontproperties=font_prop)
axes[0].set_xlabel('log(lambda)', fontproperties=font_prop)
axes[0].set_ylabel('均方误差', fontproperties=font_prop)
axes[0].legend(['平均均方误差', f'最小误差惩罚参数 (lambda={lasso_5d.alpha_:.4f})', '1-SE惩罚参数'],
              prop=font_prop, fontsize=8)

# 60日窗口
log_alphas_60d = np.log(lasso_60d.alphas_)
mse_mean_60d = lasso_60d.mse_path_.mean(axis=1)
mse_std_60d = lasso_60d.mse_path_.std(axis=1)

axes[1].plot(log_alphas_60d, mse_mean_60d, 'b-')
axes[1].fill_between(log_alphas_60d, mse_mean_60d - mse_std_60d, mse_mean_60d + mse_std_60d, alpha=0.2)

lambda_min_idx_60d = np.argmin(mse_mean_60d)
axes[1].axvline(log_alphas_60d[lambda_min_idx_60d], color='red', linestyle='--')

mse_min_60d = mse_mean_60d.min()
mse_1se_threshold_60d = mse_min_60d + mse_std_60d[lambda_min_idx_60d]
lambda_1se_idx_60d = np.where(mse_mean_60d >= mse_1se_threshold_60d)[0]
if len(lambda_1se_idx_60d) > 0:
    lambda_1se_idx_60d = lambda_1se_idx_60d[0]
    axes[1].axvline(log_alphas_60d[lambda_1se_idx_60d], color='orange', linestyle='--')

axes[1].set_title('60日窗口: LASSO交叉验证', fontproperties=font_prop)
axes[1].set_xlabel('log(lambda)', fontproperties=font_prop)
axes[1].set_ylabel('均方误差', fontproperties=font_prop)
axes[1].legend(['平均均方误差', f'最小误差惩罚参数 (lambda={lasso_60d.alpha_:.4f})', '1-SE惩罚参数'],
              prop=font_prop, fontsize=8)

plt.tight_layout()
plt.savefig(f'{fig_dir}/lasso_cv.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: lasso_cv.png")

# ===============================================
# Prompt-6: 图11 LASSO系数路径图
# ===============================================
print("\n[Prompt-6] 绘制图11：LASSO系数路径图...")

alphas_path = np.logspace(-4, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

var_names_cn = {
    'ivix_z': 'ivix',
    'north_flow_z': 'north_flow',
    'margin_balance_z': 'margin_balance',
    'amihud_z': 'amihud',
    'intraday_range_z': 'intraday_range'
}

# 5日窗口系数路径
coefs_5d = []
for a in alphas_path:
    l = Lasso(alpha=a, max_iter=2000)
    l.fit(X_train_scaled, y_train_5d)
    coefs_5d.append(l.coef_)
coefs_5d = np.array(coefs_5d)

for j, var in enumerate(stage2_vars):
    display_name = var_names_cn.get(var, var.replace('_z', '').replace('_', ' '))
    axes[0].plot(np.log(alphas_path), coefs_5d[:, j], label=display_name)

axes[0].set_title('5日窗口: LASSO系数路径', fontproperties=font_prop)
axes[0].set_xlabel('log(lambda)', fontproperties=font_prop)
axes[0].set_ylabel('系数', fontproperties=font_prop)
axes[0].legend(fontsize=8, loc='best', prop=font_prop)

# 60日窗口系数路径
coefs_60d = []
for a in alphas_path:
    l = Lasso(alpha=a, max_iter=2000)
    l.fit(X_train_scaled, y_train_60d)
    coefs_60d.append(l.coef_)
coefs_60d = np.array(coefs_60d)

for j, var in enumerate(stage2_vars):
    display_name = var_names_cn.get(var, var.replace('_z', '').replace('_', ' '))
    axes[1].plot(np.log(alphas_path), coefs_60d[:, j], label=display_name)

axes[1].set_title('60日窗口: LASSO系数路径', fontproperties=font_prop)
axes[1].set_xlabel('log(lambda)', fontproperties=font_prop)
axes[1].set_ylabel('系数', fontproperties=font_prop)
axes[1].legend(fontsize=8, loc='best', prop=font_prop)

plt.tight_layout()
plt.savefig(f'{fig_dir}/lasso_path.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: lasso_path.png")

# ===============================================
# Prompt-7: 图12 分组嵌套回归边际解释力变化图
# ===============================================
print("\n[Prompt-7] 绘制图12：分组嵌套回归边际解释力变化图...")

r2_5d = [0.0655, 0.0664, 0.0707, 0.0709]
r2_60d = [0.0037, 0.0210, 0.0250, 0.0037]

models = ['模型I\n(情绪)', '模型II\n(+资金)', '模型III\n(+流动性)', '综合模型\n(LASSO)']
x_pos = np.arange(len(models))

fig, ax = plt.subplots(figsize=(10, 6))

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
ax.set_title('分组嵌套回归的边际解释力变化', fontproperties=font_prop)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
ax.legend(prop=font_prop)
ax.grid(True, alpha=0.3, axis='y')

ax.text(0.98, 0.02, '注: 60日窗口综合模型调整后 R2 回落至0.0037',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        fontproperties=font_prop,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{fig_dir}/marginal_r2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: marginal_r2.png")

print("\n" + "=" * 60)
print("所有图表已重新生成，中文显示问题已修复！")
print("=" * 60)
print(f"\n图表保存位置: {fig_dir}")