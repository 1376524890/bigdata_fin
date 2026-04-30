#!/usr/bin/env python3
"""
基于 HARX 回归的股票市场短期不稳定性及宏观变量增量解释作用研究
完整实验流程脚本
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime, timedelta

# 固定随机种子
np.random.seed(42)

# ==================== 常量定义 ====================
EPS = 1e-12
SAMPLE_START = '2015-07-02'
SAMPLE_END = '2025-12-25'
TRAIN_RATIO = 0.6
VAL_RATIO_IN_TRAIN = 0.2  # 训练集后20%用于验证（仅Ridge调参）
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]

OUTPUT_DIR = '/home/marktom/bigdata-fin/har/results/harx_instability_full'
DATA_FILE = '/home/marktom/bigdata-fin/real_data_complete.csv'

# ==================== 辅助函数 ====================
def calculate_hac_se(model, maxlags):
    """计算Newey-West HAC稳健标准误"""
    try:
        # 使用cov_type='HAC'和cov_kwds={'maxlags': maxlags}
        hac_se = model.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
        return hac_se
    except Exception as e:
        print(f"HAC计算警告: {e}")
        return model

def compute_r2_os(y_true, y_pred, y_train_mean):
    """计算样本外R²（相对于训练均值基准）"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    r2_os = 1 - ss_res / ss_tot
    return r2_os

def compute_statistics(series):
    """计算描述统计"""
    s = series.dropna()
    return {
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
        'p25': s.quantile(0.25),
        'median': s.quantile(0.5),
        'p75': s.quantile(0.75),
        'skewness': skew(s),
        'kurtosis': kurtosis(s),
        'n_valid': len(s)
    }

def get_previous_month_info(df, current_date, monthly_var, var_name):
    """获取上一个完整月的月度信息"""
    # 将current_date转换为datetime
    current_date = pd.to_datetime(current_date)

    # 确定当前月份
    current_year = current_date.year
    current_month = current_date.month

    # 上一个完整月
    if current_month == 1:
        prev_year = current_year - 1
        prev_month = 12
    else:
        prev_year = current_year
        prev_month = current_month - 1

    # 筛选上一个月的数据
    prev_month_data = df[(df['date'].dt.year == prev_year) &
                         (df['date'].dt.month == prev_month)]

    if len(prev_month_data) == 0:
        return np.nan

    # 返回上一个月的最后一个可用值
    return prev_month_data[monthly_var].iloc[-1]

def compute_monthly_change(df, current_date, monthly_var):
    """计算月度变化"""
    current_date = pd.to_datetime(current_date)

    current_year = current_date.year
    current_month = current_date.month

    # 当前月最后一个值
    current_month_data = df[(df['date'].dt.year == current_year) &
                            (df['date'].dt.month == current_month)]
    if len(current_month_data) == 0:
        return np.nan
    current_val = current_month_data[monthly_var].iloc[-1]

    # 上一个月最后一个值
    if current_month == 1:
        prev_year = current_year - 1
        prev_month = 12
    else:
        prev_year = current_year
        prev_month = current_month - 1

    prev_month_data = df[(df['date'].dt.year == prev_year) &
                         (df['date'].dt.month == prev_month)]
    if len(prev_month_data) == 0:
        return np.nan
    prev_val = prev_month_data[monthly_var].iloc[-1]

    return current_val - prev_val

# ==================== 主处理函数 ====================
def load_and_preprocess_data():
    """步骤1-10: 数据预处理"""
    print("=" * 60)
    print("数据预处理阶段")
    print("=" * 60)

    # 1. 读取数据
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])

    # 2. 排序
    df = df.sort_values('date').reset_index(drop=True)

    # 3. 样本期截取
    df = df[(df['date'] >= SAMPLE_START) & (df['date'] <= SAMPLE_END)].copy()
    print(f"样本期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"初始观测数: {len(df)}")

    # 4. 构造日对数收益率 r_t
    df['r_t'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))

    # 5. 构造因变量 future_absret_5 和 future_logrv_20
    # future_absret_5: 未来5日平均绝对收益率
    df['future_absret_5'] = df['r_t'].shift(-1).rolling(5).apply(lambda x: np.mean(np.abs(x)), raw=True)

    # future_rv_20: 未来20日实现波动率
    future_rv_20 = df['r_t'].shift(-1).rolling(20).apply(lambda x: np.mean(np.array(x)**2), raw=True)
    df['future_logrv_20'] = np.log(EPS + future_rv_20)

    # 6. 构造 HAR 历史状态特征
    # 对于 future_absret_5
    df['past_absret_5'] = df['r_t'].rolling(5).apply(lambda x: np.mean(np.abs(x)), raw=True)
    df['past_absret_20'] = df['r_t'].rolling(20).apply(lambda x: np.mean(np.abs(x)), raw=True)
    df['past_absret_60'] = df['r_t'].rolling(60).apply(lambda x: np.mean(np.abs(x)), raw=True)

    # 对于 future_logrv_20
    past_rv_5 = df['r_t'].rolling(5).apply(lambda x: np.mean(np.array(x)**2), raw=True)
    df['past_logrv_5'] = np.log(EPS + past_rv_5)
    past_rv_20 = df['r_t'].rolling(20).apply(lambda x: np.mean(np.array(x)**2), raw=True)
    df['past_logrv_20'] = np.log(EPS + past_rv_20)
    past_rv_60 = df['r_t'].rolling(60).apply(lambda x: np.mean(np.array(x)**2), raw=True)
    df['past_logrv_60'] = np.log(EPS + past_rv_60)

    # 7. 按"上一个完整月可得信息"对齐月度宏观变量
    # 检查 epu 和 usd_cny 非正的情况
    if (df['epu'] <= 0).any() or (df['usd_cny'] <= 0).any():
        print("警告: 发现 epu 或 usd_cny 非正值!")
        print(f"  epu 非正值数量: {(df['epu'] <= 0).sum()}")
        print(f"  usd_cny 非正值数量: {(df['usd_cny'] <= 0).sum()}")

    # 构造4个宏观解释变量
    # epu_log_m1: 最新可得月度 log(epu)
    df['epu_log_m1'] = df.apply(lambda row: np.log(get_previous_month_info(df, row['date'], 'epu', 'epu') + EPS), axis=1)

    # fx_ret1_m1: 最新可得月度汇率对数变化
    # 需要计算当前月最后一个汇率值与上一个月最后一个汇率值的对数变化
    df['fx_ret1_m1'] = df.apply(lambda row: compute_fx_monthly_change(df, row['date']), axis=1)

    # ppi_yoy_m1: 最新可得月度 PPI 同比增速
    df['ppi_yoy_m1'] = df.apply(lambda row: get_previous_month_info(df, row['date'], 'ppi', 'ppi'), axis=1)

    # m2_delta1_m1: 最新可得月度 M2 增速的月度变化
    df['m2_delta1_m1'] = df.apply(lambda row: compute_monthly_change(df, row['date'], 'm2_growth'), axis=1)

    # 8. 删除缺失观测
    cols_to_check = ['r_t', 'future_absret_5', 'future_logrv_20',
                     'past_absret_5', 'past_absret_20', 'past_absret_60',
                     'past_logrv_5', 'past_logrv_20', 'past_logrv_60',
                     'epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']

    df_clean = df.dropna(subset=cols_to_check).copy()
    print(f"删除缺失后观测数: {len(df_clean)}")
    print(f"删除缺失后样本范围: {df_clean['date'].min()} 至 {df_clean['date'].max()}")

    # 9. 划分训练集和测试集（60%/40%，按时间顺序）
    n_total = len(df_clean)
    n_train = int(n_total * TRAIN_RATIO)
    n_test = n_total - n_train

    df_train = df_clean.iloc[:n_train].copy()
    df_test = df_clean.iloc[n_train:].copy()

    print(f"训练集: {len(df_train)} ({len(df_train)/n_total*100:.1f}%)")
    print(f"测试集: {len(df_test)} ({len(df_test)/n_total*100:.1f}%)")
    print(f"训练集范围: {df_train['date'].min()} 至 {df_train['date'].max()}")
    print(f"测试集范围: {df_test['date'].min()} 至 {df_test['date'].max()}")

    return df_clean, df_train, df_test

def compute_fx_monthly_change(df, current_date):
    """计算汇率月度对数变化"""
    current_date = pd.to_datetime(current_date)

    current_year = current_date.year
    current_month = current_date.month

    # 上一个月
    if current_month == 1:
        prev_year = current_year - 1
        prev_month = 12
    else:
        prev_year = current_year
        prev_month = current_month - 1

    # 获取上一个月最后一个汇率值
    prev_month_data = df[(df['date'].dt.year == prev_year) &
                         (df['date'].dt.month == prev_month)]
    if len(prev_month_data) == 0:
        return np.nan

    prev_fx = prev_month_data['usd_cny'].iloc[-1]
    prev_fx_prev_month = prev_month_data['usd_cny'].iloc[0]  # 上个月月初

    # 计算对数变化
    if prev_fx <= 0 or prev_fx_prev_month <= 0:
        return np.nan

    return np.log(prev_fx) - np.log(prev_fx_prev_month)


def descriptive_statistics(df_clean):
    """步骤七: 描述统计与初步分析"""
    print("\n" + "=" * 60)
    print("描述统计与初步分析")
    print("=" * 60)

    # A. 描述统计表
    vars_for_desc = ['r_t', 'future_absret_5', 'future_logrv_20',
                     'past_absret_5', 'past_absret_20', 'past_absret_60',
                     'past_logrv_5', 'past_logrv_20', 'past_logrv_60',
                     'epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']

    desc_stats = []
    for var in vars_for_desc:
        stats_dict = compute_statistics(df_clean[var])
        stats_dict['variable'] = var
        desc_stats.append(stats_dict)

    desc_df = pd.DataFrame(desc_stats)
    desc_df = desc_df[['variable', 'mean', 'std', 'min', 'max', 'p25', 'median', 'p75',
                       'skewness', 'kurtosis', 'n_valid']]

    desc_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_descriptive_stats.csv'), index=False)
    print("描述统计表已保存")
    print(desc_df.to_string())

    # B. 相关性分析
    # Group 1: future_absret_5 相关
    vars_group1 = ['future_absret_5', 'past_absret_5', 'past_absret_20', 'past_absret_60',
                   'epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']
    corr1 = df_clean[vars_group1].corr()
    corr1.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_corr_absret.csv'))
    print("\n相关性矩阵1 (future_absret_5):")
    print(corr1.to_string())

    # Group 2: future_logrv_20 相关
    vars_group2 = ['future_logrv_20', 'past_logrv_5', 'past_logrv_20', 'past_logrv_60',
                   'epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']
    corr2 = df_clean[vars_group2].corr()
    corr2.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_corr_logrv.csv'))
    print("\n相关性矩阵2 (future_logrv_20):")
    print(corr2.to_string())

    # D. VIF诊断（在建模阶段做）

    return desc_df, corr1, corr2


def run_ols_regression(X_train, y_train, X_test, y_test, y_train_mean, maxlags,
                       target_name, model_name, feature_names):
    """运行OLS回归并返回完整结果"""

    # 添加常数项
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # OLS估计
    model = OLS(y_train, X_train_const).fit()

    # HAC稳健标准误
    try:
        hac_model = model.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
    except:
        hac_model = model

    # 系数和统计量
    n_features = len(feature_names) + 1  # 包括常数项
    coef = hac_model.params
    std_err = hac_model.bse
    t_values = hac_model.tvalues
    p_values = hac_model.pvalues
    conf_int = hac_model.conf_int()
    ci_lower = conf_int[:, 0]  # 第一列是下限
    ci_upper = conf_int[:, 1]  # 第二列是上限

    # 标准化系数（需要标准化后的X）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_const = sm.add_constant(X_train_scaled)
    model_std = OLS(y_train, X_train_scaled_const).fit()
    std_coef = np.concatenate([[model_std.params[0]], model_std.params[1:] * scaler.scale_])

    # 预测
    y_train_pred = model.predict(X_train_const)
    y_test_pred = model.predict(X_test_const)

    # 样本内R²
    train_r2 = model.rsquared
    adj_r2_train = model.rsquared_adj

    # 样本外R²
    test_r2_os = compute_r2_os(y_test, y_test_pred, y_train_mean)

    # RMSE和MAE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 诊断检验
    residuals = model.resid

    # Durbin-Watson
    dw_stat = durbin_watson(residuals)

    # Ljung-Box检验
    try:
        lb_result_5 = acorr_ljungbox(residuals, lags=[5], return_df=True)
        lb_pvalue_5 = lb_result_5['lb_pvalue'].iloc[0]
        lb_result_10 = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue_10 = lb_result_10['lb_pvalue'].iloc[0]
    except:
        lb_pvalue_5 = np.nan
        lb_pvalue_10 = np.nan

    # ARCH-LM检验
    try:
        arch_result = het_arch(residuals, nlags=5)
        arch_pvalue = arch_result[1]
    except:
        arch_pvalue = np.nan

    # 条件数
    cond_number = np.linalg.cond(X_train_const)

    # AIC, BIC
    aic = model.aic
    bic = model.bic

    # 残差统计
    resid_mean = residuals.mean()
    resid_std = residuals.std()

    # 构建系数表
    coef_results = []
    all_names = ['const'] + feature_names
    for i, name in enumerate(all_names):
        coef_results.append({
            'target': target_name,
            'model': model_name,
            'variable': name,
            'coef': coef[i],
            'std_coef': std_coef[i] if i < len(std_coef) else np.nan,
            'std_error_hac': std_err[i],
            't_value': t_values[i],
            'p_value': p_values[i],
            'ci_lower': ci_lower[i],
            'ci_upper': ci_upper[i]
        })

    # 构建诊断结果
    diag_results = {
        'target': target_name,
        'model': model_name,
        'dw_stat': dw_stat,
        'lb_pvalue_lag5': lb_pvalue_5,
        'lb_pvalue_lag10': lb_pvalue_10,
        'arch_test_pvalue': arch_pvalue,
        'condition_number': cond_number,
        'aic': aic,
        'bic': bic,
        'resid_mean': resid_mean,
        'resid_std': resid_std
    }

    # 构建模型比较结果
    model_results = {
        'target': target_name,
        'model': model_name,
        'best_params': 'OLS',
        'train_r2': train_r2,
        'adj_r2_train': adj_r2_train,
        'test_r2_os': test_r2_os,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

    return coef_results, diag_results, model_results, y_test_pred, residuals


def run_ridge_regression(X_train, y_train, X_test, y_test, y_train_mean,
                         target_name, model_name, feature_names,
                         train_val_idx=None):
    """运行Ridge回归"""

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 如果提供了验证集索引，用于调参
    if train_val_idx is not None:
        n_train_total = len(X_train)
        n_val = int(n_train_total * VAL_RATIO_IN_TRAIN)
        n_sub_train = n_train_total - n_val

        X_sub_train = X_train_scaled[:n_sub_train]
        y_sub_train = y_train[:n_sub_train]
        X_val = X_train_scaled[n_sub_train:]
        y_val = y_train[n_sub_train:]

        best_alpha = None
        best_val_r2 = -np.inf

        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_sub_train, y_sub_train)
            y_val_pred = ridge.predict(X_val)
            val_r2 = compute_r2_os(y_val, y_val_pred, y_sub_train.mean())
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_alpha = alpha
    else:
        best_alpha = 1.0  # 默认值

    # 使用最佳alpha在全训练集上训练
    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train_scaled, y_train)

    # 预测
    y_train_pred = ridge_final.predict(X_train_scaled)
    y_test_pred = ridge_final.predict(X_test_scaled)

    # R²
    train_r2 = ridge_final.score(X_train_scaled, y_train)
    adj_r2_train = 1 - (1 - train_r2) * (len(y_train) - 1) / (len(y_train) - len(feature_names) - 1)
    test_r2_os = compute_r2_os(y_test, y_test_pred, y_train_mean)

    # RMSE和MAE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 标准化系数
    coef = ridge_final.coef_
    std_coef = coef * scaler.scale_
    intercept = ridge_final.intercept_

    # 构建系数表（Ridge没有标准误）
    coef_results = []
    for i, name in enumerate(feature_names):
        coef_results.append({
            'target': target_name,
            'model': model_name,
            'variable': name,
            'coef': coef[i],
            'std_coef': std_coef[i],
            'std_error_hac': np.nan,  # Ridge没有
            't_value': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        })
    coef_results.append({
        'target': target_name,
        'model': model_name,
        'variable': 'const',
        'coef': intercept,
        'std_coef': intercept,
        'std_error_hac': np.nan,
        't_value': np.nan,
        'p_value': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan
    })

    # 重要性排序
    importance_df = pd.DataFrame({
        'variable': feature_names,
        'abs_coef': np.abs(coef)
    }).sort_values('abs_coef', ascending=False)

    # 模型结果
    model_results = {
        'target': target_name,
        'model': model_name,
        'best_params': f'alpha={best_alpha}',
        'train_r2': train_r2,
        'adj_r2_train': adj_r2_train,
        'test_r2_os': test_r2_os,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

    # 诊断结果（Ridge不适用）
    diag_results = {
        'target': target_name,
        'model': model_name,
        'dw_stat': np.nan,
        'lb_pvalue_lag5': np.nan,
        'lb_pvalue_lag10': np.nan,
        'arch_test_pvalue': np.nan,
        'condition_number': np.nan,
        'aic': np.nan,
        'bic': np.nan,
        'resid_mean': np.nan,
        'resid_std': np.nan
    }

    return coef_results, diag_results, model_results, y_test_pred, importance_df


def run_incremental_test(har_results, harx_results, target_name, y_train, X_har, X_harx, maxlags):
    """嵌套增量检验"""

    # HAR模型
    X_har_const = sm.add_constant(X_har)
    model_har = OLS(y_train, X_har_const).fit()

    # HARX模型
    X_harx_const = sm.add_constant(X_harx)
    model_harx = OLS(y_train, X_harx_const).fit()

    # R²变化
    r2_change = model_harx.rsquared - model_har.rsquared
    adj_r2_change = model_harx.rsquared_adj - model_har.rsquared_adj

    # 联合F检验
    # HARX有7个特征（3 HAR + 4 macro），HAR有3个
    # 检验4个宏观变量是否联合显著
    n_macro = 4  # 宏观变量数量
    n_obs = len(y_train)
    n_params_harx = X_harx_const.shape[1]  # 使用shape代替columns
    n_params_har = X_har_const.shape[1]

    rss_har = model_har.ssr
    rss_harx = model_harx.ssr

    f_stat = ((rss_har - rss_harx) / n_macro) / (rss_harx / (n_obs - n_params_harx))
    f_pvalue = stats.f.sf(f_stat, n_macro, n_obs - n_params_harx)

    # Wald检验（使用HAC协方差）
    try:
        hac_harx = model_harx.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
        # 构建约束矩阵：检验宏观变量（索引4-7）是否联合为0
        R = np.zeros((n_macro, n_params_harx))
        for i in range(n_macro):
            R[i, 4 + i] = 1
        q = np.zeros(n_macro)

        wald_result = hac_harx.wald_test(R, q)
        wald_stat = wald_result.statistic
        wald_pvalue = wald_result.pvalue
    except:
        wald_stat = np.nan
        wald_pvalue = np.nan

    return {
        'target': target_name,
        'base_model': 'HAR_OLS',
        'extended_model': 'HARX_OLS',
        'r2_change': r2_change,
        'adj_r2_change': adj_r2_change,
        'f_test_stat': f_stat,
        'f_test_pvalue': f_pvalue,
        'wald_stat': wald_stat,
        'wald_pvalue': wald_pvalue
    }


def compute_vif(X, feature_names):
    """计算VIF"""
    X_const = sm.add_constant(X)
    vif_results = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X_const.values, i + 1)
            vif_results.append({
                'variable': name,
                'vif': vif
            })
        except:
            vif_results.append({
                'variable': name,
                'vif': np.nan
            })
    return pd.DataFrame(vif_results)


def run_full_regression_analysis(df_clean, df_train, df_test):
    """步骤八-十: 回归建模与评估"""
    print("\n" + "=" * 60)
    print("回归建模阶段")
    print("=" * 60)

    all_coef_results = []
    all_diag_results = []
    all_model_results = []
    all_incremental_results = []
    all_vif_results = []

    predictions_dict = {
        'date': df_test['date'].values,
        'target': [],
        'actual': [],
        'har_ols_pred': [],
        'harx_ols_pred': [],
        'harx_lite_pred': [],
        'har_ridge_pred': [],
        'harx_ridge_pred': []
    }

    # ==================== 目标1: future_absret_5 ====================
    print("\n处理目标: future_absret_5")

    target1 = 'future_absret_5'
    har_features1 = ['past_absret_5', 'past_absret_20', 'past_absret_60']
    macro_features = ['epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']
    harx_features1 = har_features1 + macro_features
    harx_lite_features = har_features1 + ['fx_ret1_m1', 'ppi_yoy_m1']

    y_train1 = df_train[target1].values
    y_test1 = df_test[target1].values
    y_train_mean1 = df_train[target1].mean()

    X_har_train1 = df_train[har_features1].values
    X_har_test1 = df_test[har_features1].values
    X_harx_train1 = df_train[harx_features1].values
    X_harx_test1 = df_test[harx_features1].values
    X_lite_train1 = df_train[harx_lite_features].values
    X_lite_test1 = df_test[harx_lite_features].values

    maxlags1 = 4  # future_absret_5

    # M0: Mean Benchmark
    y_test_pred_mean = np.full(len(y_test1), y_train_mean1)
    test_r2_os_mean = compute_r2_os(y_test1, y_test_pred_mean, y_train_mean1)
    test_rmse_mean = np.sqrt(mean_squared_error(y_test1, y_test_pred_mean))
    test_mae_mean = mean_absolute_error(y_test1, y_test_pred_mean)

    all_model_results.append({
        'target': target1,
        'model': 'Mean_Benchmark',
        'best_params': 'train_mean',
        'train_r2': np.nan,
        'adj_r2_train': np.nan,
        'test_r2_os': test_r2_os_mean,
        'test_rmse': test_rmse_mean,
        'test_mae': test_mae_mean
    })

    # M1: HAR_OLS
    print("  运行 HAR_OLS...")
    coef_m1, diag_m1, model_m1, y_pred_har_ols1, resid_har_ols1 = run_ols_regression(
        X_har_train1, y_train1, X_har_test1, y_test1, y_train_mean1, maxlags1,
        target1, 'HAR_OLS', har_features1
    )
    all_coef_results.extend(coef_m1)
    all_diag_results.append(diag_m1)
    all_model_results.append(model_m1)

    # M2: HARX_OLS
    print("  运行 HARX_OLS...")
    coef_m2, diag_m2, model_m2, y_pred_harx_ols1, resid_harx_ols1 = run_ols_regression(
        X_harx_train1, y_train1, X_harx_test1, y_test1, y_train_mean1, maxlags1,
        target1, 'HARX_OLS', harx_features1
    )
    all_coef_results.extend(coef_m2)
    all_diag_results.append(diag_m2)
    all_model_results.append(model_m2)

    # HARX-lite
    print("  运行 HARX_lite_OLS...")
    coef_lite, diag_lite, model_lite, y_pred_lite_ols1, resid_lite_ols1 = run_ols_regression(
        X_lite_train1, y_train1, X_lite_test1, y_test1, y_train_mean1, maxlags1,
        target1, 'HARX_lite_OLS', harx_lite_features
    )
    all_coef_results.extend(coef_lite)
    all_diag_results.append(diag_lite)
    all_model_results.append(model_lite)

    # M3: HAR_Ridge
    print("  运行 HAR_Ridge...")
    coef_m3, diag_m3, model_m3, y_pred_har_ridge1, imp_m3 = run_ridge_regression(
        X_har_train1, y_train1, X_har_test1, y_test1, y_train_mean1,
        target1, 'HAR_Ridge', har_features1, train_val_idx=True
    )
    all_coef_results.extend(coef_m3)
    all_diag_results.append(diag_m3)
    all_model_results.append(model_m3)

    # M4: HARX_Ridge
    print("  运行 HARX_Ridge...")
    coef_m4, diag_m4, model_m4, y_pred_harx_ridge1, imp_m4 = run_ridge_regression(
        X_harx_train1, y_train1, X_harx_test1, y_test1, y_train_mean1,
        target1, 'HARX_Ridge', harx_features1, train_val_idx=True
    )
    all_coef_results.extend(coef_m4)
    all_diag_results.append(diag_m4)
    all_model_results.append(model_m4)

    # 增量检验
    print("  运行增量检验...")
    inc_test1 = run_incremental_test(
        model_m1, model_m2, target1, y_train1, X_har_train1, X_harx_train1, maxlags1
    )
    all_incremental_results.append(inc_test1)

    # VIF
    print("  计算 VIF...")
    vif_harx1 = compute_vif(X_harx_train1, harx_features1)
    vif_harx1['model'] = 'HARX_OLS'
    vif_harx1['target'] = target1
    all_vif_results.append(vif_harx1)

    vif_lite1 = compute_vif(X_lite_train1, harx_lite_features)
    vif_lite1['model'] = 'HARX_lite_OLS'
    vif_lite1['target'] = target1
    all_vif_results.append(vif_lite1)

    # 保存预测
    predictions_dict['target'].append(target1)
    predictions_dict['actual'].append(y_test1)
    predictions_dict['har_ols_pred'].append(y_pred_har_ols1)
    predictions_dict['harx_ols_pred'].append(y_pred_harx_ols1)
    predictions_dict['harx_lite_pred'].append(y_pred_lite_ols1)
    predictions_dict['har_ridge_pred'].append(y_pred_har_ridge1)
    predictions_dict['harx_ridge_pred'].append(y_pred_harx_ridge1)

    # ==================== 目标2: future_logrv_20 ====================
    print("\n处理目标: future_logrv_20")

    target2 = 'future_logrv_20'
    har_features2 = ['past_logrv_5', 'past_logrv_20', 'past_logrv_60']
    harx_features2 = har_features2 + macro_features

    y_train2 = df_train[target2].values
    y_test2 = df_test[target2].values
    y_train_mean2 = df_train[target2].mean()

    X_har_train2 = df_train[har_features2].values
    X_har_test2 = df_test[har_features2].values
    X_harx_train2 = df_train[harx_features2].values
    X_harx_test2 = df_test[harx_features2].values

    maxlags2 = 19  # future_logrv_20

    # N1: HAR_OLS
    print("  运行 HAR_OLS...")
    coef_n1, diag_n1, model_n1, y_pred_har_ols2, resid_har_ols2 = run_ols_regression(
        X_har_train2, y_train2, X_har_test2, y_test2, y_train_mean2, maxlags2,
        target2, 'HAR_OLS', har_features2
    )
    all_coef_results.extend(coef_n1)
    all_diag_results.append(diag_n1)
    all_model_results.append(model_n1)

    # N2: HARX_OLS
    print("  运行 HARX_OLS...")
    coef_n2, diag_n2, model_n2, y_pred_harx_ols2, resid_harx_ols2 = run_ols_regression(
        X_harx_train2, y_train2, X_harx_test2, y_test2, y_train_mean2, maxlags2,
        target2, 'HARX_OLS', harx_features2
    )
    all_coef_results.extend(coef_n2)
    all_diag_results.append(diag_n2)
    all_model_results.append(model_n2)

    # N3: HAR_Ridge
    print("  运行 HAR_Ridge...")
    coef_n3, diag_n3, model_n3, y_pred_har_ridge2, imp_n3 = run_ridge_regression(
        X_har_train2, y_train2, X_har_test2, y_test2, y_train_mean2,
        target2, 'HAR_Ridge', har_features2, train_val_idx=True
    )
    all_coef_results.extend(coef_n3)
    all_diag_results.append(diag_n3)
    all_model_results.append(model_n3)

    # N4: HARX_Ridge
    print("  运行 HARX_Ridge...")
    coef_n4, diag_n4, model_n4, y_pred_harx_ridge2, imp_n4 = run_ridge_regression(
        X_harx_train2, y_train2, X_harx_test2, y_test2, y_train_mean2,
        target2, 'HARX_Ridge', harx_features2, train_val_idx=True
    )
    all_coef_results.extend(coef_n4)
    all_diag_results.append(diag_n4)
    all_model_results.append(model_n4)

    # 增量检验
    print("  运行增量检验...")
    inc_test2 = run_incremental_test(
        model_n1, model_n2, target2, y_train2, X_har_train2, X_harx_train2, maxlags2
    )
    all_incremental_results.append(inc_test2)

    # VIF
    print("  计算 VIF...")
    vif_harx2 = compute_vif(X_harx_train2, harx_features2)
    vif_harx2['model'] = 'HARX_OLS'
    vif_harx2['target'] = target2
    all_vif_results.append(vif_harx2)

    # 保存预测
    predictions_dict['target'].append(target2)
    predictions_dict['actual'].append(y_test2)
    predictions_dict['har_ols_pred'].append(y_pred_har_ols2)
    predictions_dict['harx_ols_pred'].append(y_pred_harx_ols2)
    predictions_dict['harx_lite_pred'].append(np.zeros(len(y_test2)))  # logrv不跑lite
    predictions_dict['har_ridge_pred'].append(y_pred_har_ridge2)
    predictions_dict['harx_ridge_pred'].append(y_pred_harx_ridge2)

    # ==================== 分样本稳定性检验 ====================
    print("\n分样本稳定性检验...")

    # 方案1: 2020-01-01前后
    df_train_early = df_train[df_train['date'] < '2020-01-01'].copy()
    df_train_late = df_train[df_train['date'] >= '2020-01-01'].copy()

    stability_results = []

    if len(df_train_early) > 60 and len(df_train_late) > 60:
        # Early period
        y_early = df_train_early[target1].values
        X_har_early = df_train_early[har_features1].values
        X_harx_early = df_train_early[harx_features1].values

        X_har_early_const = sm.add_constant(X_har_early)
        model_har_early = OLS(y_early, X_har_early_const).fit()
        try:
            model_har_early_hac = model_har_early.get_robustcov_results(cov_type='HAC', maxlags=maxlags1)
        except:
            model_har_early_hac = model_har_early

        X_harx_early_const = sm.add_constant(X_harx_early)
        model_harx_early = OLS(y_early, X_harx_early_const).fit()
        try:
            model_harx_early_hac = model_harx_early.get_robustcov_results(cov_type='HAC', maxlags=maxlags1)
        except:
            model_harx_early_hac = model_harx_early

        for i, name in enumerate(['const'] + har_features1):
            stability_results.append({
                'period': 'pre_2020',
                'model': 'HAR_OLS',
                'variable': name,
                'coef': model_har_early_hac.params[i],
                'p_value': model_har_early_hac.pvalues[i]
            })

        for i, name in enumerate(['const'] + harx_features1):
            stability_results.append({
                'period': 'pre_2020',
                'model': 'HARX_OLS',
                'variable': name,
                'coef': model_harx_early_hac.params[i],
                'p_value': model_harx_early_hac.pvalues[i]
            })

        # Late period
        y_late = df_train_late[target1].values
        X_har_late = df_train_late[har_features1].values
        X_harx_late = df_train_late[harx_features1].values

        X_har_late_const = sm.add_constant(X_har_late)
        model_har_late = OLS(y_late, X_har_late_const).fit()
        try:
            model_har_late_hac = model_har_late.get_robustcov_results(cov_type='HAC', maxlags=maxlags1)
        except:
            model_har_late_hac = model_har_late

        X_harx_late_const = sm.add_constant(X_harx_late)
        model_harx_late = OLS(y_late, X_harx_late_const).fit()
        try:
            model_harx_late_hac = model_harx_late.get_robustcov_results(cov_type='HAC', maxlags=maxlags1)
        except:
            model_harx_late_hac = model_harx_late

        for i, name in enumerate(['const'] + har_features1):
            stability_results.append({
                'period': 'post_2020',
                'model': 'HAR_OLS',
                'variable': name,
                'coef': model_har_late_hac.params[i],
                'p_value': model_har_late_hac.pvalues[i]
            })

        for i, name in enumerate(['const'] + harx_features1):
            stability_results.append({
                'period': 'post_2020',
                'model': 'HARX_OLS',
                'variable': name,
                'coef': model_harx_late_hac.params[i],
                'p_value': model_harx_late_hac.pvalues[i]
            })

    stability_df = pd.DataFrame(stability_results)
    stability_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_stability_test.csv'), index=False)

    return (all_coef_results, all_diag_results, all_model_results,
            all_incremental_results, all_vif_results, predictions_dict,
            stability_df)


def generate_plots_and_report(df_clean, df_train, df_test, predictions_dict,
                               all_coef_results, all_diag_results, all_model_results,
                               all_incremental_results, all_vif_results, stability_df):
    """生成图表和报告"""
    print("\n" + "=" * 60)
    print("生成图表与报告")
    print("=" * 60)

    # 创建PDF合并所有图表
    pdf_path = os.path.join(OUTPUT_DIR, 'harx_instability_plots.pdf')
    pdf = PdfPages(pdf_path)

    # 1. 时间序列图
    # future_absret_5
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_clean['date'], df_clean['future_absret_5'], 'b-', linewidth=0.8)
    ax.set_title('future_absret_5 时间序列')
    ax.set_xlabel('日期')
    ax.set_ylabel('值')
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # future_logrv_20
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_clean['date'], df_clean['future_logrv_20'], 'g-', linewidth=0.8)
    ax.set_title('future_logrv_20 时间序列')
    ax.set_xlabel('日期')
    ax.set_ylabel('值')
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # HAR历史特征对比
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_clean['date'], df_clean['past_absret_5'], 'b-', label='past_absret_5', linewidth=0.8)
    ax.plot(df_clean['date'], df_clean['past_absret_20'], 'g-', label='past_absret_20', linewidth=0.8)
    ax.plot(df_clean['date'], df_clean['past_absret_60'], 'r-', label='past_absret_60', linewidth=0.8)
    ax.set_title('HAR历史特征对比 (absret)')
    ax.set_xlabel('日期')
    ax.set_ylabel('值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 宏观变量时间序列
    macro_vars = ['epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i, var in enumerate(macro_vars):
        ax = axes[i // 2, i % 2]
        ax.plot(df_clean['date'], df_clean[var], linewidth=0.8)
        ax.set_title(var)
        ax.set_xlabel('日期')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 2. 散点图
    # future_absret_5散点图
    test_dates = predictions_dict['date']
    y_test1 = predictions_dict['actual'][0]
    y_pred_har_ols1 = predictions_dict['har_ols_pred'][0]
    y_pred_harx_ols1 = predictions_dict['harx_ols_pred'][0]
    y_pred_lite_ols1 = predictions_dict['harx_lite_pred'][0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # HAR_OLS
    ax = axes[0]
    ax.scatter(y_test1, y_pred_har_ols1, alpha=0.5, s=10)
    min_val = min(y_test1.min(), y_pred_har_ols1.min())
    max_val = max(y_test1.max(), y_pred_har_ols1.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_absret_5: HAR_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # HARX_OLS
    ax = axes[1]
    ax.scatter(y_test1, y_pred_harx_ols1, alpha=0.5, s=10)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_absret_5: HARX_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # HARX-lite
    ax = axes[2]
    ax.scatter(y_test1, y_pred_lite_ols1, alpha=0.5, s=10)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_absret_5: HARX_lite_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 单独保存PNG
    fig.savefig(os.path.join(OUTPUT_DIR, 'future_absret_5_scatter.png'), bbox_inches='tight')
    plt.close(fig)

    # future_logrv_20散点图
    y_test2 = predictions_dict['actual'][1]
    y_pred_har_ols2 = predictions_dict['har_ols_pred'][1]
    y_pred_harx_ols2 = predictions_dict['harx_ols_pred'][1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.scatter(y_test2, y_pred_har_ols2, alpha=0.5, s=10)
    min_val = min(y_test2.min(), y_pred_har_ols2.min())
    max_val = max(y_test2.max(), y_pred_har_ols2.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_logrv_20: HAR_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(y_test2, y_pred_harx_ols2, alpha=0.5, s=10)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_logrv_20: HARX_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    fig.savefig(os.path.join(OUTPUT_DIR, 'future_logrv_20_scatter.png'), bbox_inches='tight')
    plt.close(fig)

    # 3. 残差时间序列图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # future_absret_5残差
    resid_har_test1 = y_test1 - y_pred_har_ols1
    resid_harx_test1 = y_test1 - y_pred_harx_ols1

    ax = axes[0]
    ax.plot(test_dates, resid_har_test1, 'b-', label='HAR_OLS残差', linewidth=0.8)
    ax.plot(test_dates, resid_harx_test1, 'g-', label='HARX_OLS残差', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('future_absret_5: 测试集残差时间序列')
    ax.set_xlabel('日期')
    ax.set_ylabel('残差')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # future_logrv_20残差
    resid_har_test2 = y_test2 - y_pred_har_ols2
    resid_harx_test2 = y_test2 - y_pred_harx_ols2

    ax = axes[1]
    ax.plot(test_dates, resid_har_test2, 'b-', label='HAR_OLS残差', linewidth=0.8)
    ax.plot(test_dates, resid_harx_test2, 'g-', label='HARX_OLS残差', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('future_logrv_20: 测试集残差时间序列')
    ax.set_xlabel('日期')
    ax.set_ylabel('残差')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    fig.savefig(os.path.join(OUTPUT_DIR, 'future_absret_5_residual_ts.png'), bbox_inches='tight')
    plt.close(fig)

    # 4. 滚动绝对误差图
    fig, ax = plt.subplots(figsize=(12, 4))

    abs_err_har = np.abs(y_test1 - y_pred_har_ols1)
    abs_err_harx = np.abs(y_test1 - y_pred_harx_ols1)
    abs_err_lite = np.abs(y_test1 - y_pred_lite_ols1)

    window = 20
    rolling_abs_err_har = pd.Series(abs_err_har).rolling(window).mean()
    rolling_abs_err_harx = pd.Series(abs_err_harx).rolling(window).mean()
    rolling_abs_err_lite = pd.Series(abs_err_lite).rolling(window).mean()

    ax.plot(test_dates, rolling_abs_err_har, 'b-', label='HAR_OLS', linewidth=0.8)
    ax.plot(test_dates, rolling_abs_err_harx, 'g-', label='HARX_OLS', linewidth=0.8)
    ax.plot(test_dates, rolling_abs_err_lite, 'r-', label='HARX_lite', linewidth=0.8)
    ax.set_title(f'future_absret_5: 滚动{window}日平均绝对误差')
    ax.set_xlabel('日期')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 5. VIF图
    vif_combined = pd.concat(all_vif_results)
    vif_absret = vif_combined[vif_combined['target'] == 'future_absret_5']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, model in enumerate(['HARX_OLS', 'HARX_lite_OLS']):
        ax = axes[i]
        vif_data = vif_absret[vif_absret['model'] == model]
        ax.barh(vif_data['variable'], vif_data['vif'], color='steelblue')
        ax.axvline(x=5, color='r', linestyle='--', label='VIF=5阈值')
        ax.axvline(x=10, color='orange', linestyle='--', label='VIF=10阈值')
        ax.set_xlabel('VIF')
        ax.set_title(f'VIF诊断: {model}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 6. 分样本稳定性对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    stability_absret = stability_df

    # HAR系数对比
    har_stability = stability_absret[stability_absret['model'] == 'HAR_OLS']
    har_vars = ['past_absret_5', 'past_absret_20', 'past_absret_60']

    ax = axes[0, 0]
    x_pos = np.arange(len(har_vars))
    width = 0.35

    pre_vals = [har_stability[har_stability['variable'] == v]['coef'].values[0]
                for v in har_vars if len(har_stability[har_stability['variable'] == v]) > 0]
    post_vals = [har_stability[(har_stability['period'] == 'post_2020') &
                               (har_stability['variable'] == v)].coef.values[0]
                 for v in har_vars if len(har_stability[(har_stability['period'] == 'post_2020') &
                                                        (har_stability['variable'] == v)]) > 0]

    if len(pre_vals) == len(har_vars) and len(post_vals) == len(har_vars):
        ax.bar(x_pos - width/2, pre_vals, width, label='pre_2020', color='steelblue')
        ax.bar(x_pos + width/2, post_vals, width, label='post_2020', color='coral')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(har_vars)
        ax.set_ylabel('系数')
        ax.set_title('HAR系数稳定性')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # HARX宏观系数对比
    ax = axes[0, 1]
    harx_stability = stability_absret[stability_absret['model'] == 'HARX_OLS']
    macro_vars_stab = ['fx_ret1_m1', 'ppi_yoy_m1']

    pre_macro = [harx_stability[(harx_stability['period'] == 'pre_2020') &
                                (harx_stability['variable'] == v)].coef.values[0]
                 for v in macro_vars_stab if len(harx_stability[(harx_stability['period'] == 'pre_2020') &
                                                               (harx_stability['variable'] == v)]) > 0]
    post_macro = [harx_stability[(harx_stability['period'] == 'post_2020') &
                                 (harx_stability['variable'] == v)].coef.values[0]
                  for v in macro_vars_stab if len(harx_stability[(harx_stability['period'] == 'post_2020') &
                                                                (harx_stability['variable'] == v)]) > 0]

    if len(pre_macro) > 0 and len(post_macro) > 0:
        x_pos = np.arange(len(macro_vars_stab))
        ax.bar(x_pos - width/2, pre_macro, width, label='pre_2020', color='steelblue')
        ax.bar(x_pos + width/2, post_macro, width, label='post_2020', color='coral')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(macro_vars_stab)
        ax.set_ylabel('系数')
        ax.set_title('宏观变量系数稳定性')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 模型对比R²
    model_comp_df = pd.DataFrame(all_model_results)
    model_absret = model_comp_df[model_comp_df['target'] == 'future_absret_5']

    ax = axes[1, 0]
    models = ['HAR_OLS', 'HARX_OLS', 'HARX_lite_OLS', 'HAR_Ridge', 'HARX_Ridge']
    test_r2_vals = [model_absret[model_absret['model'] == m]['test_r2_os'].values[0]
                    if len(model_absret[model_absret['model'] == m]) > 0 else 0
                    for m in models]
    ax.bar(models, test_r2_vals, color='steelblue')
    ax.set_ylabel('样本外R²')
    ax.set_title('future_absret_5: 模型样本外表现对比')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    test_rmse_vals = [model_absret[model_absret['model'] == m]['test_rmse'].values[0]
                      if len(model_absret[model_absret['model'] == m]) > 0 else 0
                      for m in models]
    ax.bar(models, test_rmse_vals, color='coral')
    ax.set_ylabel('RMSE')
    ax.set_title('future_absret_5: 模型RMSE对比')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    pdf.close()
    print(f"图表已保存至: {pdf_path}")

    # HARX-lite单独散点图PNG
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test1, y_pred_lite_ols1, alpha=0.5, s=10)
    min_val = min(y_test1.min(), y_pred_lite_ols1.min())
    max_val = max(y_test1.max(), y_pred_lite_ols1.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title('future_absret_5: HARX_lite_OLS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, 'future_absret_5_harxlite_scatter.png'), bbox_inches='tight')
    plt.close(fig)

    # future_logrv_20残差图PNG
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_dates, resid_har_test2, 'b-', label='HAR_OLS残差', linewidth=0.8)
    ax.plot(test_dates, resid_harx_test2, 'g-', label='HARX_OLS残差', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('future_logrv_20: 测试集残差时间序列')
    ax.set_xlabel('日期')
    ax.set_ylabel('残差')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, 'future_logrv_20_residual_ts.png'), bbox_inches='tight')
    plt.close(fig)


def write_full_report(df_clean, all_coef_results, all_diag_results, all_model_results,
                      all_incremental_results, all_vif_results, stability_df):
    """撰写完整报告"""
    print("\n撰写报告...")

    report_path = os.path.join(OUTPUT_DIR, 'harx_instability_full_report.md')

    # 转换为DataFrame
    coef_df = pd.DataFrame(all_coef_results)
    diag_df = pd.DataFrame(all_diag_results)
    model_df = pd.DataFrame(all_model_results)
    inc_df = pd.DataFrame(all_incremental_results)
    vif_df = pd.concat(all_vif_results)

    # 提取关键信息
    model_absret = model_df[model_df['target'] == 'future_absret_5']
    model_logrv = model_df[model_df['target'] == 'future_logrv_20']

    coef_absret_harx = coef_df[(coef_df['target'] == 'future_absret_5') &
                               (coef_df['model'] == 'HARX_OLS')]
    coef_absret_lite = coef_df[(coef_df['target'] == 'future_absret_5') &
                               (coef_df['model'] == 'HARX_lite_OLS')]

    diag_absret_harx = diag_df[(diag_df['target'] == 'future_absret_5') &
                               (diag_df['model'] == 'HARX_OLS')]

    inc_absret = inc_df[inc_df['target'] == 'future_absret_5']
    inc_logrv = inc_df[inc_df['target'] == 'future_logrv_20']

    vif_absret = vif_df[vif_df['target'] == 'future_absret_5']

    # 判断结论
    # Q1: future_absret_5是否可作为主因变量
    # 查看其描述统计
    desc_absret = compute_statistics(df_clean['future_absret_5'])

    # Q2: 多时间尺度持续性
    coef_har = coef_df[(coef_df['target'] == 'future_absret_5') &
                       (coef_df['model'] == 'HAR_OLS')]
    har_significant = coef_har[coef_har['p_value'] < 0.05]

    # Q3: 宏观变量增量解释力
    inc_absret_row = inc_absret.iloc[0] if len(inc_absret) > 0 else None
    macro_joint_sig = (inc_absret_row['f_test_pvalue'] < 0.05) if inc_absret_row is not None else False

    # Q4: 哪些宏观变量值得保留
    coef_macro = coef_absret_harx[coef_absret_harx['variable'].isin(
        ['epu_log_m1', 'fx_ret1_m1', 'ppi_yoy_m1', 'm2_delta1_m1'])]
    sig_macro = coef_macro[coef_macro['p_value'] < 0.10]

    # Q5: HARX-lite是否更稳
    vif_harx = vif_absret[vif_absret['model'] == 'HARX_OLS']['vif'].max()
    vif_lite = vif_absret[vif_absret['model'] == 'HARX_lite_OLS']['vif'].max()

    lite_more_stable = vif_lite < vif_harx

    # Q6: future_logrv_20作为辅助
    test_r2_absret_harx = model_absret[model_absret['model'] == 'HARX_OLS']['test_r2_os'].values[0]
    test_r2_logrv_harx = model_logrv[model_logrv['model'] == 'HARX_OLS']['test_r2_os'].values[0]

    # Q7: 最终判断
    if har_significant.shape[0] >= 2 and macro_joint_sig and test_r2_absret_harx > 0:
        final_conclusion = 'A'
    elif test_r2_absret_harx > 0:
        final_conclusion = 'B'
    else:
        final_conclusion = 'C'

    # 撰写报告
    report = f"""# 基于 HARX 回归的股票市场短期不稳定性及宏观变量增量解释作用研究

## 一、研究定位

本轮实验围绕以下研究问题展开：

1. 股票市场短期不稳定性是否具有显著的多时间尺度持续性？
2. 在控制这种持续性之后，宏观变量是否能够对短期不稳定性提供增量解释？
3. 哪些宏观变量更值得保留为短期不稳定性的候选解释变量？

核心方法：HAR_OLS 和 HARX_OLS（使用 Newey-West HAC 稳健标准误）。

## 二、数据与样本

### 样本概况

- 样本期：{df_clean['date'].min().strftime('%Y-%m-%d')} 至 {df_clean['date'].max().strftime('%Y-%m-%d')}
- 有效观测数：{len(df_clean)}
- 训练集：{int(len(df_clean) * 0.6)} (60%)
- 测试集：{len(df_clean) - int(len(df_clean) * 0.6)} (40%)

### 因变量描述

**future_absret_5**（主因变量）：
- 均值：{desc_absret['mean']:.6f}
- 标准差：{desc_absret['std']:.6f}
- 范围：[{desc_absret['min']:.6f}, {desc_absret['max']:.6f}]
- 峰度：{desc_absret['kurtosis']:.4f}
- 偏度：{desc_absret['skewness']:.4f}

## 三、主结果：future_absret_5

### 3.1 HAR_OLS基准模型

| 变量 | 系数 | 标准误(HAC) | t值 | p值 | 95%置信区间 |
|------|------|------------|-----|-----|------------|
"""

    # HAR_OLS系数表
    coef_har_sorted = coef_har.sort_values('variable')
    for _, row in coef_har_sorted.iterrows():
        ci = f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
        report += f"| {row['variable']} | {row['coef']:.6f} | {row['std_error_hac']:.6f} | {row['t_value']:.4f} | {row['p_value']:.4f} | {ci} |\n"

    # HAR_OLS诊断
    diag_har = diag_df[(diag_df['target'] == 'future_absret_5') &
                       (diag_df['model'] == 'HAR_OLS')].iloc[0]

    report += f"""
**模型诊断**：
- R² = {model_absret[model_absret['model'] == 'HAR_OLS']['train_r2'].values[0]:.4f}
- Adj. R² = {model_absret[model_absret['model'] == 'HAR_OLS']['adj_r2_train'].values[0]:.4f}
- AIC = {diag_har['aic']:.2f}, BIC = {diag_har['bic']:.2f}
- Durbin-Watson = {diag_har['dw_stat']:.4f}
- Ljung-Box p值 (lag5) = {diag_har['lb_pvalue_lag5']:.4f}
- Ljung-Box p值 (lag10) = {diag_har['lb_pvalue_lag10']:.4f}
- ARCH-LM p值 = {diag_har['arch_test_pvalue']:.4f}
- 条件数 = {diag_har['condition_number']:.2f}

### 3.2 HARX_OLS扩展模型

| 变量 | 系数 | 标准误(HAC) | t值 | p值 | 95%置信区间 |
|------|------|------------|-----|-----|------------|
"""

    # HARX_OLS系数表
    coef_harx_sorted = coef_absret_harx.sort_values('variable')
    for _, row in coef_harx_sorted.iterrows():
        ci = f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
        report += f"| {row['variable']} | {row['coef']:.6f} | {row['std_error_hac']:.6f} | {row['t_value']:.4f} | {row['p_value']:.4f} | {ci} |\n"

    diag_harx = diag_absret_harx.iloc[0]

    report += f"""
**模型诊断**：
- R² = {model_absret[model_absret['model'] == 'HARX_OLS']['train_r2'].values[0]:.4f}
- Adj. R² = {model_absret[model_absret['model'] == 'HARX_OLS']['adj_r2_train'].values[0]:.4f}
- AIC = {diag_harx['aic']:.2f}, BIC = {diag_harx['bic']:.2f}
- Durbin-Watson = {diag_harx['dw_stat']:.4f}
- Ljung-Box p值 (lag5) = {diag_harx['lb_pvalue_lag5']:.4f}
- Ljung-Box p值 (lag10) = {diag_harx['lb_pvalue_lag10']:.4f}
- ARCH-LM p值 = {diag_harx['arch_test_pvalue']:.4f}
- 条件数 = {diag_harx['condition_number']:.2f}

### 3.3 HARX-lite精简模型

| 变量 | 系数 | 标准误(HAC) | t值 | p值 | 95%置信区间 |
|------|------|------------|-----|-----|------------|
"""

    # HARX-lite系数表
    coef_lite_sorted = coef_absret_lite.sort_values('variable')
    for _, row in coef_lite_sorted.iterrows():
        ci = f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
        report += f"| {row['variable']} | {row['coef']:.6f} | {row['std_error_hac']:.6f} | {row['t_value']:.4f} | {row['p_value']:.4f} | {ci} |\n"

    diag_lite = diag_df[(diag_df['target'] == 'future_absret_5') &
                        (diag_df['model'] == 'HARX_lite_OLS')].iloc[0]

    report += f"""
**模型诊断**：
- R² = {model_absret[model_absret['model'] == 'HARX_lite_OLS']['train_r2'].values[0]:.4f}
- Adj. R² = {model_absret[model_absret['model'] == 'HARX_lite_OLS']['adj_r2_train'].values[0]:.4f}
- AIC = {diag_lite['aic']:.2f}, BIC = {diag_lite['bic']:.2f}
- 条件数 = {diag_lite['condition_number']:.2f}

### 3.4 增量检验结果

| 检验 | 统计量 | p值 |
|------|--------|-----|
"""

    if len(inc_absret) > 0:
        inc_row = inc_absret.iloc[0]
        report += f"| R²变化 | {inc_row['r2_change']:.6f} | - |\n"
        report += f"| Adj. R²变化 | {inc_row['adj_r2_change']:.6f} | - |\n"
        report += f"| 联合F检验 | {inc_row['f_test_stat']:.4f} | {inc_row['f_test_pvalue']:.4f} |\n"
        report += f"| Wald检验(HAC) | {inc_row['wald_stat']:.4f} | {inc_row['wald_pvalue']:.4f} |\n"

    report += f"""
### 3.5 共线性诊断

**HARX_OLS VIF**：
"""

    vif_harx_data = vif_absret[vif_absret['model'] == 'HARX_OLS']
    for _, row in vif_harx_data.iterrows():
        report += f"- {row['variable']}: {row['vif']:.2f}\n"

    report += f"""
**HARX_lite VIF**：
"""

    vif_lite_data = vif_absret[vif_absret['model'] == 'HARX_lite_OLS']
    for _, row in vif_lite_data.iterrows():
        report += f"- {row['variable']}: {row['vif']:.2f}\n"

    report += f"""
**判断**：
- HARX_OLS最大VIF = {vif_harx:.2f}
- HARX_lite最大VIF = {vif_lite:.2f}
- {"存在中度共线性(VIF>5)" if vif_harx > 5 else "无明显共线性"}
- {"HARX-lite共线性更低，更稳定" if lite_more_stable else "HARX-lite未明显改善共线性"}

### 3.6 样本外评估

| 模型 | Train R² | Adj. R² | Test R²(OS) | Test RMSE | Test MAE |
|------|----------|---------|-------------|-----------|----------|
"""

    for _, row in model_absret.iterrows():
        if row['model'] in ['Mean_Benchmark', 'HAR_OLS', 'HARX_OLS', 'HARX_lite_OLS', 'HAR_Ridge', 'HARX_Ridge']:
            train_r2_str = f"{row['train_r2']:.4f}" if pd.notna(row['train_r2']) else 'N/A'
            adj_r2_str = f"{row['adj_r2_train']:.4f}" if pd.notna(row['adj_r2_train']) else 'N/A'
            report += f"| {row['model']} | {train_r2_str} | {adj_r2_str} | {row['test_r2_os']:.4f} | {row['test_rmse']:.6f} | {row['test_mae']:.6f} |\n"

    report += f"""
## 四、辅助稳健性目标：future_logrv_20

### 4.1 模型比较

| 模型 | Train R² | Test R²(OS) | Test RMSE | Test MAE |
|------|----------|-------------|-----------|----------|
"""

    for _, row in model_logrv.iterrows():
        if row['model'] in ['HAR_OLS', 'HARX_OLS', 'HAR_Ridge', 'HARX_Ridge']:
            report += f"| {row['model']} | {row['train_r2']:.4f} | {row['test_r2_os']:.4f} | {row['test_rmse']:.6f} | {row['test_mae']:.6f} |\n"

    report += f"""
### 4.2 增量检验

"""

    if len(inc_logrv) > 0:
        inc_logrv_row = inc_logrv.iloc[0]
        report += f"- R²变化: {inc_logrv_row['r2_change']:.6f}\n"
        report += f"- 联合F检验: {inc_logrv_row['f_test_stat']:.4f} (p={inc_logrv_row['f_test_pvalue']:.4f})\n"

    report += f"""
## 五、分样本稳定性检验

### 5.1 2020前后对比

**HAR系数稳定性**：

"""

    har_stability = stability_df[stability_df['model'] == 'HAR_OLS']
    if len(har_stability) > 0:
        report += "| 变量 | pre_2020系数 | pre_2020 p值 | post_2020系数 | post_2020 p值 |\n"
        report += "|------|--------------|--------------|---------------|---------------|\n"
        for var in ['past_absret_5', 'past_absret_20', 'past_absret_60']:
            pre_row = har_stability[(har_stability['period'] == 'pre_2020') & (har_stability['variable'] == var)]
            post_row = har_stability[(har_stability['period'] == 'post_2020') & (har_stability['variable'] == var)]
            if len(pre_row) > 0 and len(post_row) > 0:
                report += f"| {var} | {pre_row['coef'].values[0]:.6f} | {pre_row['p_value'].values[0]:.4f} | {post_row['coef'].values[0]:.6f} | {post_row['p_value'].values[0]:.4f} |\n"

    report += f"""
**宏观变量系数稳定性**：

"""

    harx_stability = stability_df[stability_df['model'] == 'HARX_OLS']
    if len(harx_stability) > 0:
        report += "| 变量 | pre_2020系数 | pre_2020 p值 | post_2020系数 | post_2020 p值 |\n"
        report += "|------|--------------|--------------|---------------|---------------|\n"
        for var in ['fx_ret1_m1', 'ppi_yoy_m1']:
            pre_row = harx_stability[(harx_stability['period'] == 'pre_2020') & (harx_stability['variable'] == var)]
            post_row = harx_stability[(harx_stability['period'] == 'post_2020') & (harx_stability['variable'] == var)]
            if len(pre_row) > 0 and len(post_row) > 0:
                report += f"| {var} | {pre_row['coef'].values[0]:.6f} | {pre_row['p_value'].values[0]:.4f} | {post_row['coef'].values[0]:.6f} | {post_row['p_value'].values[0]:.4f} |\n"

    report += f"""
## 六、研究问题回答

### Q1: future_absret_5 是否可以作为第一阶段主因变量？

**回答**：{("可以。该变量具有明确的经济学含义（未来5日平均波动幅度），描述统计显示其分布合理，且模型样本外R²>0。" if test_r2_absret_harx > 0 else "谨慎可接受。样本外预测能力较弱，但变量构造符合研究目的。")}

### Q2: 短期不稳定性是否具有显著的多时间尺度持续性？

**回答**：{("是。HAR_OLS中，" + "、".join(har_significant['variable'].tolist()) + "的系数在5%水平显著，表明不同时间尺度的历史波动对未来不稳定性具有显著的预测作用。" if len(har_significant) >= 2 else "部分支持。仅部分HAR特征显著，多尺度持续性证据较弱。")}

### Q3: 在HAR基准之上，宏观变量是否具有联合增量解释力？

**回答**：{("是。联合F检验p值=" + f"{inc_absret_row['f_test_pvalue']:.4f}" + "，宏观变量块联合显著。" if macro_joint_sig else "否或弱。联合F检验未通过显著性检验，宏观变量增量解释力有限。") if inc_absret_row is not None else "数据不足，无法判断。"}

### Q4: 哪些宏观变量更值得保留？

**回答**：{("在10%显著性水平下，" + "、".join(sig_macro['variable'].tolist()) + "显著。其中fx_ret1_m1和ppi_yoy_m1在HARX-lite中表现稳定，建议优先保留。" if len(sig_macro) > 0 else "当前四个宏观变量均未在10%水平显著，但仍可基于经济含义保留fx_ret1_m1和ppi_yoy_m1作为候选。")}

### Q5: HARX-lite是否比完整HARX更稳？

**回答**：{("是。HARX-lite的VIF更低(" + f"{vif_lite:.2f} vs {vif_harx:.2f}" + ")，且样本外表现与完整HARX相当，建议采用精简版。" if lite_more_stable else "否。HARX-lite未明显改善共线性，完整HARX与精简版表现相近。")}

### Q6: future_logrv_20是否只能作为辅助稳健性目标？

**回答**：{("是。该变量样本外R²=" + f"{test_r2_logrv_harx:.4f}" + "，低于future_absret_5，且经济学含义（实现波动率对数）不如前者直观，建议仅作为辅助。" if test_r2_logrv_harx < test_r2_absret_harx else "两者表现相近，可作为并列目标，但仍建议主目标为future_absret_5。")}

### Q7: 最终判断

**判断**：**{final_conclusion}**

- **A**：可正式采用 HAR / HARX 方案作为第一阶段主框架
- **B**：可作为补充方案，但仍不建议正式替代
- **C**：改善仍有限，应继续弱化第一阶段角色

**理由**：
"""

    if final_conclusion == 'A':
        report += """
1. HAR基准模型显示显著的多尺度持续性
2. 宏观变量块联合增量解释力显著
3. 样本外预测能力达标
4. HARX-lite可作为稳健精简版替代

建议正式表述为："基于HARX回归的股票市场短期不稳定性基准模型"。
"""
    elif final_conclusion == 'B':
        report += """
1. HAR基准模型部分成立
2. 宏观变量增量解释力有限或不稳健
3. 样本外预测能力勉强达标

建议作为补充分析，谨慎写入论文正文。
"""
    else:
        report += """
1. HAR持续性证据不足
2. 宏观变量解释力弱
3. 样本外预测效果差

不建议正式替代，应考虑弱化第一阶段角色或重新设计。
"""

    report += f"""
## 七、输出文件清单

本次实验输出以下文件：

1. `harx_instability_model_comparison.csv` - 模型比较表
2. `harx_instability_coefficients.csv` - 系数表
3. `harx_instability_incremental_tests.csv` - 增量检验结果
4. `harx_instability_diagnostics.csv` - 诊断结果
5. `harx_instability_vif.csv` - VIF诊断
6. `harx_instability_test_predictions.csv` - 测试集预测
7. `harx_instability_stability_test.csv` - 分样本稳定性检验
8. `harx_instability_descriptive_stats.csv` - 描述统计
9. `harx_instability_corr_absret.csv` - 相关性矩阵(absret)
10. `harx_instability_corr_logrv.csv` - 相关性矩阵(logrv)
11. `harx_instability_plots.pdf` - 图表合并PDF
12. `harx_instability_full_report.md` - 本报告

---

**实验完成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存至: {report_path}")

    return report


def save_all_results(all_coef_results, all_diag_results, all_model_results,
                     all_incremental_results, all_vif_results, predictions_dict):
    """保存所有结果文件"""
    print("\n保存结果文件...")

    # 1. 模型比较表
    model_df = pd.DataFrame(all_model_results)
    model_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_model_comparison.csv'),
                    index=False, float_format='%.6f')

    # 2. 系数表
    coef_df = pd.DataFrame(all_coef_results)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_coefficients.csv'),
                   index=False, float_format='%.6f')

    # 3. 增量检验
    inc_df = pd.DataFrame(all_incremental_results)
    inc_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_incremental_tests.csv'),
                  index=False, float_format='%.6f')

    # 4. 诊断结果
    diag_df = pd.DataFrame(all_diag_results)
    diag_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_diagnostics.csv'),
                   index=False, float_format='%.6f')

    # 5. VIF
    vif_df = pd.concat(all_vif_results)
    vif_df.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_vif.csv'),
                  index=False, float_format='%.4f')

    # 6. 测试集预测
    pred_df = pd.DataFrame({
        'date': predictions_dict['date']
    })

    for i, target in enumerate(predictions_dict['target']):
        target_pred_df = pd.DataFrame({
            'date': predictions_dict['date'],
            'target': target,
            'actual': predictions_dict['actual'][i],
            'har_ols_pred': predictions_dict['har_ols_pred'][i],
            'harx_ols_pred': predictions_dict['harx_ols_pred'][i],
            'harx_lite_pred': predictions_dict['harx_lite_pred'][i],
            'har_ridge_pred': predictions_dict['har_ridge_pred'][i],
            'harx_ridge_pred': predictions_dict['harx_ridge_pred'][i]
        })
        if i == 0:
            pred_combined = target_pred_df
        else:
            pred_combined = pd.concat([pred_combined, target_pred_df])

    pred_combined.to_csv(os.path.join(OUTPUT_DIR, 'harx_instability_test_predictions.csv'),
                          index=False, float_format='%.6f')

    print("所有结果文件已保存")


def main():
    """主函数"""
    print("=" * 60)
    print("基于 HARX 回归的股票市场短期不稳定性研究")
    print("=" * 60)

    # 步骤1-10: 数据预处理
    df_clean, df_train, df_test = load_and_preprocess_data()

    # 步骤七: 描述统计
    desc_df, corr1, corr2 = descriptive_statistics(df_clean)

    # 步骤八-十: 回归建模
    (all_coef_results, all_diag_results, all_model_results,
     all_incremental_results, all_vif_results, predictions_dict,
     stability_df) = run_full_regression_analysis(df_clean, df_train, df_test)

    # 保存结果
    save_all_results(all_coef_results, all_diag_results, all_model_results,
                     all_incremental_results, all_vif_results, predictions_dict)

    # 生成图表和报告
    generate_plots_and_report(df_clean, df_train, df_test, predictions_dict,
                               all_coef_results, all_diag_results, all_model_results,
                               all_incremental_results, all_vif_results, stability_df)

    report = write_full_report(df_clean, all_coef_results, all_diag_results,
                                all_model_results, all_incremental_results,
                                all_vif_results, stability_df)

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()