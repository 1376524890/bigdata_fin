#!/usr/bin/env python3
"""
第一阶段替代模型实验：Ridge / Elastic Net / PCR 与 Baseline 比较
严格遵循数据口径、样本期、变量定义和评价指标
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "real_data_complete.csv"
RESULT_DIR = BASE_DIR / "har" / "results"
FIGURE_DIR = RESULT_DIR / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (5, 60)
K_MONTHS = 12
MACRO_VARS = ["cpi", "ppi", "m2_growth", "epu", "usd_cny"]


def compute_future_return(log_ret: pd.Series, h: int) -> pd.Series:
    """计算未来h期对数收益累积和 R_t^{(h)} = sum_{s=1}^{h} r_{t+s}"""
    future_parts = [log_ret.shift(-s) for s in range(1, h + 1)]
    return pd.concat(future_parts, axis=1).sum(axis=1)


def load_and_prepare_data():
    """加载并准备数据"""
    print("\n【1】加载并准备数据...")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 过滤样本期（严格遵循）
    df = df[(df["date"] >= "2015-07-02") & (df["date"] <= "2025-12-25")].copy()

    # 计算对数收益率
    df["log_return"] = np.log(df["hs300_close"] / df["hs300_close"].shift(1))

    # 计算未来收益窗口 R_t^{(h)}
    for h in WINDOWS:
        df[f"R_{h}d"] = compute_future_return(df["log_return"], h)

    # 时间索引
    df["year_month"] = df["date"].dt.to_period("M")

    print(f"样本期: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"有效观测数: {len(df)}")
    return df


def split_train_test(df, train_ratio=0.6):
    """划分训练集和测试集（按时间顺序，前60%训练后40%测试）"""
    print("\n【2】划分训练集与测试集（60%/40%）...")

    # 先删除缺失值
    df_valid = df.dropna(subset=["R_5d", "R_60d"] + MACRO_VARS).copy()
    df_valid = df_valid.reset_index(drop=True)  # 重置索引

    train_size = int(len(df_valid) * train_ratio)
    df_valid["is_train"] = False
    df_valid.loc[:train_size - 1, "is_train"] = True

    df_train = df_valid[df_valid["is_train"]].copy()
    df_test = df_valid[~df_valid["is_train"]].copy()
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"训练集: {len(df_train)} ({df_train['date'].min().strftime('%Y-%m-%d')} 至 {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"测试集: {len(df_test)} ({df_test['date'].min().strftime('%Y-%m-%d')} 至 {df_test['date'].max().strftime('%Y-%m-%d')})")

    return df_valid, df_train, df_test, train_size


def build_monthly_available_panel(df, macro_vars=MACRO_VARS):
    """构建月度可得信息面板（滞后一期）- 严格遵循信息原则"""
    monthly_panel = df.groupby("year_month")[macro_vars].last().sort_index()
    monthly_available = monthly_panel.shift(1)  # 上一个完整月可得信息
    return monthly_available


def build_60_dim_features(df, monthly_available, macro_vars=MACRO_VARS, K=12):
    """构建60维完整月滞后特征：每个宏观变量 lagm1 到 lagm12"""
    feature_names = []
    feature_data = []

    for var in macro_vars:
        for lag in range(1, K + 1):
            feature_names.append(f"{var}_lagm{lag}")

    # 为每个交易日构建特征（使用位置索引）
    X_full = np.full((len(df), len(feature_names)), np.nan)

    for pos_idx in range(len(df)):
        row = df.iloc[pos_idx]
        current_month = pd.Period(row["date"], freq="M")
        for var_idx, var in enumerate(macro_vars):
            for lag in range(1, K + 1):
                target_month = current_month - lag
                col_idx = var_idx * K + (lag - 1)
                if target_month in monthly_available.index:
                    X_full[pos_idx, col_idx] = monthly_available.loc[target_month, var]

    return feature_names, X_full


def build_15_dim_features(df, monthly_available, macro_vars=MACRO_VARS, K=12):
    """构建15维压缩特征：每个宏观变量 m1, m3avg, m12avg"""
    feature_names = []
    feature_data = []

    for var in macro_vars:
        feature_names.extend([f"{var}_m1", f"{var}_m3avg", f"{var}_m12avg"])

    X_full = np.full((len(df), len(feature_names)), np.nan)

    for pos_idx in range(len(df)):
        row = df.iloc[pos_idx]
        current_month = pd.Period(row["date"], freq="M")
        for var_idx, var in enumerate(macro_vars):
            # m1: 最近一个月
            m1_idx = current_month - 1
            if m1_idx in monthly_available.index:
                X_full[pos_idx, var_idx * 3 + 0] = monthly_available.loc[m1_idx, var]

            # m3avg: 最近三个月平均
            m3_vals = []
            for lag in range(1, 4):
                target_month = current_month - lag
                if target_month in monthly_available.index:
                    m3_vals.append(monthly_available.loc[target_month, var])
            if len(m3_vals) > 0:
                X_full[pos_idx, var_idx * 3 + 1] = np.mean(m3_vals)

            # m12avg: 最近12个月平均
            m12_vals = []
            for lag in range(1, 13):
                target_month = current_month - lag
                if target_month in monthly_available.index:
                    m12_vals.append(monthly_available.loc[target_month, var])
            if len(m12_vals) > 0:
                X_full[pos_idx, var_idx * 3 + 2] = np.mean(m12_vals)

    return feature_names, X_full


def beta_weight_vector(K, a, b):
    """计算Beta权重向量"""
    x = (np.arange(K) + 1) / (K + 1)
    w = (x ** (a - 1)) * ((1 - x) ** (b - 1))
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    return w


def build_midas_term(dates, monthly_available, var_name, K, a, b):
    """为给定日期序列构建MIDAS项"""
    weights = beta_weight_vector(K, a, b)
    out = []

    for d in dates:
        current_month = pd.Period(d, freq="M")
        lags = []
        for ell in range(K):
            target_month = current_month - (ell + 1)
            if target_month in monthly_available.index:
                lags.append(monthly_available.loc[target_month, var_name])
            else:
                lags.append(np.nan)

        if np.any(pd.isna(lags)):
            out.append(np.nan)
        else:
            out.append(np.dot(weights, np.asarray(lags)))

    return np.asarray(out)


def fit_single_midas_nls(df_hist, y_col, monthly_available, var_name, K=12):
    """单变量MIDAS模型的NLS估计（Baseline复现）"""
    y = df_hist[y_col].values
    dates = df_hist["date"]

    def objective(params):
        alpha, beta, a, b = params
        a = max(a, 1e-4)
        b = max(b, 1e-4)

        x_midas = build_midas_term(dates, monthly_available, var_name, K, a, b)
        mask = (~np.isnan(y)) & (~np.isnan(x_midas))
        if mask.sum() < 5:
            return 1e10
        resid = y[mask] - (alpha + beta * x_midas[mask])
        return np.sum(resid ** 2)

    x0 = np.array([np.nanmean(y), 0.0, 1.5, 1.5])
    bounds = [(None, None), (None, None), (1e-4, 50), (1e-4, 50)]

    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")

    alpha, beta, a, b = res.x
    x_midas = build_midas_term(dates, monthly_available, var_name, K, a, b)
    fitted = alpha + beta * x_midas

    return {
        "success": bool(res.success),
        "alpha": alpha,
        "beta": beta,
        "a": a,
        "b": b,
        "objective": float(res.fun),
        "x_midas": x_midas,
        "fitted": fitted,
    }


def run_baseline_stage1(df_train, df_test, y_col, monthly_available, macro_vars=MACRO_VARS, K=12):
    """复现Baseline_Stage1：单变量MIDAS + VIF筛选 + 精简模型"""
    print(f"\n  运行 Baseline_Stage1 ({y_col})...")

    # 1) 在训练集上拟合每个单变量MIDAS
    uni_fits = {}
    uni_preds_train = {var: [] for var in macro_vars}

    for var in macro_vars:
        fit = fit_single_midas_nls(df_train, y_col, monthly_available, var, K)
        uni_fits[var] = fit
        # 训练集预测
        alpha, beta, a, b = fit["alpha"], fit["beta"], fit["a"], fit["b"]
        x_midas_train = build_midas_term(df_train["date"].values, monthly_available, var, K, a, b)
        uni_preds_train[var] = alpha + beta * x_midas_train

    # 2) VIF检验（在训练集上）
    X_vif_list = []
    valid_vars_for_vif = []

    for var in macro_vars:
        fit = uni_fits[var]
        x_midas = build_midas_term(df_train["date"].values, monthly_available, var, K, fit["a"], fit["b"])
        if not np.all(np.isnan(x_midas)):
            X_vif_list.append(x_midas)
            valid_vars_for_vif.append(var)

    vif_data = []
    if len(X_vif_list) > 1:
        X_vif = np.column_stack(X_vif_list)
        mask_vif = ~np.isnan(X_vif).any(axis=1)
        X_vif_clean = X_vif[mask_vif]
        if len(X_vif_clean) > 1:
            for i, var in enumerate(valid_vars_for_vif):
                try:
                    vif_val = variance_inflation_factor(X_vif_clean, i)
                    vif_data.append({"Variable": var, "VIF": vif_val})
                except:
                    vif_data.append({"Variable": var, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_data)
    print(f"    VIF检验结果:")
    for _, row in vif_df.iterrows():
        print(f"      {row['Variable']}: VIF={row['VIF']:.2f}")

    # 3) VIF<10筛选
    selected_vars = []
    for _, row in vif_df.iterrows():
        if row["VIF"] < 10 and not np.isnan(row["VIF"]):
            selected_vars.append(row["Variable"])

    if len(selected_vars) == 0:
        # 选择训练集上最优单变量
        best_r2_train = -np.inf
        best_var = None
        y_train = df_train[y_col].values
        for var in macro_vars:
            pred = uni_preds_train[var]
            mask = (~np.isnan(y_train)) & (~np.isnan(pred))
            if mask.sum() > 0:
                y_mean = np.nanmean(y_train[mask])
                ss_res = np.sum((y_train[mask] - pred[mask]) ** 2)
                ss_tot = np.sum((y_train[mask] - y_mean) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
                if r2 > best_r2_train:
                    best_r2_train = r2
                    best_var = var
        selected_vars = [best_var] if best_var else [macro_vars[0]]

    print(f"    VIF筛选后保留变量: {selected_vars}")

    # 4) 训练集拟合和测试集预测
    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    if len(selected_vars) == 1:
        var = selected_vars[0]
        fit = uni_fits[var]
        alpha, beta, a, b = fit["alpha"], fit["beta"], fit["a"], fit["b"]
        pred_train = alpha + beta * build_midas_term(df_train["date"].values, monthly_available, var, K, a, b)
        pred_test = alpha + beta * build_midas_term(df_test["date"].values, monthly_available, var, K, a, b)
    else:
        # 多变量MIDAS (简化：用训练集估计的参数)
        # 这里用线性组合简化实现
        X_train_list = []
        for var in selected_vars:
            fit = uni_fits[var]
            x_midas = build_midas_term(df_train["date"].values, monthly_available, var, K, fit["a"], fit["b"])
            X_train_list.append(x_midas)
        X_train = np.column_stack(X_train_list)

        mask = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
        if mask.sum() > 5:
            X_const = add_constant(X_train[mask])
            model = OLS(y_train[mask], X_const).fit()
            alpha = model.params[0]
            betas = model.params[1:]

            pred_train = alpha + X_train @ betas
            X_test_list = []
            for var in selected_vars:
                fit = uni_fits[var]
                x_midas = build_midas_term(df_test["date"].values, monthly_available, var, K, fit["a"], fit["b"])
                X_test_list.append(x_midas)
            X_test = np.column_stack(X_test_list)
            pred_test = alpha + X_test @ betas
        else:
            # fallback to single variable
            var = selected_vars[0]
            fit = uni_fits[var]
            alpha, beta, a, b = fit["alpha"], fit["beta"], fit["a"], fit["b"]
            pred_train = alpha + beta * build_midas_term(df_train["date"].values, monthly_available, var, K, a, b)
            pred_test = alpha + beta * build_midas_term(df_test["date"].values, monthly_available, var, K, a, b)

    # 5) 计算指标
    # 样本内R2
    mask_train = (~np.isnan(y_train)) & (~np.isnan(pred_train))
    if mask_train.sum() > 0:
        y_mean_train = np.nanmean(y_train[mask_train])
        ss_res_in = np.sum((y_train[mask_train] - pred_train[mask_train]) ** 2)
        ss_tot_in = np.sum((y_train[mask_train] - y_mean_train) ** 2)
        r2_in = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else np.nan
        rmse_in = np.sqrt(np.mean((y_train[mask_train] - pred_train[mask_train]) ** 2))
        mae_in = np.mean(np.abs(y_train[mask_train] - pred_train[mask_train]))
    else:
        r2_in = rmse_in = mae_in = np.nan

    # 样本外R2 (Campbell-Thompson定义)
    mask_test = (~np.isnan(y_test)) & (~np.isnan(pred_test))
    if mask_test.sum() > 0:
        y_mean_train = np.nanmean(y_train)  # 使用训练集均值作为基准
        ss_res_out = np.sum((y_test[mask_test] - pred_test[mask_test]) ** 2)
        ss_tot_out = np.sum((y_test[mask_test] - y_mean_train) ** 2)
        r2_os = 1 - ss_res_out / ss_tot_out if ss_tot_out > 0 else np.nan
        rmse_out = np.sqrt(np.mean((y_test[mask_test] - pred_test[mask_test]) ** 2))
        mae_out = np.mean(np.abs(y_test[mask_test] - pred_test[mask_test]))
    else:
        r2_os = rmse_out = mae_out = np.nan

    print(f"    样本内 R²: {r2_in:.4f}, RMSE: {rmse_in:.6f}")
    print(f"    样本外 R²_OS: {r2_os:.4f}, RMSE: {rmse_out:.6f}")

    return {
        "model": "Baseline_Stage1",
        "y_col": y_col,
        "selected_vars": selected_vars,
        "r2_in": r2_in,
        "rmse_in": rmse_in,
        "mae_in": mae_in,
        "r2_os": r2_os,
        "rmse_out": rmse_out,
        "mae_out": mae_out,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "uni_fits": uni_fits,
        "vif": vif_df,
    }


def run_ridge_model(df_train, df_test, y_col, X_train, X_test, feature_names, feature_type):
    """运行Ridge模型（训练集内时间序列交叉验证调参）"""
    print(f"\n  运行 Ridge ({y_col}, {feature_type})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    # 处理缺失值
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_train.sum() < 10:
        return None

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化（在训练集上拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # 时间序列交叉验证（只在训练集内）
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-4, 4, 50)

    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train_clean)

    best_alpha = ridge_cv.alpha_
    print(f"    最优 alpha: {best_alpha:.4f}")

    # 预测
    pred_train = ridge_cv.predict(X_train_scaled)
    pred_test = ridge_cv.predict(X_test_scaled)

    # 计算指标
    y_mean_train = np.nanmean(y_train_clean)

    # 样本内
    ss_res_in = np.sum((y_train_clean - pred_train) ** 2)
    ss_tot_in = np.sum((y_train_clean - y_mean_train) ** 2)
    r2_in = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else np.nan
    rmse_in = np.sqrt(np.mean((y_train_clean - pred_train) ** 2))
    mae_in = np.mean(np.abs(y_train_clean - pred_train))

    # 样本外
    if len(y_test_clean) > 0:
        ss_res_out = np.sum((y_test_clean - pred_test) ** 2)
        ss_tot_out = np.sum((y_test_clean - y_mean_train) ** 2)
        r2_os = 1 - ss_res_out / ss_tot_out if ss_tot_out > 0 else np.nan
        rmse_out = np.sqrt(np.mean((y_test_clean - pred_test) ** 2))
        mae_out = np.mean(np.abs(y_test_clean - pred_test))
    else:
        r2_os = rmse_out = mae_out = np.nan

    print(f"    样本内 R²: {r2_in:.4f}, RMSE: {rmse_in:.6f}")
    print(f"    样本外 R²_OS: {r2_os:.4f}, RMSE: {rmse_out:.6f}")

    # 系数
    coef_dict = dict(zip(feature_names, ridge_cv.coef_))

    return {
        "model": f"Ridge_{feature_type}",
        "y_col": y_col,
        "feature_type": feature_type,
        "alpha": best_alpha,
        "r2_in": r2_in,
        "rmse_in": rmse_in,
        "mae_in": mae_in,
        "r2_os": r2_os,
        "rmse_out": rmse_out,
        "mae_out": mae_out,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "scaler": scaler,
    }


def run_elasticnet_model(df_train, df_test, y_col, X_train, X_test, feature_names, feature_type):
    """运行Elastic Net模型（训练集内时间序列交叉验证调参）"""
    print(f"\n  运行 Elastic Net ({y_col}, {feature_type})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    # 处理缺失值
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_train.sum() < 10:
        return None

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # 时间序列交叉验证（只在训练集内）
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-4, 2, 30)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    enet_cv = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=tscv,
        random_state=42,
        max_iter=10000,
        n_jobs=-1
    )
    enet_cv.fit(X_train_scaled, y_train_clean)

    best_alpha = enet_cv.alpha_
    best_l1_ratio = enet_cv.l1_ratio_
    print(f"    最优 alpha: {best_alpha:.4f}, l1_ratio: {best_l1_ratio:.2f}")

    # 预测
    pred_train = enet_cv.predict(X_train_scaled)
    pred_test = enet_cv.predict(X_test_scaled)

    # 计算指标
    y_mean_train = np.nanmean(y_train_clean)

    # 样本内
    ss_res_in = np.sum((y_train_clean - pred_train) ** 2)
    ss_tot_in = np.sum((y_train_clean - y_mean_train) ** 2)
    r2_in = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else np.nan
    rmse_in = np.sqrt(np.mean((y_train_clean - pred_train) ** 2))
    mae_in = np.mean(np.abs(y_train_clean - pred_train))

    # 样本外
    if len(y_test_clean) > 0:
        ss_res_out = np.sum((y_test_clean - pred_test) ** 2)
        ss_tot_out = np.sum((y_test_clean - y_mean_train) ** 2)
        r2_os = 1 - ss_res_out / ss_tot_out if ss_tot_out > 0 else np.nan
        rmse_out = np.sqrt(np.mean((y_test_clean - pred_test) ** 2))
        mae_out = np.mean(np.abs(y_test_clean - pred_test))
    else:
        r2_os = rmse_out = mae_out = np.nan

    print(f"    样本内 R²: {r2_in:.4f}, RMSE: {rmse_in:.6f}")
    print(f"    样本外 R²_OS: {r2_os:.4f}, RMSE: {rmse_out:.6f}")

    # 系数
    coef_dict = dict(zip(feature_names, enet_cv.coef_))

    # 识别非零系数变量
    nonzero_vars = [feature_names[i] for i in range(len(feature_names)) if np.abs(enet_cv.coef_[i]) > 1e-8]
    print(f"    非零系数变量数: {len(nonzero_vars)}")

    return {
        "model": f"ElasticNet_{feature_type}",
        "y_col": y_col,
        "feature_type": feature_type,
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "r2_in": r2_in,
        "rmse_in": rmse_in,
        "mae_in": mae_in,
        "r2_os": r2_os,
        "rmse_out": rmse_out,
        "mae_out": mae_out,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "nonzero_vars": nonzero_vars,
        "scaler": scaler,
    }


def run_pcr_model(df_train, df_test, y_col, X_train, X_test, feature_names, feature_type):
    """运行PCR模型（主成分回归，训练集内时间序列交叉验证选择成分数）"""
    print(f"\n  运行 PCR ({y_col}, {feature_type})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    # 处理缺失值
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_train.sum() < 10:
        return None

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # 时间序列交叉验证选择最优成分数
    tscv = TimeSeriesSplit(n_splits=5)
    max_components = min(X_train_scaled.shape[1], 20)  # 最多20个成分

    best_n_components = 1
    best_cv_score = -np.inf

    for n_comp in range(1, max_components + 1):
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_cv_train = X_train_scaled[train_idx]
            y_cv_train = y_train_clean[train_idx]
            X_cv_val = X_train_scaled[val_idx]
            y_cv_val = y_train_clean[val_idx]

            # PCA
            pca = PCA(n_components=n_comp)
            X_pca_train = pca.fit_transform(X_cv_train)
            X_pca_val = pca.transform(X_cv_val)

            # OLS回归
            X_const = add_constant(X_pca_train)
            model = OLS(y_cv_train, X_const).fit()
            pred_val = model.predict(add_constant(X_pca_val))

            # R2
            y_mean = np.mean(y_cv_train)
            ss_res = np.sum((y_cv_val - pred_val) ** 2)
            ss_tot = np.sum((y_cv_val - y_mean) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            cv_scores.append(r2)

        mean_score = np.mean(cv_scores)
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_n_components = n_comp

    print(f"    最优成分数: {best_n_components}")

    # 用最优成分数拟合最终模型
    pca = PCA(n_components=best_n_components)
    X_pca_train = pca.fit_transform(X_train_scaled)
    X_pca_test = pca.transform(X_test_scaled)

    X_const_train = add_constant(X_pca_train)
    model = OLS(y_train_clean, X_const_train).fit()

    pred_train = model.predict(X_const_train)
    pred_test = model.predict(add_constant(X_pca_test))

    # 计算指标
    y_mean_train = np.nanmean(y_train_clean)

    # 样本内
    ss_res_in = np.sum((y_train_clean - pred_train) ** 2)
    ss_tot_in = np.sum((y_train_clean - y_mean_train) ** 2)
    r2_in = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else np.nan
    rmse_in = np.sqrt(np.mean((y_train_clean - pred_train) ** 2))
    mae_in = np.mean(np.abs(y_train_clean - pred_train))

    # 样本外
    if len(y_test_clean) > 0:
        ss_res_out = np.sum((y_test_clean - pred_test) ** 2)
        ss_tot_out = np.sum((y_test_clean - y_mean_train) ** 2)
        r2_os = 1 - ss_res_out / ss_tot_out if ss_tot_out > 0 else np.nan
        rmse_out = np.sqrt(np.mean((y_test_clean - pred_test) ** 2))
        mae_out = np.mean(np.abs(y_test_clean - pred_test))
    else:
        r2_os = rmse_out = mae_out = np.nan

    print(f"    样本内 R²: {r2_in:.4f}, RMSE: {rmse_in:.6f}")
    print(f"    样本外 R²_OS: {r2_os:.4f}, RMSE: {rmse_out:.6f}")

    # 主成分对原始变量的载荷（重要性）
    loadings = pca.components_
    importance_dict = {}
    for i, var in enumerate(feature_names):
        # 该变量在各主成分上的加权重要性
        var_importance = 0
        for comp_idx in range(best_n_components):
            weight = np.abs(model.params[comp_idx + 1])  # 系数权重
            loading = np.abs(loadings[comp_idx, i])
            var_importance += weight * loading
        importance_dict[var] = var_importance

    return {
        "model": f"PCR_{feature_type}",
        "y_col": y_col,
        "feature_type": feature_type,
        "n_components": best_n_components,
        "r2_in": r2_in,
        "rmse_in": rmse_in,
        "mae_in": mae_in,
        "r2_os": r2_os,
        "rmse_out": rmse_out,
        "mae_out": mae_out,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": model.params,
        "importance": importance_dict,
        "pca": pca,
        "scaler": scaler,
    }


def aggregate_macro_importance(coef_dict, feature_names, feature_type, macro_vars=MACRO_VARS):
    """汇总各宏观变量的整体重要性"""
    agg_importance = {}

    for var in macro_vars:
        if feature_type == "60dim":
            # 60维特征：lagm1到lagm12
            var_features = [f"{var}_lagm{i}" for i in range(1, 13)]
        else:
            # 15维特征：m1, m3avg, m12avg
            var_features = [f"{var}_m1", f"{var}_m3avg", f"{var}_m12avg"]

        total_importance = 0
        for feat in var_features:
            if feat in coef_dict:
                total_importance += np.abs(coef_dict[feat])

        agg_importance[var] = total_importance

    return agg_importance


def save_results(all_results, df_train, df_test, feature_names_60, feature_names_15):
    """保存所有结果"""
    print("\n【4】保存结果...")

    # 1) stage1_model_comparison.csv
    comparison_records = []
    for result in all_results:
        comparison_records.append({
            "Window": result["y_col"],
            "Model": result["model"],
            "Feature_Type": result.get("feature_type", "MIDAS"),
            "R2_InSample": result["r2_in"],
            "RMSE_InSample": result["rmse_in"],
            "MAE_InSample": result["mae_in"],
            "R2_OutSample": result["r2_os"],
            "RMSE_OutSample": result["rmse_out"],
            "MAE_OutSample": result["mae_out"],
        })
    pd.DataFrame(comparison_records).to_csv(RESULT_DIR / "stage1_model_comparison.csv", index=False)
    print("   已保存: stage1_model_comparison.csv")

    # 2) stage1_coefficients_or_importance.csv
    coef_records = []
    for result in all_results:
        if "coef" in result and isinstance(result["coef"], dict):
            for var, val in result["coef"].items():
                coef_records.append({
                    "Window": result["y_col"],
                    "Model": result["model"],
                    "Variable": var,
                    "Coefficient/Importance": val,
                })
        elif "importance" in result:
            for var, val in result["importance"].items():
                coef_records.append({
                    "Window": result["y_col"],
                    "Model": result["model"],
                    "Variable": var,
                    "Coefficient/Importance": val,
                })
    pd.DataFrame(coef_records).to_csv(RESULT_DIR / "stage1_coefficients_or_importance.csv", index=False)
    print("   已保存: stage1_coefficients_or_importance.csv")

    # 3) stage1_macro_importance_aggregated.csv
    agg_records = []
    for result in all_results:
        if "coef" in result and isinstance(result["coef"], dict):
            feature_type = result.get("feature_type", "MIDAS")
            agg_importance = aggregate_macro_importance(
                result["coef"],
                feature_names_60 if feature_type == "60dim" else feature_names_15,
                feature_type,
                MACRO_VARS
            )
            for var, imp in agg_importance.items():
                agg_records.append({
                    "Window": result["y_col"],
                    "Model": result["model"],
                    "Macro_Variable": var,
                    "Aggregated_Importance": imp,
                })
    pd.DataFrame(agg_records).to_csv(RESULT_DIR / "stage1_macro_importance_aggregated.csv", index=False)
    print("   已保存: stage1_macro_importance_aggregated.csv")

    # 4) stage1_test_predictions.csv
    pred_records = []
    for pos_idx in range(len(df_test)):
        row = df_test.iloc[pos_idx]
        pred_record = {"date": row["date"]}
        for result in all_results:
            if result.get("pred_test") is not None:
                pred_record[f"{result['model']}_{result['y_col']}"] = result["pred_test"][pos_idx] if pos_idx < len(result["pred_test"]) else np.nan
        pred_records.append(pred_record)

    # 添加真实值
    for pos_idx in range(len(df_test)):
        row = df_test.iloc[pos_idx]
        for h in WINDOWS:
            pred_records[pos_idx][f"true_R_{h}d"] = row[f"R_{h}d"]

    pd.DataFrame(pred_records).to_csv(RESULT_DIR / "stage1_test_predictions.csv", index=False)
    print("   已保存: stage1_test_predictions.csv")


def generate_plots(all_results, df_train, df_test):
    """生成图表"""
    print("\n【5】生成图表...")

    fig_path = FIGURE_DIR / "stage1_alternative_models_plots.pdf"

    with PdfPages(fig_path) as pdf:
        for h in WINDOWS:
            y_col = f"R_{h}d"
            y_test = df_test[y_col].values

            # 获取该窗口的所有模型结果
            window_results = [r for r in all_results if r["y_col"] == y_col]

            # 1) 真实值vs预测值对比图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"测试集拟合图 - {y_col}", fontsize=14)

            for ax, result in zip(axes.flatten(), window_results[:4]):
                pred_test = result.get("pred_test")
                if pred_test is not None:
                    mask = ~np.isnan(y_test)
                    ax.scatter(y_test[mask], pred_test[mask], alpha=0.5, s=10)
                    ax.plot([y_test[mask].min(), y_test[mask].max()],
                            [y_test[mask].min(), y_test[mask].max()], 'r--', lw=1)
                    ax.set_xlabel("真实值")
                    ax.set_ylabel("预测值")
                    ax.set_title(f"{result['model']}\nR²_OS={result['r2_os']:.4f}")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # 2) 残差图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"残差诊断图 - {y_col}", fontsize=14)

            for ax, result in zip(axes.flatten(), window_results[:4]):
                pred_test = result.get("pred_test")
                if pred_test is not None:
                    mask = ~np.isnan(y_test)
                    residuals = y_test[mask] - pred_test[mask]

                    # 残差时间序列
                    ax.plot(df_test["date"].values[mask], residuals, alpha=0.7)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel("日期")
                    ax.set_ylabel("残差")
                    ax.set_title(f"{result['model']}\nRMSE={result['rmse_out']:.6f}")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    print(f"   已保存图表: {fig_path}")


def generate_report(all_results):
    """生成分析报告"""
    print("\n【6】生成分析报告...")

    report_path = RESULT_DIR / "stage1_alternative_models_report.md"

    # 找出每个窗口的baseline结果
    baseline_5d = [r for r in all_results if r["model"] == "Baseline_Stage1" and r["y_col"] == "R_5d"][0]
    baseline_60d = [r for r in all_results if r["model"] == "Baseline_Stage1" and r["y_col"] == "R_60d"][0]

    # 找出替代模型中样本外R2最高的
    alt_5d = [r for r in all_results if r["model"] != "Baseline_Stage1" and r["y_col"] == "R_5d"]
    alt_60d = [r for r in all_results if r["model"] != "Baseline_Stage1" and r["y_col"] == "R_60d"]

    best_alt_5d = max(alt_5d, key=lambda x: x["r2_os"]) if alt_5d else None
    best_alt_60d = max(alt_60d, key=lambda x: x["r2_os"]) if alt_60d else None

    # 判断是否改善
    improved_5d = best_alt_5d["r2_os"] > baseline_5d["r2_os"] if best_alt_5d else False
    improved_60d = best_alt_60d["r2_os"] > baseline_60d["r2_os"] if best_alt_60d else False

    report_content = f"""# 第一阶段替代模型实验报告

## 实验设置

- **数据**: real_data_complete.csv
- **样本期**: 2015-07-02 至 2025-12-25
- **目标变量**: R_5d, R_60d
- **宏观变量**: cpi, ppi, m2_growth, epu, usd_cny
- **信息原则**: 每个交易日使用"上一个完整月可得信息"
- **训练测试划分**: 前60%训练，后40%测试（时间顺序）

## 特征方案

1. **60维完整月滞后特征**: 每个宏观变量 lagm1 到 lagm12（共5×12=60维）
2. **15维压缩特征**: 每个宏观变量 m1, m3avg, m12avg（共5×3=15维）

## 模型比较结果

### R_5d 窗口

| 模型 | 特征类型 | 样本内R² | 样本外R²_OS | RMSE(测试) | MAE(测试) |
|------|----------|----------|-------------|------------|-----------|
"""

    for r in [baseline_5d] + alt_5d:
        report_content += f"| {r['model']} | {r.get('feature_type', 'MIDAS')} | {r['r2_in']:.4f} | {r['r2_os']:.4f} | {r['rmse_out']:.6f} | {r['mae_out']:.6f} |\n"

    report_content += f"""
### R_60d 窗口

| 模型 | 特征类型 | 样本内R² | 样本外R²_OS | RMSE(测试) | MAE(测试) |
|------|----------|----------|-------------|------------|-----------|
"""

    for r in [baseline_60d] + alt_60d:
        report_content += f"| {r['model']} | {r.get('feature_type', 'MIDAS')} | {r['r2_in']:.4f} | {r['r2_os']:.4f} | {r['rmse_out']:.6f} | {r['mae_out']:.6f} |\n"

    report_content += f"""

## 核心问题回答

### 1）样本外是否比baseline改善？

**R_5d窗口**:
- Baseline R²_OS = {baseline_5d['r2_os']:.4f}
- 最佳替代模型 R²_OS = {best_alt_5d['r2_os']:.4f if best_alt_5d else 'N/A'}
- **结论**: {'有改善 (+%.4f)' % (best_alt_5d['r2_os'] - baseline_5d['r2_os']) if improved_5d else '无改善'}

**R_60d窗口**:
- Baseline R²_OS = {baseline_60d['r2_os']:.4f}
- 最佳替代模型 R²_OS = {best_alt_60d['r2_os']:.4f if best_alt_60d else 'N/A'}
- **结论**: {'有改善 (+%.4f)' % (best_alt_60d['r2_os'] - baseline_60d['r2_os']) if improved_60d else '无改善'}

### 2）改善是否在5d和60d上都存在？

**结论**: {'两个窗口都有改善' if improved_5d and improved_60d else '仅在5d改善' if improved_5d and not improved_60d else '仅在60d改善' if improved_60d and not improved_5d else '两个窗口都无改善'}

### 3）解释结构是否更稳定？

分析各模型的系数稳定性：

"""

    # 添加系数稳定性分析
    for r in all_results:
        if "coef" in r and isinstance(r["coef"], dict):
            nonzero_count = sum(1 for v in r["coef"].values() if np.abs(v) > 1e-8)
            total_count = len(r["coef"])
            report_content += f"- **{r['model']} ({r['y_col']})**: {nonzero_count}/{total_count} 个非零系数\n"

    report_content += f"""
**稳定性评估**:
- Ridge模型通过L2正则化约束系数，通常比OLS更稳定
- Elastic Net通过L1+L2正则化实现变量选择和系数约束，理论上最稳定
- PCR通过降维减少噪声，可能更稳定但牺牲可解释性

### 4）是否值得替换原一阶段主模型？

**综合评估**:

"""

    # 给出最终建议
    if improved_5d and improved_60d:
        report_content += f"- 样本外表现: 替代模型在两个窗口均有改善，**建议替换**\n"
    elif improved_5d or improved_60d:
        report_content += f"- 样本外表现: 仅在一个窗口有改善，**不建议替换**（收益有限）\n"
    else:
        report_content += f"- 样本外表现: 替代模型均无改善，**强烈不建议替换**\n"

    report_content += f"""
- 可解释性: MIDAS模型具有明确的经济学解释（Beta权重函数），替代模型解释性较弱
- 实施复杂度: Ridge/ElasticNet较简单，PCR需要额外的主成分解释步骤
- 稳定性: 正则化模型理论上更稳定，但需要更多验证

**最终建议**: {'可以考虑替换为' + best_alt_5d['model'] if improved_5d and improved_60d else '保持现有MIDAS模型为主基准'}

## 详细参数

"""

    for r in all_results:
        if "alpha" in r:
            report_content += f"- **{r['model']}**: alpha={r['alpha']:.4f}\n"
        if "l1_ratio" in r:
            report_content += f"  l1_ratio={r['l1_ratio']:.2f}\n"
        if "n_components" in r:
            report_content += f"- **{r['model']}**: n_components={r['n_components']}\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"   已保存报告: {report_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("第一阶段替代模型实验：Ridge / Elastic Net / PCR vs Baseline")
    print("=" * 70)

    # 加载并准备数据
    df = load_and_prepare_data()

    # 划分训练测试集
    df_valid, df_train, df_test, train_size = split_train_test(df)

    # 构建月度可得信息面板
    monthly_available = build_monthly_available_panel(df_valid)

    # 构建特征矩阵
    print("\n【3】构建特征矩阵...")
    feature_names_60, X_60 = build_60_dim_features(df_valid, monthly_available)
    feature_names_15, X_15 = build_15_dim_features(df_valid, monthly_available)

    # 使用位置索引切分（df_valid已重置索引）
    train_size = len(df_train)
    X_60_train = X_60[:train_size]
    X_60_test = X_60[train_size:]
    X_15_train = X_15[:train_size]
    X_15_test = X_15[train_size:]

    print(f"   60维特征: {len(feature_names_60)} 个")
    print(f"   15维特征: {len(feature_names_15)} 个")

    # 运行所有模型
    all_results = []

    for h in WINDOWS:
        y_col = f"R_{h}d"
        print(f"\n{'='*60}")
        print(f"目标变量: {y_col}")
        print("="*60)

        # 1) Baseline_Stage1
        baseline_result = run_baseline_stage1(
            df_train, df_test, y_col, monthly_available, MACRO_VARS, K_MONTHS
        )
        all_results.append(baseline_result)

        # 2) Ridge - 60维特征
        ridge_60_result = run_ridge_model(
            df_train, df_test, y_col, X_60_train, X_60_test, feature_names_60, "60dim"
        )
        if ridge_60_result:
            all_results.append(ridge_60_result)

        # 3) Ridge - 15维特征
        ridge_15_result = run_ridge_model(
            df_train, df_test, y_col, X_15_train, X_15_test, feature_names_15, "15dim"
        )
        if ridge_15_result:
            all_results.append(ridge_15_result)

        # 4) Elastic Net - 60维特征
        enet_60_result = run_elasticnet_model(
            df_train, df_test, y_col, X_60_train, X_60_test, feature_names_60, "60dim"
        )
        if enet_60_result:
            all_results.append(enet_60_result)

        # 5) Elastic Net - 15维特征
        enet_15_result = run_elasticnet_model(
            df_train, df_test, y_col, X_15_train, X_15_test, feature_names_15, "15dim"
        )
        if enet_15_result:
            all_results.append(enet_15_result)

        # 6) PCR - 60维特征
        pcr_60_result = run_pcr_model(
            df_train, df_test, y_col, X_60_train, X_60_test, feature_names_60, "60dim"
        )
        if pcr_60_result:
            all_results.append(pcr_60_result)

        # 7) PCR - 15维特征
        pcr_15_result = run_pcr_model(
            df_train, df_test, y_col, X_15_train, X_15_test, feature_names_15, "15dim"
        )
        if pcr_15_result:
            all_results.append(pcr_15_result)

    # 保存结果
    save_results(all_results, df_train, df_test, feature_names_60, feature_names_15)

    # 生成图表
    generate_plots(all_results, df_train, df_test)

    # 生成报告
    generate_report(all_results)

    print("\n" + "=" * 70)
    print("第一阶段替代模型实验完成！")
    print(f"结果保存位置: {RESULT_DIR}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()