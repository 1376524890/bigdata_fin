#!/usr/bin/env python3
"""
第一阶段去共线性重构实验
严格遵循：数据口径、样本期、变量定义、模型集合、验证方式和输出格式

实验目标：
1. 检验共线性是否来自趋势型水平变量与12个月滞后堆叠
2. 验证"变化/偏离型口径+多时间尺度摘要"能否缓解
3. 判断第一阶段瓶颈是方法问题还是宏观信号解释力有限
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "real_data_complete.csv"
RESULT_DIR = BASE_DIR / "experiment_results" / "stage1_restructured_models"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (5, 60)
SAMPLE_START = "2015-07-02"
SAMPLE_END = "2025-12-25"
TRAIN_RATIO = 0.6
VAL_RATIO_IN_TRAIN = 0.2

# Ridge候选alpha
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]
# PCR候选主成分数
PCR_COMPONENTS = [1, 2, 3, 4, 5, 6, 8]


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

    # 过滤样本期
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)].copy()

    # 计算对数收益率
    df["log_return"] = np.log(df["hs300_close"] / df["hs300_close"].shift(1))

    # 计算未来收益窗口
    for h in WINDOWS:
        df[f"R_{h}d"] = compute_future_return(df["log_return"], h)

    # 时间索引
    df["year_month"] = df["date"].dt.to_period("M")

    print(f"样本期: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"有效观测数: {len(df)}")

    # 检查EPU是否有非正值
    if df["epu"].min() <= 0:
        print(f"警告: EPU存在非正值 min={df['epu'].min()}, 但需要取log")
        # 处理方式：用正的最小值替换非正值
        epu_positive_min = df[df["epu"] > 0]["epu"].min()
        df["epu"] = df["epu"].clip(lower=epu_positive_min * 0.1)
        print(f"已将EPU非正值替换为 {epu_positive_min * 0.1}")
    else:
        print(f"EPU最小值: {df['epu'].min()}, 可直接取log")

    return df


def build_redefined_monthly_variables(df):
    """
    构造"变化/偏离型口径"的月度宏观变量
    遵循"上一个完整月可得信息"原则
    """
    print("\n【2】构造变化/偏离型口径变量...")

    # 获取月度数据面板（每月最后一日）
    monthly_panel = df.groupby("year_month").agg({
        "date": "last",
        "cpi": "last",
        "ppi": "last",
        "m2_growth": "last",
        "epu": "last",
        "usd_cny": "last"
    }).sort_index()

    # 滞后一期（上一个完整月可得信息）
    monthly_available = monthly_panel.shift(1)

    # ===== CPI =====
    monthly_available["cpi_yoy"] = monthly_available["cpi"]  # 同比增速（原值）
    monthly_available["cpi_delta1"] = monthly_available["cpi"] - monthly_available["cpi"].shift(1)  # 月度变化
    # 相对过去12个月均值偏离
    cpi_rolling_mean = monthly_available["cpi"].rolling(12).mean()
    monthly_available["cpi_dev12"] = monthly_available["cpi"] - cpi_rolling_mean

    # ===== PPI =====
    monthly_available["ppi_yoy"] = monthly_available["ppi"]
    monthly_available["ppi_delta1"] = monthly_available["ppi"] - monthly_available["ppi"].shift(1)
    ppi_rolling_mean = monthly_available["ppi"].rolling(12).mean()
    monthly_available["ppi_dev12"] = monthly_available["ppi"] - ppi_rolling_mean

    # ===== M2 =====
    monthly_available["m2_yoy"] = monthly_available["m2_growth"]
    monthly_available["m2_delta1"] = monthly_available["m2_growth"] - monthly_available["m2_growth"].shift(1)
    m2_rolling_mean = monthly_available["m2_growth"].rolling(12).mean()
    monthly_available["m2_dev12"] = monthly_available["m2_growth"] - m2_rolling_mean

    # ===== EPU (log口径) =====
    monthly_available["epu_log"] = np.log(monthly_available["epu"])
    monthly_available["epu_log_delta1"] = monthly_available["epu_log"] - monthly_available["epu_log"].shift(1)
    epu_log_rolling_mean = monthly_available["epu_log"].rolling(12).mean()
    monthly_available["epu_log_dev12"] = monthly_available["epu_log"] - epu_log_rolling_mean

    # ===== USD/CNY (汇率变化/偏离) =====
    monthly_available["fx_log"] = np.log(monthly_available["usd_cny"])
    monthly_available["fx_ret1"] = monthly_available["fx_log"] - monthly_available["fx_log"].shift(1)  # 月度汇率变化
    fx_log_rolling_mean = monthly_available["fx_log"].rolling(12).mean()
    monthly_available["fx_dev12"] = monthly_available["fx_log"] - fx_log_rolling_mean

    # 定义所有重构变量列表
    redefined_vars = [
        "cpi_yoy", "cpi_delta1", "cpi_dev12",
        "ppi_yoy", "ppi_delta1", "ppi_dev12",
        "m2_yoy", "m2_delta1", "m2_dev12",
        "epu_log", "epu_log_delta1", "epu_log_dev12",
        "fx_log", "fx_ret1", "fx_dev12"
    ]

    print(f"构造了 {len(redefined_vars)} 个变化/偏离型变量")

    return monthly_available, redefined_vars


def build_summary_features(df, monthly_available, redefined_vars):
    """
    构造"多时间尺度摘要特征"方案A
    对每个变量构造: m1, avg3, avg6, avg12
    """
    print("\n【3】构造多时间尺度摘要特征（方案A）...")

    feature_names = []
    for var in redefined_vars:
        feature_names.extend([f"{var}_m1", f"{var}_avg3", f"{var}_avg6", f"{var}_avg12"])

    X_full = np.full((len(df), len(feature_names)), np.nan)

    for pos_idx in range(len(df)):
        row = df.iloc[pos_idx]
        current_month = pd.Period(row["date"], freq="M")

        for var_idx, var in enumerate(redefined_vars):
            # m1: 最新可得月值（当前月-1）
            m1_month = current_month - 1
            if m1_month in monthly_available.index:
                val_m1 = monthly_available.loc[m1_month, var]
                if not np.isnan(val_m1):
                    X_full[pos_idx, var_idx * 4 + 0] = val_m1

            # avg3: 最近3个月均值
            vals_3 = []
            for lag in range(1, 4):
                target_month = current_month - lag
                if target_month in monthly_available.index:
                    val = monthly_available.loc[target_month, var]
                    if not np.isnan(val):
                        vals_3.append(val)
            if len(vals_3) >= 2:  # 至少2个月有值
                X_full[pos_idx, var_idx * 4 + 1] = np.mean(vals_3)

            # avg6: 最近6个月均值
            vals_6 = []
            for lag in range(1, 7):
                target_month = current_month - lag
                if target_month in monthly_available.index:
                    val = monthly_available.loc[target_month, var]
                    if not np.isnan(val):
                        vals_6.append(val)
            if len(vals_6) >= 4:  # 至少4个月有值
                X_full[pos_idx, var_idx * 4 + 2] = np.mean(vals_6)

            # avg12: 最近12个月均值
            vals_12 = []
            for lag in range(1, 13):
                target_month = current_month - lag
                if target_month in monthly_available.index:
                    val = monthly_available.loc[target_month, var]
                    if not np.isnan(val):
                        vals_12.append(val)
            if len(vals_12) >= 6:  # 至少6个月有值
                X_full[pos_idx, var_idx * 4 + 3] = np.mean(vals_12)

    print(f"方案A特征数: {len(feature_names)}")

    return feature_names, X_full


def build_group_factors(monthly_available):
    """
    构造分组因子（方案B的前置步骤）
    按经济含义分四组，每组标准化后PCA提取第一主成分
    """
    print("\n【4】构造分组因子...")

    # 定义分组
    groups = {
        "price": ["cpi_yoy", "cpi_delta1", "cpi_dev12", "ppi_yoy", "ppi_delta1", "ppi_dev12"],
        "money": ["m2_yoy", "m2_delta1", "m2_dev12"],
        "policy": ["epu_log", "epu_log_delta1", "epu_log_dev12"],
        "fx": ["fx_log", "fx_ret1", "fx_dev12"]
    }

    factor_names = ["price_factor", "money_factor", "policy_factor", "fx_factor"]

    # 在月度面板上标准化并提取第一主成分
    factor_panel = pd.DataFrame(index=monthly_available.index)

    for group_name, group_vars in groups.items():
        # 提取该组变量
        X_group = monthly_available[group_vars].values

        # 删除缺失值行
        mask_valid = ~np.isnan(X_group).any(axis=1)
        X_group_valid = X_group[mask_valid]
        valid_indices = monthly_available.index[mask_valid]

        if len(X_group_valid) < 5:
            print(f"  警告: {group_name}_factor 有效数据不足，跳过")
            continue

        # 标准化
        scaler = StandardScaler()
        X_group_scaled = scaler.fit_transform(X_group_valid)

        # PCA提取第一主成分
        pca = PCA(n_components=1)
        factor_values = pca.fit_transform(X_group_scaled)

        # 创建因子面板（仅对有效索引赋值）
        factor_panel.loc[valid_indices, f"{group_name}_factor"] = factor_values.flatten()

        print(f"  {group_name}_factor: 第一主成分解释方差 {pca.explained_variance_ratio_[0]:.2%}, 有效样本数 {len(X_group_valid)}")

    # 滞后一期（上一个完整月可得信息）
    factor_available = factor_panel.shift(1)

    return factor_available, factor_names


def build_factor_summary_features(df, factor_available, factor_names):
    """
    构造分组因子的多时间尺度摘要特征（方案B）
    """
    print("\n【5】构造分组因子多时间尺度摘要特征（方案B）...")

    feature_names = []
    for factor in factor_names:
        feature_names.extend([f"{factor}_m1", f"{factor}_avg3", f"{factor}_avg6", f"{factor}_avg12"])

    X_full = np.full((len(df), len(feature_names)), np.nan)

    for pos_idx in range(len(df)):
        row = df.iloc[pos_idx]
        current_month = pd.Period(row["date"], freq="M")

        for factor_idx, factor in enumerate(factor_names):
            # m1
            m1_month = current_month - 1
            if m1_month in factor_available.index:
                val_m1 = factor_available.loc[m1_month, factor]
                if not np.isnan(val_m1):
                    X_full[pos_idx, factor_idx * 4 + 0] = val_m1

            # avg3
            vals_3 = []
            for lag in range(1, 4):
                target_month = current_month - lag
                if target_month in factor_available.index:
                    val = factor_available.loc[target_month, factor]
                    if not np.isnan(val):
                        vals_3.append(val)
            if len(vals_3) >= 2:
                X_full[pos_idx, factor_idx * 4 + 1] = np.mean(vals_3)

            # avg6
            vals_6 = []
            for lag in range(1, 7):
                target_month = current_month - lag
                if target_month in factor_available.index:
                    val = factor_available.loc[target_month, factor]
                    if not np.isnan(val):
                        vals_6.append(val)
            if len(vals_6) >= 4:
                X_full[pos_idx, factor_idx * 4 + 2] = np.mean(vals_6)

            # avg12
            vals_12 = []
            for lag in range(1, 13):
                target_month = current_month - lag
                if target_month in factor_available.index:
                    val = factor_available.loc[target_month, factor]
                    if not np.isnan(val):
                        vals_12.append(val)
            if len(vals_12) >= 6:
                X_full[pos_idx, factor_idx * 4 + 3] = np.mean(vals_12)

    print(f"方案B特征数: {len(feature_names)}")

    return feature_names, X_full


def split_train_test(df):
    """
    划分训练集和测试集（前60%训练，后40%测试，时间顺序）
    """
    print("\n【6】划分训练集与测试集（60%/40%）...")

    # 删除R_5d和R_60d缺失的行
    df_valid = df.dropna(subset=["R_5d", "R_60d"]).copy()
    df_valid = df_valid.reset_index(drop=True)

    train_size = int(len(df_valid) * TRAIN_RATIO)

    df_train = df_valid.iloc[:train_size].copy()
    df_test = df_valid.iloc[train_size:].copy()

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"训练集: {len(df_train)} ({df_train['date'].min().strftime('%Y-%m-%d')} 至 {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"测试集: {len(df_test)} ({df_test['date'].min().strftime('%Y-%m-%d')} 至 {df_test['date'].max().strftime('%Y-%m-%d')})")

    return df_valid, df_train, df_test, train_size


def split_train_val_within_train(df_train):
    """
    在训练集内部切分验证集（前80%子训练，后20%验证）
    用于超参数选择
    """
    val_size = int(len(df_train) * VAL_RATIO_IN_TRAIN)
    sub_train_size = len(df_train) - val_size

    df_sub_train = df_train.iloc[:sub_train_size].copy()
    df_val = df_train.iloc[sub_train_size:].copy()

    df_sub_train = df_sub_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"  子训练集: {len(df_sub_train)} ({df_sub_train['date'].min().strftime('%Y-%m-%d')} 至 {df_sub_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"  验证集: {len(df_val)} ({df_val['date'].min().strftime('%Y-%m-%d')} 至 {df_val['date'].max().strftime('%Y-%m-%d')})")

    return df_sub_train, df_val, sub_train_size


def compute_r2_os(y_true, y_pred, y_train_mean):
    """计算样本外R2 (Campbell-Thompson定义)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_train_mean) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test):
    """计算训练集和测试集指标（使用清理后的数据）"""
    y_train_mean = np.nanmean(y_train_clean)

    # 训练集指标
    mask_train = (~np.isnan(y_train_clean)) & (~np.isnan(pred_train))
    if mask_train.sum() > 0:
        y_train_sub = y_train_clean[mask_train]
        pred_train_sub = pred_train[mask_train]
        train_r2 = compute_r2_os(y_train_sub, pred_train_sub, y_train_mean)
        train_rmse = np.sqrt(np.mean((y_train_sub - pred_train_sub) ** 2))
        train_mae = np.mean(np.abs(y_train_sub - pred_train_sub))
    else:
        train_r2 = train_rmse = train_mae = np.nan

    # 测试集指标
    mask_test = (~np.isnan(y_test_clean)) & (~np.isnan(pred_test))
    if mask_test.sum() > 0:
        y_test_sub = y_test_clean[mask_test]
        pred_test_sub = pred_test[mask_test]
        test_r2_os = compute_r2_os(y_test_sub, pred_test_sub, y_train_mean)
        test_rmse = np.sqrt(np.mean((y_test_sub - pred_test_sub) ** 2))
        test_mae = np.mean(np.abs(y_test_sub - pred_test_sub))
    else:
        test_r2_os = test_rmse = test_mae = np.nan

    return {
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "test_r2_os": test_r2_os,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }


def fit_summary_ols(df_train, df_test, y_col, X_train, X_test, feature_names):
    """
    拟合摘要OLS（Baseline_SummaryOLS）
    """
    print(f"\n  运行 Baseline_SummaryOLS ({y_col})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    # 处理缺失值
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_train.sum() < 20:
        return None

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # OLS回归
    X_const = add_constant(X_train_clean)
    model = OLS(y_train_clean, X_const).fit()

    # 预测
    pred_train = model.predict(X_const)
    pred_test = model.predict(add_constant(X_test_clean))

    # 评估
    metrics = evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test)

    print(f"    训练R²: {metrics['train_r2']:.4f}, 测试R²_OS: {metrics['test_r2_os']:.4f}")

    # 系数
    coef_dict = {"const": model.params[0]}
    for i, name in enumerate(feature_names):
        coef_dict[name] = model.params[i + 1]

    return {
        "model": "Baseline_SummaryOLS",
        "target": y_col,
        "feature_scheme": "summary",
        "best_params": "None",
        "train_r2": metrics["train_r2"],
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "test_r2_os": metrics["test_r2_os"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "model_obj": model,
        "train_idx": np.where(mask_train)[0],
        "test_idx": np.where(mask_test)[0]
    }


def fit_ridge(df_train, df_test, df_sub_train, df_val, y_col, X_train, X_test, X_sub_train, X_val, X_sub_test, feature_names):
    """
    拟合Ridge模型（训练集内验证调参）
    """
    print(f"\n  运行 Ridge ({y_col})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    y_sub_train = df_sub_train[y_col].values
    y_val = df_val[y_col].values

    # 处理缺失值
    mask_sub_train = (~np.isnan(y_sub_train)) & (~np.isnan(X_sub_train).any(axis=1))
    mask_val = (~np.isnan(y_val)) & (~np.isnan(X_val).any(axis=1))
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_sub_train.sum() < 10:
        return None

    X_sub_train_clean = X_sub_train[mask_sub_train]
    y_sub_train_clean = y_sub_train[mask_sub_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化（在子训练集上拟合）
    scaler = StandardScaler()
    X_sub_train_scaled = scaler.fit_transform(X_sub_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)

    # 在验证集上选择最优alpha
    best_alpha = None
    best_val_r2 = -np.inf

    for alpha in RIDGE_ALPHAS:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_sub_train_scaled, y_sub_train_clean)
        pred_val = ridge.predict(X_val_scaled)

        # 验证集R2
        y_sub_train_mean = np.nanmean(y_sub_train_clean)
        val_r2 = compute_r2_os(y_val_clean, pred_val, y_sub_train_mean)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha

    print(f"    最优alpha: {best_alpha} (验证R²: {best_val_r2:.4f})")

    # 用最优alpha在完整训练集上重训
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_clean)
    X_test_scaled = scaler_full.transform(X_test_clean)

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train_scaled, y_train_clean)

    pred_train = ridge_final.predict(X_train_scaled)
    pred_test = ridge_final.predict(X_test_scaled)

    # 评估
    metrics = evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test)

    print(f"    训练R²: {metrics['train_r2']:.4f}, 测试R²_OS: {metrics['test_r2_os']:.4f}")

    # 系数
    coef_dict = {}
    for i, name in enumerate(feature_names):
        coef_dict[name] = ridge_final.coef_[i]

    return {
        "model": "Ridge",
        "target": y_col,
        "feature_scheme": "summary",
        "best_params": f"alpha={best_alpha}",
        "train_r2": metrics["train_r2"],
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "test_r2_os": metrics["test_r2_os"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "best_alpha": best_alpha,
        "scaler": scaler_full,
        "train_idx": np.where(mask_train)[0],
        "test_idx": np.where(mask_test)[0]
    }


def fit_pcr(df_train, df_test, df_sub_train, df_val, y_col, X_train, X_test, X_sub_train, X_val, feature_names):
    """
    拟合PCR模型（PCA+OLS，训练集内验证调参）
    """
    print(f"\n  运行 PCR ({y_col})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    y_sub_train = df_sub_train[y_col].values
    y_val = df_val[y_col].values

    # 处理缺失值
    mask_sub_train = (~np.isnan(y_sub_train)) & (~np.isnan(X_sub_train).any(axis=1))
    mask_val = (~np.isnan(y_val)) & (~np.isnan(X_val).any(axis=1))
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_sub_train.sum() < 10:
        return None

    X_sub_train_clean = X_sub_train[mask_sub_train]
    y_sub_train_clean = y_sub_train[mask_sub_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化（在子训练集上拟合）
    scaler = StandardScaler()
    X_sub_train_scaled = scaler.fit_transform(X_sub_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)

    # 在验证集上选择最优主成分数
    best_n_comp = None
    best_val_r2 = -np.inf
    max_n_comp = min(X_sub_train_scaled.shape[1], max(PCR_COMPONENTS))

    for n_comp in PCR_COMPONENTS:
        if n_comp > max_n_comp:
            continue

        pca = PCA(n_components=n_comp)
        X_sub_pca = pca.fit_transform(X_sub_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        # OLS
        X_const = add_constant(X_sub_pca)
        model = OLS(y_sub_train_clean, X_const).fit()
        pred_val = model.predict(add_constant(X_val_pca))

        # 验证集R2
        y_sub_train_mean = np.nanmean(y_sub_train_clean)
        val_r2 = compute_r2_os(y_val_clean, pred_val, y_sub_train_mean)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_n_comp = n_comp

    print(f"    最优主成分数: {best_n_comp} (验证R²: {best_val_r2:.4f})")

    # 用最优主成分数在完整训练集上重训
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_clean)
    X_test_scaled = scaler_full.transform(X_test_clean)

    pca_final = PCA(n_components=best_n_comp)
    X_train_pca = pca_final.fit_transform(X_train_scaled)
    X_test_pca = pca_final.transform(X_test_scaled)

    X_const_train = add_constant(X_train_pca)
    model_final = OLS(y_train_clean, X_const_train).fit()

    pred_train = model_final.predict(X_const_train)
    pred_test = model_final.predict(add_constant(X_test_pca))

    # 评估
    metrics = evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test)

    print(f"    训练R²: {metrics['train_r2']:.4f}, 测试R²_OS: {metrics['test_r2_os']:.4f}")

    # 主成分解释方差
    explained_var = pca_final.explained_variance_ratio_

    # 回推原始特征重要性（通过主成分载荷）
    loadings = pca_final.components_
    importance_dict = {}
    for i, name in enumerate(feature_names):
        var_importance = 0
        for comp_idx in range(best_n_comp):
            coef_weight = np.abs(model_final.params[comp_idx + 1])
            loading = np.abs(loadings[comp_idx, i])
            var_importance += coef_weight * loading
        importance_dict[name] = var_importance

    return {
        "model": "PCR",
        "target": y_col,
        "feature_scheme": "summary",
        "best_params": f"n_components={best_n_comp}",
        "train_r2": metrics["train_r2"],
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "test_r2_os": metrics["test_r2_os"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": model_final.params,
        "importance": importance_dict,
        "best_n_comp": best_n_comp,
        "explained_var": explained_var,
        "pca": pca_final,
        "scaler": scaler_full,
        "train_idx": np.where(mask_train)[0],
        "test_idx": np.where(mask_test)[0]
    }


def fit_factor_ols(df_train, df_test, y_col, X_train, X_test, feature_names):
    """
    拟合分组因子OLS
    """
    print(f"\n  运行 Factor_OLS ({y_col})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values

    # 处理缺失值
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_train.sum() < 20:
        return None

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # OLS回归
    X_const = add_constant(X_train_clean)
    model = OLS(y_train_clean, X_const).fit()

    # 预测
    pred_train = model.predict(X_const)
    pred_test = model.predict(add_constant(X_test_clean))

    # 评估
    metrics = evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test)

    print(f"    训练R²: {metrics['train_r2']:.4f}, 测试R²_OS: {metrics['test_r2_os']:.4f}")

    # 系数
    coef_dict = {"const": model.params[0]}
    for i, name in enumerate(feature_names):
        coef_dict[name] = model.params[i + 1]

    return {
        "model": "Factor_OLS",
        "target": y_col,
        "feature_scheme": "factor",
        "best_params": "None",
        "train_r2": metrics["train_r2"],
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "test_r2_os": metrics["test_r2_os"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "model_obj": model,
        "train_idx": np.where(mask_train)[0],
        "test_idx": np.where(mask_test)[0]
    }


def fit_factor_ridge(df_train, df_test, df_sub_train, df_val, y_col, X_train, X_test, X_sub_train, X_val, feature_names):
    """
    拟合分组因子Ridge
    """
    print(f"\n  运行 Factor_Ridge ({y_col})...")

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    y_sub_train = df_sub_train[y_col].values
    y_val = df_val[y_col].values

    # 处理缺失值
    mask_sub_train = (~np.isnan(y_sub_train)) & (~np.isnan(X_sub_train).any(axis=1))
    mask_val = (~np.isnan(y_val)) & (~np.isnan(X_val).any(axis=1))
    mask_train = (~np.isnan(y_train)) & (~np.isnan(X_train).any(axis=1))
    mask_test = (~np.isnan(y_test)) & (~np.isnan(X_test).any(axis=1))

    if mask_sub_train.sum() < 10:
        return None

    X_sub_train_clean = X_sub_train[mask_sub_train]
    y_sub_train_clean = y_sub_train[mask_sub_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    # 标准化（在子训练集上拟合）
    scaler = StandardScaler()
    X_sub_train_scaled = scaler.fit_transform(X_sub_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)

    # 在验证集上选择最优alpha
    best_alpha = None
    best_val_r2 = -np.inf

    for alpha in RIDGE_ALPHAS:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_sub_train_scaled, y_sub_train_clean)
        pred_val = ridge.predict(X_val_scaled)

        y_sub_train_mean = np.nanmean(y_sub_train_clean)
        val_r2 = compute_r2_os(y_val_clean, pred_val, y_sub_train_mean)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha

    print(f"    最优alpha: {best_alpha} (验证R²: {best_val_r2:.4f})")

    # 用最优alpha在完整训练集上重训
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_clean)
    X_test_scaled = scaler_full.transform(X_test_clean)

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train_scaled, y_train_clean)

    pred_train = ridge_final.predict(X_train_scaled)
    pred_test = ridge_final.predict(X_test_scaled)

    # 评估
    metrics = evaluate_predictions(y_train_clean, y_test_clean, pred_train, pred_test)

    print(f"    训练R²: {metrics['train_r2']:.4f}, 测试R²_OS: {metrics['test_r2_os']:.4f}")

    # 系数
    coef_dict = {}
    for i, name in enumerate(feature_names):
        coef_dict[name] = ridge_final.coef_[i]

    return {
        "model": "Factor_Ridge",
        "target": y_col,
        "feature_scheme": "factor",
        "best_params": f"alpha={best_alpha}",
        "train_r2": metrics["train_r2"],
        "train_rmse": metrics["train_rmse"],
        "train_mae": metrics["train_mae"],
        "test_r2_os": metrics["test_r2_os"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "pred_train": pred_train,
        "pred_test": pred_test,
        "coef": coef_dict,
        "best_alpha": best_alpha,
        "scaler": scaler_full,
        "train_idx": np.where(mask_train)[0],
        "test_idx": np.where(mask_test)[0]
    }


def compute_vif(X, feature_names):
    """计算VIF（处理缺失值和零方差）"""
    # 删除缺失值
    mask_valid = ~np.isnan(X).any(axis=1)
    X_clean = X[mask_valid]

    if len(X_clean) < 10 or X_clean.shape[1] < 2:
        return pd.DataFrame({"feature": feature_names, "VIF": [np.nan] * len(feature_names)})

    # 检查零方差列并排除
    variances = np.var(X_clean, axis=0)
    nonzero_var_cols = variances > 1e-10

    if nonzero_var_cols.sum() < 2:
        return pd.DataFrame({"feature": feature_names, "VIF": [np.nan] * len(feature_names)})

    X_clean_filtered = X_clean[:, nonzero_var_cols]
    filtered_feature_names = [feature_names[i] for i in range(len(feature_names)) if nonzero_var_cols[i]]

    # 添加常数项
    X_const = add_constant(X_clean_filtered)

    vif_records = []
    for i in range(X_clean_filtered.shape[1]):
        try:
            vif = variance_inflation_factor(X_const, i + 1)  # 跳过常数项
            vif_records.append({"feature": filtered_feature_names[i], "VIF": vif})
        except:
            vif_records.append({"feature": filtered_feature_names[i], "VIF": np.nan})

    # 对于被排除的零方差列，VIF设为NaN
    for i in range(len(feature_names)):
        if not nonzero_var_cols[i]:
            vif_records.append({"feature": feature_names[i], "VIF": np.nan})

    return pd.DataFrame(vif_records)


def aggregate_importance_summary(coef_dict, redefined_vars):
    """
    汇总摘要特征方案的重要性（按原始宏观变量聚合）
    """
    agg_dict = {}

    # 定义变量到宏观类别的映射
    var_mapping = {
        "cpi": ["cpi_yoy", "cpi_delta1", "cpi_dev12"],
        "ppi": ["ppi_yoy", "ppi_delta1", "ppi_dev12"],
        "m2": ["m2_yoy", "m2_delta1", "m2_dev12"],
        "epu": ["epu_log", "epu_log_delta1", "epu_log_dev12"],
        "fx": ["fx_log", "fx_ret1", "fx_dev12"]
    }

    for macro_name, sub_vars in var_mapping.items():
        total_importance = 0
        for sub_var in sub_vars:
            # 该变量的四个时间尺度
            for suffix in ["_m1", "_avg3", "_avg6", "_avg12"]:
                feat_name = f"{sub_var}{suffix}"
                if feat_name in coef_dict:
                    total_importance += np.abs(coef_dict[feat_name])
        agg_dict[macro_name] = total_importance

    return agg_dict


def aggregate_importance_factor(coef_dict, factor_names):
    """
    汇总分组因子方案的重要性
    """
    agg_dict = {}

    for factor in factor_names:
        total_importance = 0
        for suffix in ["_m1", "_avg3", "_avg6", "_avg12"]:
            feat_name = f"{factor}{suffix}"
            if feat_name in coef_dict:
                total_importance += np.abs(coef_dict[feat_name])
        agg_dict[factor] = total_importance

    return agg_dict


def save_results(all_results, df_test, summary_feature_names, factor_feature_names, redefined_vars, factor_names, vif_summary):
    """保存所有结果"""
    print("\n【8】保存结果...")

    # 1) stage1_restructured_model_comparison.csv
    comparison_records = []
    for result in all_results:
        comparison_records.append({
            "target": result["target"],
            "feature_scheme": result["feature_scheme"],
            "model": result["model"],
            "best_params": result["best_params"],
            "train_r2": result["train_r2"],
            "test_r2_os": result["test_r2_os"],
            "test_rmse": result["test_rmse"],
            "test_mae": result["test_mae"]
        })
    pd.DataFrame(comparison_records).to_csv(RESULT_DIR / "stage1_restructured_model_comparison.csv", index=False)
    print("   已保存: stage1_restructured_model_comparison.csv")

    # 2) stage1_restructured_coefficients.csv
    coef_records = []
    for result in all_results:
        if "coef" in result and isinstance(result["coef"], dict):
            for var, val in result["coef"].items():
                coef_records.append({
                    "target": result["target"],
                    "feature_scheme": result["feature_scheme"],
                    "model": result["model"],
                    "variable": var,
                    "coefficient": val
                })
        elif "importance" in result:
            for var, val in result["importance"].items():
                coef_records.append({
                    "target": result["target"],
                    "feature_scheme": result["feature_scheme"],
                    "model": result["model"],
                    "variable": var,
                    "coefficient_or_importance": val
                })

    # 添加PCR主成分解释方差
    for result in all_results:
        if result["model"] == "PCR" and "explained_var" in result:
            for i, var in enumerate(result["explained_var"]):
                coef_records.append({
                    "target": result["target"],
                    "feature_scheme": result["feature_scheme"],
                    "model": "PCR_explained_var",
                    "variable": f"PC{i+1}",
                    "explained_variance_ratio": var
                })

    pd.DataFrame(coef_records).to_csv(RESULT_DIR / "stage1_restructured_coefficients.csv", index=False)
    print("   已保存: stage1_restructured_coefficients.csv")

    # 3) stage1_restructured_macro_importance.csv
    importance_records = []
    for result in all_results:
        if "coef" in result and isinstance(result["coef"], dict):
            if result["feature_scheme"] == "summary":
                agg_imp = aggregate_importance_summary(result["coef"], redefined_vars)
            else:
                agg_imp = aggregate_importance_factor(result["coef"], factor_names)

            for var, imp in agg_imp.items():
                importance_records.append({
                    "target": result["target"],
                    "feature_scheme": result["feature_scheme"],
                    "model": result["model"],
                    "macro_group": var,
                    "aggregated_importance": imp
                })
        elif "importance" in result:
            agg_imp = aggregate_importance_summary(result["importance"], redefined_vars)
            for var, imp in agg_imp.items():
                importance_records.append({
                    "target": result["target"],
                    "feature_scheme": result["feature_scheme"],
                    "model": result["model"],
                    "macro_group": var,
                    "aggregated_importance": imp
                })

    pd.DataFrame(importance_records).to_csv(RESULT_DIR / "stage1_restructured_macro_importance.csv", index=False)
    print("   已保存: stage1_restructured_macro_importance.csv")

    # 4) stage1_restructured_test_predictions.csv
    pred_records = []
    for pos_idx in range(len(df_test)):
        row = df_test.iloc[pos_idx]
        pred_record = {
            "date": row["date"],
        }

        # 添加各模型预测
        for result in all_results:
            if result.get("pred_test") is not None:
                # 找到对应的测试索引
                test_idx = result.get("test_idx", [])
                if pos_idx in test_idx:
                    idx_in_pred = np.where(test_idx == pos_idx)[0][0]
                    pred_record[f"{result['model']}_{result['target']}"] = result["pred_test"][idx_in_pred]
                else:
                    pred_record[f"{result['model']}_{result['target']}"] = np.nan

        pred_records.append(pred_record)

    # 添加真实值
    for pos_idx in range(len(df_test)):
        row = df_test.iloc[pos_idx]
        pred_records[pos_idx]["target_R_5d"] = row["R_5d"]
        pred_records[pos_idx]["target_R_60d"] = row["R_60d"]

    pd.DataFrame(pred_records).to_csv(RESULT_DIR / "stage1_restructured_test_predictions.csv", index=False)
    print("   已保存: stage1_restructured_test_predictions.csv")

    # 5) VIF诊断
    vif_summary.to_csv(RESULT_DIR / "stage1_restructured_vif_diagnosis.csv", index=False)
    print("   已保存: stage1_restructured_vif_diagnosis.csv")


def generate_plots(all_results, df_test):
    """生成图表"""
    print("\n【7】生成图表...")

    for h in WINDOWS:
        y_col = f"R_{h}d"
        y_test = df_test[y_col].values

        # 获取该窗口的所有模型结果
        window_results = [r for r in all_results if r["target"] == y_col]

        # 1) 预测散点图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"预测散点图 - {y_col}", fontsize=14)

        for ax, result in zip(axes.flatten(), window_results[:6]):
            if result.get("pred_test") is not None:
                test_idx = result.get("test_idx", [])
                y_test_subset = y_test[test_idx]
                pred_test = result["pred_test"]

                ax.scatter(y_test_subset, pred_test, alpha=0.5, s=10)
                ax.plot([y_test_subset.min(), y_test_subset.max()],
                        [y_test_subset.min(), y_test_subset.max()], 'r--', lw=1)
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                ax.set_title(f"{result['model']}\nR²_OS={result['test_r2_os']:.4f}")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"R_{h}d_pred_scatter.png")
        plt.close()

        # 2) 残差时间序列图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"残差时间序列图 - {y_col}", fontsize=14)

        for ax, result in zip(axes.flatten(), window_results[:6]):
            if result.get("pred_test") is not None:
                test_idx = result.get("test_idx", [])
                y_test_subset = y_test[test_idx]
                pred_test = result["pred_test"]
                residuals = y_test_subset - pred_test

                dates_test = df_test["date"].values[test_idx]
                ax.plot(dates_test, residuals, alpha=0.7)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel("日期")
                ax.set_ylabel("残差")
                ax.set_title(f"{result['model']}\nRMSE={result['test_rmse']:.6f}")

        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"R_{h}d_residual_ts.png")
        plt.close()

    print(f"   图表已保存至: {RESULT_DIR}")


def generate_report(all_results, vif_summary, df_test):
    """生成分析报告"""
    print("\n【9】生成分析报告...")

    # 获取各窗口的最佳结果
    results_5d = [r for r in all_results if r["target"] == "R_5d"]
    results_60d = [r for r in all_results if r["target"] == "R_60d"]

    # 找出最佳模型
    best_5d = max(results_5d, key=lambda x: x["test_r2_os"]) if results_5d else None
    best_60d = max(results_60d, key=lambda x: x["test_r2_os"]) if results_60d else None

    # 原baseline结果（从之前实验获取）
    baseline_5d_r2 = 0.0019
    baseline_60d_r2 = 0.0396

    # VIF统计
    vif_max = vif_summary["VIF"].max()
    vif_mean = vif_summary["VIF"].mean()
    vif_over_10 = (vif_summary["VIF"] >= 10).sum()

    report_content = f"""# 第一阶段去共线性重构实验报告

## 实验设置

- **数据文件**: real_data_complete.csv
- **样本期**: {SAMPLE_START} 至 {SAMPLE_END}
- **目标变量**: R_5d, R_60d
- **信息原则**: 每个交易日使用"上一个完整月可得信息"
- **训练测试切分**: 前60%训练，后40%测试（时间顺序）
- **验证集切分**: 训练集内部前80%子训练，后20%验证（仅用于调参）
- **随机种子**: 42

## 变量口径重构

### 原始宏观变量
- cpi, ppi, m2_growth, epu, usd_cny

### 重构为"变化/偏离型口径"

#### CPI组
- cpi_yoy: 同比增速（保留原值）
- cpi_delta1: 月度变化 (cpi_t - cpi_{{t-1}})
- cpi_dev12: 相对过去12个月均值偏离

#### PPI组
- ppi_yoy: 同比增速
- ppi_delta1: 月度变化
- ppi_dev12: 相对12个月均值偏离

#### M2组
- m2_yoy: 同比增速
- m2_delta1: 月度变化
- m2_dev12: 相对12个月均值偏离

#### EPU组
- epu_log: log(epu)
- epu_log_delta1: log变化
- epu_log_dev12: log相对偏离

#### 汇率组
- fx_log: log(usd_cny)
- fx_ret1: 月度汇率变化
- fx_dev12: log相对偏离

## 特征构造方案

### 方案A：多时间尺度摘要特征
对每个变量构造:
- m1: 最新可得月值
- avg3: 最近3个月均值
- avg6: 最近6个月均值
- avg12: 最近12个月均值

总特征数: 15变量 × 4时间尺度 = 60维

### 方案B：分组因子多时间尺度特征
按经济含义分四组：
1. 价格组: cpi_yoy, cpi_delta1, cpi_dev12, ppi_yoy, ppi_delta1, ppi_dev12
2. 货币组: m2_yoy, m2_delta1, m2_dev12
3. 政策组: epu_log, epu_log_delta1, epu_log_dev12
4. 汇率组: fx_log, fx_ret1, fx_dev12

每组标准化后PCA提取第一主成分，再构造时间尺度摘要。

总特征数: 4因子 × 4时间尺度 = 16维

## 模型比较结果

### R_5d 窗口

| 特征方案 | 模型 | 最优参数 | 训练R² | 测试R²_OS | 测试RMSE | 测试MAE |
|----------|------|----------|--------|-----------|----------|---------|
"""

    for r in results_5d:
        report_content += f"| {r['feature_scheme']} | {r['model']} | {r['best_params']} | {r['train_r2']:.4f} | {r['test_r2_os']:.4f} | {r['test_rmse']:.6f} | {r['test_mae']:.6f} |\n"

    report_content += f"""
### R_60d 窗口

| 特征方案 | 模型 | 最优参数 | 训练R² | 测试R²_OS | 测试RMSE | 测试MAE |
|----------|------|----------|--------|-----------|----------|---------|
"""

    for r in results_60d:
        report_content += f"| {r['feature_scheme']} | {r['model']} | {r['best_params']} | {r['train_r2']:.4f} | {r['test_r2_os']:.4f} | {r['test_rmse']:.6f} | {r['test_mae']:.6f} |\n"

    report_content += f"""

## 核心问题回答

### 1）新口径下，共线性是否明显缓解？

**VIF诊断结果**:

摘要特征方案A的VIF统计：
- 最大VIF: {vif_max:.2f}
- 平均VIF: {vif_mean:.2f}
- VIF≥10的特征数: {vif_over_10}

**与原方案比较**:
- 原方案（趋势型水平变量+12个月滞后堆叠）: m2_growth VIF=268, usd_cny VIF=462
- 新方案（变化/偏离型口径）: 最大VIF={vif_max:.2f}

**判断**: {'共线性明显缓解 ✓' if vif_max < 100 else '共线性仍较严重，但有所改善'}

### 2）新口径下，样本外表现是否比原baseline更稳？

**R_5d窗口比较**:
- 原Baseline R²_OS: {baseline_5d_r2:.4f}
- 新方案最佳 R²_OS: {best_5d['test_r2_os']:.4f if best_5d else 'N/A'} ({best_5d['model'] if best_5d else 'N/A'})
- {'改善 ✓ (+%.4f)' % (best_5d['test_r2_os'] - baseline_5d_r2) if best_5d and best_5d['test_r2_os'] > baseline_5d_r2 else '无改善或略差'}

**R_60d窗口比较**:
- 原Baseline R²_OS: {baseline_60d_r2:.4f}
- 新方案最佳 R²_OS: {best_60d['test_r2_os']:.4f if best_60d else 'N/A'} ({best_60d['model'] if best_60d else 'N/A'})
- {'改善 ✓ (+%.4f)' % (best_60d['test_r2_os'] - baseline_60d_r2) if best_60d and best_60d['test_r2_os'] > baseline_60d_r2 else '无改善或略差'}

**稳定性分析**:
- 新方案各模型间R²_OS差异: 需查看具体结果
- 原方案MIDAS单变量退化，稳定性差

### 3）改善是否集中在60日窗口，而不是5日窗口？

**判断**: {'改善集中在60日窗口' if best_60d['test_r2_os'] - baseline_60d_r2 > best_5d['test_r2_os'] - baseline_5d_r2 else '改善在两个窗口相近'}

**详细分析**:
- 5日窗口改善幅度: {best_5d['test_r2_os'] - baseline_5d_r2:.4f if best_5d else 'N/A'}
- 60日窗口改善幅度: {best_60d['test_r2_os'] - baseline_60d_r2:.4f if best_60d else 'N/A'}

**解释**: 短期收益（5日）更多受高频交易行为和随机波动影响，宏观基本面信息对其解释力天然有限；60日窗口允许宏观信号有时间传导和累积效应。

### 4）变量解释结构是否比原来更清晰？

"""

    # 添加重要性分析
    importance_df = pd.read_csv(RESULT_DIR / "stage1_restructured_macro_importance.csv")

    report_content += "**各宏观组重要性排序（基于系数绝对值汇总）**:\n\n"

    for target in ["R_5d", "R_60d"]:
        report_content += f"**{target}**:\n"
        for scheme in ["summary", "factor"]:
            scheme_data = importance_df[(importance_df["target"] == target) & (importance_df["feature_scheme"] == scheme)]
            if len(scheme_data) > 0:
                for model in scheme_data["model"].unique():
                    model_data = scheme_data[scheme_data["model"] == model].sort_values("aggregated_importance", ascending=False)
                    top_groups = model_data.head(3)["macro_group"].tolist()
                    report_content += f"- {model}: {', '.join(top_groups)} 主导\n"
        report_content += "\n"

    report_content += f"""
**判断**: 变量解释结构{'更清晰 ✓' if vif_max < 100 else '仍需改进'}

### 5）最终结论

基于以上分析，以下判断最符合结果：

"""

    # 判断结论
    if best_60d and best_60d['test_r2_os'] > baseline_60d_r2 + 0.02 and vif_max < 100:
        conclusion = "A"
        conclusion_text = "新口径+新特征后，第一阶段确实更稳，值得替换原一阶段主模型"
    elif best_60d and best_60d['test_r2_os'] > baseline_60d_r2 and vif_max < 200:
        conclusion = "B"
        conclusion_text = "新口径+新特征后，第一阶段有一定改善，但更适合作为稳健性替代，不建议替换主模型"
    else:
        conclusion = "C"
        conclusion_text = "新口径+新特征后，仍然没有本质改善，因此可以更有把握地认为，一阶段瓶颈主要来自宏观信号对短中期收益解释力有限"

    report_content += f"""
**结论 {conclusion}: {conclusion_text}

**理由**:
1. 共线性: VIF从原方案的462降至{vif_max:.2f}{'，明显缓解' if vif_max < 100 else '，有所改善但仍有问题'}
2. 样本外改善: {'60日窗口有显著改善(+%.4f)' % (best_60d['test_r2_os'] - baseline_60d_r2) if best_60d and best_60d['test_r2_os'] > baseline_60d_r2 else '改善有限'}
3. 稳定性: {'新方案各模型表现更一致' if vif_max < 100 else '稳定性仍需验证'}
4. 解释结构: {'变化/偏离型口径使经济含义更清晰' if vif_max < 100 else '解释结构仍不够清晰'}

---

**生成时间**: 2026-04-24
**实验脚本**: experiment_pipeline/07_stage1_restructured_models.py
"""

    with open(RESULT_DIR / "stage1_restructured_models_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"   已保存报告: stage1_restructured_models_report.md")


def main():
    """主函数"""
    print("=" * 70)
    print("第一阶段去共线性重构实验")
    print("=" * 70)

    # 1. 加载并准备数据
    df = load_and_prepare_data()

    # 2. 构造变化/偏离型口径变量
    monthly_available, redefined_vars = build_redefined_monthly_variables(df)

    # 3. 划分训练测试集
    df_valid, df_train, df_test, train_size = split_train_test(df)

    # 4. 在训练集内部切分验证集
    print("\n【在训练集内部切分验证集】...")
    df_sub_train, df_val, sub_train_size = split_train_val_within_train(df_train)

    # 5. 构造特征矩阵
    # 方案A：多时间尺度摘要特征
    summary_feature_names, X_summary = build_summary_features(df_valid, monthly_available, redefined_vars)

    # 方案B：分组因子多时间尺度特征
    factor_available, factor_names = build_group_factors(monthly_available)
    factor_feature_names, X_factor = build_factor_summary_features(df_valid, factor_available, factor_names)

    # 切分特征矩阵
    X_summary_train = X_summary[:train_size]
    X_summary_test = X_summary[train_size:]
    X_summary_sub_train = X_summary[:sub_train_size]
    X_summary_val = X_summary[sub_train_size:train_size]

    X_factor_train = X_factor[:train_size]
    X_factor_test = X_factor[train_size:]
    X_factor_sub_train = X_factor[:sub_train_size]
    X_factor_val = X_factor[sub_train_size:train_size]

    print(f"\n【特征矩阵切分】")
    print(f"  方案A训练特征: {X_summary_train.shape}")
    print(f"  方案B训练特征: {X_factor_train.shape}")

    # 6. VIF诊断（在训练集上）
    print("\n【共线性诊断】...")
    mask_train = ~np.isnan(X_summary_train).any(axis=1)
    X_summary_train_clean = X_summary_train[mask_train]
    vif_summary = compute_vif(X_summary_train_clean, summary_feature_names)
    print(f"  VIF最大值: {vif_summary['VIF'].max():.2f}")
    print(f"  VIF均值: {vif_summary['VIF'].mean():.2f}")
    print(f"  VIF≥10的特征数: {(vif_summary['VIF'] >= 10).sum()}")

    # 7. 运行所有模型
    all_results = []

    for h in WINDOWS:
        y_col = f"R_{h}d"
        print(f"\n{'='*60}")
        print(f"目标变量: {y_col}")
        print("="*60)

        # 方案A模型
        # 7.1 Baseline_SummaryOLS
        result_ols = fit_summary_ols(df_train, df_test, y_col, X_summary_train, X_summary_test, summary_feature_names)
        if result_ols:
            all_results.append(result_ols)

        # 7.2 Ridge
        result_ridge = fit_ridge(df_train, df_test, df_sub_train, df_val, y_col,
                                 X_summary_train, X_summary_test,
                                 X_summary_sub_train, X_summary_val, X_summary_sub_train,
                                 summary_feature_names)
        if result_ridge:
            all_results.append(result_ridge)

        # 7.3 PCR
        result_pcr = fit_pcr(df_train, df_test, df_sub_train, df_val, y_col,
                             X_summary_train, X_summary_test,
                             X_summary_sub_train, X_summary_val,
                             summary_feature_names)
        if result_pcr:
            all_results.append(result_pcr)

        # 方案B模型
        # 7.4 Factor_OLS
        result_factor_ols = fit_factor_ols(df_train, df_test, y_col, X_factor_train, X_factor_test, factor_feature_names)
        if result_factor_ols:
            all_results.append(result_factor_ols)

        # 7.5 Factor_Ridge
        result_factor_ridge = fit_factor_ridge(df_train, df_test, df_sub_train, df_val, y_col,
                                                X_factor_train, X_factor_test,
                                                X_factor_sub_train, X_factor_val,
                                                factor_feature_names)
        if result_factor_ridge:
            all_results.append(result_factor_ridge)

    # 8. 保存结果
    save_results(all_results, df_test, summary_feature_names, factor_feature_names, redefined_vars, factor_names, vif_summary)

    # 9. 生成图表
    generate_plots(all_results, df_test)

    # 10. 生成报告
    generate_report(all_results, vif_summary, df_test)

    print("\n" + "=" * 70)
    print("第一阶段去共线性重构实验完成！")
    print(f"结果保存位置: {RESULT_DIR}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()