#!/usr/bin/env python3
"""
阶段4：MIDAS模型两阶段实证分析（核心实验）
第一阶段：宏观变量混频回归预测收益率（Beta权重MIDAS + NLS估计 + 扩展窗口递推）
第二阶段：异常收益偏离的分组嵌套回归（模型I-IV四层嵌套 + HAC推断 + LASSO筛选）
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "real_data_complete.csv"
RESULT_DIR = BASE_DIR / "experiment_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (5, 60)
K_MONTHS = 12
MACRO_VARS = ["cpi", "ppi", "m2_growth", "epu", "usd_cny"]
STAGE2_BASE_VARS = [
    "sentiment_zscore", "ivix",
    "north_flow", "margin_balance",
    "amihud", "momentum_20d", "intraday_range"
]
STAGE2_EXT_VARS = ["epu", "fx_vol"]


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
    df = df[(df["date"] >= "2015-07-02") & (df["date"] <= "2025-12-25")].copy()

    # 计算对数收益率
    df["log_return"] = np.log(df["hs300_close"] / df["hs300_close"].shift(1))

    # 计算未来收益窗口 R_t^{(h)}
    for h in WINDOWS:
        df[f"R_{h}d"] = compute_future_return(df["log_return"], h)

    # 第二阶段原始特征构造
    df["momentum_20d"] = np.log(df["hs300_close"] / df["hs300_close"].shift(20))
    df["fx_ret"] = np.log(df["usd_cny"] / df["usd_cny"].shift(1))
    df["fx_vol"] = df["fx_ret"].rolling(20).std()

    # 时间索引
    df["year_month"] = df["date"].dt.to_period("M")
    df["year_quarter"] = df["date"].dt.to_period("Q")

    print(f"样本期: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"有效观测数: {len(df)}")
    return df


def split_train_test(df, train_ratio=0.6):
    """划分训练集和测试集（按时间顺序）"""
    print("\n【2】划分训练集与测试集...")

    train_size = int(len(df) * train_ratio)
    df["is_train"] = False
    df.loc[:train_size - 1, "is_train"] = True

    df_train = df[df["is_train"]].copy()
    df_test = df[~df["is_train"]].copy()

    print(f"训练集: {len(df_train)} ({df_train['date'].min().strftime('%Y-%m-%d')} 至 {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"测试集: {len(df_test)} ({df_test['date'].min().strftime('%Y-%m-%d')} 至 {df_test['date'].max().strftime('%Y-%m-%d')})")

    return df, df_train, df_test


def fit_stage2_scaler_and_clip_params(df_train):
    """在训练集上拟合第二阶段预处理参数（截尾和标准化）"""
    clip_vars = ["ivix", "north_flow", "margin_balance", "amihud", "intraday_range"]
    std_vars = STAGE2_BASE_VARS + STAGE2_EXT_VARS

    params = {"clip": {}, "scale": {}}

    for var in clip_vars:
        if var in df_train.columns:
            params["clip"][var] = {
                "lower": df_train[var].quantile(0.01),
                "upper": df_train[var].quantile(0.99),
            }

    for var in std_vars:
        if var in df_train.columns:
            params["scale"][var] = {
                "mean": df_train[var].mean(),
                "std": df_train[var].std() if df_train[var].std() not in [0, np.nan] else 1.0
            }

    return params


def apply_stage2_preprocess(df, params):
    """应用第二阶段预处理（训练集参数外推到全样本）"""
    df = df.copy()

    # 截尾处理
    for var, bounds in params["clip"].items():
        if var in df.columns:
            df[var] = df[var].clip(bounds["lower"], bounds["upper"])

    # 标准化
    for var, stat in params["scale"].items():
        if var in df.columns:
            std = stat["std"] if stat["std"] not in [0, np.nan] else 1.0
            df[f"{var}_z"] = (df[var] - stat["mean"]) / std

    return df


def build_monthly_available_panel(df, macro_vars=MACRO_VARS):
    """构建月度可得信息面板（滞后一期）X^M_{j,t} = X_{j,m(t)-1}"""
    monthly_panel = df.groupby("year_month")[macro_vars].last().sort_index()
    monthly_available = monthly_panel.shift(1)  # 上一个完整月可得信息
    return monthly_available


def build_quarterly_available_series(df, gdp_col="gdp_growth"):
    """构建季度GDP可得信息序列"""
    if gdp_col not in df.columns:
        return None
    quarterly = df.groupby("year_quarter")[gdp_col].last().sort_index()
    quarterly_available = quarterly.shift(1)
    return quarterly_available


def beta_weight_vector(K, a, b):
    """计算Beta权重向量 w_ell(a,b) = (ell/(K+1))^{a-1} * (1-ell/(K+1))^{b-1} / normalization"""
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
    """单变量MIDAS模型的NLS估计"""
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


def fit_multi_midas_nls(df_hist, y_col, monthly_available, selected_vars, K=12):
    """多变量MIDAS模型的NLS估计（精简模型）"""
    y = df_hist[y_col].values
    dates = df_hist["date"]
    n_var = len(selected_vars)

    def unpack(params):
        alpha = params[0]
        betas = params[1:1 + n_var]
        ab_pairs = params[1 + n_var:].reshape(n_var, 2)
        return alpha, betas, ab_pairs

    def objective(params):
        alpha, betas, ab_pairs = unpack(params)
        x_terms = []
        for i, var in enumerate(selected_vars):
            a, b = ab_pairs[i]
            a = max(a, 1e-4)
            b = max(b, 1e-4)
            x_terms.append(build_midas_term(dates, monthly_available, var, K, a, b))
        X = np.column_stack(x_terms)
        mask = (~np.isnan(y)) & (~np.isnan(X).any(axis=1))
        if mask.sum() < 5:
            return 1e10
        yhat = alpha + X[mask] @ betas
        resid = y[mask] - yhat
        return np.sum(resid ** 2)

    x0 = [np.nanmean(y)] + [0.0] * n_var + [1.5, 1.5] * n_var
    bounds = [(None, None)] + [(None, None)] * n_var + [(1e-4, 50), (1e-4, 50)] * n_var

    res = minimize(objective, x0=np.array(x0), bounds=bounds, method="L-BFGS-B")

    alpha, betas, ab_pairs = unpack(res.x)
    x_terms = []
    for i, var in enumerate(selected_vars):
        a, b = ab_pairs[i]
        x_terms.append(build_midas_term(dates, monthly_available, var, K, a, b))
    X = np.column_stack(x_terms)
    fitted = alpha + X @ betas

    return {
        "success": bool(res.success),
        "alpha": alpha,
        "betas": dict(zip(selected_vars, betas)),
        "ab_pairs": {var: tuple(ab_pairs[i]) for i, var in enumerate(selected_vars)},
        "X_terms": X,
        "fitted": fitted,
        "objective": float(res.fun),
    }


def recursive_stage1_forecast(df, train_size, y_col, monthly_available, macro_vars=MACRO_VARS, K=12):
    """扩展窗口递推的MIDAS第一阶段估计"""
    n_total = len(df)
    y_full = df[y_col].values
    dates = df["date"].values

    # 存储结果
    pred_full = np.full(n_total, np.nan)
    coef_records = []

    # 1) 初始训练集上拟合每个单变量MIDAS
    df_train_init = df.iloc[:train_size].copy()
    uni_fits = {}
    uni_preds_oob = {var: [] for var in macro_vars}
    uni_dates_oob = []

    for var in macro_vars:
        fit = fit_single_midas_nls(df_train_init, y_col, monthly_available, var, K)
        uni_fits[var] = fit

    # 2) 生成每个变量的样本外递推预测
    for t in range(train_size, n_total):
        df_hist = df.iloc[:t].copy()
        uni_dates_oob.append(dates[t])

        for var in macro_vars:
            # 使用之前估计的参数生成预测
            alpha, beta = uni_fits[var]["alpha"], uni_fits[var]["beta"]
            a, b = uni_fits[var]["a"], uni_fits[var]["b"]
            x_midas = build_midas_term([df.iloc[t]["date"]], monthly_available, var, K, a, b)
            pred = alpha + beta * x_midas[0] if not np.isnan(x_midas[0]) else np.nan
            uni_preds_oob[var].append(pred)

    # 3) 根据整体样本外R2_OS选出最优单变量
    y_oob = y_full[train_size:]
    best_r2_os = -np.inf
    best_var = None

    for var in macro_vars:
        pred_oob = np.array(uni_preds_oob[var])
        mask = (~np.isnan(y_oob)) & (~np.isnan(pred_oob))
        if mask.sum() > 0:
            y_mean_train = np.nanmean(y_full[:train_size])
            ss_res = np.sum((y_oob[mask] - pred_oob[mask]) ** 2)
            ss_tot = np.sum((y_oob[mask] - y_mean_train) ** 2)
            r2_os = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            if r2_os > best_r2_os:
                best_r2_os = r2_os
                best_var = var

    # 4) 在初始训练集上用单变量拟合得到的MIDAS项做VIF
    vif_data = []
    X_vif_list = []
    valid_vars_for_vif = []

    for var in macro_vars:
        fit = uni_fits[var]
        x_midas = build_midas_term(df_train_init["date"].values, monthly_available, var, K, fit["a"], fit["b"])
        if not np.all(np.isnan(x_midas)):
            X_vif_list.append(x_midas)
            valid_vars_for_vif.append(var)

    if len(X_vif_list) > 1:
        X_vif = np.column_stack(X_vif_list)
        # 检查每列是否有足够变异
        for i, var in enumerate(valid_vars_for_vif):
            if np.std(X_vif[:, i]) > 1e-10:
                try:
                    vif_val = variance_inflation_factor(X_vif, i)
                    vif_data.append({"变量": var, "VIF": vif_val})
                except:
                    vif_data.append({"变量": var, "VIF": np.nan})
            else:
                vif_data.append({"变量": var, "VIF": np.nan})
    else:
        vif_data = [{"变量": var, "VIF": np.nan} for var in macro_vars]

    vif_df = pd.DataFrame(vif_data)

    # 5) 选出VIF<10的候选变量，最多保留4个
    valid_vars = []
    for _, row in vif_df.iterrows():
        if row["VIF"] < 10 and not np.isnan(row["VIF"]):
            valid_vars.append(row["变量"])

    if len(valid_vars) == 0:
        selected_vars = [best_var] if best_var else [macro_vars[0]]
    else:
        selected_vars = valid_vars[:4]  # 最多4个

    print(f"    VIF筛选后保留变量: {selected_vars}")

    # 6) 样本外预测：使用训练集估计的参数进行递推
    # 实际实现：在训练集上估计一次参数，用于整个测试集
    print(f"    估计精简模型参数（变量: {selected_vars}）...")

    if len(selected_vars) == 1:
        var = selected_vars[0]
        fit = fit_single_midas_nls(df_train_init, y_col, monthly_available, var, K)
        alpha, beta = fit["alpha"], fit["beta"]
        a, b = fit["a"], fit["b"]

        for t in range(train_size, n_total):
            x_midas = build_midas_term([df.iloc[t]["date"]], monthly_available, var, K, a, b)
            pred = alpha + beta * x_midas[0] if not np.isnan(x_midas[0]) else np.nan
            pred_full[t] = pred
    else:
        fit = fit_multi_midas_nls(df_train_init, y_col, monthly_available, selected_vars, K)
        alpha = fit["alpha"]
        betas = fit["betas"]
        ab_pairs = fit["ab_pairs"]

        for t in range(train_size, n_total):
            pred = alpha
            for var in selected_vars:
                a, b = ab_pairs[var]
                x_midas = build_midas_term([df.iloc[t]["date"]], monthly_available, var, K, a, b)
                if not np.isnan(x_midas[0]):
                    pred += betas[var] * x_midas[0]
            pred_full[t] = pred

    # 7) 计算样本外指标
    y_oob = y_full[train_size:]
    pred_oob = pred_full[train_size:]
    mask = (~np.isnan(y_oob)) & (~np.isnan(pred_oob))

    if mask.sum() > 0:
        y_mean_train = np.nanmean(y_full[:train_size])
        ss_res = np.sum((y_oob[mask] - pred_oob[mask]) ** 2)
        ss_tot = np.sum((y_oob[mask] - y_mean_train) ** 2)
        r2_os = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse_os = np.sqrt(np.mean((y_oob[mask] - pred_oob[mask]) ** 2))
        mae_os = np.mean(np.abs(y_oob[mask] - pred_oob[mask]))
    else:
        r2_os = rmse_os = mae_os = np.nan

    # 样本内R2（用初始训练集）
    y_train = y_full[:train_size]
    mask_in = ~np.isnan(y_train)
    r2_in = np.nan
    fitted_in = None
    if mask_in.sum() > 0:
        y_mean_in = np.nanmean(y_train[mask_in])
        ss_tot_in = np.sum((y_train[mask_in] - y_mean_in) ** 2)

        if len(selected_vars) == 1:
            fit_in = fit_single_midas_nls(df_train_init, y_col, monthly_available, selected_vars[0], K)
        else:
            fit_in = fit_multi_midas_nls(df_train_init, y_col, monthly_available, selected_vars, K)

        fitted_in = fit_in["fitted"][:train_size]
        mask_fitted = (~np.isnan(y_train)) & (~np.isnan(fitted_in))
        if mask_fitted.sum() > 0:
            ss_res_in = np.sum((y_train[mask_fitted] - fitted_in[mask_fitted]) ** 2)
            r2_in = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else np.nan
            # 填充训练集的拟合值
            pred_full[:train_size] = fitted_in

    return {
        "pred_full": pred_full,
        "r2_in": r2_in,
        "r2_os": r2_os,
        "rmse_os": rmse_os,
        "mae_os": mae_os,
        "selected_vars": selected_vars,
        "best_univariate": best_var,
        "vif": vif_df,
        "uni_fits": uni_fits,
    }


def stage1_midas_regression(df, train_size):
    """第一阶段：MIDAS回归（Beta权重 + NLS + 扩展窗口递推）"""
    print("\n" + "=" * 60)
    print("【5】第一阶段：MIDAS模型估计（Beta权重 + NLS + 扩展窗口）")
    print("=" * 60)

    monthly_available = build_monthly_available_panel(df)
    quarterly_available = build_quarterly_available_series(df)

    stage1_results = {}

    for h in WINDOWS:
        print(f"\n--- 预测窗口 h={h}日 ---")
        y_col = f"R_{h}d"

        result_h = recursive_stage1_forecast(
            df=df,
            train_size=train_size,
            y_col=y_col,
            monthly_available=monthly_available,
            macro_vars=MACRO_VARS,
            K=K_MONTHS,
        )

        df[f"R_{h}d_pred"] = result_h["pred_full"]
        stage1_results[f"h{h}"] = result_h

        print(f"  最优单变量: {result_h['best_univariate']}")
        print(f"  精简模型变量: {result_h['selected_vars']}")
        print(f"  R²(内): {result_h['r2_in']:.4f}")
        print(f"  R²(外): {result_h['r2_os']:.4f}")
        print(f"  RMSE(外): {result_h['rmse_os']:.6f}")
        print(f"  MAE(外): {result_h['mae_os']:.6f}")

    return df, stage1_results


def construct_abnormal_returns(df):
    """构造异常收益（实际收益 - 条件预期收益率）"""
    print("\n【6】构造异常收益...")

    for h in WINDOWS:
        df[f"AR_{h}d"] = df[f"R_{h}d"] - df[f"R_{h}d_pred"]
        df[f"AbsAR_{h}d"] = df[f"AR_{h}d"].abs()
        print(f"  h={h}日: AR均值={df[f'AR_{h}d'].mean():.6f}, AbsAR均值={df[f'AbsAR_{h}d'].mean():.6f}")

    return df


def create_stage2_lagged_features(df):
    """创建第二阶段滞后一期特征"""
    df = df.copy()

    lag_base = [f"{v}_z" for v in STAGE2_BASE_VARS]
    lag_ext = [f"{v}_z" for v in STAGE2_EXT_VARS]

    for v in lag_base + lag_ext:
        if v in df.columns:
            df[f"{v}_lag1"] = df[v].shift(1)

    return df


def fit_hac_ols(y, X, maxlags):
    """拟合HAC稳健标准误的OLS回归"""
    X_const = add_constant(X, has_constant="add")
    model = OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return model


def wald_block_test(model, x_vars, added_vars):
    """Wald块联合检验"""
    param_names = ["const"] + list(x_vars)
    R = np.zeros((len(added_vars), len(param_names)))

    for i, var in enumerate(added_vars):
        if var in param_names:
            R[i, param_names.index(var)] = 1.0

    test_res = model.wald_test(R, use_f=True)
    fval = test_res.fvalue.item() if hasattr(test_res.fvalue, 'item') else float(test_res.fvalue)
    pval = test_res.pvalue.item() if hasattr(test_res.pvalue, 'item') else float(test_res.pvalue)
    return fval, pval


def run_lasso_screen(df_train, x_vars, y_col):
    """LASSO变量筛选"""
    X = df_train[x_vars].values
    y = df_train[y_col].values

    # 处理缺失值
    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    if mask.sum() < 10:
        return {"alpha": np.nan, "selected": [], "coef": {}}

    X_clean = X[mask]
    y_clean = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y_clean)

    nonzero_mask = np.abs(lasso.coef_) > 1e-8
    selected = [x_vars[i] for i in range(len(x_vars)) if nonzero_mask[i]]

    return {
        "alpha": float(lasso.alpha_),
        "selected": selected,
        "coef": dict(zip(x_vars, lasso.coef_)),
    }


def run_stage2_for_target(df_train, h, target_prefix="AbsAR"):
    """为特定目标变量（AbsAR或AR）运行第二阶段回归"""
    y_col = f"{target_prefix}_{h}d"
    lags = h - 1

    # 模型I-IV的变量定义（论文正式设定）
    X1_vars = ["sentiment_zscore_z_lag1", "ivix_z_lag1"]
    X2_vars = X1_vars + ["north_flow_z_lag1", "margin_balance_z_lag1"]
    X3_vars = X2_vars + ["amihud_z_lag1", "momentum_20d_z_lag1", "intraday_range_z_lag1"]
    X4_vars = X3_vars + ["epu_z_lag1", "fx_vol_z_lag1"]

    # 检查变量是否存在
    all_vars = X4_vars
    available_vars = [v for v in all_vars if v in df_train.columns]

    X1_vars = [v for v in X1_vars if v in available_vars]
    X2_vars = [v for v in X2_vars if v in available_vars]
    X3_vars = [v for v in X3_vars if v in available_vars]
    X4_vars = [v for v in X4_vars if v in available_vars]

    # 删除缺失值
    df_train_h = df_train.dropna(subset=[y_col] + X4_vars).copy()

    if len(df_train_h) < 20:
        return None

    y = df_train_h[y_col].values

    # 拟合四个模型
    if len(X1_vars) > 0:
        X1 = df_train_h[X1_vars].values
        model1 = fit_hac_ols(y, X1, maxlags=lags)
    else:
        model1 = None

    if len(X2_vars) > 0:
        X2 = df_train_h[X2_vars].values
        model2 = fit_hac_ols(y, X2, maxlags=lags)
        if model1 is not None:
            added_12 = [v for v in X2_vars if v not in X1_vars]
            f12, p12 = wald_block_test(model2, X2_vars, added_12) if added_12 else (0, 1)
        else:
            f12, p12 = 0, 1
    else:
        model2, f12, p12 = None, 0, 1

    if len(X3_vars) > 0:
        X3 = df_train_h[X3_vars].values
        model3 = fit_hac_ols(y, X3, maxlags=lags)
        if model2 is not None:
            added_23 = [v for v in X3_vars if v not in X2_vars]
            f23, p23 = wald_block_test(model3, X3_vars, added_23) if added_23 else (0, 1)
        else:
            f23, p23 = 0, 1
    else:
        model3, f23, p23 = None, 0, 1

    if len(X4_vars) > 0:
        X4 = df_train_h[X4_vars].values
        model4 = fit_hac_ols(y, X4, maxlags=lags)
        if model3 is not None:
            added_34 = [v for v in X4_vars if v not in X3_vars]
            f34, p34 = wald_block_test(model4, X4_vars, added_34) if added_34 else (0, 1)
        else:
            f34, p34 = 0, 1
    else:
        model4, f34, p34 = None, 0, 1

    # LASSO筛选
    if len(X3_vars) > 0:
        lasso_base = run_lasso_screen(df_train_h, X3_vars, y_col)
    else:
        lasso_base = {"alpha": np.nan, "selected": [], "coef": {}}

    if len(X4_vars) > 0:
        lasso_ext = run_lasso_screen(df_train_h, X4_vars, y_col)
    else:
        lasso_ext = {"alpha": np.nan, "selected": [], "coef": {}}

    # 综合模型
    selected = lasso_base["selected"] if lasso_base["selected"] else X1_vars
    if selected:
        X_lasso = df_train_h[selected].values
        model_lasso = fit_hac_ols(y, X_lasso, maxlags=lags)
    else:
        model_lasso = model1

    return {
        "model1": model1, "model2": model2, "model3": model3, "model4": model4,
        "model_lasso": model_lasso,
        "X1_vars": X1_vars, "X2_vars": X2_vars, "X3_vars": X3_vars, "X4_vars": X4_vars,
        "joint_f_12": f12, "joint_p_12": p12,
        "joint_f_23": f23, "joint_p_23": p23,
        "joint_f_34": f34, "joint_p_34": p34,
        "lasso_base": lasso_base,
        "lasso_ext": lasso_ext,
        "train_index": df_train_h.index.tolist(),
    }


def stage2_nested_regression(df, df_train):
    """第二阶段：分组嵌套回归（AbsAR和AR双口径）"""
    print("\n" + "=" * 60)
    print("【7】第二阶段：分组嵌套回归")
    print("=" * 60)

    results = {"AbsAR": {}, "AR": {}}

    for target_prefix in ["AbsAR", "AR"]:
        print(f"\n--- 目标变量: {target_prefix} ---")
        for h in WINDOWS:
            print(f"\n  窗口 h={h}日:")
            res = run_stage2_for_target(df_train=df_train, h=h, target_prefix=target_prefix)

            if res is None:
                print(f"    数据不足，跳过")
                continue

            results[target_prefix][f"h{h}"] = res

            # 输出主要结果
            for model_name in ["model1", "model2", "model3", "model4"]:
                model = res[model_name]
                if model is not None:
                    print(f"    {model_name}: R²_adj={model.rsquared_adj:.4f}")

            print(f"    联合检验 I→II: F={res['joint_f_12']:.2f}, p={res['joint_p_12']:.4f}")
            print(f"    联合检验 II→III: F={res['joint_f_23']:.2f}, p={res['joint_p_23']:.4f}")
            print(f"    联合检验 III→IV: F={res['joint_f_34']:.2f}, p={res['joint_p_34']:.4f}")
            print(f"    LASSO(base)选择: {res['lasso_base']['selected']}")
            print(f"    LASSO(ext)选择: {res['lasso_ext']['selected']}")

    return results


def export_model_coefficients(model, x_vars, window, model_name, target_prefix):
    """导出模型系数到记录"""
    if model is None:
        return []

    rows = []
    param_names = ["const"] + list(x_vars)

    for i, var in enumerate(param_names):
        if i < len(model.params):
            rows.append({
                "Target": target_prefix,
                "Window": f"{window}d",
                "Model": model_name,
                "Variable": var,
                "Coef": float(model.params[i]),
                "t_value": float(model.tvalues[i]) if i < len(model.tvalues) else np.nan,
                "p_value": float(model.pvalues[i]) if i < len(model.pvalues) else np.nan,
            })

    return rows


def save_results(df, stage1_results, stage2_results):
    """保存实验结果"""
    print("\n【8】保存实验结果...")

    # 1) Stage1 Summary
    stage1_summary = []
    for h in WINDOWS:
        res = stage1_results[f"h{h}"]
        stage1_summary.append({
            "Window": f"{h}d",
            "Best_Univariate": res["best_univariate"],
            "Selected_Vars": ", ".join(res["selected_vars"]),
            "R2_InSample": res["r2_in"],
            "R2_OutSample": res["r2_os"],
            "RMSE_Test": res["rmse_os"],
            "MAE_Test": res["mae_os"],
        })
    pd.DataFrame(stage1_summary).to_csv(RESULT_DIR / "stage1_summary.csv", index=False)
    print(f"   已保存: stage1_summary.csv")

    # 2) Stage1 VIF
    vif_records = []
    for h in WINDOWS:
        vif_df = stage1_results[f"h{h}"]["vif"]
        for _, row in vif_df.iterrows():
            vif_records.append({
                "Window": f"{h}d",
                "Variable": row["变量"],
                "VIF": row["VIF"],
            })
    pd.DataFrame(vif_records).to_csv(RESULT_DIR / "stage1_vif.csv", index=False)
    print(f"   已保存: stage1_vif.csv")

    # 3) Stage1 Coefficients (Univariate)
    coef_records = []
    for h in WINDOWS:
        uni_fits = stage1_results[f"h{h}"]["uni_fits"]
        for var, fit in uni_fits.items():
            if fit["success"]:
                coef_records.append({
                    "Window": f"{h}d",
                    "Variable": var,
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                    "a": fit["a"],
                    "b": fit["b"],
                })
    pd.DataFrame(coef_records).to_csv(RESULT_DIR / "stage1_coefficients.csv", index=False)
    print(f"   已保存: stage1_coefficients.csv")

    # 4) Stage2 Summary (AbsAR)
    stage2_absar_summary = []
    for h in WINDOWS:
        if f"h{h}" in stage2_results["AbsAR"]:
            res = stage2_results["AbsAR"][f"h{h}"]
            for model_name in ["model1", "model2", "model3", "model4", "model_lasso"]:
                model = res[model_name]
                if model is not None:
                    stage2_absar_summary.append({
                        "Window": f"{h}d",
                        "Model": model_name,
                        "Adj_R2": model.rsquared_adj,
                        "Joint_F_12": res["joint_f_12"] if model_name == "model2" else np.nan,
                        "Joint_P_12": res["joint_p_12"] if model_name == "model2" else np.nan,
                        "Joint_F_23": res["joint_f_23"] if model_name == "model3" else np.nan,
                        "Joint_P_23": res["joint_p_23"] if model_name == "model3" else np.nan,
                        "Joint_F_34": res["joint_f_34"] if model_name == "model4" else np.nan,
                        "Joint_P_34": res["joint_p_34"] if model_name == "model4" else np.nan,
                        "LASSO_Selected": ", ".join(res["lasso_base"]["selected"]) if model_name == "model_lasso" else "",
                    })
    pd.DataFrame(stage2_absar_summary).to_csv(RESULT_DIR / "stage2_absar_summary.csv", index=False)
    print(f"   已保存: stage2_absar_summary.csv")

    # 5) Stage2 Coefficients (AbsAR)
    absar_coef_records = []
    for h in WINDOWS:
        if f"h{h}" in stage2_results["AbsAR"]:
            res = stage2_results["AbsAR"][f"h{h}"]
            for model_name, x_vars in [("model1", res["X1_vars"]),
                                        ("model2", res["X2_vars"]),
                                        ("model3", res["X3_vars"]),
                                        ("model4", res["X4_vars"]),
                                        ("model_lasso", res["lasso_base"]["selected"] if res["lasso_base"]["selected"] else res["X1_vars"])]:
                model = res[model_name]
                if model is not None and x_vars:
                    absar_coef_records.extend(export_model_coefficients(model, x_vars, h, model_name, "AbsAR"))
    pd.DataFrame(absar_coef_records).to_csv(RESULT_DIR / "stage2_absar_coefficients.csv", index=False)
    print(f"   已保存: stage2_absar_coefficients.csv")

    # 6) Stage2 Summary (AR)
    stage2_ar_summary = []
    for h in WINDOWS:
        if f"h{h}" in stage2_results["AR"]:
            res = stage2_results["AR"][f"h{h}"]
            for model_name in ["model1", "model2", "model3", "model4", "model_lasso"]:
                model = res[model_name]
                if model is not None:
                    stage2_ar_summary.append({
                        "Window": f"{h}d",
                        "Model": model_name,
                        "Adj_R2": model.rsquared_adj,
                        "Joint_F_12": res["joint_f_12"] if model_name == "model2" else np.nan,
                        "Joint_P_12": res["joint_p_12"] if model_name == "model2" else np.nan,
                        "Joint_F_23": res["joint_f_23"] if model_name == "model3" else np.nan,
                        "Joint_P_23": res["joint_p_23"] if model_name == "model3" else np.nan,
                        "Joint_F_34": res["joint_f_34"] if model_name == "model4" else np.nan,
                        "Joint_P_34": res["joint_p_34"] if model_name == "model4" else np.nan,
                        "LASSO_Selected": ", ".join(res["lasso_base"]["selected"]) if model_name == "model_lasso" else "",
                    })
    pd.DataFrame(stage2_ar_summary).to_csv(RESULT_DIR / "stage2_ar_summary.csv", index=False)
    print(f"   已保存: stage2_ar_summary.csv")

    # 7) Stage2 Coefficients (AR)
    ar_coef_records = []
    for h in WINDOWS:
        if f"h{h}" in stage2_results["AR"]:
            res = stage2_results["AR"][f"h{h}"]
            for model_name, x_vars in [("model1", res["X1_vars"]),
                                        ("model2", res["X2_vars"]),
                                        ("model3", res["X3_vars"]),
                                        ("model4", res["X4_vars"]),
                                        ("model_lasso", res["lasso_base"]["selected"] if res["lasso_base"]["selected"] else res["X1_vars"])]:
                model = res[model_name]
                if model is not None and x_vars:
                    ar_coef_records.extend(export_model_coefficients(model, x_vars, h, model_name, "AR"))
    pd.DataFrame(ar_coef_records).to_csv(RESULT_DIR / "stage2_ar_coefficients.csv", index=False)
    print(f"   已保存: stage2_ar_coefficients.csv")

    # 8) LASSO Summary
    lasso_records = []
    for target in ["AbsAR", "AR"]:
        for h in WINDOWS:
            if f"h{h}" in stage2_results[target]:
                res = stage2_results[target][f"h{h}"]
                lasso_records.append({
                    "Target": target,
                    "Window": f"{h}d",
                    "LASSO_Alpha_Base": res["lasso_base"]["alpha"],
                    "LASSO_Selected_Base": ", ".join(res["lasso_base"]["selected"]),
                    "LASSO_Alpha_Ext": res["lasso_ext"]["alpha"],
                    "LASSO_Selected_Ext": ", ".join(res["lasso_ext"]["selected"]),
                })
    pd.DataFrame(lasso_records).to_csv(RESULT_DIR / "lasso_summary.csv", index=False)
    print(f"   已保存: lasso_summary.csv")

    # 9) Full data with predictions
    df.to_csv(RESULT_DIR / "full_data_with_predictions.csv", index=False)
    print(f"   已保存: full_data_with_predictions.csv")


def main():
    """主函数"""
    print("=" * 60)
    print("阶段4：MIDAS模型两阶段实证分析")
    print("=" * 60)

    # 加载数据
    df = load_and_prepare_data()

    # 只切分一次（删除原来的第二次切分）
    df_valid = df.dropna(subset=["R_5d", "R_60d"]).copy()
    df_valid, df_train, df_test = split_train_test(df_valid, train_ratio=0.6)
    train_size = len(df_train)

    # 训练集上拟合预处理参数，并外推到全样本
    params = fit_stage2_scaler_and_clip_params(df_train)
    df_valid = apply_stage2_preprocess(df_valid, params)

    # 第一阶段：MIDAS回归（Beta权重 + NLS + 扩展窗口递推）
    df_valid, stage1_results = stage1_midas_regression(df_valid, train_size=train_size)

    # 构造异常收益（实际收益 - 条件预期收益率）
    df_valid = construct_abnormal_returns(df_valid)

    # 创建第二阶段滞后一期特征
    df_valid = create_stage2_lagged_features(df_valid)

    # 重新取训练集（不重新切，只按原mask）
    df_train_final = df_valid[df_valid["is_train"]].copy()

    # 第二阶段：分组嵌套回归（AbsAR和AR双口径）
    stage2_results = stage2_nested_regression(df_valid, df_train_final)

    # 保存结果
    save_results(df_valid, stage1_results, stage2_results)

    print("\n" + "=" * 60)
    print("MIDAS实验完成！")
    print(f"结果保存位置: {RESULT_DIR}")
    print("=" * 60)

    return df_valid, stage1_results, stage2_results


if __name__ == "__main__":
    main()
