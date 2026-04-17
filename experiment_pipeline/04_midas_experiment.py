#!/usr/bin/env python3
"""
阶段4：MIDAS模型两阶段实证分析（核心实验）
第一阶段：宏观变量混频回归预测收益率
第二阶段：异常收益偏离的分组嵌套回归
"""

import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

warnings.filterwarnings('ignore')

DATA_PATH = '/home/marktom/bigdata-fin/real_data_complete.csv'
RESULT_DIR = '/home/marktom/bigdata-fin/experiment_results'
os.makedirs(RESULT_DIR, exist_ok=True)


def load_and_prepare_data():
    """加载并准备数据"""
    print("\n【1】加载并准备数据...")

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 过滤样本期
    df = df[(df['date'] >= '2015-07-02') & (df['date'] <= '2025-12-25')].copy()

    # 计算收益率
    df['log_return'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))
    for h in [5, 60]:
        df[f'R_{h}d'] = df['log_return'].shift(-h).rolling(window=h).sum().values

    df = df.dropna(subset=['R_5d', 'R_60d']).reset_index(drop=True)

    print(f"样本期: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"有效观测数: {len(df)}")
    return df


def split_train_test(df, train_ratio=0.6):
    """划分训练集和测试集"""
    print("\n【2】划分训练集与测试集...")

    train_size = int(len(df) * train_ratio)
    df['is_train'] = False
    df.loc[:train_size-1, 'is_train'] = True

    df_train = df[df['is_train']].copy()
    df_test = df[~df['is_train']].copy()

    print(f"训练集: {len(df_train)} ({df_train['date'].min().strftime('%Y-%m-%d')} 至 {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"测试集: {len(df_test)} ({df_test['date'].min().strftime('%Y-%m-%d')} 至 {df_test['date'].max().strftime('%Y-%m-%d')})")

    return df, df_train, df_test


def preprocess_data(df, df_train):
    """数据预处理（缩尾和标准化）"""
    print("\n【3】数据预处理...")

    # 连续变量
    continuous_vars = ['ivix', 'north_flow', 'margin_balance', 'amihud', 'volatility_20d', 'intraday_range']
    available_vars = [v for v in continuous_vars if v in df.columns]

    # 缩尾处理（训练集1%分位数）
    for var in available_vars:
        lower = df_train[var].quantile(0.01)
        upper = df_train[var].quantile(0.99)
        df[var] = df[var].clip(lower, upper)

    # 标准化
    for var in available_vars:
        mean = df_train[var].mean()
        std = df_train[var].std()
        df[f'{var}_z'] = (df[var] - mean) / std

    print(f"   缩尾和标准化完成（变量: {available_vars}）")
    return df


def construct_midas_weights(df):
    """构造MIDAS加权项"""
    print("\n【4】构造MIDAS加权项...")

    macro_vars = ['cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny']
    df['year_month'] = df['date'].dt.to_period('M')

    # 获取滞后一期的月度数据
    for var in macro_vars:
        monthly_data = df.groupby('year_month')[var].last()
        monthly_dict = monthly_data.shift(1).to_dict()
        df[f'{var}_monthly'] = df['year_month'].map(monthly_dict)

    # 构造MIDAS加权项（12个月等权重）
    K = 12
    for var in macro_vars:
        df[f'{var}_MIDAS'] = np.nan
        for i in range(len(df)):
            current_month = df.loc[i, 'year_month']
            midas_val = 0
            count = 0
            for lag in range(1, K+1):
                lag_month = current_month - lag
                if lag_month in df['year_month'].values:
                    month_data = df[df['year_month'] == lag_month][var].values
                    if len(month_data) > 0:
                        midas_val += month_data[-1]
                        count += 1
            if count == K:
                df.loc[i, f'{var}_MIDAS'] = midas_val / K

    df = df.dropna(subset=[f'{var}_MIDAS' for var in macro_vars]).reset_index(drop=True)
    print(f"   MIDAS构造后观测数: {len(df)}")
    return df


def stage1_midas_regression(df, df_train, df_test):
    """第一阶段：MIDAS回归"""
    print("\n" + "=" * 60)
    print("【5】第一阶段：MIDAS模型估计")
    print("=" * 60)

    macro_vars = ['cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny']
    midas_results = {}

    for h in [5, 60]:
        print(f"\n--- 预测窗口 h={h}日 ---")

        y_train = df_train[f'R_{h}d'].values
        y_test = df_test[f'R_{h}d'].values

        # 单变量模型
        univariate_results = {}
        for var in macro_vars:
            X_train = df_train[f'{var}_MIDAS'].values.reshape(-1, 1)
            X_test = df_test[f'{var}_MIDAS'].values.reshape(-1, 1)

            X_const = add_constant(X_train)
            model = OLS(y_train, X_const).fit()

            pred_train = model.predict(X_const)
            pred_test = model.predict(add_constant(X_test))

            r2_train = 1 - np.sum((y_train - pred_train)**2) / np.sum((y_train - y_train.mean())**2)
            y_test_mean = df_train[f'R_{h}d'].mean()
            r2_os = 1 - np.sum((y_test - pred_test)**2) / np.sum((y_test - y_test_mean)**2)
            rmse_test = np.sqrt(np.mean((y_test - pred_test)**2))

            coef = model.params[1] if len(model.params) > 1 else model.params[0]
            tval = model.tvalues[1] if len(model.tvalues) > 1 else model.tvalues[0]
            pval = model.pvalues[1] if len(model.pvalues) > 1 else model.pvalues[0]

            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f"  {var:10s}: R²(内)={r2_train:.4f}, R²(外)={r2_os:.4f}, coef={coef:.6f}, t={tval:.2f}{sig}")

            univariate_results[var] = {
                'r2_train': r2_train, 'r2_os': r2_os, 'rmse_test': rmse_test,
                'pred_train': pred_train, 'pred_test': pred_test,
                'params': model.params, 'tvalues': model.tvalues, 'pvalues': model.pvalues
            }

        # 找出最优单变量
        best_var = max(univariate_results.keys(), key=lambda v: univariate_results[v]['r2_os'])
        print(f"\n  ★ 最优单变量: {best_var} (样本外R²={univariate_results[best_var]['r2_os']:.4f})")

        # VIF检验
        print("\n  【VIF检验】")
        X_vif = df_train[[f'{var}_MIDAS' for var in macro_vars]].dropna()
        vif_data = pd.DataFrame({'变量': macro_vars, 'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(len(macro_vars))]})
        for _, row in vif_data.iterrows():
            print(f"    {row['变量']:10s}: VIF={row['VIF']:.2f}")

        # 精简多变量模型（仅保留VIF<10的变量）
        valid_vars = [var for var in macro_vars if vif_data[vif_data['变量']==var]['VIF'].values[0] < 10]
        selected_vars = valid_vars if valid_vars else [best_var]
        print(f"\n  多变量模型入选: {', '.join(selected_vars)}")

        X_multi_train = df_train[[f'{var}_MIDAS' for var in selected_vars]].values
        X_multi_test = df_test[[f'{var}_MIDAS' for var in selected_vars]].values
        X_multi_const = add_constant(X_multi_train)
        multi_model = OLS(y_train, X_multi_const).fit()

        pred_multi_train = multi_model.predict(X_multi_const)
        pred_multi_test = multi_model.predict(add_constant(X_multi_test))
        r2_multi_train = 1 - np.sum((y_train - pred_multi_train)**2) / np.sum((y_train - y_train.mean())**2)
        r2_multi_os = 1 - np.sum((y_test - pred_multi_test)**2) / np.sum((y_test - df_train[f'R_{h}d'].mean())**2)

        print(f"  多变量: R²(内)={r2_multi_train:.4f}, R²(外)={r2_multi_os:.4f}")

        # 保存预测值
        df.loc[df['is_train'], f'R_{h}d_pred'] = pred_multi_train
        df.loc[~df['is_train'], f'R_{h}d_pred'] = pred_multi_test

        midas_results[f'h{h}'] = {
            'best_var': best_var, 'univariate': univariate_results,
            'multivariate': {'r2_train': r2_multi_train, 'r2_os': r2_multi_os},
            'vif': vif_data
        }

    return df, midas_results


def construct_abnormal_returns(df):
    """构造异常收益"""
    print("\n【6】构造异常收益...")

    for h in [5, 60]:
        df[f'AR_{h}d'] = df[f'R_{h}d'] - df[f'R_{h}d_pred']
        df[f'AbsAR_{h}d'] = np.abs(df[f'AR_{h}d'])
        print(f"  h={h}日: AR均值={df[f'AR_{h}d'].mean():.6f}, AbsAR均值={df[f'AbsAR_{h}d'].mean():.6f}")

    return df


def stage2_nested_regression(df, df_train, df_test):
    """第二阶段：分组嵌套回归"""
    print("\n" + "=" * 60)
    print("【7】第二阶段：分组嵌套回归")
    print("=" * 60)

    all_stage2_vars = ['ivix_z', 'north_flow_z', 'margin_balance_z', 'amihud_z', 'intraday_range_z']
    all_stage2_vars = [v for v in all_stage2_vars if v in df.columns]

    stage2_results = {}

    for h in [5, 60]:
        print(f"\n--- 预测窗口 h={h}日 ---")

        y_train = df_train[f'AbsAR_{h}d'].values
        lags = h - 1

        # 模型I: 情绪与风险感知
        X1_vars = ['ivix_z']
        X1_vars = [v for v in X1_vars if v in df_train.columns]
        X1 = df_train[X1_vars].values
        X1_const = add_constant(X1)
        model1 = OLS(y_train, X1_const).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        print(f"\n  模型I (情绪): R²_adj={model1.rsquared_adj:.4f}")
        for i, var in enumerate(X1_vars):
            print(f"    {var}: coef={model1.params[i+1]:.6f}, t={model1.tvalues[i+1]:.2f}")

        # 模型II: 加入资金与杠杆
        X2_vars = ['ivix_z', 'north_flow_z', 'margin_balance_z']
        X2_vars = [v for v in X2_vars if v in df_train.columns]
        X2 = df_train[X2_vars].values
        X2_const = add_constant(X2)
        model2 = OLS(y_train, X2_const).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        # 联合检验
        ssr_restricted = model1.ssr
        ssr_unrestricted = model2.ssr
        q = len(X2_vars) - len(X1_vars)
        n, k = len(y_train), len(X2_vars) + 1
        f_stat = ((ssr_restricted - ssr_unrestricted) / q) / (ssr_unrestricted / (n - k))
        f_pval = 1 - stats.f.cdf(f_stat, q, n - k)

        print(f"\n  模型II (+资金): R²_adj={model2.rsquared_adj:.4f}, 联合F={f_stat:.2f}, p={f_pval:.4f}")
        for i, var in enumerate(X2_vars):
            print(f"    {var}: coef={model2.params[i+1]:.6f}, t={model2.tvalues[i+1]:.2f}")

        # 模型III: 完整模型
        X3_vars = all_stage2_vars
        X3 = df_train[X3_vars].values
        X3_const = add_constant(X3)
        model3 = OLS(y_train, X3_const).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        # 联合检验（流动性组）
        ssr_restricted2 = model2.ssr
        ssr_unrestricted2 = model3.ssr
        q2 = len(X3_vars) - len(X2_vars)
        k2 = len(X3_vars) + 1
        f_stat2 = ((ssr_restricted2 - ssr_unrestricted2) / q2) / (ssr_unrestricted2 / (n - k2))
        f_pval2 = 1 - stats.f.cdf(f_stat2, q2, n - k2)

        print(f"\n  模型III (+流动性): R²_adj={model3.rsquared_adj:.4f}, 联合F={f_stat2:.2f}, p={f_pval2:.4f}")
        for i, var in enumerate(X3_vars):
            print(f"    {var}: coef={model3.params[i+1]:.6f}, t={model3.tvalues[i+1]:.2f}")

        # LASSO筛选
        print(f"\n  【LASSO筛选】")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_train[all_stage2_vars])
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y_train)

        nonzero_mask = np.abs(lasso.coef_) > 1e-6
        lasso_selected = [all_stage2_vars[i] for i in range(len(all_stage2_vars)) if nonzero_mask[i]]
        print(f"    最优λ={lasso.alpha_:.6f}, 保留: {', '.join(lasso_selected) if lasso_selected else '无'}")

        # 综合模型
        if len(lasso_selected) > 0:
            X4 = df_train[lasso_selected].values
            X4_const = add_constant(X4)
            model4 = OLS(y_train, X4_const).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        else:
            model4 = model1
            lasso_selected = X1_vars

        print(f"\n  综合模型: R²_adj={model4.rsquared_adj:.4f}")

        stage2_results[f'h{h}'] = {
            'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4,
            'lasso_selected': lasso_selected,
            'joint_test_f1': f_stat, 'joint_test_p1': f_pval,
            'joint_test_f2': f_stat2, 'joint_test_p2': f_pval2
        }

    return stage2_results


def save_results(df, midas_results, stage2_results):
    """保存结果"""
    print("\n【8】保存实验结果...")

    # 第一阶段结果
    stage1_summary = []
    for h in [5, 60]:
        best_var = midas_results[f'h{h}']['best_var']
        uni = midas_results[f'h{h}']['univariate'][best_var]
        multi = midas_results[f'h{h}']['multivariate']
        stage1_summary.append({
            'Window': f'{h}d', 'Model': f'Univariate_{best_var}',
            'R2_InSample': uni['r2_train'], 'R2_OutSample': uni['r2_os'], 'RMSE_Test': uni['rmse_test']
        })
        stage1_summary.append({
            'Window': f'{h}d', 'Model': 'Multivariate',
            'R2_InSample': multi['r2_train'], 'R2_OutSample': multi['r2_os'], 'RMSE_Test': uni['rmse_test']
        })
    pd.DataFrame(stage1_summary).to_csv(f'{RESULT_DIR}/stage1_results.csv', index=False)
    print(f"   已保存: {RESULT_DIR}/stage1_results.csv")

    # 第二阶段结果
    stage2_summary = []
    for h in [5, 60]:
        for i, model_name in enumerate(['model1', 'model2', 'model3', 'model4']):
            model = stage2_results[f'h{h}'][model_name]
            stage2_summary.append({
                'Window': f'{h}d', 'Model': model_name,
                'Adj_R2': model.rsquared_adj,
                'LASSO_Selected': ','.join(stage2_results[f'h{h}']['lasso_selected']) if model_name == 'model4' else ''
            })
    pd.DataFrame(stage2_summary).to_csv(f'{RESULT_DIR}/stage2_results.csv', index=False)
    print(f"   已保存: {RESULT_DIR}/stage2_results.csv")

    # 完整数据
    df.to_csv(f'{RESULT_DIR}/full_data_with_predictions.csv', index=False)
    print(f"   已保存: {RESULT_DIR}/full_data_with_predictions.csv")


def main():
    """主函数"""
    print("=" * 60)
    print("阶段4：MIDAS模型两阶段实证分析")
    print("=" * 60)

    df = load_and_prepare_data()
    df, df_train, df_test = split_train_test(df)
    df = preprocess_data(df, df_train)
    df = construct_midas_weights(df)

    # 重新划分
    df['is_train'] = False
    train_size = int(len(df) * 0.6)
    df.loc[:train_size-1, 'is_train'] = True
    df_train = df[df['is_train']].copy()
    df_test = df[~df['is_train']].copy()

    df, midas_results = stage1_midas_regression(df, df_train, df_test)
    df = construct_abnormal_returns(df)
    stage2_results = stage2_nested_regression(df, df_train, df_test)
    save_results(df, midas_results, stage2_results)

    print("\n" + "=" * 60)
    print("MIDAS实验完成！")
    print(f"结果保存位置: {RESULT_DIR}")
    print("=" * 60)

    return df, midas_results, stage2_results


if __name__ == '__main__':
    main()
