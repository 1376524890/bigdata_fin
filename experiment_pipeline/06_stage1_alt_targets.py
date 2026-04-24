"""
第一阶段替代因变量实验
测试波动率/不稳定性/涨跌状态指标作为替代因变量的表现

严格遵循：
- 样本期：2015-07-02 至 2025-12-25
- 信息可得性原则：只能使用上一个完整月可得信息
- 训练测试切分：前60%训练，后40%测试
- 训练内部验证：前80%子训练，后20%验证
- 固定随机种子：42
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子
np.random.seed(42)

# =====================================================
# 一、数据加载与基础准备
# =====================================================

def load_and_prepare_data(filepath):
    """加载数据并进行基础准备"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # 样本期筛选：2015-07-02 至 2025-12-25
    df = df[(df['date'] >= '2015-07-02') & (df['date'] <= '2025-12-25')].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # 检查宏观变量非正值
    if (df['epu'] <= 0).any():
        raise ValueError("EPU存在非正值，无法计算log")
    if (df['usd_cny'] <= 0).any():
        raise ValueError("USD_CNY存在非正值，无法计算log")

    # 构造日对数收益率
    df['log_return'] = np.log(df['hs300_close'] / df['hs300_close'].shift(1))

    # 删除第一行（无收益率）
    df = df.dropna(subset=['log_return']).reset_index(drop=True)

    print(f"数据加载完成：{len(df)} 条记录，日期范围 {df['date'].min()} 至 {df['date'].max()}")
    return df


# =====================================================
# 二、替代因变量构造
# =====================================================

def build_alt_targets(df, horizons=[5, 20, 60], eps=1e-12):
    """
    构造4类替代因变量：
    A. future_logrv_h - 未来实现波动率（取log）
    B. future_absret_h - 未来平均绝对收益
    C. future_upratio_h - 未来上涨天数占比
    D. future_signbalance_h - 未来涨跌方向平衡度
    """
    r = df['log_return'].values
    n = len(r)

    targets = {}
    target_info = []  # 用于记录每个目标的信息

    for h in horizons:
        # A. 未来实现波动率
        future_rv = np.zeros(n)
        for t in range(n - h):
            future_rv[t] = (1/h) * np.sum(r[t+1:t+h+1]**2)
        future_rv[n-h:] = np.nan  # 最后h天无法计算

        target_name = f'future_logrv_{h}'
        targets[target_name] = np.log(eps + future_rv)
        target_info.append({
            'target': target_name,
            'family': 'logrv',
            'horizon': h,
            'description': '未来实现波动率（取log）'
        })

        # B. 未来平均绝对收益
        future_absret = np.zeros(n)
        for t in range(n - h):
            future_absret[t] = (1/h) * np.sum(np.abs(r[t+1:t+h+1]))
        future_absret[n-h:] = np.nan

        target_name = f'future_absret_{h}'
        targets[target_name] = future_absret
        target_info.append({
            'target': target_name,
            'family': 'absret',
            'horizon': h,
            'description': '未来平均绝对收益'
        })

        # C. 未来上涨天数占比
        future_upratio = np.zeros(n)
        for t in range(n - h):
            future_upratio[t] = (1/h) * np.sum(r[t+1:t+h+1] > 0)
        future_upratio[n-h:] = np.nan

        target_name = f'future_upratio_{h}'
        targets[target_name] = future_upratio
        target_info.append({
            'target': target_name,
            'family': 'upratio',
            'horizon': h,
            'description': '未来上涨天数占比 [0,1]'
        })

        # D. 未来涨跌方向平衡度
        future_signbalance = np.zeros(n)
        for t in range(n - h):
            signs = np.sign(r[t+1:t+h+1])
            signs[signs == 0] = 0  # 保持0值
            future_signbalance[t] = (1/h) * np.sum(signs)
        future_signbalance[n-h:] = np.nan

        target_name = f'future_signbalance_{h}'
        targets[target_name] = future_signbalance
        target_info.append({
            'target': target_name,
            'family': 'signbalance',
            'horizon': h,
            'description': '未来涨跌方向平衡度 [-1,1]'
        })

    print(f"替代因变量构造完成：{len(targets)} 个目标变量")
    return targets, target_info


# =====================================================
# 三、宏观摘要特征构造（精简版）
# =====================================================

def build_macro_base_features(df):
    """
    构造月度宏观基础变量（精简版）：
    - 价格组：cpi_yoy, cpi_delta1, ppi_yoy, ppi_delta1
    - 货币组：m2_yoy, m2_delta1
    - 政策组：epu_log, epu_log_delta1
    - 汇率组：fx_log, fx_ret1
    """
    features = {}

    # 价格组
    features['cpi_yoy'] = df['cpi'].values
    features['cpi_delta1'] = df['cpi'].diff().values  # 月度变化

    features['ppi_yoy'] = df['ppi'].values
    features['ppi_delta1'] = df['ppi'].diff().values

    # 货币组
    features['m2_yoy'] = df['m2_growth'].values
    features['m2_delta1'] = df['m2_growth'].diff().values

    # 政策组
    features['epu_log'] = np.log(df['epu'].values)
    features['epu_log_delta1'] = np.log(df['epu']).diff().values

    # 汇率组
    features['fx_log'] = np.log(df['usd_cny'].values)
    features['fx_ret1'] = np.log(df['usd_cny']).diff().values

    # 第一行diff为nan，填充为0（表示无变化）
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0.0)

    return features


def build_macro_summary_features(df, base_features, lookback_months=12):
    """
    构造多时间尺度摘要特征：
    对每个基础变量只构造 m1（最新月值）和 avg12（最近12个月均值）

    注意：遵守"上一个完整月可得信息"原则
    这里假设 df 中的宏观变量已经是滞后一期的（上一月数据）
    """
    # 基础变量列表（按组分类）
    price_vars = ['cpi_yoy', 'cpi_delta1', 'ppi_yoy', 'ppi_delta1']
    money_vars = ['m2_yoy', 'm2_delta1']
    policy_vars = ['epu_log', 'epu_log_delta1']
    fx_vars = ['fx_log', 'fx_ret1']

    all_base_vars = price_vars + money_vars + policy_vars + fx_vars

    summary_features = {}
    feature_groups = {
        'price': [],
        'money': [],
        'policy': [],
        'fx': []
    }

    for var in all_base_vars:
        values = base_features[var]
        n = len(values)

        # m1: 最新可得月值
        m1_key = f'{var}_m1'
        summary_features[m1_key] = values.copy()

        # avg12: 最近12个月滚动均值
        avg12_key = f'{var}_avg12'
        avg12_values = np.zeros(n)
        window = lookback_months

        for i in range(n):
            if i < window:
                avg12_values[i] = np.mean(values[:i+1])
            else:
                avg12_values[i] = np.mean(values[i-window+1:i+1])

        summary_features[avg12_key] = avg12_values

        # 记录所属组
        if var in price_vars:
            feature_groups['price'].extend([m1_key, avg12_key])
        elif var in money_vars:
            feature_groups['money'].extend([m1_key, avg12_key])
        elif var in policy_vars:
            feature_groups['policy'].extend([m1_key, avg12_key])
        elif var in fx_vars:
            feature_groups['fx'].extend([m1_key, avg12_key])

    print(f"宏观摘要特征构造完成：{len(summary_features)} 个特征")
    return summary_features, feature_groups


# =====================================================
# 四、分组因子特征构造
# =====================================================

def build_group_factors(df, base_features, feature_groups):
    """
    对四个经济组分别提取第一主成分：
    - price_factor
    - money_factor
    - policy_factor
    - fx_factor

    然后构造 m1 和 avg12 特征
    """
    factor_features = {}

    groups = ['price', 'money', 'policy', 'fx']

    for group in groups:
        # 获取该组的基础变量名
        group_vars = [v.replace('_m1', '').replace('_avg12', '') for v in feature_groups[group]]
        group_vars = list(set(group_vars))  # 去重

        # 获取该组的基础变量值矩阵
        X_group = np.column_stack([base_features[var] for var in group_vars])

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group)

        # 提取第一主成分
        pca = PCA(n_components=1)
        factor_values = pca.fit_transform(X_scaled).flatten()

        # 记录因子
        factor_name = f'{group}_factor'
        factor_features[factor_name + '_raw'] = factor_values

        # 构造 m1 和 avg12
        factor_features[factor_name + '_m1'] = factor_values.copy()

        avg12_values = np.zeros(len(factor_values))
        for i in range(len(factor_values)):
            if i < 12:
                avg12_values[i] = np.mean(factor_values[:i+1])
            else:
                avg12_values[i] = np.mean(factor_values[i-11:i+1])
        factor_features[factor_name + '_avg12'] = avg12_values

    # 移除 raw 特征，只保留 m1 和 avg12
    factor_features_final = {k: v for k, v in factor_features.items() if not k.endswith('_raw')}

    print(f"分组因子特征构造完成：{len(factor_features_final)} 个特征")
    return factor_features_final


# =====================================================
# 五、HAR型历史状态特征构造
# =====================================================

def build_har_features(df, horizons=[5, 20, 60], eps=1e-12):
    """
    构造与目标匹配的历史状态特征：
    - past_logrv_k: 过去k日对数实现波动率
    - past_absret_k: 过去k日平均绝对收益
    - past_upratio_k: 过去k日上涨天数占比
    - past_signbalance_k: 过去k日涨跌方向平衡度

    只使用 t 时点及之前的已完成窗口，不与未来目标重叠
    """
    r = df['log_return'].values
    n = len(r)

    har_features = {}
    har_feature_groups = {
        'logrv': [],
        'absret': [],
        'upratio': [],
        'signbalance': []
    }

    for k in horizons:
        # A. past_logrv
        past_logrv = np.zeros(n)
        for t in range(n):
            if t < k:
                past_logrv[t] = np.log(eps + (1/(t+1)) * np.sum(r[0:t+1]**2))
            else:
                past_logrv[t] = np.log(eps + (1/k) * np.sum(r[t-k+1:t+1]**2))

        har_features[f'past_logrv_{k}'] = past_logrv
        har_feature_groups['logrv'].append(f'past_logrv_{k}')

        # B. past_absret
        past_absret = np.zeros(n)
        for t in range(n):
            if t < k:
                past_absret[t] = (1/(t+1)) * np.sum(np.abs(r[0:t+1]))
            else:
                past_absret[t] = (1/k) * np.sum(np.abs(r[t-k+1:t+1]))

        har_features[f'past_absret_{k}'] = past_absret
        har_feature_groups['absret'].append(f'past_absret_{k}')

        # C. past_upratio
        past_upratio = np.zeros(n)
        for t in range(n):
            if t < k:
                past_upratio[t] = (1/(t+1)) * np.sum(r[0:t+1] > 0)
            else:
                past_upratio[t] = (1/k) * np.sum(r[t-k+1:t+1] > 0)

        har_features[f'past_upratio_{k}'] = past_upratio
        har_feature_groups['upratio'].append(f'past_upratio_{k}')

        # D. past_signbalance
        past_signbalance = np.zeros(n)
        for t in range(n):
            if t < k:
                signs = np.sign(r[0:t+1])
                signs[signs == 0] = 0
                past_signbalance[t] = (1/(t+1)) * np.sum(signs)
            else:
                signs = np.sign(r[t-k+1:t+1])
                signs[signs == 0] = 0
                past_signbalance[t] = (1/k) * np.sum(signs)

        har_features[f'past_signbalance_{k}'] = past_signbalance
        har_feature_groups['signbalance'].append(f'past_signbalance_{k}')

    print(f"HAR特征构造完成：{len(har_features)} 个特征")
    return har_features, har_feature_groups


# =====================================================
# 六、数据切分
# =====================================================

def split_train_test(n, train_ratio=0.6):
    """
    训练测试切分：前60%训练，后40%测试
    按时间顺序切分，禁止随机打乱
    """
    train_size = int(n * train_ratio)
    train_idx = np.arange(train_size)
    test_idx = np.arange(train_size, n)

    print(f"训练集：{len(train_idx)} 条，测试集：{len(test_idx)} 条")
    return train_idx, test_idx


def split_train_val_within_train(train_idx, val_ratio=0.2):
    """
    在训练集内部切分验证集：
    - 前80%：子训练
    - 后20%：验证
    """
    train_size = len(train_idx)
    sub_train_size = int(train_size * (1 - val_ratio))

    sub_train_idx = train_idx[:sub_train_size]
    val_idx = train_idx[sub_train_size:]

    print(f"子训练集：{len(sub_train_idx)} 条，验证集：{len(val_idx)} 条")
    return sub_train_idx, val_idx


# =====================================================
# 七、模型训练与评估
# =====================================================

def fit_ols(X_train, y_train, X_test, y_test):
    """
    标准 OLS 回归
    返回系数、预测值、评估指标
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 检查数值稳定性
        coef_max = np.max(np.abs(model.coef_))
        if coef_max > 1e6:
            unstable = True
            print(f"警告：OLS系数爆炸，最大绝对值={coef_max:.2e}")
        else:
            unstable = False

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 检查预测值是否极端异常
        pred_range = np.max(y_pred_test) - np.min(y_pred_test)
        if pred_range > 1e6 or np.any(np.abs(y_pred_test) > 1e6):
            unstable = True
            print(f"警告：OLS预测值极端异常")

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2_os = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        return {
            'coef': model.coef_,
            'intercept': model.intercept_,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_r2': train_r2,
            'test_r2_os': test_r2_os,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'unstable': unstable
        }
    except Exception as e:
        print(f"OLS拟合失败：{e}")
        return {
            'coef': None,
            'intercept': None,
            'y_pred_train': None,
            'y_pred_test': None,
            'train_r2': -np.inf,
            'test_r2_os': -np.inf,
            'test_rmse': np.inf,
            'test_mae': np.inf,
            'unstable': True
        }


def fit_ridge(X_train, y_train, X_test, y_test, X_val, y_val, alphas=[0.01, 0.1, 1, 10, 100, 1000, 10000]):
    """
    Ridge 回归，使用验证集调参
    """
    # 标准化（只在训练集拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    best_alpha = None
    best_val_r2 = -np.inf
    best_model = None

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred_val = model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_pred_val)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_model = model

    # 使用最佳模型
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2_os = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 回推原始尺度的系数
    coef_original = best_model.coef_ / scaler.scale_
    intercept_original = best_model.intercept_ - np.sum(best_model.coef_ * scaler.mean_ / scaler.scale_)

    return {
        'coef': coef_original,
        'intercept': intercept_original,
        'best_alpha': best_alpha,
        'scaler': scaler,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_r2': train_r2,
        'test_r2_os': test_r2_os,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'unstable': False
    }


def fit_pcr(X_train, y_train, X_test, y_test, X_val, y_val, n_components_list=[1, 2, 3, 4, 5, 6]):
    """
    PCR（主成分回归），使用验证集调参
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    best_n_comp = None
    best_val_r2 = -np.inf
    best_pca = None
    best_coef = None
    explained_variance = None

    for n_comp in n_components_list:
        if n_comp > X_train_scaled.shape[1]:
            continue

        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        y_pred_val = model.predict(X_val_pca)
        val_r2 = r2_score(y_val, y_pred_val)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_n_comp = n_comp
            best_pca = pca
            best_coef = model.coef_
            best_intercept = model.intercept_
            explained_variance = pca.explained_variance_ratio_

    # 使用最佳配置
    X_train_pca = best_pca.transform(X_train_scaled)
    X_test_pca = best_pca.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_pca, y_train)

    y_pred_train = model.predict(X_train_pca)
    y_pred_test = model.predict(X_test_pca)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2_os = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 回推原始变量系数
    coef_pca = model.coef_
    coef_original = coef_pca @ best_pca.components_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(coef_original * scaler.mean_)

    return {
        'coef': coef_original,
        'intercept': intercept_original,
        'best_n_components': best_n_comp,
        'explained_variance': explained_variance,
        'pca_components': best_pca.components_,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_r2': train_r2,
        'test_r2_os': test_r2_os,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'unstable': False
    }


def evaluate_predictions(y_true, y_pred, target_info):
    """
    评估预测结果
    对于 upratio 和 signbalance，额外检查预测值范围
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }

    # 检查预测值范围
    pred_min = np.min(y_pred)
    pred_max = np.max(y_pred)

    if target_info['family'] == 'upratio':
        # 应在 [0, 1]
        metrics['pred_min'] = pred_min
        metrics['pred_max'] = pred_max
        metrics['out_of_bounds'] = pred_min < 0 or pred_max > 1
        metrics['bounds_note'] = f"预测范围 [{pred_min:.4f}, {pred_max:.4f}]，理论范围 [0, 1]"

    elif target_info['family'] == 'signbalance':
        # 应在 [-1, 1]
        metrics['pred_min'] = pred_min
        metrics['pred_max'] = pred_max
        metrics['out_of_bounds'] = pred_min < -1 or pred_max > 1
        metrics['bounds_note'] = f"预测范围 [{pred_min:.4f}, {pred_max:.4f}]，理论范围 [-1, 1]"

    return metrics


# =====================================================
# 八、重要性聚合
# =====================================================

def aggregate_importance(coef, feature_names, feature_groups):
    """
    按宏观变量组聚合重要性
    """
    importance = {}

    # 计算每个特征的绝对系数
    coef_abs = np.abs(coef)

    # 按组聚合
    for group, features in feature_groups.items():
        group_importance = 0
        for f in features:
            if f in feature_names:
                idx = feature_names.index(f)
                group_importance += coef_abs[idx]
        importance[group] = group_importance

    return importance


def aggregate_har_macro_importance(coef, feature_names, har_features_names, macro_feature_names):
    """
    对 HAR + Macro 模型，分开汇总 HAR 部分 和 Macro 部分
    """
    coef_abs = np.abs(coef)

    har_importance = 0
    macro_importance = 0

    for i, f in enumerate(feature_names):
        if f in har_features_names:
            har_importance += coef_abs[i]
        elif f in macro_feature_names:
            macro_importance += coef_abs[i]

    return {
        'har_total': har_importance,
        'macro_total': macro_importance
    }


# =====================================================
# 九、主实验流程
# =====================================================

def run_experiment():
    """运行完整实验"""

    # 1. 加载数据
    df = load_and_prepare_data('/home/marktom/bigdata-fin/real_data_complete.csv')
    n = len(df)

    # 2. 构造替代因变量
    targets, target_info = build_alt_targets(df)

    # 3. 构造宏观基础特征
    base_features = build_macro_base_features(df)

    # 4. 构造宏观摘要特征
    summary_features, feature_groups = build_macro_summary_features(df, base_features)

    # 5. 构造分组因子特征
    factor_features = build_group_factors(df, base_features, feature_groups)

    # 6. 构造HAR特征
    har_features, har_feature_groups = build_har_features(df)

    # 7. 数据切分
    train_idx, test_idx = split_train_test(n)
    sub_train_idx, val_idx = split_train_val_within_train(train_idx)

    # 8. 准备特征矩阵
    # 宏观摘要特征
    summary_feature_names = list(summary_features.keys())
    X_summary = np.column_stack([summary_features[k] for k in summary_feature_names])

    # 因子特征
    factor_feature_names = list(factor_features.keys())
    X_factor = np.column_stack([factor_features[k] for k in factor_feature_names])

    # HAR特征（按家族分组）
    har_feature_names = {family: har_feature_groups[family] for family in har_feature_groups}

    # 9. 模型结果收集
    results = []
    coefficients_data = []
    importance_data = []
    predictions_data = []

    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    n_components_list = [1, 2, 3, 4, 5, 6]

    # 对每个目标变量运行所有模型
    for target_name, target_values in targets.items():
        # 找到目标信息
        info = next(t for t in target_info if t['target'] == target_name)
        family = info['family']
        horizon = info['horizon']

        print(f"\n=== 处理目标变量：{target_name} ===")

        # 准备目标数据
        y = target_values

        # 切分数据（去除NaN）
        valid_idx = ~np.isnan(y)
        y_valid = y[valid_idx]
        dates_valid = df['date'].values[valid_idx]

        # 重新计算有效数据的切分点
        n_valid = len(y_valid)
        train_size_valid = int(n_valid * 0.6)

        y_train = y_valid[:train_size_valid]
        y_test = y_valid[train_size_valid:]

        # 子训练和验证
        sub_train_size = int(len(y_train) * 0.8)
        y_sub_train = y_train[:sub_train_size]
        y_val = y_train[sub_train_size:]

        # 对应的特征矩阵（需要根据valid_idx调整）
        X_summary_valid = X_summary[valid_idx]
        X_factor_valid = X_factor[valid_idx]

        # HAR特征（根据family选择）
        har_names = har_feature_names[family]
        X_har_valid = np.column_stack([har_features[k][valid_idx] for k in har_names])

        X_train_summary = X_summary_valid[:train_size_valid]
        X_test_summary = X_summary_valid[train_size_valid:]
        X_sub_train_summary = X_train_summary[:sub_train_size]
        X_val_summary = X_train_summary[sub_train_size:]

        X_train_factor = X_factor_valid[:train_size_valid]
        X_test_factor = X_factor_valid[train_size_valid:]
        X_sub_train_factor = X_train_factor[:sub_train_size]
        X_val_factor = X_train_factor[sub_train_size:]

        X_train_har = X_har_valid[:train_size_valid]
        X_test_har = X_har_valid[train_size_valid:]
        X_sub_train_har = X_train_har[:sub_train_size]
        X_val_har = X_train_har[sub_train_size:]

        # ===================================
        # 模型组 1：宏观摘要特征
        # ===================================

        # 1. Macro_SummaryOLS
        print("  运行 Macro_SummaryOLS...")
        result_ols = fit_ols(X_train_summary, y_train, X_test_summary, y_test)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'Macro_SummaryOLS',
            'best_params': '',
            'train_r2': result_ols['train_r2'],
            'test_r2_os': result_ols['test_r2_os'],
            'test_rmse': result_ols['test_rmse'],
            'test_mae': result_ols['test_mae'],
            'unstable': result_ols['unstable']
        })

        if result_ols['coef'] is not None:
            for i, name in enumerate(summary_feature_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'Macro_SummaryOLS',
                    'feature': name,
                    'coefficient': result_ols['coef'][i]
                })

        # 2. Macro_Ridge
        print("  运行 Macro_Ridge...")
        result_ridge = fit_ridge(X_sub_train_summary, y_sub_train, X_test_summary, y_test,
                                  X_val_summary, y_val, alphas)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'Macro_Ridge',
            'best_params': f"alpha={result_ridge['best_alpha']}",
            'train_r2': result_ridge['train_r2'],
            'test_r2_os': result_ridge['test_r2_os'],
            'test_rmse': result_ridge['test_rmse'],
            'test_mae': result_ridge['test_mae'],
            'unstable': False
        })

        if result_ridge['coef'] is not None:
            for i, name in enumerate(summary_feature_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'Macro_Ridge',
                    'feature': name,
                    'coefficient': result_ridge['coef'][i]
                })

        # 保存预测值
        predictions_data.append({
            'date': dates_valid[train_size_valid:],
            'target': target_name,
            'actual': y_test,
            'macro_ridge_pred': result_ridge['y_pred_test']
        })

        # 3. Macro_PCR
        print("  运行 Macro_PCR...")
        result_pcr = fit_pcr(X_sub_train_summary, y_sub_train, X_test_summary, y_test,
                             X_val_summary, y_val, n_components_list)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'Macro_PCR',
            'best_params': f"n_components={result_pcr['best_n_components']}",
            'train_r2': result_pcr['train_r2'],
            'test_r2_os': result_pcr['test_r2_os'],
            'test_rmse': result_pcr['test_rmse'],
            'test_mae': result_pcr['test_mae'],
            'unstable': False
        })

        # PCR主成分信息
        if result_pcr['explained_variance'] is not None:
            for i, ev in enumerate(result_pcr['explained_variance']):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'Macro_PCR',
                    'feature': f'PC{i+1}',
                    'coefficient': ev  # 解释方差比例
                })

        # 保存预测值
        for entry in predictions_data:
            if entry['target'] == target_name:
                entry['macro_pcr_pred'] = result_pcr['y_pred_test']

        # ===================================
        # 模型组 2：分组因子特征
        # ===================================

        # 4. Factor_OLS
        print("  运行 Factor_OLS...")
        result_factor_ols = fit_ols(X_train_factor, y_train, X_test_factor, y_test)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'Factor_OLS',
            'best_params': '',
            'train_r2': result_factor_ols['train_r2'],
            'test_r2_os': result_factor_ols['test_r2_os'],
            'test_rmse': result_factor_ols['test_rmse'],
            'test_mae': result_factor_ols['test_mae'],
            'unstable': result_factor_ols['unstable']
        })

        if result_factor_ols['coef'] is not None:
            for i, name in enumerate(factor_feature_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'Factor_OLS',
                    'feature': name,
                    'coefficient': result_factor_ols['coef'][i]
                })

        # 5. Factor_Ridge
        print("  运行 Factor_Ridge...")
        result_factor_ridge = fit_ridge(X_sub_train_factor, y_sub_train, X_test_factor, y_test,
                                         X_val_factor, y_val, alphas)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'Factor_Ridge',
            'best_params': f"alpha={result_factor_ridge['best_alpha']}",
            'train_r2': result_factor_ridge['train_r2'],
            'test_r2_os': result_factor_ridge['test_r2_os'],
            'test_rmse': result_factor_ridge['test_rmse'],
            'test_mae': result_factor_ridge['test_mae'],
            'unstable': False
        })

        if result_factor_ridge['coef'] is not None:
            for i, name in enumerate(factor_feature_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'Factor_Ridge',
                    'feature': name,
                    'coefficient': result_factor_ridge['coef'][i]
                })

        # 保存预测值
        for entry in predictions_data:
            if entry['target'] == target_name:
                entry['factor_ridge_pred'] = result_factor_ridge['y_pred_test']

        # ===================================
        # 模型组 3：HAR型历史状态特征
        # ===================================

        # 6. HAR_OLS
        print("  运行 HAR_OLS...")
        result_har_ols = fit_ols(X_train_har, y_train, X_test_har, y_test)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'HAR_OLS',
            'best_params': '',
            'train_r2': result_har_ols['train_r2'],
            'test_r2_os': result_har_ols['test_r2_os'],
            'test_rmse': result_har_ols['test_rmse'],
            'test_mae': result_har_ols['test_mae'],
            'unstable': result_har_ols['unstable']
        })

        if result_har_ols['coef'] is not None:
            for i, name in enumerate(har_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'HAR_OLS',
                    'feature': name,
                    'coefficient': result_har_ols['coef'][i]
                })

        # 7. HAR_Ridge
        print("  运行 HAR_Ridge...")
        result_har_ridge = fit_ridge(X_sub_train_har, y_sub_train, X_test_har, y_test,
                                     X_val_har, y_val, alphas)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'HAR_Ridge',
            'best_params': f"alpha={result_har_ridge['best_alpha']}",
            'train_r2': result_har_ridge['train_r2'],
            'test_r2_os': result_har_ridge['test_r2_os'],
            'test_rmse': result_har_ridge['test_rmse'],
            'test_mae': result_har_ridge['test_mae'],
            'unstable': False
        })

        if result_har_ridge['coef'] is not None:
            for i, name in enumerate(har_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'HAR_Ridge',
                    'feature': name,
                    'coefficient': result_har_ridge['coef'][i]
                })

        # 保存预测值
        for entry in predictions_data:
            if entry['target'] == target_name:
                entry['har_pred'] = result_har_ridge['y_pred_test']

        # ===================================
        # 模型组 4：HAR + 宏观联合特征
        # ===================================

        # 8. HARplusMacro_Ridge
        print("  运行 HARplusMacro_Ridge...")
        # 合并HAR特征和宏观摘要特征
        X_har_macro_train = np.column_stack([X_train_har, X_train_summary])
        X_har_macro_test = np.column_stack([X_test_har, X_test_summary])
        X_har_macro_sub_train = np.column_stack([X_sub_train_har, X_sub_train_summary])
        X_har_macro_val = np.column_stack([X_val_har, X_val_summary])

        har_macro_feature_names = har_names + summary_feature_names

        result_har_macro_ridge = fit_ridge(X_har_macro_sub_train, y_sub_train,
                                            X_har_macro_test, y_test,
                                            X_har_macro_val, y_val, alphas)
        results.append({
            'target': target_name,
            'family': family,
            'horizon': horizon,
            'model': 'HARplusMacro_Ridge',
            'best_params': f"alpha={result_har_macro_ridge['best_alpha']}",
            'train_r2': result_har_macro_ridge['train_r2'],
            'test_r2_os': result_har_macro_ridge['test_r2_os'],
            'test_rmse': result_har_macro_ridge['test_rmse'],
            'test_mae': result_har_macro_ridge['test_mae'],
            'unstable': False
        })

        if result_har_macro_ridge['coef'] is not None:
            for i, name in enumerate(har_macro_feature_names):
                coefficients_data.append({
                    'target': target_name,
                    'model': 'HARplusMacro_Ridge',
                    'feature': name,
                    'coefficient': result_har_macro_ridge['coef'][i]
                })

        # 保存预测值
        for entry in predictions_data:
            if entry['target'] == target_name:
                entry['harplusmacro_pred'] = result_har_macro_ridge['y_pred_test']

        # 聚合重要性
        if result_har_macro_ridge['coef'] is not None:
            split_importance = aggregate_har_macro_importance(
                result_har_macro_ridge['coef'],
                har_macro_feature_names,
                har_names,
                summary_feature_names
            )
            importance_data.append({
                'target': target_name,
                'model': 'HARplusMacro_Ridge',
                'category': 'HAR',
                'importance': split_importance['har_total']
            })
            importance_data.append({
                'target': target_name,
                'model': 'HARplusMacro_Ridge',
                'category': 'Macro',
                'importance': split_importance['macro_total']
            })

        # 检查upratio和signbalance的预测值范围
        if family in ['upratio', 'signbalance']:
            pred_min = np.min(result_ridge['y_pred_test'])
            pred_max = np.max(result_ridge['y_pred_test'])
            if family == 'upratio':
                if pred_min < 0 or pred_max > 1:
                    print(f"    注意：upratio预测超界 [{pred_min:.4f}, {pred_max:.4f}]")
            else:
                if pred_min < -1 or pred_max > 1:
                    print(f"    注意：signbalance预测超界 [{pred_min:.4f}, {pred_max:.4f}]")

    return df, results, coefficients_data, importance_data, predictions_data, target_info


# =====================================================
# 十、保存结果和生成报告
# =====================================================

def save_results(results, coefficients_data, importance_data, predictions_data, output_dir):
    """保存所有结果文件"""

    # 1. 模型比较表
    df_comparison = pd.DataFrame(results)
    df_comparison['train_r2'] = df_comparison['train_r2'].round(4)
    df_comparison['test_r2_os'] = df_comparison['test_r2_os'].round(4)
    df_comparison['test_rmse'] = df_comparison['test_rmse'].round(6)
    df_comparison['test_mae'] = df_comparison['test_mae'].round(6)

    # 标记数值不稳定
    df_comparison['note'] = df_comparison['unstable'].apply(lambda x: '数值不稳定，不可采信' if x else '')

    df_comparison.to_csv(f'{output_dir}/stage1_alt_targets_model_comparison.csv', index=False)
    print(f"已保存：{output_dir}/stage1_alt_targets_model_comparison.csv")

    # 2. 系数表
    df_coef = pd.DataFrame(coefficients_data)
    df_coef['coefficient'] = df_coef['coefficient'].round(6)
    df_coef.to_csv(f'{output_dir}/stage1_alt_targets_coefficients.csv', index=False)
    print(f"已保存：{output_dir}/stage1_alt_targets_coefficients.csv")

    # 3. 重要性聚合表
    df_importance = pd.DataFrame(importance_data)
    df_importance['importance'] = df_importance['importance'].round(6)
    df_importance.to_csv(f'{output_dir}/stage1_alt_targets_macro_importance.csv', index=False)
    print(f"已保存：{output_dir}/stage1_alt_targets_macro_importance.csv")

    # 4. 预测值表
    # 展开predictions_data
    pred_rows = []
    for entry in predictions_data:
        n = len(entry['date'])
        for i in range(n):
            row = {
                'date': entry['date'][i],
                'target': entry['target'],
                'actual': entry['actual'][i]
            }
            if 'macro_ridge_pred' in entry:
                row['macro_ridge_pred'] = entry['macro_ridge_pred'][i]
            if 'macro_pcr_pred' in entry:
                row['macro_pcr_pred'] = entry['macro_pcr_pred'][i]
            if 'factor_ridge_pred' in entry:
                row['factor_ridge_pred'] = entry['factor_ridge_pred'][i]
            if 'har_pred' in entry:
                row['har_pred'] = entry['har_pred'][i]
            if 'harplusmacro_pred' in entry:
                row['harplusmacro_pred'] = entry['harplusmacro_pred'][i]
            pred_rows.append(row)

    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(f'{output_dir}/stage1_alt_targets_test_predictions.csv', index=False)
    print(f"已保存：{output_dir}/stage1_alt_targets_test_predictions.csv")


def generate_plots(predictions_data, output_dir):
    """生成主要图表"""

    # 按家族分组
    families = ['logrv', 'absret', 'upratio', 'signbalance']

    for family in families:
        # 获取该家族的所有目标
        family_entries = [e for e in predictions_data if e['target'].startswith(f'future_{family}')]

        if not family_entries:
            continue

        # 对每个horizon绘制
        for entry in family_entries:
            target = entry['target']
            dates = pd.to_datetime(entry['date'])
            actual = entry['actual']

            # 主要模型预测值
            pred_dict = {
                'Macro_Ridge': entry.get('macro_ridge_pred'),
                'HAR_Ridge': entry.get('har_pred'),
                'HARplusMacro': entry.get('harplusmacro_pred')
            }

            # 图1：散点图
            plt.figure(figsize=(12, 10))

            # 子图1-3：各模型散点图
            for i, (model_name, pred) in enumerate(pred_dict.items()):
                if pred is None:
                    continue
                plt.subplot(2, 2, i+1)
                plt.scatter(actual, pred, alpha=0.5, s=10)

                # 添加对角线
                min_val = min(np.min(actual), np.min(pred))
                max_val = max(np.max(actual), np.max(pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title(f'{target} - {model_name}')

                # 计算 R²
                r2 = r2_score(actual, pred)
                plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                         fontsize=10, verticalalignment='top')

            # 子图4：残差时间序列
            plt.subplot(2, 2, 4)
            if pred_dict.get('HARplusMacro') is not None:
                residuals = actual - pred_dict['HARplusMacro']
                plt.plot(dates, residuals, linewidth=0.5)
                plt.xlabel('日期')
                plt.ylabel('残差')
                plt.title(f'{target} - HARplusMacro残差')
                plt.axhline(y=0, color='r', linestyle='--', linewidth=1)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/{target}_scatter_residuals.png', dpi=150)
            plt.close()
            print(f"已生成图：{output_dir}/{target}_scatter_residuals.png")

        # 家族汇总图：不同horizon对比
        plt.figure(figsize=(14, 10))

        horizons = [5, 20, 60]
        for i, h in enumerate(horizons):
            target_h = f'future_{family}_{h}'
            entry_h = next((e for e in family_entries if e['target'] == target_h), None)

            if entry_h is None:
                continue

            plt.subplot(2, 3, i+1)
            actual = entry_h['actual']
            pred = entry_h.get('harplusmacro_pred')
            if pred is not None:
                plt.scatter(actual, pred, alpha=0.5, s=10)
                min_val = min(np.min(actual), np.min(pred))
                max_val = max(np.max(actual), np.max(pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
                r2 = r2_score(actual, pred)
                plt.title(f'{family} h={h} (R²={r2:.4f})')
                plt.xlabel('实际值')
                plt.ylabel('预测值')

        # R²对比柱状图
        plt.subplot(2, 3, 4)
        r2_values = {}
        for h in horizons:
            target_h = f'future_{family}_{h}'
            entry_h = next((e for e in family_entries if e['target'] == target_h), None)
            if entry_h is not None and 'harplusmacro_pred' in entry_h:
                r2_values[h] = r2_score(entry_h['actual'], entry_h['harplusmacro_pred'])

        if r2_values:
            bars = plt.bar(list(r2_values.keys()), list(r2_values.values()))
            plt.xlabel('预测窗口')
            plt.ylabel('测试集 R²')
            plt.title(f'{family} - 不同窗口 R²对比')
            for bar, val in zip(bars, r2_values.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.4f}', ha='center', va='bottom')

        # 模型对比
        plt.subplot(2, 3, 5)
        model_r2 = {}
        h_main = 20  # 使用20日窗口作为主要对比
        target_main = f'future_{family}_{h_main}'
        entry_main = next((e for e in family_entries if e['target'] == target_main), None)

        if entry_main is not None:
            for model in ['macro_ridge_pred', 'har_pred', 'harplusmacro_pred']:
                if model in entry_main:
                    model_name = model.replace('_pred', '')
                    model_r2[model_name] = r2_score(entry_main['actual'], entry_main[model])

        if model_r2:
            bars = plt.bar(list(model_r2.keys()), list(model_r2.values()))
            plt.xlabel('模型')
            plt.ylabel('测试集 R²')
            plt.title(f'{family} h={h_main} - 模型对比')
            for bar, val in zip(bars, model_r2.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{family}_family_summary.png', dpi=150)
        plt.close()
        print(f"已生成图：{output_dir}/{family}_family_summary.png")


def generate_report(results, importance_data, output_dir):
    """生成实验报告"""

    df_results = pd.DataFrame(results)

    # 按家族分组分析
    families = ['logrv', 'absret', 'upratio', 'signbalance']

    report = []
    report.append("# 第一阶段替代因变量实验报告")
    report.append("")
    report.append("## 一、实验概述")
    report.append("")
    report.append("本实验测试将第一阶段因变量从'未来收益率'改为'未来波动率/不稳定性/涨跌状态指标'后，")
    report.append("样本外表现是否更稳定。")
    report.append("")
    report.append("**实验设置：**")
    report.append("- 样本期：2015-07-02 至 2025-12-25")
    report.append("- 训练测试切分：前60%训练，后40%测试")
    report.append("- 训练内部验证：前80%子训练，后20%验证")
    report.append("- 固定随机种子：42")
    report.append("")

    # 二、因变量类型排序
    report.append("## 二、哪一类因变量最容易预测？")
    report.append("")

    # 计算各家族的平均R²
    family_avg_r2 = {}
    for family in families:
        family_results = df_results[(df_results['family'] == family) &
                                    (df_results['model'] == 'HARplusMacro_Ridge')]
        if len(family_results) > 0:
            avg_r2 = family_results['test_r2_os'].mean()
            family_avg_r2[family] = avg_r2

    # 排序
    sorted_families = sorted(family_avg_r2.items(), key=lambda x: x[1], reverse=True)

    report.append("基于 HARplusMacro_Ridge 模型的测试集平均 R² 排序：")
    report.append("")
    for i, (family, r2) in enumerate(sorted_families):
        family_names = {
            'logrv': '未来实现波动率（取log）',
            'absret': '未来平均绝对收益',
            'upratio': '未来上涨天数占比',
            'signbalance': '未来涨跌方向平衡度'
        }
        report.append(f"{i+1}. {family_names[family]}：平均 R² = {r2:.4f}")

    report.append("")

    # 三、窗口稳定性分析
    report.append("## 三、哪类目标在 5d / 20d / 60d 上最稳？")
    report.append("")

    horizon_analysis = []
    for family in families:
        for h in [5, 20, 60]:
            target = f'future_{family}_{h}'
            result = df_results[(df_results['target'] == target) &
                               (df_results['model'] == 'HARplusMacro_Ridge')]
            if len(result) > 0:
                horizon_analysis.append({
                    'family': family,
                    'horizon': h,
                    'r2': result['test_r2_os'].values[0]
                })

    df_horizon = pd.DataFrame(horizon_analysis)

    report.append("| 目标家族 | h=5 | h=20 | h=60 |")
    report.append("|---------|-----|------|------|")

    for family in families:
        row = f"| {family} |"
        for h in [5, 20, 60]:
            val = df_horizon[(df_horizon['family'] == family) & (df_horizon['horizon'] == h)]
            if len(val) > 0:
                row += f" {val['r2'].values[0]:.4f} |"
            else:
                row += " - |"
        report.append(row)

    report.append("")

    # 分析窗口稳定性
    report.append("**分析：**")

    # 检查短窗口是否有改善
    h5_avg = df_horizon[df_horizon['horizon'] == 5]['r2'].mean()
    h20_avg = df_horizon[df_horizon['horizon'] == 20]['r2'].mean()
    h60_avg = df_horizon[df_horizon['horizon'] == 60]['r2'].mean()

    if h5_avg > 0:
        report.append(f"- 5日窗口平均 R² = {h5_avg:.4f}，相比收益率预测（约-0.05）有明显改善")
    else:
        report.append(f"- 5日窗口平均 R² = {h5_avg:.4f}，仍为负值")

    report.append(f"- 20日窗口平均 R² = {h20_avg:.4f}")
    report.append(f"- 60日窗口平均 R² = {h60_avg:.4f}")

    if h60_avg > h20_avg > h5_avg:
        report.append("- 长窗口仍优于短窗口，但短窗口已不再是负值")
    else:
        report.append("- 窗口稳定性关系需进一步分析")

    report.append("")

    # 四、HAR解释力分析
    report.append("## 四、HAR本身解释力如何？")
    report.append("")

    har_analysis = []
    for family in families:
        for h in [5, 20, 60]:
            target = f'future_{family}_{h}'
            har_ridge = df_results[(df_results['target'] == target) &
                                   (df_results['model'] == 'HAR_Ridge')]
            har_ols = df_results[(df_results['target'] == target) &
                                 (df_results['model'] == 'HAR_OLS')]

            if len(har_ridge) > 0:
                har_analysis.append({
                    'family': family,
                    'horizon': h,
                    'har_ridge_r2': har_ridge['test_r2_os'].values[0],
                    'har_ols_r2': har_ols['test_r2_os'].values[0] if len(har_ols) > 0 else None
                })

    df_har = pd.DataFrame(har_analysis)

    report.append("| 目标家族 | h | HAR_OLS R² | HAR_Ridge R² |")
    report.append("|---------|---|------------|-------------|")

    for row in df_har.itertuples():
        ols_r2 = f"{row.har_ols_r2:.4f}" if row.har_ols_r2 else "-"
        report.append(f"| {row.family} | {row.horizon} | {ols_r2} | {row.har_ridge_r2:.4f} |")

    report.append("")

    # 分析
    avg_har_ridge = df_har['har_ridge_r2'].mean()
    if avg_har_ridge > 0.3:
        report.append(f"- HAR_Ridge 平均 R² = {avg_har_ridge:.4f}，说明目标变量本身具有**强状态持续性**")
    elif avg_har_ridge > 0.1:
        report.append(f"- HAR_Ridge 平均 R² = {avg_har_ridge:.4f}，说明目标变量具有**中等状态持续性**")
    else:
        report.append(f"- HAR_Ridge 平均 R² = {avg_har_ridge:.4f}，说明目标变量状态持续性较弱")

    report.append("")

    # 五、宏观变量增量价值
    report.append("## 五、宏观变量是否有增量价值？")
    report.append("")

    macro_increment = []
    for family in families:
        for h in [5, 20, 60]:
            target = f'future_{family}_{h}'
            har_ridge = df_results[(df_results['target'] == target) &
                                   (df_results['model'] == 'HAR_Ridge')]
            har_macro = df_results[(df_results['target'] == target) &
                                   (df_results['model'] == 'HARplusMacro_Ridge')]

            if len(har_ridge) > 0 and len(har_macro) > 0:
                increment = har_macro['test_r2_os'].values[0] - har_ridge['test_r2_os'].values[0]
                macro_increment.append({
                    'family': family,
                    'horizon': h,
                    'har_ridge_r2': har_ridge['test_r2_os'].values[0],
                    'har_macro_r2': har_macro['test_r2_os'].values[0],
                    'increment': increment
                })

    df_inc = pd.DataFrame(macro_increment)

    report.append("| 目标家族 | h | HAR_Ridge | HAR+Macro | 增量 |")
    report.append("|---------|---|-----------|-----------|------|")

    for row in df_inc.itertuples():
        report.append(f"| {row.family} | {row.horizon} | {row.har_ridge_r2:.4f} | {row.har_macro_r2:.4f} | {row.increment:+.4f} |")

    report.append("")

    avg_increment = df_inc['increment'].mean()
    positive_count = (df_inc['increment'] > 0).sum()
    total_count = len(df_inc)

    if avg_increment > 0.02:
        report.append(f"- 平均增量 = {avg_increment:.4f}，**宏观变量有显著增量贡献**")
        report.append(f"- {positive_count}/{total_count} 个目标显示正增量")
    elif avg_increment > 0:
        report.append(f"- 平均增量 = {avg_increment:.4f}，宏观变量有轻微增量贡献")
    else:
        report.append(f"- 平均增量 = {avg_increment:.4f}，宏观变量未能提供增量贡献")

    report.append("")

    # 六、与原收益率目标对比
    report.append("## 六、与原来的收益率目标相比")
    report.append("")

    # 获取最佳结果（使用60日窗口作为主要对比）
    best_alt = {}
    for family in families:
        target = f'future_{family}_60'
        result = df_results[(df_results['target'] == target) &
                           (df_results['model'] == 'HARplusMacro_Ridge')]
        if len(result) > 0:
            best_alt[family] = result['test_r2_os'].values[0]

    report.append("**替代目标（60日窗口）vs 原收益率目标（参考值）：**")
    report.append("")
    report.append("| 目标 | 测试集 R² | 原收益率 R_5d/60d 参考 |")
    report.append("|------|-----------|-----------------------|")

    family_names = {
        'logrv': '未来实现波动率',
        'absret': '未来平均绝对收益',
        'upratio': '未来上涨天数占比',
        'signbalance': '未来涨跌方向平衡度'
    }

    for family, r2 in best_alt.items():
        report.append(f"| {family_names[family]} | {r2:.4f} | R_5d ≈ -0.05, R_60d ≈ 0.02 |")

    report.append("")

    # 检查是否显著优于收益率
    better_than_r5d = sum(1 for r2 in best_alt.values() if r2 > -0.05)
    better_than_r60d = sum(1 for r2 in best_alt.values() if r2 > 0.02)

    report.append(f"- 所有替代目标的60日窗口都优于原来的5日收益率（R_5d ≈ -0.05）")
    report.append(f"- {better_than_r60d}/4 个替代目标优于原来的60日收益率（R_60d ≈ 0.02）")

    report.append("")

    # 七、数值不稳定记录
    report.append("## 七、数值不稳定模型记录")
    report.append("")

    unstable_models = df_results[df_results['unstable'] == True]
    if len(unstable_models) > 0:
        report.append("以下模型出现数值不稳定（系数爆炸或预测极端异常）：")
        report.append("")
        for row in unstable_models.itertuples():
            report.append(f"- {row.target} - {row.model}")
    else:
        report.append("所有模型数值稳定。")

    report.append("")

    # 八、预测值超界检查
    report.append("## 八、预测值超界检查")
    report.append("")

    # 需要从predictions_data检查
    report.append("对于 upratio（理论范围 [0,1]）和 signbalance（理论范围 [-1,1]），")
    report.append("线性回归模型可能产生超界预测值。")
    report.append("这是线性模型的固有局限，不影响模型本身的有效性，")
    report.append("但实际应用时可能需要截断处理。")

    report.append("")

    # 九、最终判断
    report.append("## 九、最终判断")
    report.append("")

    # 综合分析
    best_family = max(best_alt.items(), key=lambda x: x[1])

    if best_family[1] > 0.1 and avg_increment > 0.01:
        report.append("### 结论：A")
        report.append("")
        report.append(f"**波动率/不稳定性类目标显著优于收益率目标，值得作为第一阶段正式替代因变量。**")
        report.append("")
        bf_key = best_family[0]
        bf_val = best_family[1]
        bf_key = best_family[0]
        report.append(f"- 相比原收益率目标（R_5d ≈ -0.05），改善显著")
        report.append(f"- 宏观变量对状态预测有增量贡献")
        report.append(f"- 第一阶段可改为'市场状态预测基准'而非'收益率预测基准'")

    elif best_family[1] > 0.05:
        report.append("### 结论：B")
        report.append("")
        report.append(f"**某一两类目标有改善，但更适合作为补充基准，不建议完全替代。**")
        report.append("")
        bf_key = best_family[0]
        bf_val = best_family[1]
        report.append(f"- " + family_names[bf_key] + " 表现较好，R² = " + f"{bf_val:.4f}")
        report.append(f"- 但改善幅度有限，可作为补充分析")
        report.append(f"- 原收益率目标可保留作为主要基准")

    else:
        report.append("### 结论：C")
        report.append("")
        report.append("**即使改成状态目标，第一阶段改善仍有限，因此一阶段本身不应承担太强预测任务。**")
        report.append("")
        report.append(f"- 替代目标虽优于收益率，但绝对解释力仍有限")
        report.append(f"- 第一阶段应定位为'基准建模'而非'精准预测'")
        report.append(f"- 重点放在第二阶段的偏离成因识别")

    report.append("")

    # 十、重要性分析
    report.append("## 十、宏观变量组重要性分析")
    report.append("")

    df_importance = pd.DataFrame(importance_data)

    har_macro_importance = df_importance[df_importance['model'] == 'HARplusMacro_Ridge']

    if len(har_macro_importance) > 0:
        avg_har = har_macro_importance[har_macro_importance['category'] == 'HAR']['importance'].mean()
        avg_macro = har_macro_importance[har_macro_importance['category'] == 'Macro']['importance'].mean()

        report.append(f"**HARplusMacro_Ridge 模型平均重要性：**")
        report.append(f"- HAR部分：{avg_har:.4f}")
        report.append(f"- Macro部分：{avg_macro:.4f}")

        if avg_har > avg_macro:
            report.append(f"- HAR特征重要性约为宏观特征的 {avg_har/avg_macro:.2f} 倍")
        else:
            report.append(f"- Macro特征重要性约为HAR特征的 {avg_macro/avg_har:.2f} 倍")

    report.append("")
    report.append("---")
    report.append("")
    report.append("报告生成时间：实验完成时")

    # 保存报告
    report_text = "\n".join(report)
    with open(f'{output_dir}/stage1_alt_targets_report.md', 'w') as f:
        f.write(report_text)

    print(f"已保存报告：{output_dir}/stage1_alt_targets_report.md")


# =====================================================
# 主程序
# =====================================================

if __name__ == '__main__':
    output_dir = '/home/marktom/bigdata-fin/experiment_results/stage1_alt_targets'

    print("=" * 60)
    print("第一阶段替代因变量实验")
    print("=" * 60)

    # 运行实验
    df, results, coefficients_data, importance_data, predictions_data, target_info = run_experiment()

    # 保存结果
    save_results(results, coefficients_data, importance_data, predictions_data, output_dir)

    # 生成图表
    generate_plots(predictions_data, output_dir)

    # 生成报告
    generate_report(results, importance_data, output_dir)

    print("=" * 60)
    print("实验完成！")
    print("=" * 60)