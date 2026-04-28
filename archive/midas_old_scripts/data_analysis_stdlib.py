#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据描述统计与分析脚本 (纯Python标准库版本)
"""

import csv
import math
from datetime import datetime
from collections import defaultdict

def read_csv(filepath):
    """读取CSV文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def to_float(value):
    """转换为浮点数"""
    try:
        if value is None or value == '' or value == 'NA':
            return None
        return float(value)
    except:
        return None

def calculate_stats(values):
    """计算描述统计量"""
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return None

    n = len(clean_values)
    mean = sum(clean_values) / n

    # 标准差
    variance = sum((x - mean) ** 2 for x in clean_values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)

    # 排序计算中位数和分位数
    sorted_vals = sorted(clean_values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    p25 = sorted_vals[int(n * 0.25)]
    p75 = sorted_vals[int(n * 0.75)]
    p1 = sorted_vals[int(n * 0.01)] if n > 100 else sorted_vals[0]
    p99 = sorted_vals[int(n * 0.99)] if n > 100 else sorted_vals[-1]

    # 偏度 (简化计算)
    if n > 2 and std > 0:
        skewness = sum((x - mean) ** 3 for x in clean_values) / (n * std ** 3)
    else:
        skewness = 0

    # 峰度 (简化计算)
    if n > 3 and std > 0:
        kurtosis = sum((x - mean) ** 4 for x in clean_values) / (n * std ** 4) - 3
    else:
        kurtosis = 0

    return {
        'count': n,
        'mean': mean,
        'std': std,
        'min': min_val,
        'p25': p25,
        'median': median,
        'p75': p75,
        'max': max_val,
        'p1': p1,
        'p99': p99,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def calculate_correlation(x, y):
    """计算相关系数"""
    clean_pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(clean_pairs) < 2:
        return None

    n = len(clean_pairs)
    mean_x = sum(p[0] for p in clean_pairs) / n
    mean_y = sum(p[1] for p in clean_pairs) / n

    var_x = sum((p[0] - mean_x) ** 2 for p in clean_pairs) / (n - 1)
    var_y = sum((p[1] - mean_y) ** 2 for p in clean_pairs) / (n - 1)

    if var_x == 0 or var_y == 0:
        return 0

    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in clean_pairs) / (n - 1)

    return cov / math.sqrt(var_x * var_y)

# 读取数据
print("读取数据文件...")
data = read_csv('/home/marktom/bigdata-fin/real_data_complete.csv')
print(f"总观测数: {len(data)}")

# 解析日期
for row in data:
    row['date_parsed'] = datetime.strptime(row['date'], '%Y-%m-%d')

# 排序
data.sort(key=lambda x: x['date_parsed'])

# 提取数值列
print("\n解析数值变量...")

# 定义变量映射
vars_mapping = {
    'hs300_close': '沪深300收盘价',
    'market_return': '日收益率',
    'ivix': '隐含波动率指数',
    'north_flow': '北向资金净流入',
    'margin_balance': '融资融券余额',
    'usd_cny': '美元兑人民币汇率',
    'gdp_growth': 'GDP同比增速',
    'cpi': 'CPI同比增速',
    'ppi': 'PPI同比增速',
    'm2_growth': 'M2同比增速',
    'epu': '经济政策不确定性指数',
    'volatility_20d': '20日波动率',
    'volatility_60d': '60日波动率',
    'momentum_20d': '20日动量',
    'intraday_range': '日内振幅',
    'sentiment_zscore': '情绪标准分',
    'amihud': 'Amihud非流动性指标'
}

# 提取数值
numeric_data = {}
for var in vars_mapping.keys():
    if var in data[0]:
        numeric_data[var] = [to_float(row[var]) for row in data]
    else:
        print(f"  警告: {var} 不在数据中")

# 计算日对数收益率
print("\n计算日对数收益率...")
closes = numeric_data.get('hs300_close', [])
daily_returns = []
for i in range(len(closes)):
    if i == 0 or closes[i] is None or closes[i-1] is None:
        daily_returns.append(None)
    else:
        daily_returns.append(math.log(closes[i] / closes[i-1]))
numeric_data['daily_return'] = daily_returns

# 计算未来5日累计收益率
print("计算未来5日累计收益率...")
future_5d = []
for i in range(len(daily_returns)):
    if i + 5 >= len(daily_returns):
        future_5d.append(None)
    else:
        # 计算未来5日的累计对数收益率
        cum_ret = 0
        valid = True
        for j in range(1, 6):
            if daily_returns[i + j] is None:
                valid = False
                break
            cum_ret += daily_returns[i + j]
        if valid:
            future_5d.append(cum_ret)
        else:
            future_5d.append(None)
numeric_data['future_return_5d'] = future_5d

# 计算未来60日累计收益率
print("计算未来60日累计收益率...")
future_60d = []
for i in range(len(daily_returns)):
    if i + 60 >= len(daily_returns):
        future_60d.append(None)
    else:
        cum_ret = 0
        valid = True
        for j in range(1, 61):
            if daily_returns[i + j] is None:
                valid = False
                break
            cum_ret += daily_returns[i + j]
        if valid:
            future_60d.append(cum_ret)
        else:
            future_60d.append(None)
numeric_data['future_return_60d'] = future_60d

# 更新变量映射
vars_mapping['daily_return'] = '日对数收益率'
vars_mapping['future_return_5d'] = '未来5日累计收益率'
vars_mapping['future_return_60d'] = '未来60日累计收益率'

# 筛选有效样本
print("\n筛选有效样本...")
valid_indices = []
for i in range(len(data)):
    if (numeric_data.get('future_return_5d', [None]*len(data))[i] is not None and
        numeric_data.get('future_return_60d', [None]*len(data))[i] is not None and
        numeric_data.get('daily_return', [None]*len(data))[i] is not None):
        valid_indices.append(i)

print(f"有效样本数: {len(valid_indices)}")

# 获取样本时间范围
valid_dates = [data[i]['date_parsed'] for i in valid_indices]
start_date = min(valid_dates)
end_date = max(valid_dates)
print(f"样本时间区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
print(f"时间跨度: {(end_date - start_date).days} 天")

# 提取有效样本数据
valid_data = {}
for var, values in numeric_data.items():
    valid_data[var] = [values[i] for i in valid_indices]

# 描述性统计
print("\n" + "="*80)
print("描述性统计")
print("="*80)

# 定义要报告的变量顺序
report_vars = ['hs300_close', 'daily_return', 'future_return_5d', 'future_return_60d',
               'cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny',
               'sentiment_zscore', 'ivix', 'north_flow', 'margin_balance',
               'amihud', 'momentum_20d', 'intraday_range']

stats_results = {}
for var in report_vars:
    if var in valid_data:
        stats = calculate_stats(valid_data[var])
        stats_results[var] = stats
        if stats:
            print(f"\n{vars_mapping.get(var, var)}:")
            print(f"  观测数: {stats['count']}, 均值: {stats['mean']:.6f}, 标准差: {stats['std']:.6f}")
            print(f"  最小值: {stats['min']:.6f}, 25%分位: {stats['p25']:.6f}, 中位数: {stats['median']:.6f}")
            print(f"  75%分位: {stats['p75']:.6f}, 最大值: {stats['max']:.6f}")
            print(f"  偏度: {stats['skewness']:.4f}, 峰度: {stats['kurtosis']:.4f}")

# 缺失值分析
print("\n" + "="*80)
print("缺失值统计")
print("="*80)

all_vars = list(vars_mapping.keys())
print(f"{'变量名':<30} {'缺失数':>10} {'缺失率%':>10} {'有效观测':>10}")
print("-" * 60)
for var in all_vars:
    if var in numeric_data:
        total = len(numeric_data[var])
        missing = sum(1 for v in numeric_data[var] if v is None)
        missing_rate = missing / total * 100 if total > 0 else 0
        valid = total - missing
        print(f"{vars_mapping.get(var, var):<30} {missing:>10} {missing_rate:>10.2f} {valid:>10}")

# 相关性分析
print("\n" + "="*80)
print("相关性分析")
print("="*80)

# 宏观变量间相关性
print("\n【宏观变量间相关性】")
macro_vars = ['cpi', 'ppi', 'm2_growth', 'epu', 'usd_cny']
print(f"{'':>15}", end="")
for v in macro_vars:
    print(f"{v:>12}", end="")
print()

for v1 in macro_vars:
    if v1 in valid_data:
        print(f"{v1:>15}", end="")
        for v2 in macro_vars:
            if v2 in valid_data:
                corr = calculate_correlation(valid_data[v1], valid_data[v2])
                if corr is not None:
                    print(f"{corr:>12.4f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
        print()

# 市场状态变量间相关性
print("\n【市场状态变量间相关性】")
market_vars = ['sentiment_zscore', 'ivix', 'north_flow', 'margin_balance', 'amihud', 'momentum_20d', 'intraday_range']
available_market = [v for v in market_vars if v in valid_data]
if len(available_market) > 1:
    print(f"{'':>20}", end="")
    for v in available_market:
        short_name = v[:12]
        print(f"{short_name:>14}", end="")
    print()

    for v1 in available_market:
        short_name = v1[:20]
        print(f"{short_name:>20}", end="")
        for v2 in available_market:
            corr = calculate_correlation(valid_data[v1], valid_data[v2])
            if corr is not None:
                print(f"{corr:>14.4f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

# 宏观变量与未来收益相关性
print("\n【宏观变量与未来收益相关性】")
print(f"{'变量':<30} {'与未来5日收益':>15} {'与未来60日收益':>15}")
print("-" * 60)
for var in macro_vars:
    if var in valid_data:
        corr_5d = calculate_correlation(valid_data[var], valid_data.get('future_return_5d', []))
        corr_60d = calculate_correlation(valid_data[var], valid_data.get('future_return_60d', []))
        print(f"{vars_mapping.get(var, var):<30} {corr_5d:>15.4f} {corr_60d:>15.4f}")

# 市场状态变量与未来收益相关性
print("\n【市场状态变量与未来收益相关性】")
print(f"{'变量':<30} {'与未来5日收益':>15} {'与未来60日收益':>15}")
print("-" * 60)
for var in available_market:
    corr_5d = calculate_correlation(valid_data[var], valid_data.get('future_return_5d', []))
    corr_60d = calculate_correlation(valid_data[var], valid_data.get('future_return_60d', []))
    print(f"{vars_mapping.get(var, var):<30} {corr_5d:>15.4f} {corr_60d:>15.4f}")

# 极端值分析
print("\n" + "="*80)
print("极端值分析")
print("="*80)

extreme_vars = ['north_flow', 'margin_balance', 'ivix', 'amihud', 'intraday_range']
for var in extreme_vars:
    if var in valid_data and valid_data[var]:
        stats = calculate_stats(valid_data[var])
        if stats:
            below_p1 = sum(1 for v in valid_data[var] if v is not None and v < stats['p1'])
            above_p99 = sum(1 for v in valid_data[var] if v is not None and v > stats['p99'])
            print(f"{vars_mapping.get(var, var)}:")
            print(f"  1%分位数: {stats['p1']:.6f}, 99%分位数: {stats['p99']:.6f}")
            print(f"  低于1%: {below_p1}个, 高于99%: {above_p99}个")

# 年份分布
print("\n" + "="*80)
print("样本年份分布")
print("="*80)

year_counts = defaultdict(int)
for i in valid_indices:
    year = data[i]['date_parsed'].year
    year_counts[year] += 1

print(f"{'年份':<10} {'观测数':>10}")
print("-" * 20)
for year in sorted(year_counts.keys()):
    print(f"{year:<10} {year_counts[year]:>10}")

# 训练集/测试集划分
print("\n" + "="*80)
print("训练集/测试集划分 (60%/40%)")
print("="*80)

total_valid = len(valid_indices)
train_size = int(total_valid * 0.6)
train_indices = valid_indices[:train_size]
test_indices = valid_indices[train_size:]

print(f"总样本量: {total_valid}")
print(f"训练集: {len(train_indices)} ({len(train_indices)/total_valid*100:.1f}%)")
if train_indices:
    train_dates = [data[i]['date_parsed'] for i in train_indices]
    print(f"  时间区间: {min(train_dates).strftime('%Y-%m-%d')} 至 {max(train_dates).strftime('%Y-%m-%d')}")

print(f"测试集: {len(test_indices)} ({len(test_indices)/total_valid*100:.1f}%)")
if test_indices:
    test_dates = [data[i]['date_parsed'] for i in test_indices]
    print(f"  时间区间: {min(test_dates).strftime('%Y-%m-%d')} 至 {max(test_dates).strftime('%Y-%m-%d')}")

# 保存结果到文件
print("\n" + "="*80)
print("保存分析结果")
print("="*80)

# 保存描述统计
with open('/home/marktom/bigdata-fin/desc_stats.txt', 'w', encoding='utf-8') as f:
    f.write("描述性统计结果\n")
    f.write("="*80 + "\n\n")
    f.write(f"样本时间区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}\n")
    f.write(f"总观测数: {len(valid_indices)}\n\n")

    for var in report_vars:
        if var in stats_results and stats_results[var]:
            stats = stats_results[var]
            f.write(f"\n{vars_mapping.get(var, var)} ({var}):\n")
            f.write(f"  观测数: {stats['count']}\n")
            f.write(f"  均值: {stats['mean']:.6f}\n")
            f.write(f"  标准差: {stats['std']:.6f}\n")
            f.write(f"  最小值: {stats['min']:.6f}\n")
            f.write(f"  25%分位数: {stats['p25']:.6f}\n")
            f.write(f"  中位数: {stats['median']:.6f}\n")
            f.write(f"  75%分位数: {stats['p75']:.6f}\n")
            f.write(f"  最大值: {stats['max']:.6f}\n")
            f.write(f"  偏度: {stats['skewness']:.4f}\n")
            f.write(f"  峰度: {stats['kurtosis']:.4f}\n")

print("描述统计已保存至: desc_stats.txt")

# 保存缺失值统计
with open('/home/marktom/bigdata-fin/missing_stats.txt', 'w', encoding='utf-8') as f:
    f.write("缺失值统计\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'变量名':<30} {'缺失数':>10} {'缺失率%':>10} {'有效观测':>10}\n")
    f.write("-" * 60 + "\n")
    for var in all_vars:
        if var in numeric_data:
            total = len(numeric_data[var])
            missing = sum(1 for v in numeric_data[var] if v is None)
            missing_rate = missing / total * 100 if total > 0 else 0
            valid = total - missing
            f.write(f"{vars_mapping.get(var, var):<30} {missing:>10} {missing_rate:>10.2f} {valid:>10}\n")

print("缺失值统计已保存至: missing_stats.txt")

# 保存相关性矩阵
with open('/home/marktom/bigdata-fin/correlation_matrix.txt', 'w', encoding='utf-8') as f:
    f.write("相关性矩阵\n")
    f.write("="*80 + "\n\n")

    f.write("【宏观变量间相关性】\n")
    f.write(f"{'':>15}")
    for v in macro_vars:
        f.write(f"{v:>12}")
    f.write("\n")

    for v1 in macro_vars:
        if v1 in valid_data:
            f.write(f"{v1:>15}")
            for v2 in macro_vars:
                if v2 in valid_data:
                    corr = calculate_correlation(valid_data[v1], valid_data[v2])
                    if corr is not None:
                        f.write(f"{corr:>12.4f}")
                    else:
                        f.write(f"{'N/A':>12}")
            f.write("\n")

    f.write("\n【市场状态变量间相关性】\n")
    if len(available_market) > 1:
        for v1 in available_market:
            for v2 in available_market:
                corr = calculate_correlation(valid_data[v1], valid_data[v2])
                f.write(f"{vars_mapping.get(v1, v1)} vs {vars_mapping.get(v2, v2)}: {corr:.4f}\n")

    f.write("\n【与未来收益相关性】\n")
    f.write(f"{'变量':<30} {'与未来5日收益':>15} {'与未来60日收益':>15}\n")
    f.write("-" * 60 + "\n")
    for var in macro_vars + available_market:
        if var in valid_data:
            corr_5d = calculate_correlation(valid_data[var], valid_data.get('future_return_5d', []))
            corr_60d = calculate_correlation(valid_data[var], valid_data.get('future_return_60d', []))
            f.write(f"{vars_mapping.get(var, var):<30} {corr_5d:>15.4f} {corr_60d:>15.4f}\n")

print("相关性矩阵已保存至: correlation_matrix.txt")

print("\n" + "="*80)
print("分析完成!")
print("="*80)
