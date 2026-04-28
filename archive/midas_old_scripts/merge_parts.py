#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合论文三部分：第一部分 + 第二部分 + 第三部分
"""

# 读取第一部分和第三部分（整合稿）
with open('/home/marktom/bigdata-fin/第一部分_第三部分_整合稿_编号参考文献版.md', 'r', encoding='utf-8') as f:
    part1_3_content = f.read()

# 读取第二部分
with open('/home/marktom/bigdata-fin/第二部分_数据获取与分析.md', 'r', encoding='utf-8') as f:
    part2_content = f.read()

# 第一部分在第一和第三部分之间插入第二部分
# 找到"---"的位置（第一部分和第三部分的分隔）
parts = part1_3_content.split('---\n\n## 第三部分')

if len(parts) == 2:
    part1 = parts[0].strip()
    part3 = '## 第三部分' + parts[1]

    # 组合：第一部分 + 分隔线 + 第二部分 + 分隔线 + 第三部分
    combined = part1 + '\n\n---\n\n' + part2_content + '\n\n---\n\n' + part3

    # 保存到文件
    with open('/home/marktom/bigdata-fin/论文_完整版_三大部分.md', 'w', encoding='utf-8') as f:
        f.write(combined)

    print("整合完成！")
    print(f"输出文件: /home/marktom/bigdata-fin/论文_完整版_三大部分.md")
else:
    print("错误：无法正确分割第一和第三部分")
    print(f"找到 {len(parts)} 个部分")
