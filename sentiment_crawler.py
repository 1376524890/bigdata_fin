#!/usr/bin/env python3
"""
情绪数据爬虫脚本
爬取: 东方财富股吧情绪、雪球热帖、百度指数
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import random
import re
import os
import warnings
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/marktom/bigdata-fin/real_data'

print("=" * 60)
print("情绪数据爬虫脚本")
print("=" * 60)

# 通用headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# ============================================================================
# 1. 东方财富股吧情绪数据
# ============================================================================
print("\n【1】东方财富股吧情绪数据...")

def get_eastmoney_guba_sentiment():
    """获取东方财富股吧热门帖子情绪"""

    # 股吧热门帖子API
    url = "http://gbapi.eastmoney.com/topic/api/v1/hotspot/list"

    headers = HEADERS.copy()
    headers['Referer'] = 'http://guba.eastmoney.com/'

    try:
        params = {
            'ps': 50,  # 每页数量
            'p': 1,    # 页码
        }

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('list'):
                posts = data['data']['list']
                result = []
                for post in posts:
                    result.append({
                        'post_id': post.get('post_id', ''),
                        'title': post.get('title', ''),
                        'content': post.get('content', ''),
                        'author': post.get('author', ''),
                        'read_count': post.get('read_count', 0),
                        'comment_count': post.get('comment_count', 0),
                        'like_count': post.get('like_count', 0),
                        'post_date': post.get('post_date', ''),
                        'source': 'eastmoney_guba'
                    })
                return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")
    return pd.DataFrame()

# 获取股吧数据
guba_data = get_eastmoney_guba_sentiment()
if len(guba_data) > 0:
    guba_data.to_csv(f'{OUTPUT_DIR}/eastmoney_guba_hot.csv', index=False)
    print(f"   股吧热门帖子: {len(guba_data)}条")

# 获取上证指数股吧情绪
def get_stock_guba_posts(stock_code='sh000001', pages=5):
    """获取特定股票股吧帖子"""

    url = f"http://guba.eastmoney.com/list,{stock_code}.html"
    headers = HEADERS.copy()
    headers['Referer'] = f'http://guba.eastmoney.com/list,{stock_code}.html'

    all_posts = []

    for page in range(1, pages + 1):
        try:
            params = {'f': 'pag', 'p': page}
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                posts = soup.find_all('div', class_='listitem')

                for post in posts:
                    try:
                        title_elem = post.find('a', class_='title')
                        read_elem = post.find('span', class_='read')
                        comment_elem = post.find('span', class_='reply')

                        title = title_elem.text.strip() if title_elem else ''
                        read_count = read_elem.text.replace('万', '0000').replace('阅读', '').strip() if read_elem else '0'
                        comment_count = comment_elem.text.replace('评论', '').strip() if comment_elem else '0'

                        # 情绪判断（简单规则）
                        positive_words = ['涨', '牛', '利好', '突破', '上涨', '反弹', '看好', '强势']
                        negative_words = ['跌', '熊', '利空', '暴跌', '下跌', '破位', '看空', '弱势']

                        sentiment = 0
                        for w in positive_words:
                            if w in title:
                                sentiment += 1
                        for w in negative_words:
                            if w in title:
                                sentiment -= 1

                        all_posts.append({
                            'title': title,
                            'read_count': int(read_count) if read_count.isdigit() else 0,
                            'comment_count': int(comment_count) if comment_count.isdigit() else 0,
                            'sentiment': sentiment,  # -1负面, 0中性, 1正面
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
                    except:
                        continue

            time.sleep(random.uniform(0.5, 1.5))  # 随机延迟
        except Exception as e:
            print(f"   第{page}页错误: {e}")

    return pd.DataFrame(all_posts)

print("   获取上证指数股吧帖子...")
guba_sh = get_stock_guba_posts('sh000001', pages=3)
if len(guba_sh) > 0:
    guba_sh.to_csv(f'{OUTPUT_DIR}/eastmoney_guba_sh000001.csv', index=False)
    # 计算情绪指标
    sentiment_score = guba_sh['sentiment'].mean()
    print(f"   上证股吧帖子: {len(guba_sh)}条")
    print(f"   情绪得分: {sentiment_score:.3f} ({'偏多' if sentiment_score > 0 else '偏空' if sentiment_score < 0 else '中性'})")

# ============================================================================
# 2. 东方财富市场情绪指数
# ============================================================================
print("\n【2】东方财富市场情绪指数...")

def get_eastmoney_market_sentiment():
    """获取东方财富市场情绪指数"""

    url = "http://push2.eastmoney.com/api/qt/clist/get"

    headers = HEADERS.copy()
    headers['Referer'] = 'http://quote.eastmoney.com/'

    try:
        # 获取涨跌停数据
        params = {
            'fid': 'f3',
            'po': '1',
            'pz': '5000',
            'pn': '1',
            'np': '1',
            'fltt': '2',
            'invt': '2',
            'fs': 'm:1,m:2,m:3',
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18'
        }

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('diff'):
                stocks = data['data']['diff']

                # 统计涨跌
                up_count = 0
                down_count = 0
                limit_up = 0
                limit_down = 0

                for stock in stocks:
                    try:
                        change = float(stock.get('f3', 0))
                        if change > 0:
                            up_count += 1
                            if change >= 9.9:
                                limit_up += 1
                        elif change < 0:
                            down_count += 1
                            if change <= -9.9:
                                limit_down += 1
                    except:
                        continue

                total = up_count + down_count
                sentiment = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'up_count': up_count,
                    'down_count': down_count,
                    'limit_up': limit_up,
                    'limit_down': limit_down,
                    'up_ratio': up_count / total if total > 0 else 0.5,
                    'sentiment_score': (up_count - down_count) / total if total > 0 else 0
                }

                return sentiment
    except Exception as e:
        print(f"   错误: {e}")

    return None

sentiment_data = get_eastmoney_market_sentiment()
if sentiment_data:
    df = pd.DataFrame([sentiment_data])
    df.to_csv(f'{OUTPUT_DIR}/eastmoney_market_sentiment.csv', index=False)
    print(f"   上涨: {sentiment_data['up_count']}, 下跌: {sentiment_data['down_count']}")
    print(f"   涨停: {sentiment_data['limit_up']}, 跌停: {sentiment_data['limit_down']}")
    print(f"   情绪得分: {sentiment_data['sentiment_score']:.3f}")

# ============================================================================
# 3. 雪球热帖数据
# ============================================================================
print("\n【3】雪球热帖数据...")

def get_xueqiu_hot_posts(count=50):
    """获取雪球热门帖子"""

    url = "https://xueqiu.com/statuses/hot/listV2.json"

    headers = HEADERS.copy()
    headers['Referer'] = 'https://xueqiu.com/'
    headers['Origin'] = 'https://xueqiu.com'

    try:
        params = {'since_id': -1, 'max_id': -1, 'size': count}

        # 首先获取cookie
        session = requests.Session()
        session.get('https://xueqiu.com', headers=headers, timeout=15)

        resp = session.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('items'):
                posts = []
                for item in data['items']:
                    original = item.get('original_status', {})
                    posts.append({
                        'id': original.get('id', ''),
                        'user': original.get('user', {}).get('screen_name', ''),
                        'title': original.get('title', ''),
                        'text': original.get('text', '')[:200],  # 截取前200字
                        'retweet_count': original.get('retweet_count', 0),
                        'reply_count': original.get('reply_count', 0),
                        'like_count': original.get('like_count', 0),
                        'view_count': original.get('view_count', 0),
                        'created_at': original.get('created_at', 0),
                        'source': 'xueqiu'
                    })

                return pd.DataFrame(posts)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

xueqiu_data = get_xueqiu_hot_posts(50)
if len(xueqiu_data) > 0:
    # 转换时间戳
    xueqiu_data['date'] = pd.to_datetime(xueqiu_data['created_at'], unit='ms').dt.strftime('%Y-%m-%d')
    xueqiu_data.to_csv(f'{OUTPUT_DIR}/xueqiu_hot_posts.csv', index=False)
    print(f"   雪球热帖: {len(xueqiu_data)}条")

# 获取雪球沪深300讨论
def get_xueqiu_stock_posts(symbol='SH000300', count=30):
    """获取雪球特定股票讨论"""

    url = f"https://xueqiu.com/query/v1/symbol/search/status.json"

    headers = HEADERS.copy()
    headers['Referer'] = f'https://xueqiu.com/S/{symbol}'

    try:
        params = {
            'symbol': symbol,
            'count': count,
            'comment': 0,
            'symbol': symbol,
            'hl': 'true',
            'source': 'all',
            'sort': 'time',
            'page': 1
        }

        session = requests.Session()
        session.get('https://xueqiu.com', headers=headers, timeout=15)

        resp = session.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('list'):
                posts = []
                for item in data['list']:
                    posts.append({
                        'id': item.get('id', ''),
                        'user': item.get('user', {}).get('screen_name', ''),
                        'text': item.get('text', '')[:200],
                        'retweet_count': item.get('retweet_count', 0),
                        'reply_count': item.get('reply_count', 0),
                        'like_count': item.get('like_count', 0),
                        'created_at': item.get('created_at', 0),
                        'symbol': symbol,
                        'source': 'xueqiu'
                    })

                return pd.DataFrame(posts)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

print("   获取雪球沪深300讨论...")
xueqiu_hs300 = get_xueqiu_stock_posts('SH000300', 30)
if len(xueqiu_hs300) > 0:
    xueqiu_hs300['date'] = pd.to_datetime(xueqiu_hs300['created_at'], unit='ms').dt.strftime('%Y-%m-%d')
    xueqiu_hs300.to_csv(f'{OUTPUT_DIR}/xueqiu_hs300_posts.csv', index=False)
    print(f"   沪深300讨论: {len(xueqiu_hs300)}条")

# ============================================================================
# 4. 东方财富恐慌指数
# ============================================================================
print("\n【4】东方财富恐慌贪婪指数...")

def get_eastmoney_fear_greed():
    """获取东方财富恐慌贪婪指数"""

    url = "http://emdata.eastmoney.com/fxjss/api/data"

    headers = HEADERS.copy()
    headers['Referer'] = 'http://data.eastmoney.com/xg/feargreed/'

    try:
        params = {
            'type': 'RPTA_APP_WEB_FEARGREED',
            'sty': 'ALL',
            'source': 'WEB',
            'client': 'WEB'
        }

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('result') and data['result'].get('data'):
                records = data['result']['data']
                result = []
                for r in records:
                    result.append({
                        'date': r.get('TRADE_DATE', ''),
                        'fear_greed_index': r.get('CLOSE', 0),
                        'level': r.get('LEVEL', '')
                    })
                return pd.DataFrame(result)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

fear_greed = get_eastmoney_fear_greed()
if len(fear_greed) > 0:
    fear_greed['date'] = pd.to_datetime(fear_greed['date']).dt.strftime('%Y-%m-%d')
    fear_greed.to_csv(f'{OUTPUT_DIR}/eastmoney_fear_greed.csv', index=False)
    print(f"   恐慌贪婪指数: {len(fear_greed)}条")
    # 显示最新值
    latest = fear_greed.iloc[-1]
    print(f"   最新值: {latest['fear_greed_index']} ({latest['level']})")

# ============================================================================
# 5. 百度搜索指数（需要登录，尝试公开数据）
# ============================================================================
print("\n【5】百度搜索指数（尝试公开接口）...")

def get_baidu_index_keyword(keyword='股市', days=30):
    """尝试获取百度指数（公开接口有限制）"""

    # 百度指数API需要登录，这里尝试替代方案
    # 使用百度热搜作为替代

    url = "http://top.baidu.com/board?platform=wise&tab=realtime"

    headers = HEADERS.copy()
    headers['Referer'] = 'http://top.baidu.com/'

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 查找热搜数据
            items = soup.find_all('div', class_='category-wrap_iQLoo')

            results = []
            for item in items:
                try:
                    title_elem = item.find('div', class_='c-single-text-ellipsis')
                    hot_elem = item.find('div', class_='hot-index_1Bl1a')

                    if title_elem and hot_elem:
                        title = title_elem.text.strip()
                        hot = hot_elem.text.strip()

                        results.append({
                            'title': title,
                            'hot_index': hot,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'baidu_hot'
                        })
                except:
                    continue

            if results:
                return pd.DataFrame(results)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

print("   获取百度热搜榜...")
baidu_hot = get_baidu_index_keyword()
if len(baidu_hot) > 0:
    baidu_hot.to_csv(f'{OUTPUT_DIR}/baidu_hot_search.csv', index=False)
    print(f"   百度热搜: {len(baidu_hot)}条")
    # 查找股市相关
    stock_related = baidu_hot[baidu_hot['title'].str.contains('股|市|金融|经济', na=False)]
    if len(stock_related) > 0:
        print(f"   股市相关热搜: {len(stock_related)}条")

# ============================================================================
# 6. 新浪财经情绪数据
# ============================================================================
print("\n【6】新浪财经情绪数据...")

def get_sina_finance_sentiment():
    """获取新浪财经市场情绪"""

    url = "https://tousu.sina.com.cn/api/company/getObjectionAdd"

    headers = HEADERS.copy()
    headers['Referer'] = 'https://finance.sina.com.cn/'

    # 获取大盘数据
    market_url = "https://hq.sinajs.cn/list=sh000001,sz399001,sh000300"

    try:
        resp = requests.get(market_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            content = resp.text

            # 解析大盘数据
            def parse_sina_data(data_str):
                parts = data_str.split(',')
                if len(parts) >= 32:
                    return {
                        'name': parts[0].split('="')[1] if '="' in parts[0] else '',
                        'open': float(parts[1]) if parts[1] else 0,
                        'last_close': float(parts[2]) if parts[2] else 0,
                        'current': float(parts[3]) if parts[3] else 0,
                        'high': float(parts[4]) if parts[4] else 0,
                        'low': float(parts[5]) if parts[5] else 0,
                        'volume': int(parts[8]) if parts[8] else 0,
                        'amount': float(parts[9]) if parts[9] else 0,
                    }
                return None

            results = []
            for line in content.strip().split('\n'):
                if 'var hq_str_' in line:
                    code = line.split('hq_str_')[1].split('=')[0]
                    data_str = line.split('="')[1].rstrip('";')
                    data = parse_sina_data(data_str)
                    if data:
                        data['code'] = code
                        data['date'] = datetime.now().strftime('%Y-%m-%d')
                        results.append(data)

            if results:
                return pd.DataFrame(results)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

sina_data = get_sina_finance_sentiment()
if len(sina_data) > 0:
    sina_data.to_csv(f'{OUTPUT_DIR}/sina_market_data.csv', index=False)
    print(f"   新浪大盘数据: {len(sina_data)}条")

# ============================================================================
# 7. 淘股吧情绪数据
# ============================================================================
print("\n【7】淘股吧热门帖子...")

def get_taoguba_hot():
    """获取淘股吧热门帖子"""

    url = "https://www.taoguba.com.cn/new/Home/getTopicList"

    headers = HEADERS.copy()
    headers['Referer'] = 'https://www.taoguba.com.cn/'
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        data = {
            'type': 'hot',
            'pageNo': 1,
            'pageSize': 30
        }

        resp = requests.post(url, data=data, headers=headers, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            if result.get('data') and result['data'].get('list'):
                posts = []
                for item in result['data']['list']:
                    posts.append({
                        'title': item.get('title', ''),
                        'author': item.get('userName', ''),
                        'read_count': item.get('readCount', 0),
                        'reply_count': item.get('replyCount', 0),
                        'like_count': item.get('praiseCount', 0),
                        'create_time': item.get('createTime', ''),
                        'source': 'taoguba'
                    })
                return pd.DataFrame(posts)
    except Exception as e:
        print(f"   错误: {e}")

    return pd.DataFrame()

taoguba = get_taoguba_hot()
if len(taoguba) > 0:
    taoguba['date'] = datetime.now().strftime('%Y-%m-%d')
    taoguba.to_csv(f'{OUTPUT_DIR}/taoguba_hot.csv', index=False)
    print(f"   淘股吧热帖: {len(taoguba)}条")

# ============================================================================
# 8. 计算综合情绪指数
# ============================================================================
print("\n【8】计算综合情绪指数...")

def calculate_sentiment_index():
    """计算综合情绪指数"""

    sentiment_scores = {}

    # 1. 东方财富市场情绪
    try:
        market_sent = pd.read_csv(f'{OUTPUT_DIR}/eastmoney_market_sentiment.csv')
        sentiment_scores['eastmoney'] = market_sent['sentiment_score'].iloc[0]
    except:
        pass

    # 2. 恐慌贪婪指数
    try:
        fear_greed = pd.read_csv(f'{OUTPUT_DIR}/eastmoney_fear_greed.csv')
        latest = fear_greed.iloc[-1]
        # 归一化到-1到1
        fg_value = latest['fear_greed_index']
        sentiment_scores['fear_greed'] = (fg_value - 50) / 50  # 50是中性
    except:
        pass

    # 3. 股吧情绪
    try:
        guba = pd.read_csv(f'{OUTPUT_DIR}/eastmoney_guba_sh000001.csv')
        sentiment_scores['guba'] = guba['sentiment'].mean()
    except:
        pass

    # 计算综合得分
    if sentiment_scores:
        composite = np.mean(list(sentiment_scores.values()))

        result = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'eastmoney_sentiment': sentiment_scores.get('eastmoney', 0),
            'fear_greed_sentiment': sentiment_scores.get('fear_greed', 0),
            'guba_sentiment': sentiment_scores.get('guba', 0),
            'composite_sentiment': composite,
            'interpretation': '贪婪/看多' if composite > 0.2 else ('恐惧/看空' if composite < -0.2 else '中性')
        }

        df = pd.DataFrame([result])
        df.to_csv(f'{OUTPUT_DIR}/composite_sentiment_index.csv', index=False)

        print(f"   东方财富情绪: {result['eastmoney_sentiment']:.3f}")
        print(f"   恐慌贪婪指数: {result['fear_greed_sentiment']:.3f}")
        print(f"   股吧情绪: {result['guba_sentiment']:.3f}")
        print(f"   综合情绪指数: {result['composite_sentiment']:.3f} ({result['interpretation']})")

        return df

    return None

calculate_sentiment_index()

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 60)
print("情绪数据爬取完成！")
print("=" * 60)

# 统计文件
files = [f for f in os.listdir(OUTPUT_DIR) if 'sentiment' in f or 'hot' in f or 'guba' in f or 'xueqiu' in f or 'taoguba' in f or 'baidu' in f]
print(f"\n生成的情绪数据文件: {len(files)}个")
for f in files:
    filepath = os.path.join(OUTPUT_DIR, f)
    try:
        df = pd.read_csv(filepath)
        print(f"  - {f}: {len(df)}条")
    except:
        pass