# modules/trend_detector.py
# 데이터베이스에서 기사를 로드하고 새로운 트렌드를 감지하는 모듈

import sqlite3
from loguru import logger
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter
import re

# SQLite DB 파일 경로 (data_collector.py와 동일)
DB_FILE = 'trends.db'

def get_articles_from_db(days_ago: int = 30) -> List[Dict[str, Any]]:
    """
    지정된 기간(days_ago) 내에 수집된 모든 기사를 데이터베이스에서 로드합니다.
    """
    conn = None
    articles = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 'created_at' 필드를 기준으로 최근 N일간의 기사만 가져옵니다.
        # ISO 형식의 문자열 비교가 가능합니다.
        cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        cursor.execute(f"SELECT id, source, category, title, content, publish_date, url, keywords, created_at FROM articles WHERE created_at >= '{cutoff_date}' ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        for row in rows:
            articles.append({
                "id": row[0],
                "source": row[1],
                "category": row[2],
                "title": row[3],
                "content": row[4],
                "publish_date": row[5],
                "url": row[6],
                "keywords": row[7], # 수집 시 사용된 키워드
                "created_at": row[8] # DB에 저장된 시간
            })
        logger.info(f"Loaded {len(articles)} articles from DB within last {days_ago} days.")
    except sqlite3.Error as e:
        logger.error(f"Error loading articles from database: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return articles

def extract_keywords_from_content(text: str) -> List[str]:
    """
    텍스트에서 알파벳과 한글 단어만 추출하여 소문자로 변환합니다.
    이 함수는 간단한 키워드 추출을 위한 것이며, 더 정교한 NLP 전처리는
    trend_analyzer.py의 preprocess_text를 활용할 수 있습니다.
    """
    # 알파벳과 한글 단어만 매칭 (숫자, 특수문자 등 제외)
    words = re.findall(r'[a-zA-Z가-힣]+', text.lower())
    # 길이가 2 이상인 단어만 필터링 (너무 짧은 단어 제외)
    return [word for word in words if len(word) > 1]


def detect_trending_keywords(articles: List[Dict[str, Any]], lookback_days: int = 7, threshold_percent_increase: float = 100.0) -> List[Dict[str, Any]]:
    """
    최근 N일간의 키워드 언급량과 이전 N일간의 언급량을 비교하여
    언급량이 급증한 트렌드 키워드를 감지합니다.
    
    Args:
        articles: 분석할 뉴스 기사 리스트 (DB에서 로드된 형태).
        lookback_days: 최근 N일 (현재 기간) 및 이전 N일 (비교 기간)의 일수.
        threshold_percent_increase: 언급량 증가율이 이 임계값(%)을 초과해야 트렌드로 감지.
        
    Returns:
        급증한 키워드와 그 증가율, 관련 기사 수 등을 포함하는 리스트.
    """
    if not articles:
        logger.warning("No articles provided for trend detection.")
        return []

    logger.info(f"Detecting trending keywords based on last {lookback_days} days vs. previous {lookback_days} days.")

    # 현재 날짜 및 기간 설정
    now = datetime.now()
    current_period_start = now - timedelta(days=lookback_days)
    previous_period_start = now - timedelta(days=lookback_days * 2)

    current_keywords = Counter()
    previous_keywords = Counter()
    
    # 각 기사를 순회하며 기간별 키워드 카운트
    for article in articles:
        try:
            # created_at이 ISO 형식 문자열이므로 datetime 객체로 변환
            article_date = datetime.fromisoformat(article['created_at'])
            
            # 기사 제목과 내용에서 키워드 추출
            content_keywords = extract_keywords_from_content(article['title'] + " " + article['content'])

            if article_date >= current_period_start:
                current_keywords.update(content_keywords)
            elif article_date >= previous_period_start:
                previous_keywords.update(content_keywords)
        except Exception as e:
            logger.warning(f"Error processing article date or keywords: {e} for article ID {article.get('id')}")
            continue

    trending_results = []

    # 현재 기간에 등장한 모든 키워드에 대해 증가율 계산
    for keyword, current_count in current_keywords.items():
        previous_count = previous_keywords.get(keyword, 0)

        # 이전 기간에 언급이 없었거나 매우 적었지만 현재 급증한 경우
        if previous_count == 0 and current_count > 0:
            # 이전 언급이 0인 경우, 현재 언급이 1개 이상이면 '새로운' 것으로 간주
            # 또는 특정 최소 언급량 이상일 경우만 고려 (예: current_count > 2)
            if current_count > 1: # 최소 2회 이상 언급되어야 새로운 트렌드로 간주
                trending_results.append({
                    "keyword": keyword,
                    "current_mentions": current_count,
                    "previous_mentions": previous_count,
                    "percent_increase": float('inf'), # 무한대 증가율
                    "reason": "New or significantly increased mentions in current period"
                })
        elif previous_count > 0:
            percent_increase = ((current_count - previous_count) / previous_count) * 100
            if percent_increase >= threshold_percent_increase:
                trending_results.append({
                    "keyword": keyword,
                    "current_mentions": current_count,
                    "previous_mentions": previous_count,
                    "percent_increase": round(percent_increase, 2),
                    "reason": f"Mention increased by {round(percent_increase, 2)}%"
                })
    
    # 언급량(current_mentions) 기준으로 내림차순 정렬
    trending_results.sort(key=lambda x: x['current_mentions'], reverse=True)
    
    logger.info(f"Detected {len(trending_results)} trending keywords.")
    return trending_results

def get_articles_by_keywords(keywords: List[str], days_ago: int = 30) -> List[Dict[str, Any]]:
    """
    주어진 키워드 리스트를 포함하는 기사들을 데이터베이스에서 로드합니다.
    """
    conn = None
    articles = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        # 키워드 리스트를 SQL 쿼리에 사용할 수 있는 형태로 변환
        # 각 키워드가 title 또는 content에 포함되어 있는지 확인
        keyword_conditions = [f"(title LIKE '%{kw}%' OR content LIKE '%{kw}%')" for kw in keywords]
        where_clause = " OR ".join(keyword_conditions)
        
        query = f"SELECT id, source, category, title, content, publish_date, url, keywords, created_at FROM articles WHERE created_at >= '{cutoff_date}' AND ({where_clause}) ORDER BY created_at DESC"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            articles.append({
                "id": row[0],
                "source": row[1],
                "category": row[2],
                "title": row[3],
                "content": row[4],
                "publish_date": row[5],
                "url": row[6],
                "keywords": row[7],
                "created_at": row[8]
            })
        logger.info(f"Loaded {len(articles)} articles from DB matching keywords: {keywords}.")
    except sqlite3.Error as e:
        logger.error(f"Error loading articles by keywords from database: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return articles


if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    print("--- Trend Detector 모듈 테스트 ---")
    
    # 1. DB에서 기사 로드 테스트 (최근 30일)
    all_recent_articles = get_articles_from_db(days_ago=30)
    print(f"\n총 {len(all_recent_articles)}개의 최근 기사 로드 완료.")
    
    if all_recent_articles:
        # 2. 트렌딩 키워드 감지 테스트 (최근 7일 vs 이전 7일, 100% 증가율)
        trending_keywords = detect_trending_keywords(all_recent_articles, lookback_days=7, threshold_percent_increase=50.0)
        
        if trending_keywords:
            print("\n감지된 트렌딩 키워드:")
            for trend in trending_keywords:
                print(f"- 키워드: '{trend['keyword']}', 현재 언급: {trend['current_mentions']}, 이전 언급: {trend['previous_mentions']}, 증가율: {trend['percent_increase']}%")
                
            # 3. 특정 트렌딩 키워드에 대한 기사 로드 테스트 (첫 번째 트렌딩 키워드 사용)
            if trending_keywords:
                test_keyword = trending_keywords[0]['keyword']
                related_articles = get_articles_by_keywords([test_keyword], days_ago=14)
                print(f"\n'{test_keyword}' 관련 기사 ({len(related_articles)}개):")
                for i, article in enumerate(related_articles[:3]): # 상위 3개만 출력
                    print(f"  - 제목: {article['title']} (ID: {article['id']})")
                    print(f"    URL: {article['url']}")
                    print(f"    발행일: {article['publish_date']}")
        else:
            print("\n감지된 트렌딩 키워드가 없습니다. 더 많은 데이터를 수집하거나 기간/임계값을 조정해보세요.")
    else:
        print("\n데이터베이스에 충분한 기사가 없어 트렌드 감지를 수행할 수 없습니다.")

