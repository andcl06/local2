# data_collector.py
# Google Custom Search API를 사용하여 데이터를 수집하고 SQLite DB에 저장하는 모듈

import requests
import os
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any
from datetime import datetime
import re
import sqlite3 # SQLite 라이브러리 임포트

# .env 파일에서 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Google Custom Search API 엔드포인트
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"

# --- SQLite DB 설정 ---
# 데이터베이스 파일 경로 (프로젝트 루트에 'trends.db' 파일 생성)
DB_FILE = 'trends.db'

def init_db():
    """
    SQLite 데이터베이스를 초기화하고 articles 테이블을 생성합니다.
    테이블이 이미 존재하면 생성하지 않습니다.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                category TEXT,
                title TEXT NOT NULL,
                content TEXT,
                publish_date TEXT,
                url TEXT UNIQUE, -- URL은 중복되지 않도록 UNIQUE 제약 조건 추가
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        logger.info(f"SQLite database '{DB_FILE}' and 'articles' table initialized.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def clean_text(text: str) -> str:
    """텍스트 정제 함수"""
    cleaned_text = re.sub(r'<.*?>|http\S+|www.\S+|\S*@\S*\s?|[^\w\s\.\,\?!\-]', '', text, flags=re.MULTILINE)
    return cleaned_text.strip()

def scrape_google_news_api(keyword: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Google Custom Search API를 사용하여 뉴스 기사를 수집하고 DB에 저장합니다.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Google API 키 또는 CSE ID가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return []

    all_articles = []
    
    logger.info(f"'{keyword}' 키워드로 Google Custom Search API 호출 시작. {num_results}개 결과 요청.")
    
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': keyword,
        'lr': 'lang_ko', 
        'num': num_results,
    }

    conn = None
    try:
        response = requests.get(GOOGLE_API_URL, params=params, headers={'User-Agent': 'MyApp/1.0'})
        response.raise_for_status() 
        
        response_json = response.json()
        
        if 'items' in response_json:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            articles_count = 0
            for item in response_json['items']:
                article = {
                    "source": item.get('displayLink', '출처 없음'),
                    "category": "모빌리티 트렌드", # 고정 값
                    "title": clean_text(item.get('title', '')),
                    "content": clean_text(item.get('snippet', '')),
                    "publish_date": datetime.now().isoformat(), # 현재 시간으로 저장
                    "url": item.get('link', ''),
                    "keywords": keyword
                }
                
                # DB에 기사 저장
                try:
                    cursor.execute('''
                        INSERT INTO articles (source, category, title, content, publish_date, url, keywords)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (article['source'], article['category'], article['title'], 
                          article['content'], article['publish_date'], article['url'], article['keywords']))
                    conn.commit()
                    all_articles.append(article) # 성공적으로 저장된 기사만 리스트에 추가
                    articles_count += 1
                except sqlite3.IntegrityError:
                    logger.warning(f"Duplicate URL found, skipping: {article['url']}")
                    # 중복 URL인 경우, 이미 저장된 기사이므로 건너뜀
                except sqlite3.Error as db_err:
                    logger.error(f"Error saving article to DB: {db_err} - {article['url']}", exc_info=True)
                    conn.rollback() # 오류 발생 시 롤백
            
            logger.success(f"'{keyword}' 키워드 API 호출 성공. 총 {articles_count}개 기사 DB에 저장 및 수집.")
        else:
            logger.warning(f"API 응답에 'items'가 없습니다. 결과가 없거나 API 응답 형식이 다릅니다: {response_json}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google API 호출 중 오류 발생: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"API 응답 처리 또는 DB 작업 중 예상치 못한 오류 발생: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

    return all_articles

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    test_key = os.getenv("GOOGLE_API_KEY")
    test_cse = os.getenv("GOOGLE_CSE_ID")
    
    # DB 초기화 함수 호출
    init_db() 

    print("--- Google Custom Search API 및 SQLite DB 테스트 ---")
    if test_key and test_cse:
        test_keyword = "자율주행 보험" # 테스트 키워드 변경
        collected_data = scrape_google_news_api(test_keyword, num_results=5)
        
        if collected_data:
            print(f"\n총 {len(collected_data)}개의 기사 수집 및 DB 저장 완료.")
            print("\nDB에서 저장된 기사 확인:")
            conn = None
            try:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                # 최근 저장된 기사 5개 조회
                cursor.execute("SELECT id, title, url FROM articles ORDER BY created_at DESC LIMIT 5")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"ID: {row[0]}, 제목: {row[1]}, URL: {row[2]}")
            except sqlite3.Error as e:
                print(f"DB 조회 중 오류 발생: {e}")
            finally:
                if conn:
                    conn.close()
        else:
            print("\n수집된 기사가 없거나 DB 저장에 실패했습니다. 키워드, API 설정 또는 DB 오류를 확인해주세요.")
    else:
        print("\n[경고] .env 파일에 GOOGLE_API_KEY 또는 GOOGLE_CSE_ID가 설정되지 않아 테스트를 실행할 수 없습니다.")

