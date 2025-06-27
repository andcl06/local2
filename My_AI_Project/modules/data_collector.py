# data_collector.py
# Google Custom Search API를 사용하여 데이터를 수집하는 모듈

import requests
import os
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any
from datetime import datetime
import re

# .env 파일에서 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Google Custom Search API 엔드포인트
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"

def clean_text(text: str) -> str:
    """텍스트 정제 함수"""
    cleaned_text = re.sub(r'<.*?>|http\S+|www.\S+|\S*@\S*\s?|[^\w\s\.\,\?!\-]', '', text, flags=re.MULTILINE)
    return cleaned_text.strip()

def scrape_google_news_api(keyword: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Google Custom Search API를 사용하여 뉴스 기사를 수집합니다.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Google API 키 또는 CSE ID가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return []

    all_articles = []
    
    logger.info(f"'{keyword}' 키워드로 Google Custom Search API 호출 시작. {num_results}개 결과 요청.")
    
    # 요청 파라미터 설정
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': keyword,
        # 'searchType': 'news', # 뉴스만 검색 (주석 처리된 상태 유지)
        'lr': 'lang_ko',       # 언어설정
        'num': num_results,    # 가져올 결과 수 (최대 10개)
    }

    try:
        # API GET 요청 보내기
        response = requests.get(GOOGLE_API_URL, params=params, headers={'User-Agent': 'MyApp/1.0'})
        response.raise_for_status() # HTTP 오류 시 예외 발생
        
        response_json = response.json()
        
        if 'items' in response_json:
            for item in response_json['items']:
                article = {
                    "source": item.get('displayLink', '출처 없음'),
                    "category": "모빌리티 트렌드",
                    "title": clean_text(item.get('title', '')),
                    "content": clean_text(item.get('snippet', '')),
                    "publish_date": datetime.now().isoformat(),
                    "url": item.get('link', ''),
                    "keywords": keyword
                }
                all_articles.append(article)
            logger.success(f"'{keyword}' 키워드 API 호출 성공. 총 {len(all_articles)}개 기사 수집.")
        else:
            logger.warning(f"API 응답에 'items'가 없습니다. 결과가 없거나 API 응답 형식이 다릅니다: {response_json}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google API 호출 중 오류 발생: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"API 응답 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)

    return all_articles

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    test_key = os.getenv("GOOGLE_API_KEY")
    test_cse = os.getenv("GOOGLE_CSE_ID")
    print("--- Google Custom Search API 테스트 ---")
    if test_key and test_cse:
        test_keyword = "self-driving car liability"
        collected_data = scrape_google_news_api(test_keyword, num_results=5)
        if collected_data:
            print(f"\n총 {len(collected_data)}개의 기사 수집 완료.")
            for i, article in enumerate(collected_data[:5]):
                print(f"\n--- 기사 {i+1} ---")
                print(f"제목: {article['title']}")
                print(f"링크: {article['url']}")
                print(f"요약: {article['content'][:100]}...")
                print(f"출처: {article['source']}")
                print("-" * 20)
        else:
            print("\n수집된 기사가 없습니다. 키워드나 API 설정을 확인해주세요.")
    else:
        print("\n[경고] .env 파일에 GOOGLE_API_KEY 또는 GOOGLE_CSE_ID가 설정되지 않아 테스트를 실행할 수 없습니다.")