# modules/report_generator.py
# '한 페이지 자동 요약 보고서'를 순수 텍스트(TXT) 파일로 생성하는 모듈 (보험 관련 사실 정보 추출)

from loguru import logger
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import time # 재시도를 위한 time 모듈 임포트
import re # 정규표현식 모듈 임포트

# AI 인터페이스 모듈 임포트
from modules.ai_interface import call_potens_api


def retry_ai_call(prompt: str, api_key: str, history: List[Dict[str, str]] = None, max_retries: int = 2, delay_seconds: int = 15) -> str:
    """
    Potens.dev API 호출에 대한 재시도 로직을 포함한 래퍼 함수.
    history 파라미터를 추가하여 멀티턴 대화 맥락을 전달할 수 있도록 함.
    """
    for attempt in range(max_retries):
        logger.info(f"AI 호출 시도 {attempt + 1}/{max_retries}...")
        response = call_potens_api(prompt, api_key=api_key, history=history) # history 전달
        
        # 성공적인 응답으로 간주하는 조건 (API 키 에러나 일반적인 실패 메시지가 아닌 경우)
        if "API 키가 설정되지 않았습니다" not in response and \
           "API 호출에 실패했습니다" not in response and \
           "알 수 없는 오류가 발생했습니다" not in response and \
           "API 응답 형식이 올바르지 않습니다" not in response:
            logger.success(f"AI 호출 성공 (시도 {attempt + 1}).")
            return response
        else:
            logger.warning(f"AI 호출 실패 (시도 {attempt + 1}): {response}. 재시도합니다...")
            if attempt < max_retries - 1:
                time.sleep(delay_seconds) # 실패 시 지연 후 재시도
    
    logger.error(f"AI 호출 최종 실패 후 {max_retries}회 재시도. 마지막 응답: {response}")
    return "AI 응답을 가져오는 데 최종 실패했습니다. 나중에 다시 시도해주세요."

def clean_ai_response_text(text: str) -> str:
    """
    AI 응답 텍스트에서 불필요한 마크다운 기호, 여러 줄바꿈,
    그리고 AI가 자주 사용하는 서두 문구들을 제거하여 평탄화합니다.
    """
    # 마크다운 헤더, 리스트 기호, 볼드체/이탤릭체 기호 등 제거
    cleaned_text = re.sub(r'#|\*|-|\+', '', text)
    # AI가 자주 사용하는 서두 문구 제거 (예: '제공해주신 텍스트를 요약하겠습니다. 요약: ')
    # 정규표현식으로 유연하게 매칭 (띄어쓰기, 조사 변화 등 고려)
    patterns_to_remove = [
        r'제공해주신\s*텍스트를\s*요약\s*하겠\s*습니다[.:\s]*\s*요약[.:\s]*',
        r'요약해\s*드리겠습니다[.:\s]*\s*주요\s*내용\s*요약[.:\s]*',
        r'다음\s*텍스트의\s*요약입니다[.:\s]*',
        r'주요\s*내용을\s*요약\s*하면\s*다음과\s*같습니다[.:\s]*',
        r'핵심\s*내용은\s*다음과\s*같습니다[.:\s]*',
        r'요약하자면[.:\s]*',
        r'주요\s*요약[.:\s]*',
        r'텍스트를\s*요약하면\s*다음과\s*같습니다[.:\s]*', 
        r'제공된\s*텍스트에\s*대한\s*요약입니다[.:\s]*',
        r'다음은\s*ai가\s*내용을\s*요약한\s*것입니다[.:\s]*',
        r'먼저\s*최신\s*정보가\s*필요합니다[.:\s]*\s*현재\s*자율주행차\s*기술과\s*관련된\s*최신\s*트렌드를\s*확인해보겠습니다[.:\s]*' 
    ]
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text)

    # 여러 개의 줄바꿈을 하나의 공백으로 대체
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
    # 여러 개의 공백을 하나로 대체
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def summarize_content_in_chunks(
    content: str, 
    api_key: str, 
    chunk_size: int = 350, 
    delay_between_chunks: int = 3
) -> str:
    """
    긴 텍스트를 작은 청크로 나누어 순차적으로 AI에 요약을 요청하고,
    각 요약을 독립적으로 생성하여 합칩니다. (멀티턴 맥락 유지 안 함)
    """
    if not content:
        return ""

    logger.info(f"Starting chunked summarization for content (total length: {len(content)}). Chunk size: {chunk_size}")
    
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    chat_history = []
    final_summary_parts = []

    for i, chunk in enumerate(chunks):
        # 모든 프롬프트를 한글로 변경
        prompt = f"다음 텍스트를 요약해 주세요. 텍스트: {chunk}"
        if i == 0:
            prompt = f"다음 텍스트를 요약해 주세요. 텍스트: {chunk}" # 첫 번째 청크는 이전 내용 없음

        # history는 이전 AI 응답을 포함하여 AI에게 맥락을 제공
        response = retry_ai_call(prompt, api_key=api_key, history=chat_history)
        
        if "AI 응답을 가져오는 데 최종 실패했습니다" in response:
            logger.error(f"Failed to summarize chunk {i+1}. Skipping remaining chunks.")
            final_summary_parts.append(f"[청크 {i+1} 요약 실패]")
            break # 실패 시 나머지 청크 건너뛰기
        
        final_summary_parts.append(response)
        
        # chat_history 업데이트: AI의 이전 응답과 현재 사용자 프롬프트를 추가
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response})
        
        logger.info(f"Chunk {i+1}/{len(chunks)} summarized. Waiting {delay_between_chunks} seconds...")
        time.sleep(delay_between_chunks) # 각 청크 요약 후 지연

    # 모든 청크 요약본을 합쳐서 최종 요약으로 반환
    return " ".join(final_summary_parts)


def create_single_page_report(
    trend_name: str,
    trend_detection_reason: str,
    related_articles: List[Dict[str, Any]],
    api_key: str, # AI 요약을 위한 API 키
    max_articles_for_ai_summary: int = 3, # AI 요약에 사용할 최대 기사 수 제한
    delay_between_ai_calls: int = 20 # AI 호출 사이의 지연 시간 (초)
) -> str: # report_format 파라미터 제거
    """
    포착된 새로운 트렌드에 대한 한 페이지 분량의 핵심 트렌드 요약 보고서를 순수 텍스트(TXT) 파일로 생성합니다.
    Args:
        trend_name: 포착된 트렌드의 핵심 키워드/주제 (예: "전고체 배터리 상용화 임박").
        trend_detection_reason: 트렌드가 감지된 주요 이유 (예: "최근 1주일간 관련 키워드 언급량 300% 증가").
        related_articles: 해당 트렌드와 관련된 뉴스 기사 목록 (DB에서 로드된 형태).
        api_key: AI 요약을 위한 Potens.dev API 키.
        max_articles_for_ai_summary: AI 요약을 위해 전달할 최대 기사 수.
        delay_between_ai_calls: 각 AI 호출 성공 후 대기할 시간 (초).
    Returns:
        생성된 TXT 파일의 경로.
    """
    logger.info(f"Generating single-page report for trend: '{trend_name}' in TXT format.")
    
    # --- AI 요약을 위해 사용할 원본 기사 내용 준비 ---
    articles_for_ai = related_articles[:max_articles_for_ai_summary]
    
    concatenated_original_content_for_final_ai = ""
    for i, article in enumerate(articles_for_ai): 
        concatenated_original_content_for_final_ai += f"뉴스 {i+1} 요약: " 
        concatenated_original_content_for_final_ai += article['content'][:300] 
        concatenated_original_content_for_final_ai += "...\n\n"
    
    final_ai_input_content = summarize_content_in_chunks(
        content=concatenated_original_content_for_final_ai,
        api_key=api_key,
        chunk_size=350, 
        delay_between_chunks=delay_between_ai_calls # 청크 요약에도 더 긴 지연 적용
    )

    # 최종 AI 입력 내용에서 불필요한 서두 문구 및 마크다운, 줄바꿈 등을 제거하여 평탄화
    cleaned_final_ai_input_content = clean_ai_response_text(final_ai_input_content) 
    logger.debug(f"Cleaned final AI input content (after chunking): {cleaned_final_ai_input_content[:200]}...") 


    # --- AI 요약 내용 생성 ---
    # 핵심 요약 (AI 생성)
    summary_prompt = (
        f"당신은 모빌리티 트렌드 및 보험 개발 전문가입니다. "
        f"다음 핵심 뉴스 내용을 바탕으로 주요 트렌드, 배경, 그리고 산업에 미칠 핵심적인 영향(위험 또는 기회)을 2-3문장으로 간결하게 요약해 주세요. "
        f"내용: {cleaned_final_ai_input_content}" 
    )
    
    ai_core_summary_raw = retry_ai_call(summary_prompt, api_key=api_key) 
    logger.debug(f"AI 핵심 요약 (원본): {ai_core_summary_raw[:200]}...") 
    ai_core_summary = clean_ai_response_text(ai_core_summary_raw) 
    if "AI 응답을 가져오는 데 최종 실패했습니다" in ai_core_summary_raw: 
        ai_core_summary = "AI 핵심 요약을 불러오는 데 실패했습니다. Potens.dev 서버 상태를 확인해주세요."
        logger.error(f"Failed to get AI core summary for '{trend_name}'.")
    time.sleep(delay_between_ai_calls) 

    # 트렌드 상세 내용 (AI 생성)
    detail_prompt = (
        f"당신은 모빌리티 트렌드 전문가입니다. "
        f"다음 핵심 뉴스 내용을 바탕으로 '{trend_name}' 트렌드에 대해 5-7문장으로 자세히 설명해 주세요. "
        f"기술적 특징, 개발 현황 등에 초점을 맞춰 주세요. "
        f"내용: {cleaned_final_ai_input_content}" 
    )
    ai_detailed_content_raw = retry_ai_call(detail_prompt, api_key=api_key) 
    logger.debug(f"AI 상세 내용 (원본): {ai_detailed_content_raw[:200]}...") 
    ai_detailed_content = clean_ai_response_text(ai_detailed_content_raw) 
    if "AI 응답을 가져오는 데 최종 실패했습니다" in ai_detailed_content_raw: 
        ai_detailed_content = "AI 상세 내용을 불러오는 데 실패했습니다. Potens.dev 서버 상태를 확인해주세요."
        logger.error(f"Failed to get AI detailed content for '{trend_name}'.")
    time.sleep(delay_between_ai_calls) 

    # 보험 개발 시사점 (AI 분석) - 역할 변경: 이제 '보험 관련 주요 사실 요약'
    # 프롬프트 내용을 사용자 요청에 따라 변경: 한글로, 보험 개발 시사점 도출 후 정리
    insurance_prompt = ( 
        f"'{trend_name}' 트렌드와 관련된 보험 및 법적 책임에 대한 주요 사실들을 요약해 주세요. " # <-- 변경된 프롬프트
        f"한국어로 요약 내용을 제공해 주세요. "
        f"내용: {cleaned_final_ai_input_content}" 
    )
    logger.debug(f"AI 보험 관련 사실 요약 프롬프트 (전송): {insurance_prompt}") # 로그 메시지 변경
    ai_insurance_implications_raw = retry_ai_call(insurance_prompt, api_key=api_key) 
    logger.debug(f"AI 보험 관련 사실 요약 (원본): {ai_insurance_implications_raw[:200]}...") # 로그 메시지 변경
    ai_insurance_implications = clean_ai_response_text(ai_insurance_implications_raw) 
    if "AI 응답을 가져오는 데 최종 실패했습니다" in ai_insurance_implications_raw: 
        ai_insurance_implications = "AI 보험 관련 사실 요약을 불러오는 데 실패했습니다. Potens.dev 서버 상태를 확인해주세요." # 오류 메시지 변경
        logger.error(f"Failed to get AI insurance related facts for '{trend_name}'.") # 로그 메시지 변경
    time.sleep(delay_between_ai_calls) 

    # --- 보고서 내용 구성 (TXT 형식) ---
    report_content_parts = []
    report_content_parts.append(f"[긴급 보고] {datetime.now().strftime('%Y년 %m월')} 미래 모빌리티 핵심 트렌드 분석: {trend_name}\n\n")
    report_content_parts.append("Executive Summary / 핵심 요약\n")
    report_content_parts.append(f"트렌드 명: {trend_name}\n")
    report_content_parts.append(f"핵심 요약 (AI 생성): {ai_core_summary}\n")
    report_content_parts.append(f"감지 근거: {trend_detection_reason}\n\n")
    report_content_parts.append("심층 분석 및 시사점\n")
    report_content_parts.append(f"트렌드 상세 내용 (AI 생성): {ai_detailed_content}\n")
    report_content_parts.append(f"보험 개발 시사점 (AI 분석): {ai_insurance_implications}\n\n") # 섹션명은 유지 (사용자가 시사점으로 해석)
    report_content_parts.append("근거 자료\n")

    # 근거 자료 표 내용 (TXT 형식으로 변환)
    report_content_parts.append("뉴스 번호 | 제목 | 언론사 | 발행일 | 링크\n")
    report_content_parts.append("---|---|---|---|---\n")
    for i, article in enumerate(related_articles):
        publish_date_str = article.get('publish_date', '')
        formatted_date = ""
        try:
            formatted_date = datetime.fromisoformat(publish_date_str).strftime("%Y.%m.%d")
        except ValueError:
            formatted_date = publish_date_str 
        report_content_parts.append(f"[뉴스 {i+1}] | {article.get('title', 'N/A')} | {article.get('source', 'N/A')} | {formatted_date} | {article.get('url', 'N/A')}\n")

    # 파일 저장 (TXT 형식만 지원)
    output_filename = f"HEART_Insight_Report_{trend_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt" # 확장자 .txt 고정
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("".join(report_content_parts)) # 모든 내용을 문자열로 합쳐서 쓰기
    
    logger.success(f"Report '{output_filename}' generated successfully.")
    return output_filename

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    print("--- Report Generator 모듈 테스트 ---")
    
    # 필요한 모듈 임포트 (테스트용)
    from modules.trend_detector import get_articles_from_db, detect_trending_keywords, get_articles_by_keywords
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("POTENS_API_KEY")

    if not API_KEY:
        print("[경고] API 키가 설정되지 않아 AI 요약 기능을 테스트할 수 없습니다. .env 파일을 확인해주세요.")
    
    # 1. DB에서 기사 로드
    all_recent_articles = get_articles_from_db(days_ago=30)
    
    if all_recent_articles:
        # 2. 트렌딩 키워드 감지
        trending_keywords = detect_trending_keywords(all_recent_articles, lookback_days=7, threshold_percent_increase=50.0)
        
        if trending_keywords:
            test_trend_keyword = trending_keywords[0]['keyword'] 
            test_trend_reason = f"키워드 '{test_trend_keyword}' 언급량 급증 (현재 {trending_keywords[0]['current_mentions']}회, 이전 {trending_keywords[0]['previous_mentions']}회, 증가율 {trending_keywords[0]['percent_increase']}%)"
            
            # 3. 해당 트렌드와 관련된 기사 로드
            test_related_articles = get_articles_by_keywords([test_trend_keyword], days_ago=14)
            
            if test_related_articles and API_KEY:
                # 4. 보고서 생성 (TXT 형식으로만)
                print(f"\n'{test_trend_keyword}' 트렌드에 대한 보고서 생성 시작 (TXT 형식)...")
                report_path_txt = create_single_page_report(
                    trend_name=f"'{test_trend_keyword}' 관련 트렌드",
                    trend_detection_reason=test_trend_reason,
                    related_articles=test_related_articles,
                    api_key=API_KEY, 
                    max_articles_for_ai_summary=3, 
                    delay_between_ai_calls=20 # 각 AI 호출 성공 후 20초 지연
                )
                print(f"TXT 보고서가 다음 경로에 생성되었습니다: {report_path_txt}")

            elif not API_KEY:
                print("API 키가 없어 보고서 생성 시 AI 요약을 건너뛰었습니다.")
            else:
                print(f"'{test_trend_keyword}' 관련 기사를 찾을 수 없어 보고서를 생성할 수 없습니다.")
        else:
            print("\n감지된 트렌딩 키워드가 없어 보고서를 생성할 수 없습니다. 더 많은 데이터를 수집하거나 기간/임계값을 조정해보세요.")
    else:
        print("\n데이터베이스에 충분한 기사가 없어 트렌드 감지 및 보고서 생성을 수행할 수 없습니다.")

