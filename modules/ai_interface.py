import pandas as pd
import requests
import os
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any
import json # JSON 로깅을 위해 임포트

# .env 파일에서 환경 변수 로드
load_dotenv()

# Potens.dev API 엔드포인트 URL
API_URL = "https://ai.potens.ai/api/chat"

def call_potens_api(prompt_message: str, api_key: str, history: List[Dict[str, str]] = None) -> str:
    """
    주어진 프롬프트 메시지로 Potens.dev API를 호출하고 응답을 반환합니다.
    API 호출 타임아웃 시간을 5분(300초)으로 늘려 안정성을 개선합니다.
    오류 발생 시 응답 본문을 로깅합니다.
    """
    if not api_key:
        logger.error("API 키가 누락되었습니다. Potens.dev API 호출을 중단합니다.")
        return "API 키가 설정되지 않았습니다. 관리자에게 문의해주세요."

    # HTTP 헤더 설정 (인증 토큰과 데이터 타입 명시)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 요청 페이로드(본문) 설정 (멀티턴 대화 맥락을 'prompt' 메시지에 포함하여 전달)
    full_prompt = ""
    if history:
        for msg in history:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"user: {prompt_message}"
    else:
        full_prompt = prompt_message
    
    payload = {
        "prompt": full_prompt # AI에게 전달할 핵심 메시지
    }

    try:
        logger.info(f"Potens.dev API 호출 시작 (엔드포인트: {API_URL})")
        # --- 디버깅을 위해 전송될 payload 로깅 ---
        logger.debug(f"Sending payload: {json.dumps(payload, ensure_ascii=False, indent=2)}") # <-- 추가된 부분
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300) 
        response.raise_for_status() # HTTP 오류 시 예외 발생
        
        response_json = response.json()
        
        # 응답 파싱 및 반환 (공유해주신 {'message': '...'} 구조 사용)
        if "message" in response_json:
            ai_response_text = response_json["message"].strip()
            logger.info("Potens.dev API 호출 성공.")
            return ai_response_text
        else:
            logger.error(f"API 응답에 'message' 키가 없습니다: {response_json}")
            return "API 응답 형식이 올바르지 않습니다. 개발자에게 문의해주세요."

    except requests.exceptions.RequestException as e:
        # 오류 발생 시 응답 본문 로깅 추가
        error_message = f"API 호출 오류 발생 (네트워크/타임아웃/HTTP): {e}"
        if e.response is not None:
            error_message += f" Response content: {e.response.text}" 
        logger.error(error_message, exc_info=True)
        return "API 호출에 실패했습니다. 네트워크 상태를 확인해주세요."
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
        return "알 수 없는 오류가 발생했습니다."

# --- AI 토픽 요약 기능 추가 ---
def get_topic_summaries_from_ai(topic_info_data: List[Dict[str, Any]], api_key: str) -> pd.DataFrame:
    """
    토픽 정보 데이터를 받아 AI API를 호출하여 각 토픽의 의미를 요약합니다.
    """
    logger.info("AI를 통해 토픽 의미 요약을 시작합니다.")
    topic_summaries = []
    
    for topic_info in topic_info_data:
        topic_id = topic_info['Topic']
        keywords = topic_info['Keywords']
        
        prompt = f"You are an expert in mobility trends. Summarize the following topic based on its keywords. The topic keywords are: {keywords}. Provide a concise summary in Korean, less than 20 words."
        
        summary = call_potens_api(prompt, api_key=api_key)
        
        topic_summaries.append({
            "Topic": topic_id,
            "Keywords": keywords,
            "AI Summary": summary
        })
        
    logger.success("AI 토픽 요약이 완료되었습니다.")
    return pd.DataFrame(topic_summaries)


if __name__ == '__main__':
    # 모듈 테스트를 위한 코드 (Potens.dev API 테스트)
    test_api_key = os.getenv("POTENS_API_KEY")
    print("--- Potens.dev API 호출 테스트 ---")
    if test_api_key:
        # call_potens_api 함수 테스트
        test_query = "미래 모빌리티 트렌드에 대해 간략히 설명해줘."
        response_text = call_potens_api(test_query, api_key=test_api_key)
        print(f"\nAI 답변 (call_potens_api):\n{response_text}")

        # get_topic_summaries_from_ai 함수 테스트
        print("\n--- AI 토픽 요약 기능 테스트 ---")
        test_topic_data = [
            {"Topic": 0, "Keywords": "0.047*car + 0.025*기능 + 0.025*시스템 + 0.024*data"},
            {"Topic": 1, "Keywords": "0.034*수단 + 0.023*통합 + 0.023*교통 + 0.023*서비스 + 0.013*도시"}
        ]
        summaries_df = get_topic_summaries_from_ai(test_topic_data, api_key=test_api_key)
        print("\nAI 요약 결과:")
        print(summaries_df)
    else:
        print("\n[경고] .env 파일에 POTENS_API_KEY가 설정되지 않아 테스트를 실행할 수 없습니다.")

