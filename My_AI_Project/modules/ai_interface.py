# ai_interface.py
# Potens.dev API 연동을 위한 모듈

import os
import requests
import json
from loguru import logger
from typing import List, Dict, Any
from dotenv import load_dotenv

def call_potens_api(user_query: str, api_key: str, model_name: str = "potens-llm-model", history: List[Dict[str, str]] = None) -> str:
    """
    Potens.dev API를 호출하여 AI 응답을 생성합니다.
    
    Args:
        user_query (str): 사용자의 질문.
        api_key (str): Potens.dev API 키.
        model_name (str): 사용할 AI 모델 이름. (기획안에서 언급된 모델명으로 가정)
        history (List[Dict[str, str]]): 이전 대화 기록 (role, content).
        
    Returns:
        str: AI가 생성한 답변 텍스트.
    """
    if not api_key:
        logger.error("API 키가 누락되었습니다. Potens.dev API 호출을 중단합니다.")
        return "API 키가 설정되지 않았습니다. 관리자에게 문의해주세요."

    # Potens.dev API Chat Completions 엔드포인트 URL (실제 API 문서 기반으로 수정 필요)
    API_BASE_URL = "https://api.potens.dev/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 기획안의 페르소나를 시스템 프롬프트에 명시하여 답변 품질을 높입니다.
    system_prompt = (
        "당신은 현대해상 미래 모빌리티 트렌드 분석 및 보험 시사점 도출 전문가 AI 'HEART Insight AI'입니다. "
        "자율주행, UAM(도심항공교통), PBV 등 미래 모빌리티 관련 최신 트렌드를 분석하고, "
        "그에 따른 보험 상품 및 리스크 변화에 대한 핵심 시사점을 중점적으로 답변해주세요. "
        "답변은 객관적이고 전문적인 어조로, 현대해상의 비즈니스 관점에서 중요한 통찰력을 제공해야 합니다."
    )
    
    # API 요청에 사용할 메시지 리스트 구성 (시스템 메시지 + 대화 기록 + 사용자 질문)
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # Streamlit 메시지 형식을 API 요구사항에 맞게 변환
        api_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.extend(api_history)
        
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.5, # 답변의 창의성 제어 (0: 보수적, 1: 창의적)
        "max_tokens": 1024, # 최대 생성 토큰 수
        "stream": False # 스트리밍(실시간 답변) 기능 사용 여부
    }
    
    try:
        logger.info(f"Potens.dev API 호출 시작 (모델: {model_name})")
        response = requests.post(API_BASE_URL, headers=headers, data=json.dumps(data), timeout=120)
        response.raise_for_status() # HTTP 상태 코드가 200번대가 아니면 예외 발생
        
        response_json = response.json()
        
        # API 응답 구조에 따라 답변 추출 (안정성을 위해 .get() 사용)
        ai_response_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '죄송합니다. 답변을 생성하는 데 문제가 발생했습니다.')
        
        logger.info("Potens.dev API 호출 성공.")
        return ai_response_text
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP 오류 발생: {http_err.response.status_code} - {http_err.response.text}", exc_info=True)
        return f"API 호출 중 HTTP 오류가 발생했습니다: {http_err.response.status_code}. 응답 내용: {http_err.response.text[:100]}..."
    except requests.exceptions.RequestException as req_err:
        logger.error(f"API 요청 중 네트워크 오류 발생: {req_err}", exc_info=True)
        return "네트워크 오류로 API에 연결할 수 없습니다. 인터넷 연결을 확인해주세요."
    except json.JSONDecodeError as json_err:
        logger.error(f"JSON 응답 디코딩 오류: {json_err}", exc_info=True)
        return "API 응답을 처리하는 데 문제가 발생했습니다. 응답 형식이 올바르지 않습니다."
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
        return "예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

if __name__ == '__main__':
    # 모듈 자체 테스트를 위한 코드
    load_dotenv()
    test_api_key = os.getenv("POTENS_API_KEY")
    print("--- Potens.dev API 호출 테스트 ---")
    if test_api_key:
        test_query = "자율주행 레벨 4의 상용화가 보험 산업에 미치는 가장 큰 시사점은 무엇인가요?"
        response_text = call_potens_api(test_query, api_key=test_api_key)
        print(f"\nAI 답변:\n{response_text}")
    else:
        print("\n[경고] .env 파일에 POTENS_API_KEY가 설정되지 않아 테스트를 실행할 수 없습니다.")