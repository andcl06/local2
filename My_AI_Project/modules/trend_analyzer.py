# trend_analyzer.py
# NLP 기반 트렌드 분석 및 토픽 모델링 모듈 (최종 버전)

import pandas as pd
from loguru import logger
from typing import List, Dict, Any
import numpy as np
import umap # <-- umap 임포트 추가
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from konlpy.tag import Mecab

# -----------------------------------------------------
# 1. 한국어 텍스트 분석을 위한 KoNLPy-Mecab 초기화
# -----------------------------------------------------
try:
    mecab = Mecab()
    logger.info("KoNLPy Mecab initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Mecab: {e}. Please check your KoNLPy and Mecab installation.", exc_info=True)
    mecab = None

def preprocess_korean_text(text_list: List[str]) -> List[str]:
    """
    한국어 텍스트를 토큰화하고 불용어를 제거하여 전처리합니다.
    """
    if mecab is None:
        logger.warning("Mecab is not available. Skipping text preprocessing and tokenization.")
        return text_list
        
    logger.info("Starting Korean text preprocessing...")
    processed_texts = []
    
    stopwords = ['은', '는', '이', '가', '을', '를', '에', '와', '과', '하다', '있다', '없다', '되다', '이다', '것', '수', '등', '등등']
    
    for text in text_list:
        try:
            words = mecab.pos(text)
            filtered_words = [
                word[0] for word in words 
                if word[1] in ['NNG', 'NNP', 'VV', 'VA'] and word[0] not in stopwords
            ]
            processed_texts.append(' '.join(filtered_words))
        except Exception as e:
            logger.warning(f"Failed to preprocess text with Mecab: {e}. Skipping this document.")
            processed_texts.append(text)
        
    logger.info("Korean text preprocessing finished.")
    return processed_texts

def perform_topic_modeling(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    수집된 뉴스 기사를 기반으로 BERTopic 모델링을 수행합니다.
    - 시각화 생성 시 발생하는 torch 오류를 회피하는 로직이 포함됩니다.
    """
    if not articles:
        logger.warning("No articles provided for topic modeling.")
        return {"topics": [], "topic_info": pd.DataFrame().to_dict(), "fig_html": None}
        
    logger.info(f"Starting topic modeling with BERTopic on {len(articles)} documents...")
    
    df = pd.DataFrame(articles)
    documents = df['content'].tolist()
    
    preprocessed_documents = preprocess_korean_text(documents)
    
    # BERTopic 모델 초기화 및 학습
    try:
        # Sentence-Transformer 모델을 한국어 모델로 명시적으로 설정
        embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        
        # BERTopic 모델 초기화 (representation_model은 KeyBERTInspired 사용)
        representation_model = KeyBERTInspired()
        
        # --- UMAP 모델 파라미터 동적 계산 (scipy 오류 해결) ---
        num_docs = len(preprocessed_documents)
        n_neighbors = min(num_docs - 1, 15) if num_docs > 1 else 1
        
        topic_model = BERTopic(
            language="multilingual",
            calculate_probabilities=True,
            embedding_model=embedding_model,
            representation_model=representation_model,
            umap_model=umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, metric='cosine', random_state=42), # <-- 수정된 부분
            verbose=True,
        )
        
        # 모델 학습 및 토픽 추출
        topics, probabilities = topic_model.fit_transform(preprocessed_documents)
        
        # 토픽 정보 추출
        topic_info = topic_model.get_topic_info()
        
        logger.success(f"Topic modeling completed successfully. Found {len(topic_info)} topics.")
        
        # --- 2. 시각화 생성에 대한 예외 처리 ---
        fig_html = None
        try:
            fig = topic_model.visualize_topics()
            fig_html = fig.to_html(full_html=False)
            logger.info("Topic visualization generated successfully.")
        except Exception as vis_e:
            logger.error(f"Failed to generate visualization (likely a backend/torch issue): {vis_e}", exc_info=True)
            fig_html = "<div style='color:red;'>Failed to generate visualization. See terminal logs for details.</div>"
            
        return {
            "topics": topics,
            "topic_info": topic_info.to_dict(),
            "fig_html": fig_html
        }
        
    except Exception as e:
        logger.error(f"An error occurred during BERTopic modeling: {e}", exc_info=True)
        return {"topics": [], "topic_info": {}, "fig_html": "<div style='color:red;'>Failed to perform topic modeling. See terminal logs.</div>"}

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    print("--- BERTopic 모델링 테스트 ---")
    test_articles = [
        {"title": "전기차 배터리 화재 예방 기술", "content": "최근 전기차 배터리 열 폭주를 막는 기술 개발이 중요해지고 있습니다. 이는 보험 산업의 새로운 리스크 요인으로 부상합니다.", "source": "test_data"},
        {"title": "자율주행 레벨4 보험 상품 개발 동향", "content": "자율주행차의 사고 책임 소재를 운전자에서 제조사로 바꾸는 보험 상품이 연구되고 있습니다. 현대해상 같은 보험사에게 새로운 기회입니다.", "source": "test_data"},
        {"title": "도심항공교통(UAM) 실증 사업 시작", "content": "국토교통부가 서울 도심에서 UAM 실증 비행을 성공적으로 마쳤습니다. UAM 관련 보험 상품의 필요성이 커지고 있습니다.", "source": "test_data"},
        {"title": "전기차 충전 인프라 확대 정책 발표", "content": "정부에서 2030년까지 전기차 충전소 확대를 위한 대규모 예산을 투입합니다. 이는 전기차 보험 시장의 성장을 가속화할 것입니다.", "source": "test_data"},
        {"title": "PBV 시장의 새로운 비즈니스 모델", "content": "현대차와 기아는 목적 기반 모빌리티(PBV)를 통해 다양한 서비스를 제공할 계획입니다. 이에 맞는 맞춤형 보험 상품이 필요합니다.", "source": "test_data"},
        {"title": "자율주행차 사이버 보안 위협 증가", "content": "커넥티드카에 대한 해킹 위협이 증가하면서 새로운 형태의 사이버 보험이 주목받고 있습니다. 현대해상은 이에 대한 대비가 필요합니다.", "source": "test_data"},
        {"title": "전기차 화재 진압용 소방 장비", "content": "전기차 배터리 화재는 기존 소화 방식과 달라 특수 진압 장비가 필요합니다. 이는 손해율에 영향을 미칠 수 있습니다.", "source": "test_data"},
        {"title": "전기차 충전소 인프라 확대", "content": "전국에 전기차 충전소 인프라가 빠르게 확산되고 있습니다. 이는 보험 상품의 리스크 평가에 새로운 요인으로 작용합니다.", "source": "test_data"},
    ]
    
    result = perform_topic_modeling(test_articles)
    
    if result["topics"]:
        print("\n--- 토픽 모델링 결과 ---")
        topic_info_df = pd.DataFrame(result["topic_info"])
        print(topic_info_df[['Topic', 'Count', 'Name', 'Representation']].head())
        if result["fig_html"]:
            print("\n토픽 시각화 HTML이 생성되었습니다. 브라우저에서 확인할 수 있습니다.")
    else:
        print("\n토픽 모델링에 실패했습니다. 로그를 확인해주세요.")