# modules/trend_analyzer.py

import pandas as pd
from loguru import logger
from typing import List, Dict, Any

# Gensim & pyLDAvis 관련 라이브러리 임포트
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 한국어 전처리 라이브러리
from konlpy.tag import Okt

# Okt 객체 초기화
try:
    okt = Okt()
    logger.info("KoNLPy Okt initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Okt: {e}. Please check your KoNLPy and Java (JDK) installation.", exc_info=True)
    okt = None

def preprocess_korean_text(text_list: List[str]) -> List[List[str]]:
    """
    한국어 텍스트 리스트를 받아 명사만 추출하고 토큰화된 리스트의 리스트를 반환합니다.
    (Gensim에 맞게 출력을 수정)
    """
    if okt is None:
        logger.warning("Okt is not available. Skipping text preprocessing.")
        # Gensim은 단어로 분리된 리스트를 기대하므로, 최소한의 분리라도 수행
        return [text.split() for text in text_list]
        
    logger.info("Starting Korean text preprocessing using Okt...")
    processed_texts = []
    stopwords = ['것', '수', '등', '등등'] 
    
    for text in text_list:
        try:
            nouns = okt.nouns(text)
            filtered_words = [word for word in nouns if word not in stopwords and len(word) > 1]
            processed_texts.append(filtered_words)
        except Exception as e:
            logger.warning(f"Failed to preprocess text with Okt: {e}. Using original split text.")
            processed_texts.append(text.split())
            
    logger.info("Korean text preprocessing finished.")
    return processed_texts

def perform_topic_modeling(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    수집된 뉴스 기사를 기반으로 Gensim LDA 모델링 및 pyLDAvis 시각화를 수행합니다.
    """
    if not articles:
        logger.warning("No articles provided for topic modeling.")
        return {"topic_info": pd.DataFrame().to_dict(), "fig_html": None}
        
    logger.info(f"Starting topic modeling with Gensim LDA on {len(articles)} documents...")
    
    df = pd.DataFrame(articles)
    documents = df['content'].tolist()
    
    tokenized_docs = preprocess_korean_text(documents)
    
    try:
        # 1. Dictionary 생성
        id2word = corpora.Dictionary(tokenized_docs)
        
        # 2. Corpus 생성 (Term Document Frequency)
        corpus = [id2word.doc2bow(text) for text in tokenized_docs]
        
        # 3. LDA 모델 학습
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           passes=10,
                                           alpha='auto')
                                           
        logger.success("Gensim LDA topic modeling completed successfully.")
        
        # 4. pyLDAvis 시각화 생성
        vis_data = gensimvis.prepare(lda_model, corpus, id2word)
        fig_html = pyLDAvis.prepared_data_to_html(vis_data)
        logger.info("pyLDAvis visualization generated successfully.")
        
        # 5. 결과 포맷 맞추기 (표로 보여줄 데이터)
        topics = lda_model.print_topics()
        topic_info_data = []
        for t_id, topic_words in topics:
            topic_info_data.append({
                "Topic": t_id,
                "Keywords": topic_words.replace('"', '')
            })
        topic_info_df = pd.DataFrame(topic_info_data)
        
        return {
            "topic_info": topic_info_df.to_dict(),
            "fig_html": fig_html
        }
        
    except Exception as e:
        logger.error(f"An error occurred during Gensim LDA modeling: {e}", exc_info=True)
        return {"topic_info": {}, "fig_html": "<div style='color:red;'>Failed to perform topic modeling. See terminal logs.</div>"}