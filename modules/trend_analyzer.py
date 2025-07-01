# modules/trend_analyzer.py (최종 업그레이드 버전)

import pandas as pd
from loguru import logger
from typing import List, Dict, Any

import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --- 라이브러리 추가 ---
from konlpy.tag import Okt
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# --- NLTK 데이터 다운로드 (최초 1회 필요) ---
try:
    # `nltk.downloader.DownloadError` 대신 표준 에러인 `LookupError`를 사용합니다.
    nltk.data.find('corpora/stopwords')
except LookupError: # <--- 수정된 부분
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError: # <--- 수정된 부분
    logger.info("Downloading NLTK wordnet...")
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # <--- 수정된 부분
    logger.info("Downloading NLTK punkt...")
    nltk.download('punkt')
# ---------------------------------------------# ---------------------------------------------

# Okt 객체 및 영어 처리 도구 초기화
okt = Okt()
stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text_list: List[str]) -> List[List[str]]:
    """
    텍스트 리스트를 받아, 각 텍스트의 언어를 감지하고 언어에 맞는 전처리를 수행합니다.
    """
    logger.info("Starting text preprocessing with language detection...")
    processed_texts = []
    
    # 한국어 불용어
    stopwords_kr = ['것', '수', '등', '등등', '기자', '뉴스', '연합뉴스', '사진'] 

    for text in text_list:
        try:
            lang = detect(text) # 언어 감지
            
            if lang == 'ko':
                # 한국어 전처리
                nouns = okt.nouns(text)
                filtered_words = [word for word in nouns if word not in stopwords_kr and len(word) > 1]
                processed_texts.append(filtered_words)
            else: # 영어 및 기타 언어
                # 영어 전처리
                # 1. 소문자화 및 토큰화
                words = nltk.word_tokenize(text.lower())
                # 2. 정규식으로 특수문자 제거, 불용어 제거, 표제어 추출
                lemmatized_words = [
                    lemmatizer.lemmatize(word) for word in words 
                    if word.isalpha() and word not in stop_words_en and len(word) > 2
                ]
                processed_texts.append(lemmatized_words)
        except Exception as e:
            logger.warning(f"Failed to preprocess text: {e}. Skipping this document.")
            processed_texts.append([]) # 에러 발생 시 빈 리스트 추가
            
    logger.info("Text preprocessing finished.")
    return processed_texts

# perform_topic_modeling 함수는 preprocess_korean_text를 preprocess_text로 바꾸기만 하면 됩니다.
def perform_topic_modeling(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not articles:
        logger.warning("No articles provided for topic modeling.")
        return {"topic_info": pd.DataFrame().to_dict(), "fig_html": None}
        
    logger.info(f"Starting topic modeling with Gensim LDA on {len(articles)} documents...")
    
    df = pd.DataFrame(articles)
    documents = df['content'].tolist()
    
    # --- 여기가 유일한 변경점 ---
    tokenized_docs = preprocess_text(documents) # 함수 이름 변경
    
    # ... (이하 로직은 이전과 완전히 동일) ...
    try:
        id2word = corpora.Dictionary(tokenized_docs)
        corpus = [id2word.doc2bow(text) for text in tokenized_docs]
        
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=5, 
                                           random_state=100, passes=10, alpha='auto')
                                           
        logger.success("Gensim LDA topic modeling completed successfully.")
        
        vis_data = gensimvis.prepare(lda_model, corpus, id2word)
        fig_html = pyLDAvis.prepared_data_to_html(vis_data)
        logger.info("pyLDAvis visualization generated successfully.")
        
        topics = lda_model.print_topics()
        topic_info_data = []
        for t_id, topic_words in topics:
            topic_info_data.append({"Topic": t_id, "Keywords": topic_words.replace('"', '')})
        topic_info_df = pd.DataFrame(topic_info_data)
        
        return {"topic_info": topic_info_df.to_dict(), "fig_html": fig_html}
        
    except Exception as e:
        logger.error(f"An error occurred during Gensim LDA modeling: {e}", exc_info=True)
        return {"topic_info": {}, "fig_html": "<div style='color:red;'>Failed to perform topic modeling. See terminal logs.</div>"}