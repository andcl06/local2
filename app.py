# app.py

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from modules.data_collector import scrape_google_news_api
from modules.trend_analyzer import perform_topic_modeling
from modules.ai_interface import call_potens_api, get_topic_summaries_from_ai

# --- 환경 변수 로드 (로컬에서 .env 파일 사용) ---
load_dotenv()

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="최신 트렌드 분석", layout="wide")

# --- Session State 초기화 (가장 중요!) ---
# 앱 재실행 시 데이터를 유지하기 위해 session_state에 저장할 변수 초기화
if 'all_articles' not in st.session_state:
    st.session_state['all_articles'] = []
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
# ----------------------------------------

st.title("📈 최신 트렌드 분석 및 시각화 (Gensim Ver.)")
st.markdown("---")

# --- 1. 키워드 입력 기능 ---
# default_keywords를 미리 정의
default_keywords = 'electric vehicle battery, self-driving car insurance, UAM market, PBV Hyundai, MaaS service'
keywords_input = st.text_input(
    "분석할 키워드를 콤마(,)로 구분하여 입력하세요:",
    default_keywords,
    key="keywords_input_box" # 위젯의 고유 키 추가
)

# --- 분석 실행 버튼 ---
# 분석 실행 버튼을 누르면 데이터 수집 및 토픽 모델링 수행
if st.button("최신 뉴스 기반 트렌드 분석 실행하기", key="run_analysis_button"): # 위젯의 고유 키 추가
    # 입력된 키워드 문자열을 리스트로 변환
    if keywords_input:
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        st.info(f"입력된 키워드: {keywords}")
    else:
        st.warning("분석할 키워드를 입력해주세요.")
        st.stop() # 키워드 없으면 여기서 중단

    with st.spinner("최신 뉴스를 수집하고 트렌드 분석을 수행 중입니다... (약 1~2분 소요)"):
        # 1. 데이터 수집
        articles_fetched = [] # 임시 변수에 저장
        for keyword in keywords:
            articles_fetched.extend(scrape_google_news_api(keyword, num_results=5))

        # 데이터 수집 결과를 session_state에 저장
        st.session_state['all_articles'] = articles_fetched 

        if st.session_state['all_articles']:
            st.info(f"총 {len(st.session_state['all_articles'])}개의 뉴스 기사를 수집했습니다.")

            # 2. 토픽 모델링 및 시각화 수행 (수집된 데이터를 session_state에서 가져와 사용)
            analysis_output = perform_topic_modeling(st.session_state['all_articles'])
            # 분석 결과도 session_state에 저장
            st.session_state['analysis_result'] = analysis_output
            
            st.success("트렌드 분석이 완료되었습니다!")
        else:
            st.error("뉴스 기사 수집에 실패했습니다. API 키나 네트워크 설정을 확인해주세요.")
            # 실패 시 session_state 초기화
            st.session_state['all_articles'] = []
            st.session_state['analysis_result'] = None

# --- 결과 표시 (분석 결과가 session_state에 있을 때만 표시) ---
# 이제 이 블록은 '최신 뉴스 기반 트렌드 분석 실행하기' 버튼 클릭 여부와 상관없이,
# session_state['analysis_result']에 데이터가 존재하면 항상 렌더링됩니다.
if st.session_state['analysis_result'] is not None:
    st.subheader("📊 토픽 모델링 인터랙티브 시각화")
    if st.session_state['analysis_result']["fig_html"]:
        components.html(st.session_state['analysis_result']["fig_html"], width=None, height=800, scrolling=True)
    else:
        st.error("시각화 생성에 실패했습니다. 터미널 로그를 확인해주세요.")

    st.subheader("📝 주요 토픽 및 키워드")
    if "topic_info" in st.session_state['analysis_result'] and st.session_state['analysis_result']["topic_info"]:
        df_topic_info = pd.DataFrame(st.session_state['analysis_result']["topic_info"])
        st.dataframe(df_topic_info)
    else:
        st.warning("토픽 정보를 불러올 수 없습니다.")

    # --- AI 요약 기능 버튼 ---
    # 이 버튼도 analysis_result가 session_state에 있을 때만 표시
    if st.button("AI를 통해 토픽 의미 요약하기", key="ai_summary_button"): # 위젯의 고유 키 추가
        st.info("AI가 각 토픽의 의미를 분석하고 있습니다...")
        
        if st.session_state['analysis_result'] is not None and "topic_info" in st.session_state['analysis_result'] and st.session_state['analysis_result']["topic_info"]:
            # AI 인터페이스 모듈의 함수 호출
            POTENS_API_KEY = os.getenv("POTENS_API_KEY")
            
            if POTENS_API_KEY:
                # df_topic_info는 위에 정의되어 있으므로 바로 사용
                topic_summaries_df = get_topic_summaries_from_ai(df_topic_info.to_dict('records'), api_key=POTENS_API_KEY)
                
                st.subheader("🤖 AI가 분석한 토픽별 요약")
                st.dataframe(topic_summaries_df)
            else:
                st.error("AI API 키(POTENS_API_KEY)가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        else:
            st.warning("분석 결과가 없어 AI 요약을 진행할 수 없습니다. 먼저 트렌드 분석을 실행해주세요.")