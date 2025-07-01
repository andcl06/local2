# app.py

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from modules.data_collector import scrape_google_news_api
from modules.trend_analyzer import perform_topic_modeling

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="최신 트렌드 분석", layout="wide")

st.title("📈 최신 트렌드 분석 및 시각화 (Gensim Ver.)")
st.markdown("---")

# --- 분석 실행 ---
if st.button("최신 뉴스 기반 트렌드 분석 실행하기"):
    with st.spinner("최신 뉴스를 수집하고 트렌드 분석을 수행 중입니다... (약 1~2분 소요)"):
        # 1. 데이터 수집
        keywords = ['electric vehicle battery', 'self-driving car insurance', 'UAM market', 'PBV Hyundai', 'MaaS service']
        all_articles = []
        for keyword in keywords:
            articles = scrape_google_news_api(keyword, num_results=5)
            all_articles.extend(articles)

        if all_articles:
            st.info(f"총 {len(all_articles)}개의 뉴스 기사를 수집했습니다.")

            # 2. 토픽 모델링 및 시각화 수행 (Gensim + pyLDAvis 버전)
            analysis_result = perform_topic_modeling(all_articles)
            
            # 3. 결과 표시
            st.success("트렌드 분석이 완료되었습니다!")
            
            st.subheader("📊 토픽 모델링 인터랙티브 시각화")
            if analysis_result and analysis_result["fig_html"]:
                components.html(analysis_result["fig_html"], width=None, height=800, scrolling=True)
            else:
                st.error("시각화 생성에 실패했습니다. 터미널 로그를 확인해주세요.")

            st.subheader("📝 주요 토픽 및 키워드")
            if analysis_result and "topic_info" in analysis_result and analysis_result["topic_info"]:
                 st.dataframe(pd.DataFrame(analysis_result["topic_info"]))
            else:
                st.warning("토픽 정보를 불러올 수 없습니다.")

        else:
            st.error("뉴스 기사 수집에 실패했습니다. API 키나 네트워크 설정을 확인해주세요.")