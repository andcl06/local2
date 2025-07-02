# app.py
# Streamlit UI를 담당하며, 최신 뉴스 기반 트렌드 분석 및 보고서 생성을 수행하는 메인 파일입니다.

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from modules.data_collector import scrape_google_news_api
# modules.trend_analyzer는 이제 직접적인 토픽 모델링 시각화에만 사용되거나,
# 여기서는 직접적인 트렌드 감지 로직 대신 trend_detector를 사용합니다.
# from modules.trend_analyzer import perform_topic_modeling # 주석 처리 또는 제거
from modules.trend_detector import get_articles_from_db, detect_trending_keywords, get_articles_by_keywords
from modules.report_generator import create_single_page_report
# modules.ai_interface는 이제 report_generator 내부에서 호출되므로 직접 임포트 불필요
# from modules.ai_interface import call_potens_api, get_topic_summaries_from_ai

# --- 환경 변수 로드 (로컬에서 .env 파일 사용) ---
load_dotenv()

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="최신 트렌드 분석 및 보고서", layout="wide")

# --- Session State 초기화 (가장 중요!) ---
# 앱 재실행 시 데이터를 유지하기 위해 session_state에 저장할 변수 초기화
if 'all_articles' not in st.session_state:
    st.session_state['all_articles'] = []
if 'detected_trends' not in st.session_state: # 새로 추가된 감지된 트렌드 목록
    st.session_state['detected_trends'] = []
if 'selected_trend_keyword' not in st.session_state: # 사용자가 선택한 트렌드 키워드
    st.session_state['selected_trend_keyword'] = None
if 'selected_trend_articles' not in st.session_state: # 선택된 트렌드 관련 기사
    st.session_state['selected_trend_articles'] = []
if 'report_path' not in st.session_state: # 생성된 보고서 경로
    st.session_state['report_path'] = None
# ----------------------------------------

st.title("📈 HEART Insight AI: 최신 모빌리티 트렌드 분석 및 보고서")
st.markdown("---")

# --- API 키 로드 (UI에서 직접 사용하지 않지만, 모듈에 전달하기 위함) ---
POTENS_API_KEY = os.getenv("POTENS_API_KEY")
if not POTENS_API_KEY:
    st.error("Potens.dev API 키(POTENS_API_KEY)가 설정되지 않았습니다. .env 파일을 확인하거나 Streamlit Secrets에 설정해주세요.")
    st.stop()

# --- 1. 키워드 입력 기능 ---
default_keywords = 'electric vehicle battery, self-driving car insurance, UAM market, PBV Hyundai, MaaS service'
keywords_input = st.text_input(
    "분석할 키워드를 콤마(,)로 구분하여 입력하세요:",
    default_keywords,
    key="keywords_input_box"
)

# --- 분석 실행 버튼 ---
if st.button("최신 뉴스 기반 트렌드 분석 실행하기", key="run_analysis_button"):
    if keywords_input:
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        st.info(f"입력된 키워드: {keywords}")
    else:
        st.warning("분석할 키워드를 입력해주세요.")
        st.stop()

    with st.spinner("최신 뉴스를 수집하고 트렌드 감지를 수행 중입니다... (약 1~2분 소요)"):
        # 1. 데이터 수집 (data_collector.py)
        # scrape_google_news_api 함수 내부에서 DB에 저장되므로, 여기서는 반환값을 필수로 받지 않아도 됩니다.
        # 하지만, session_state에 저장하여 다른 곳에서 활용할 수 있도록 합니다.
        articles_fetched_for_session = []
        for keyword in keywords:
            articles_fetched_for_session.extend(scrape_google_news_api(keyword, num_results=5))
        
        st.session_state['all_articles'] = articles_fetched_for_session

        if st.session_state['all_articles']:
            st.success(f"총 {len(st.session_state['all_articles'])}개의 뉴스 기사를 수집했습니다.")
            
            # 2. 트렌드 감지 (trend_detector.py)
            # DB에서 최근 기사를 로드하여 트렌드 감지 (scrape_google_news_api에서 이미 DB에 저장했으므로)
            recent_articles_from_db = get_articles_from_db(days_ago=30) # 최근 30일 데이터 사용
            if recent_articles_from_db:
                detected_trends = detect_trending_keywords(recent_articles_from_db, lookback_days=7, threshold_percent_increase=50.0)
                st.session_state['detected_trends'] = detected_trends
                st.success("트렌드 감지가 완료되었습니다!")
            else:
                st.warning("데이터베이스에 최근 기사가 없어 트렌드 감지를 수행할 수 없습니다.")
                st.session_state['detected_trends'] = []
        else:
            st.error("뉴스 기사 수집에 실패했습니다. API 키나 네트워크 설정을 확인해주세요.")
            st.session_state['all_articles'] = []
            st.session_state['detected_trends'] = []
            st.session_state['selected_trend_keyword'] = None # 초기화
            st.session_state['selected_trend_articles'] = [] # 초기화
            st.session_state['report_path'] = None # 초기화

# --- 2. 감지된 트렌드 표시 및 선택 기능 ---
if st.session_state['detected_trends']:
    st.subheader("🔍 감지된 최신 트렌드")
    
    trend_options = [
        f"{trend['keyword']} (언급량: {trend['current_mentions']}회, 증가율: {trend['percent_increase']}%)" 
        for trend in st.session_state['detected_trends']
    ]
    
    selected_trend_display = st.selectbox(
        "보고서를 생성할 트렌드를 선택하세요:",
        options=["--- 트렌드 선택 ---"] + trend_options,
        key="trend_selector"
    )

    if selected_trend_display != "--- 트렌드 선택 ---":
        # 선택된 트렌드 키워드 추출
        selected_keyword_index = trend_options.index(selected_trend_display)
        st.session_state['selected_trend_keyword'] = st.session_state['detected_trends'][selected_keyword_index]['keyword']
        selected_trend_reason = f"키워드 '{st.session_state['selected_trend_keyword']}' 언급량 급증 (현재 {st.session_state['detected_trends'][selected_keyword_index]['current_mentions']}회, 이전 {st.session_state['detected_trends'][selected_keyword_index]['previous_mentions']}회, 증가율 {st.session_state['detected_trends'][selected_keyword_index]['percent_increase']}%)"
        
        st.info(f"선택된 트렌드: **{st.session_state['selected_trend_keyword']}**")

        # 선택된 트렌드에 대한 관련 기사 로드 (trend_detector.py)
        st.session_state['selected_trend_articles'] = get_articles_by_keywords([st.session_state['selected_trend_keyword']], days_ago=14)
        
        if st.session_state['selected_trend_articles']:
            st.subheader("📄 관련 뉴스 기사 (보고서 생성에 활용)")
            for i, article in enumerate(st.session_state['selected_trend_articles'][:5]): # 상위 5개만 표시
                st.markdown(f"**{i+1}. [{article['source']}] {article['title']}**")
                st.markdown(f"발행일: {pd.to_datetime(article['publish_date']).strftime('%Y.%m.%d')} | [링크]({article['url']})")
                st.markdown(f"_{article['content'][:150]}..._")
                st.markdown("---")
            
            # --- 3. 보고서 생성 버튼 ---
            if st.button(f"'{st.session_state['selected_trend_keyword']}' 트렌드 보고서 생성하기", key="generate_report_button"):
                with st.spinner("AI 분석 보고서를 생성 중입니다... (최대 5분 소요)"):
                    # 보고서 생성 (report_generator.py)
                    report_path = create_single_page_report(
                        trend_name=f"'{st.session_state['selected_trend_keyword']}' 관련 트렌드",
                        trend_detection_reason=selected_trend_reason,
                        related_articles=st.session_state['selected_trend_articles'],
                        api_key=POTENS_API_KEY,
                        max_articles_for_ai_summary=3,
                        delay_between_ai_calls=20
                    )
                    st.session_state['report_path'] = report_path
                    st.success("보고서 생성이 완료되었습니다!")
                    
                    if st.session_state['report_path'] and os.path.exists(st.session_state['report_path']):
                        st.download_button(
                            label="보고서 다운로드 (.txt)",
                            data=open(st.session_state['report_path'], 'rb').read(),
                            file_name=os.path.basename(st.session_state['report_path']),
                            mime="text/plain",
                            key="download_report_button"
                        )
                    else:
                        st.error("보고서 파일 생성에 실패했습니다. 로그를 확인해주세요.")
        else:
            st.warning("선택된 트렌드에 대한 관련 기사를 찾을 수 없습니다.")
else:
    st.info("아직 감지된 트렌드가 없습니다. 위에 키워드를 입력하고 '트렌드 분석 실행하기' 버튼을 눌러주세요.")

# --- 하단에 토픽 모델링 시각화 (기존 기능) ---
# 이 부분은 현재 트렌드 감지 및 보고서 생성 흐름에서 직접 사용되지 않지만,
# 만약 필요하다면 별도의 버튼이나 섹션으로 재활용할 수 있습니다.
# if st.session_state['analysis_result'] is not None:
#     st.subheader("📊 토픽 모델링 인터랙티브 시각화 (이전 기능)")
#     if st.session_state['analysis_result']["fig_html"]:
#         components.html(st.session_state['analysis_result']["fig_html"], width=None, height=800, scrolling=True)
#     else:
#         st.error("시각화 생성에 실패했습니다. 터미널 로그를 확인해주세요.")

#     st.subheader("📝 주요 토픽 및 키워드 (이전 기능)")
#     if "topic_info" in st.session_state['analysis_result'] and st.session_state['analysis_result']["topic_info"]:
#         df_topic_info = pd.DataFrame(st.session_state['analysis_result']["topic_info"])
#         st.dataframe(df_topic_info)
#     else:
#         st.warning("토픽 정보를 불러올 수 없습니다.")

#     # AI 요약 기능 버튼 (이전 기능)
#     # 이 버튼도 analysis_result가 session_state에 있을 때만 표시
#     # 현재는 report_generator를 통해 보고서에 직접 요약되므로 별도 버튼 불필요
#     # if st.button("AI를 통해 토픽 의미 요약하기", key="ai_summary_button"):
#     #     st.info("AI가 각 토픽의 의미를 분석하고 있습니다...")
#     #     if st.session_state['analysis_result'] is not None and "topic_info" in st.session_state['analysis_result'] and st.session_state['analysis_result']["topic_info"]:
#     #         topic_summaries_df = get_topic_summaries_from_ai(df_topic_info.to_dict('records'), api_key=POTENS_API_KEY)
#     #         st.subheader("🤖 AI가 분석한 토픽별 요약")
#     #         st.dataframe(topic_summaries_df)
#     #     else:
#     #         st.warning("분석 결과가 없어 AI 요약을 진행할 수 없습니다. 먼저 트렌드 분석을 실행해주세요.")
