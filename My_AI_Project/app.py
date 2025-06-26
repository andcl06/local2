# app.py
# Streamlit 기반 HEART Insight AI 웹 솔루션의 메인 파일

import streamlit as st
import os
from dotenv import load_dotenv
from loguru import logger
# 'modules' 폴더에 있는 커스텀 모듈들을 임포트합니다.
from modules import ai_interface
from modules import data_collector
from modules import trend_analyzer
import pandas as pd # 데이터프레임 사용을 위해 필요

# -----------------
# 1. 환경 변수 로드
# -----------------
load_dotenv()
POTENS_API_KEY = os.getenv("POTENS_API_KEY")

# -----------------
# 2. 페이지 기본 설정
# -----------------
st.set_page_config(
    page_title="현대해상 HEART Insight AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------
# 3. Streamlit 세션 상태(Session State) 초기화
# -----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "api_ready" not in st.session_state:
    st.session_state["api_ready"] = False
    
if "data_collected" not in st.session_state:
    st.session_state["data_collected"] = False
    
if "collected_data" not in st.session_state:
    st.session_state["collected_data"] = []

if "topic_analysis_result" not in st.session_state:
    st.session_state["topic_analysis_result"] = None

# -----------------
# 4. UI 구성: 메인 화면
# -----------------
st.title("🚗 현대해상 HEART Insight AI")
st.subheader("미래 모빌리티 트렌드 분석 및 보험 시사점 도출 솔루션")
st.markdown("---")

st.markdown(
    """
    **HEART Insight AI**는 급변하는 미래 모빌리티 환경의 트렌드를 심층 분석하고,
    이를 현대해상의 보험 상품 개발 및 리스크 평가에 필요한 핵심 시사점과 기회 요인으로 도출하는 AI 기반 솔루션입니다.
    """
)

# -----------------
# 5. AI 트렌드 분석 대시보드
# -----------------
st.header("📈 AI 트렌드 분석 대시보드")
st.info("이곳에 뉴스, 보고서, 특허 등에서 수집된 데이터를 기반으로 한 트렌드 예측 그래프, 키워드 네트워크, 토픽 모델링 결과 등 인터랙티브 시각화가 구현될 예정입니다.")

# --- 수집된 데이터 및 분석 결과가 있을 때만 대시보드 내용 표시 ---
if st.session_state.data_collected:
    st.markdown("### 수집된 데이터 미리보기")
    # 수집된 데이터 중 10개만 미리보기
    st.dataframe(pd.DataFrame(st.session_state.collected_data)[:10], use_container_width=True)
    st.markdown("---")

    if st.session_state.topic_analysis_result and st.session_state.topic_analysis_result['fig_html']:
        st.markdown("### 📊 토픽 모델링 시각화")
        # Plotly 그래프 표시
        st.components.v1.html(st.session_state.topic_analysis_result['fig_html'], height=600)
        st.markdown("---")

        st.markdown("### 📝 주요 트렌드 (토픽) 요약")
        # 토픽 정보를 데이터프레임으로 표시
        topic_info_df = pd.DataFrame(st.session_state.topic_analysis_result['topic_info'])
        topic_info_df.index.name = 'Topic ID'
        st.dataframe(topic_info_df[['Count', 'Name', 'Representation']], use_container_width=True)
        st.success("✅ 트렌드 분석 결과가 대시보드에 반영되었습니다!")
    else:
        st.warning("토픽 모델링 결과를 불러오는 데 실패했습니다. 로그를 확인해주세요.")

st.markdown("---")

# -----------------
# 6. 대화형 AI 챗봇 (Potens.dev API 연동)
# -----------------
st.header("💬 트렌드 분석 Q&A 챗봇")

# 기존 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 챗봇 입력 위젯
if st.session_state.api_ready:
    user_query = st.chat_input("미래 모빌리티 트렌드에 대해 무엇이 궁금하신가요? (예: 자율주행 레벨 4의 책임 소재 변화는?)")
    
    if user_query:
        # 1. 사용자 질문을 대화 기록에 추가하고 화면에 표시
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # 2. AI 응답 생성: Potens.dev API 호출
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                # ai_interface.py의 함수를 호출하여 API 통신을 수행
                ai_response = ai_interface.call_potens_api(
                    user_query=user_query,
                    api_key=POTENS_API_KEY,
                    history=st.session_state.messages # 전체 대화 기록을 문맥으로 전달
                )
                
                # 응답을 화면에 표시하고 세션 상태에 저장
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    # API 준비가 안 되었을 때 표시되는 메시지
    st.warning("⚠️ Potens.dev API 연동 준비가 필요합니다. 사이드바에서 'AI 챗봇 준비' 버튼을 눌러주세요.")
    st.markdown("API가 준비되면 이곳에서 AI와 대화할 수 있습니다.")


# -----------------
# 7. 사이드바 UI (추가 기능 및 설정)
# -----------------
with st.sidebar:
    st.header("프로젝트 정보")
    st.markdown("**프로젝트명:** HEART Insight AI")
    st.markdown("**개발:** 메이커스랩")
    st.markdown("---")
    
    st.header("기능 제어")
    # 챗봇 활성화 버튼
    if st.button("AI 챗봇 준비"):
        if POTENS_API_KEY:
            st.session_state.api_ready = True
            st.success("🎉 Potens.dev API 준비 완료! 이제 챗봇에 질문해보세요.")
            logger.info("Potens.dev API is ready.")
            st.experimental_rerun()
        else:
            st.error("API 키가 설정되지 않았습니다. `.env` 파일에 `POTENS_API_KEY`를 설정해주세요.")
            st.session_state.api_ready = False
            logger.warning("API key is missing.")

    # 데이터 수집 및 분석 기능
    st.markdown("---")
    st.header("트렌드 데이터 수집 및 분석")
    
    if st.button("트렌드 데이터 수집 및 분석 시작", help="뉴스 기사를 크롤링하고 AI 분석을 수행합니다."):
        if not st.session_state.data_collected:
            keywords = ["전기차 배터리", "자율주행 보험", "UAM 시장", "PBV 현대차", "MaaS 서비스"]
            collected_articles = []
            
            with st.spinner("뉴스 기사 데이터 수집 중..."):
                for keyword in keywords:
                    articles = data_collector.scrape_google_news(keyword, pages=1)
                    collected_articles.extend(articles)
                    st.info(f"'{keyword}' 관련 기사 {len(articles)}개 수집 완료.")
                
            if collected_articles:
                st.session_state.collected_data = collected_articles
                st.session_state.data_collected = True
                st.success(f"✅ 총 {len(st.session_state.collected_data)}개의 기사 데이터 수집 완료!")
                logger.info(f"Total articles collected: {len(st.session_state.collected_data)}")

                with st.spinner("수집된 데이터를 기반으로 AI 트렌드 분석 중..."):
                    analysis_result = trend_analyzer.perform_topic_modeling(st.session_state.collected_data)
                    st.session_state.topic_analysis_result = analysis_result
                
                if st.session_state.topic_analysis_result and st.session_state.topic_analysis_result['topics']:
                    st.success("✅ AI 트렌드 분석이 성공적으로 완료되었습니다. 대시보드를 확인하세요!")
                else:
                    st.error("❌ AI 분석 중 오류가 발생했습니다. 로그를 확인해주세요.")
            else:
                st.warning("⚠️ 데이터를 수집하지 못했습니다. 네트워크 연결 또는 크롤링 대상 사이트의 변경을 확인해주세요.")
                st.session_state.data_collected = False
                st.session_state.topic_analysis_result = None
        else:
            st.info("데이터 수집 및 분석이 이미 완료되었습니다. 앱을 새로고침하여 다시 시작하세요.")

    st.markdown("---")
    st.header("대화 초기화")
    if st.button("대화 초기화", help="모든 대화 기록을 삭제합니다."):
        st.session_state.messages = []
        st.experimental_rerun()