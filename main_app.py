# main_app.py
# 이 파일은 '보험특약개발 솔루션'의 메인 애플리케이션으로,
# 로그인, 랜딩 페이지, 최신 트렌드 분석, 문서 분석 기능을 통합합니다.

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from loguru import logger
import requests
import tiktoken # document.py에서 사용
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader # <-- TextLoader 임포트 확인
from langchain.text_splitter import RecursiveCharacterTextSplitter # document.py에서 사용
from langchain.embeddings import HuggingFaceEmbeddings # document.py에서 사용
from langchain.vectorstores import FAISS # document.py에서 사용
from langchain.memory import StreamlitChatMessageHistory # document.py에서 사용

# modules 디렉토리 내의 커스텀 모듈 임포트
# 이 파일들이 실제 디렉토리에 존재해야 합니다.
from modules.data_collector import scrape_google_news_api
from modules.trend_detector import get_articles_from_db, detect_trending_keywords, get_articles_by_keywords
from modules.report_generator import create_single_page_report

# --- 환경 변수 로드 (로컬에서 .env 파일 사용) ---
load_dotenv()

# --- Potens API 호출 함수 (두 기능에서 공통으로 사용) ---
def call_potens_api(prompt, api_key):
    """
    Potens.dev AI API를 호출하여 텍스트를 생성합니다.
    """
    url = "https://ai.potens.ai/api/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=300) # 타임아웃 추가
        response.raise_for_status()
        result = response.json()

        if isinstance(result, str):
            return result, []
        elif isinstance(result, dict):
            for key in ["response", "answer", "message", "content", "text", "data"]:
                if key in result:
                    return result[key], result.get("source_documents", [])
            return str(result), result.get("source_documents", [])
        else:
            return str(result), []
    except requests.exceptions.RequestException as e:
        logger.error(f"Potens API 요청 오류: {e}")
        return f"ERROR: Potens API 요청 중 오류 발생 - {str(e)}", []
    except Exception as e:
        logger.error(f"Potens API 처리 오류: {e}")
        return f"ERROR: Potens API 응답 처리 중 오류 발생 - {str(e)}", []

# --- 문서 분석기 관련 헬퍼 함수들 (document.py에서 가져옴) ---
def tiktoken_len(text):
    """텍스트의 토큰 길이를 계산합니다."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def get_text(uploaded_files):
    """업로드된 문서에서 텍스트를 추출합니다."""
    all_docs = []
    for doc in uploaded_files:
        file_name = doc.name
        # Streamlit 환경에서는 파일을 직접 저장해야 로더가 접근 가능
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"Uploaded: {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
        elif file_name.endswith('.txt'): # <-- TXT 파일 처리 로직 추가
            loader = TextLoader(file_name, encoding="utf-8")
        else:
            logger.warning(f"지원하지 않는 파일 형식입니다: {file_name}")
            continue

        all_docs.extend(loader.load_and_split())
    return all_docs

def get_text_chunks(texts):
    """텍스트를 청크 단위로 분할합니다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return splitter.split_documents(texts)

def get_vectorstore(chunks):
    """텍스트 청크를 기반으로 벡터 데이터베이스를 생성합니다."""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(chunks, embeddings)

# --- 사용자 DB (login.py에서 가져옴) ---
USER_DB = {
    "admin": "1234",
    "guest": "abcd",
    "user" : "qwer",
    "localai" : "asdf"
}

# --- 로그인 페이지 (login.py에서 가져옴) ---
def login_page():
    """사용자 로그인 인터페이스를 렌더링합니다."""
    st.title("🔐 로그인 페이지")

    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")

        if submitted:
            if username in USER_DB and USER_DB[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "landing"
                st.success("✅ 로그인 성공!")
                st.experimental_rerun()
            else:
                st.error("❌ 아이디 또는 비밀번호가 잘못되었습니다.")

# --- 랜딩 페이지 (login.py에서 가져옴) ---
def landing_page():
    """로그인 후 사용자가 기능을 선택하는 랜딩 페이지를 렌더링합니다."""
    st.title(f"👋 {st.session_state.username}님, 환영합니다!")
    st.subheader("원하는 기능을 선택하세요:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 최신 트렌드 분석 입장"):
            st.session_state.page = "trend"

    with col2:
        if st.button("📄 문서 분석 입장"):
            st.session_state.page = "document"

    st.markdown("---")
    if st.button("🚪 로그아웃"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "login"
        st.success("로그아웃 되었습니다.")
        st.experimental_rerun()

# --- 최신 트렌드 분석 페이지 (app.py 내용 통합) ---
def trend_analysis_page():
    """
    최신 뉴스 기반 트렌드 분석 및 보고서 생성을 수행하는 페이지입니다.
    기존 app.py의 내용을 통합합니다.
    """
    st.title("📈 HEART Insight AI: 최신 모빌리티 트렌드 분석 및 보고서")
    st.markdown("---")

    # 메인으로 돌아가기 버튼
    if st.button("⬅️ 메인으로"):
        st.session_state.page = "landing"
        st.experimental_rerun()
    st.markdown("---") # 버튼 아래 구분선 추가

    # --- API 키 로드 ---
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
            articles_fetched_for_session = []
            for keyword in keywords:
                articles_fetched_for_session.extend(scrape_google_news_api(keyword, num_results=5)) # 각 키워드당 5개 기사
            
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
                        # report_generator.py 내부에서 call_potens_api를 호출하므로, 여기서는 api_key만 전달
                        report_path = create_single_page_report(
                            trend_name=f"'{st.session_state['selected_trend_keyword']}' 관련 트렌드",
                            trend_detection_reason=selected_trend_reason,
                            related_articles=st.session_state['selected_trend_articles'],
                            api_key=POTENS_API_KEY, # API 키 전달
                            max_articles_for_ai_summary=3,
                            delay_between_ai_calls=20
                        )
                        st.session_state['report_path'] = report_path
                        st.success("보고서 생성이 완료되었습니다!")
                        
                        if st.session_state['report_path'] and os.path.exists(st.session_state['report_path']):
                            # TXT 파일로 다운로드
                            with open(st.session_state['report_path'], 'rb') as f:
                                st.download_button(
                                    label="보고서 다운로드 (.txt)",
                                    data=f.read(),
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


# --- 문서 분석 페이지 (document.py 내용 통합) ---
def document_analysis_page():
    """
    문서 기반 QA 챗봇 기능을 제공하는 페이지입니다.
    기존 document.py의 내용을 통합합니다.
    """
    st.title("📄 _Private Data :red[QA Chat]_")

    # 메인으로 돌아가기 버튼
    if st.button("⬅️ 메인으로"):
        st.session_state.page = "landing"
        st.experimental_rerun()
    st.markdown("---") # 버튼 아래 구분선 추가

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    with st.sidebar:
        # st.file_uploader의 type에 'txt' 추가
        uploaded_files = st.file_uploader("📎 문서 업로드", type=['pdf', 'docx', 'pptx', 'txt'], accept_multiple_files=True) # <-- 'txt' 추가
        # API 키는 전역에서 로드되므로, 여기서는 st.text_input으로 다시 받지 않고 전역 변수 사용
        doc_api_key = os.getenv("POTENS_API_KEY") # Potens API 키 재사용
        if not doc_api_key:
            st.warning("Potens API 키가 설정되지 않았습니다. .env 파일을 확인하거나 Streamlit Secrets에 설정해주세요.")
            st.stop()

        process = st.button("📚 문서 처리")

    if process:
        if not uploaded_files:
            st.warning("문서를 업로드해주세요.")
            st.stop()

        with st.spinner("문서를 처리 중입니다..."):
            docs = get_text(uploaded_files)
            chunks = get_text_chunks(docs)
            vectordb = get_vectorstore(chunks)
            st.session_state.vectordb = vectordb
            st.session_state.docs = docs # 'docs' 세션 상태 추가 (특약 생성에서 사용)
            st.success("✅ 문서 분석 완료! 질문을 입력해보세요.")

    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! 문서 기반 질문을 해보세요."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    history = StreamlitChatMessageHistory(key="chat_messages") # StreamlitChatMessageHistory 초기화

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not st.session_state.vectordb:
                st.warning("먼저 문서를 업로드하고 처리해야 합니다.")
                st.stop()

            with st.spinner("답변 생성 중..."):
                retriever = st.session_state.vectordb.as_retriever(search_type="similarity", k=3)
                docs = retriever.get_relevant_documents(query)

                context = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = f"""다음 문서를 참고하여 질문에 답하세요.

[문서 내용]:
{context}

[질문]:
{query}

[답변]:
"""
                # call_potens_api 함수는 전역에 정의된 것을 사용
                answer, _ = call_potens_api(final_prompt, doc_api_key) # doc_api_key 사용

                st.markdown(answer)
                with st.expander("📄 참고 문서"):
                    for doc_ref in docs: # 'doc' 변수명 충돌 피하기 위해 'doc_ref' 사용
                        st.markdown(f"**출처**: {doc_ref.metadata.get('source', '알 수 없음')}")
                        st.markdown(doc_ref.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- 특약 생성 기능 (document.py에서 가져옴) ---
    st.subheader("📑 보험 특약 생성기")

    # API 키는 이미 전역에서 로드됨
    # if not doc_api_key: # 이미 위에서 확인
    #     st.warning("먼저 API 키를 입력해주세요.")
    #     st.stop()

    if "docs" not in st.session_state: # get_text에서 저장한 docs 사용
        st.warning("문서를 먼저 업로드하고 처리해주세요.")
        st.stop()

    generate_special_contract = st.button("✨ 특약 생성 시작") # 버튼 추가

    if generate_special_contract:
        with st.spinner("특약 생성 중..."):
            all_text = "\n\n".join([doc.page_content for doc in st.session_state.docs])
            prompt = f"""
다음은 보험 약관의 내용입니다. 이 내용을 기반으로 고객 맞춤형 '특약'을 3개 제안해주세요.
각 특약은 제목과 설명을 포함해야 하며, 실제 약관처럼 작성해주세요.

[보험 약관]:
{all_text}

[결과]:
"""
            answer, _ = call_potens_api(prompt, doc_api_key) # doc_api_key 사용
            st.markdown("### ✅ 생성된 특약")
            st.markdown(answer)


# --- 메인 애플리케이션 라우팅 ---
def main_app():
    """
    애플리케이션의 메인 진입점입니다.
    로그인 상태에 따라 페이지를 라우팅합니다.
    """
    # 세션 상태 초기화 (모든 페이지에서 공통으로 사용될 변수들)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    # app.py에서 사용되는 세션 상태 변수 초기화
    if 'all_articles' not in st.session_state:
        st.session_state['all_articles'] = []
    if 'detected_trends' not in st.session_state:
        st.session_state['detected_trends'] = []
    if 'selected_trend_keyword' not in st.session_state:
        st.session_state['selected_trend_keyword'] = None
    if 'selected_trend_articles' not in st.session_state:
        st.session_state['selected_trend_articles'] = []
    if 'report_path' not in st.session_state:
        st.session_state['report_path'] = None
    
    # document.py에서 사용되는 세션 상태 변수 초기화
    # document_analysis_page 함수 내에서 초기화되므로 여기서는 제거
    # if "vectordb" not in st.session_state:
    #     st.session_state.vectordb = None
    # if 'messages' not in st.session_state:
    #     st.session_state.messages = [{
    #         "role": "assistant",
    #         "content": "안녕하세요! 문서 기반 질문을 해보세요."
    #     }]


    # 라우팅 로직
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == "landing":
            landing_page()
        elif st.session_state.page == "trend":
            trend_analysis_page()
        elif st.session_state.page == "document":
            document_analysis_page()
        else:
            st.session_state.page = "login" # 기본값 (로그인 상태인데 페이지가 이상하면 로그인으로)


if __name__ == "__main__":
    main_app()

