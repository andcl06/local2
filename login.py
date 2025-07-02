import streamlit as st

# ✅ 간단한 사용자 DB
USER_DB = {
    "admin": "1234",
    "guest": "abcd",
    "user" : "qwer",
    "localai" : "asdf"
}


def login_page():
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


def landing_page():
    st.title(f"👋 {st.session_state.username}님, 환영합니다!")
    st.subheader("원하는 기능을 선택하세요:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 최신트렌드 분석 입장"):
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


def trend_analysis_page():
    st.title("📈 최신 트렌드 분석")
    st.write("이곳에 트렌드 분석 기능을 구현하세요.")
    if st.button("⬅️ 메인으로"):
        st.session_state.page = "landing"


def document_analysis_page():
    st.title("📄 문서 분석")
    st.write("이곳에 문서 기반 QA 기능을 구현하세요.")
    if st.button("⬅️ 메인으로"):
        st.session_state.page = "landing"


def main():
    # 세션 상태 초기화
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "page" not in st.session_state:
        st.session_state.page = "login"

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
            st.session_state.page = "landing"


if __name__ == "__main__":
    main()
