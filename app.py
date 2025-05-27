import os
import streamlit as st
import sys
import platform

# ✅ 무조건 첫 Streamlit 명령어
st.set_page_config(
    page_title="KAIST 규정 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SQLite 버전 문제 해결 (Streamlit Cloud용)
if "streamlit" in sys.modules and platform.system() == "Linux":
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time
import shutil
import traceback
from tenacity import retry, stop_after_attempt, wait_fixed
import openai
import httpx
import ssl
import urllib3
import json
import re
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()
# 디버그 모드 활성화
DEBUG_MODE = False

# SSL 검증 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# LangChain 컴포넌트 임포트
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain_teddynote.document_loaders import HWPLoader
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    st.error(f"필요한 패키지를 찾을 수 없습니다: {str(e)}")
    st.info("다음 명령어로 필요한 패키지를 설치하세요:")
    st.code("pip install langchain langchain-openai langchain-community chromadb langchain-teddynote")
    st.stop()

# 디렉토리 설정
HWP_DIR = os.path.join(os.path.dirname(__file__), 'data')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')
os.makedirs(HWP_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 환경 구분 함수
def is_streamlit_cloud():
    # Streamlit Cloud(리눅스) 환경에서는 secrets.toml이 존재
    return platform.system() == "Linux" and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets

if is_streamlit_cloud():
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_EMBEDDING_MODEL = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai.gpt-4.1-mini-2025-04-14")
    OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "azure.text-embedding-3-large")

# API 키 확인
if not openai.api_key:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다!")
    st.info("Streamlit Cloud의 Settings > Secrets에서 OPENAI_API_KEY를 설정하거나, 로컬에서는 .env 파일을 생성하세요.")
    st.stop()

# CSS - 최상단에 배치하여 먼저 적용되도록 함
st.markdown("""
<style>
/* 전체 앱 스타일링 */
.stApp {
    background-color: white;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}

/* 헤더 컨테이너 - 상단 고정 */
.header-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: white;
    padding: 1rem 2rem;
    border-bottom: 1px solid #e5e7eb;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* 타이틀 컨테이너 스타일링 */
.title-container {
    display: flex;
    align-items: center;
    padding-bottom: 1rem;
}

/* 메인 타이틀 스타일링 */
.main-title {
    color: #0a1c3e;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0;
    padding: 0;
    line-height: 1.2;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 서브타이틀 스타일링 */
.subtitle {
    color: #4b5563 !important;
    font-size: 1.1rem !important;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* 채팅 영역 컨테이너 - 독립적 스크롤 */
.message-container {
    height: auto;
    overflow-y: visible;
    padding: 1rem;
    margin-top: 20px;
    padding-bottom: 150px; /* 입력창 고려 */
}

/* 사이드바 스타일링 */
.css-1d391kg {
    background-color: #ffffff;
    padding: 1rem;
    border-right: 1px solid #e9ecef;
}

/* 사이드바 헤더 */
.sidebar .block-container {
    padding-top: 1rem;
}

/* 채팅 메시지 컨테이너 */
.stChatMessage {
    background-color: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    width: auto !important;
    max-width: none !important;
    min-height: auto !important;
    max-height: none !important;
    overflow-y: visible;
}

/* 사용자 메시지: 우측 정렬 */
.stChatMessage[data-role="user"] {
    background: linear-gradient(45deg, #E8F0FE, #F8F9FA);
    border-bottom-right-radius: 5px;
    min-height: auto;
    max-width: none !important;
    width: auto !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    box-shadow: 0 2px 5px rgba(30,58,138,0.08);
}

/* 어시스턴트 메시지: 좌측 정렬 */
.stChatMessage[data-role="assistant"] {
    background: white;
    border-bottom-left-radius: 5px;
    min-height: auto;
    max-width: none !important;
    width: auto !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    box-shadow: 0 2px 5px rgba(30,58,138,0.04);
}

/* 채팅 입력 필드 - 항상 하단에 고정 */
.stChatInputContainer {
    position: fixed;
    bottom: 0;
    left: 350px; /* 사이드바 너비만큼 오른쪽으로 */
    right: 0;
    z-index: 100;
    background: white;
    padding: 1rem 2rem;
    border-top: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0 -4px 10px rgba(0,0,0,0.03);
}

/* 입력창 자체 높이/폰트/패딩 */
.stChatInput textarea, .stChatInput input {
    min-height: 100px !important;
    font-size: 1.0rem !important;
    padding: 0.8rem 1rem !important;
    border-radius: 16px !important;
    border: 2px solid #E8F0FE !important;
}

/* 채팅 아바타 */
.stChatAvatar {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    margin-right: 10px !important;
}

/* 스피너와 로딩 상태 스타일링 - 일관된 UI를 위함 */
.stSpinner {
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* 버튼 스타일링 */
.stButton button {
    background-color: White;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

/* 초기화 버튼 */
button[kind="secondary"] {
    background-color: #FF6B6B !important;
}

.stButton button:hover {
    opacity: 0.9;
}

/* 확장 패널 스타일링 */
.streamlit-expanderHeader {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    font-weight: 600;
}

/* 스크롤바 스타일링 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #2563eb;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #1e3a8a;
}

/* Streamlit 요소 조정 */
.css-1544g2n.e1fqkh3o4 {
    padding-top: 0;
}

/* 사이드바 넓이 조절 */
section[data-testid="stSidebar"] {
    width: 350px;
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    overflow-y: auto;
    background-color: #ffffff;
    padding: 1rem;
    border-right: 1px solid #e9ecef;
}

/* 사이드바 내 모든 버튼 스타일링 - 더 강력한 선택자 사용 */
section[data-testid="stSidebar"] .stButton > button {
    background-color: white !important;
    color: #1e3a8a !important;
    border: 1px solid #e5e7eb !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: 0.5rem !important;
    width: 100% !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #2563eb !important;
    color: #1e40af !important;
    background-color: #f8fafc !important;
}

/* 예시 질문 버튼 스타일링 */
.example-questions .stButton > button {
    background-color: white !important;
    color: #1e3a8a !important;
}

/* 예시 질문 버튼 - 호버 상태 */
.example-questions .stButton > button:hover {
    border-color: #2563eb !important;
    color: #1e40af !important;
    background-color: #f8fafc !important;
}

/* 사이드바의 버튼 컨테이너를 example-questions 클래스로 지정 */
.example-questions .stButton > button {
    background-color: white !important;
    color: #1e3a8a !important;
}

/* 예시 질문 버튼 - 호버 상태 */
.example-questions .stButton > button:hover {
    border-color: #2563eb !important;
    color: #1e40af !important;
    background-color: #f8fafc !important;
}

/* 추가 여백 제거 */
.block-container {
    padding-top: 0 !important;
    max-width: 100% !important;
}

/* main content 영역 */
.main .block-container {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 0 !important;
    max-width: 100% !important;
    margin-left: 350px; /* 사이드바 너비만큼 여백 */
}

/* 메시지 채팅 컨테이너를 위한 일관된 공간 */
.message-container {
    min-height: 600px;
    padding-bottom: 150px;
}

/* 사이드바 헤더 스타일링 */
.sidebar-header {
    padding: 0;
    margin: 0;
    text-align: center;
}

.sidebar-header img {
    margin: 0 auto 0.5rem;
    display: block;
}

.sidebar-header .main-title {
    font-size: 1.8rem !important;
    text-align: center;
    margin-top: 0.2rem;
    margin-bottom: 0.25rem;
}

.sidebar-header .subtitle {
    font-size: 0.9rem !important;
    text-align: center;
    margin: 0;
}

.sidebar-divider {
    margin: 1rem 0;
    border: 0;
    border-top: 1px solid #e5e7eb;
}

/* 헤더 컨테이너 - 제거 (사이드바로 이동) */
.header-container {
    display: none;
}

/* 메시지 컨테이너 - 상단 여백 제거 */
.message-container {
    height: auto;
    overflow-y: visible;
    padding: 1rem;
    margin-top: 20px;
    padding-bottom: 150px; /* 입력창 고려 */
}

/* 채팅 입력 필드 - 위치 조정 */
.stChatInputContainer {
    position: fixed;
    bottom: 0;
    left: 350px; /* 사이드바 너비만큼 오른쪽으로 */
    right: 0;
    z-index: 100;
    background: white;
    padding: 1rem 2rem;
    border-top: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0 -4px 10px rgba(0,0,0,0.03);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* 사이드바 상단 여백 완전 제거 - 최우선 적용 */
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] .element-container:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* 사이드바 내부 모든 div에 상단 여백 제거 */
section[data-testid="stSidebar"] div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] .sidebar-header {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# 메시지 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! KAIST 규정에 대해 궁금한 점이 있으시면 무엇이든 물어보세요. 어떤 도움이 필요하신가요?"}
    ]

# 대화 기록 메모리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 예시 질문 중복 방지를 위한 함수
def add_user_message(message):
    # 중복 메시지 체크: 같은 내용의 user 메시지가 이미 마지막에 있으면 추가하지 않음
    if (len(st.session_state.messages) > 0 and
            st.session_state.messages[-1]["role"] == "user" and
            st.session_state.messages[-1]["content"] == message):
        return False
    
    # 메시지 추가
    st.session_state.messages.append({"role": "user", "content": message})
    return True

# 후속 질문을 추출하는 함수 수정
def extract_follow_up_questions(text):
    """
    어시스턴트 답변에서 후속 질문을 추출하는 함수
    마크다운 형식의 관련 질문 섹션에서 질문들을 리스트로 반환
    """
    # 디버깅을 위한 텍스트 출력 (DEBUG_MODE가 True인 경우)
    if DEBUG_MODE:
        print(f"답변 텍스트 확인: {text[:200]}...")
    
    # 마크다운 패턴 처리 - 새로운 형식에 맞게 업데이트
    patterns = [
        # 마크다운 헤더 형식 (새로운 지정 형식)
        r"##\s*추천\s*질문\s*\n([\s\S]*?)(?=##|$)",
        # 마크다운 헤더 형식 (다른 변형)
        r"##\s*관련\s*질문\s*\n([\s\S]*?)(?=##|$)",
        # 이전 형식들과의 호환성 유지
        r"###?\s*추천\s*질문\s*:?\n([\s\S]*?)(?=###|$)",
        r"###?\s*관련\s*질문\s*:?\n([\s\S]*?)(?=###|$)",
        # 일반 텍스트 패턴 (백업)
        r"추천\s*질문\s*:?\n([\s\S]*?)(?=\n\n|$)"
    ]
    
    questions_section = ""
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            questions_section = match.group(1).strip()
            break
    
    if not questions_section:
        # 디버깅용 전체 텍스트의 일부 출력
        if DEBUG_MODE:
            debug_text = text[:200] + "..." if len(text) > 200 else text
            print(f"DEBUG: 텍스트에서 질문 섹션을 찾을 수 없습니다. 텍스트 시작 부분: {debug_text}")
            print(f"DEBUG: 전체 텍스트: {text}")
        
        # 대안: 간단한 자체 생성 질문 사용 (LLM 호출 없이)
        return [
            "다른 관련 규정에 대해 알려주세요",
            "이 내용을 더 자세히 설명해주세요",
            "이 규정의 예외사항이 있나요?"
        ]
    
    # 디버깅: 찾은 질문 섹션 출력
    if DEBUG_MODE:
        print(f"DEBUG: 찾은 질문 섹션: {questions_section}")
    
    # 번호가 매겨진 항목 추출 (1. 2. 3. 등의 형식)
    numbered_items = re.findall(r'^\s*\d+\.?\s*(.*?)$', questions_section, re.MULTILINE)
    if numbered_items and len(numbered_items) >= 2:  # 최소 2개 이상의 번호 매김 항목이 있으면
        # 디버깅: 번호가 매겨진 항목 확인
        if DEBUG_MODE:
            print(f"DEBUG: 번호 매김 항목 발견: {numbered_items}")
        
        # 각 항목 정리 (대괄호 등 제거)
        questions = []
        for item in numbered_items:
            # 대괄호로 둘러싸인 내용 추출 또는 그냥 항목 사용
            clean_item = re.sub(r'^\[(.*)\]$', r'\1', item.strip())
            if clean_item and len(clean_item) > 5:
                questions.append(clean_item)
        
        if len(questions) >= 2:  # 의미 있는 질문이 2개 이상 추출됐으면 사용
            return questions
    
    # 번호 매김이 실패하면 일반적인 방법으로 추출 시도
    questions = []
    for line in questions_section.split("\n"):
        # HTML 태그 제거
        line = re.sub(r'<[^>]*>', '', line)
        # 줄에서 앞부분의 불릿, 번호, 대시 등 제거
        clean_line = re.sub(r"^[\s\-–•*0-9.)\]]*\s*", "", line).strip()
        # 대괄호 안의 내용만 추출 또는 전체 라인 사용
        if '[' in clean_line and ']' in clean_line:
            bracket_content = re.search(r'\[(.*?)\]', clean_line)
            if bracket_content:
                clean_line = bracket_content.group(1).strip()
        
        if clean_line and not clean_line.startswith("관련해서") and len(clean_line) > 5:
            questions.append(clean_line)
    
    # 디버깅: 추출된 질문 확인
    if DEBUG_MODE and questions:
        print(f"DEBUG: 추출된 질문: {questions}")
    
    # 질문이 없는 경우 기본 질문 사용
    if not questions:
        return [
            "다른 관련 규정에 대해 알려주세요",
            "이 내용을 더 자세히 설명해주세요",
            "이 규정의 예외사항이 있나요?"
        ]
    
    return questions

# 후속 질문 섹션을 제거하는 함수 업데이트
def remove_follow_up_questions_section(text):
    """
    어시스턴트 답변에서 후속 질문 섹션을 제거하는 함수
    마크다운 형식 지원
    """
    # 마크다운 패턴들 - 새로운 형식 포함
    patterns = [
        r"##\s*추천\s*질문\s*\n[\s\S]*?(?=##|$)",  # 새 형식 (##)
        r"##\s*관련\s*질문\s*\n[\s\S]*?(?=##|$)",  # 새 형식 (##)
        r"####\s*추천\s*질문\s*\n[\s\S]*?(?=####|$)",  # 새 형식 (####)
        r"####\s*관련\s*질문\s*\n[\s\S]*?(?=####|$)",  # 새 형식 (####)
        r"###?\s*추천\s*질문\s*:?\n[\s\S]*?(?=###|$)",  # 기존 형식 (###)
        r"###?\s*관련\s*질문\s*:?\n[\s\S]*?(?=###|$)",  # 기존 형식 (###)
        r"추천\s*질문\s*:?\n[\s\S]*?(?=\n\n|$)"  # 일반 텍스트 형식
    ]
    
    # 원본 텍스트 저장
    original_text = text
    
    # 각 패턴 적용
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    
    # 디버깅 - 텍스트 변경 확인
    if DEBUG_MODE and original_text != result:
        print(f"DEBUG: 질문 섹션 제거됨. 원본 길이: {len(original_text)}, 처리 후 길이: {len(result)}")
    
    return result.strip()

# 사이드바 예시 질문
with st.sidebar:
    # 사이드바 상단에 로고와 타이틀 배치
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.image("kaistlogo.png", width=150)
    st.markdown('<h1 class="main-title">KAIST ChatBot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KAIST 규정에 대한 질문 및 답변</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 구분선 추가
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    with st.expander("⚙️ 설정", expanded=False):
        st.markdown("#### 📚 벡터DB 설정")
        force_rebuild = st.checkbox("벡터DB 강제 재생성", value=False)
        if force_rebuild and os.path.exists(CHROMA_DIR):
            if st.button("벡터DB 재생성", help="기존 벡터DB를 삭제하고 새로 생성합니다", key="rebuild_btn"):
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                if "retriever" in st.session_state:
                    del st.session_state.retriever
                st.success("✨ 벡터DB가 삭제되었습니다. 새로고침하여 재생성을 시작하세요.")
                st.rerun()
    
    st.markdown("### 💡 예시 질문")
    
    # 직접 HTML과 CSS로 예시 질문 버튼 스타일링
    st.markdown("""
    <style>
    .custom-button {
        background-color: white;
        color: #1e3a8a;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        width: 100%;
        text-align: left;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .custom-button:hover {
        border-color: #2563eb;
        background-color: #f8fafc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    example_questions = [
        "법인카드로 지출한 금액은 어떤 증빙서류가 필요하고, 정산 기한은 언제까지인가요?",
        "출장 중 식비와 숙박비는 각각 얼마까지 인정되며, 기준 금액을 초과하면 어떻게 되나요?",
        "상품권을 구매한 경우, 비용 처리 시 어떤 제한이 있고 어떤 서류가 필요하나요?",
        "과세 항목과 비과세 항목을 구분하는 기준은 무엇이며, 대표적인 예시는 어떤 게 있나요?",
        "같은 항목으로 중복 지출이 발생한 경우 어떻게 처리되고, 환수 대상이 될 수 있나요?",
        "강의료나 연구수당 등 인건비 항목은 어떤 소득유형으로 분류되며, 세율은 얼마인가요?",
        "사적 용도 또는 가족 명의 계좌로 지급된 지출은 어떤 절차로 확인되며, 문제가 될 경우 조치는 무엇인가요?",
        "연말 또는 회계연도 말에 몰아서 지출한 경우 규정상 문제가 될 수 있나요?",
        "비용 지출 시 카드 사용이 필수인가요, 아니면 현금 정산도 가능한가요?",
        "지출 예정일 전에 선지급이 필요한 경우 어떤 조건과 절차를 따라야 하나요?"
    ]

    # 예시 질문 버튼을 컨테이너로 감싸서 한 번만 렌더링되도록 함
    question_container = st.container()
    with question_container:
        for q in example_questions:
            if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
                if add_user_message(q):
                    st.rerun()

# 채팅 초기화 버튼을 우측에 배치
if st.session_state.messages and len(st.session_state.messages) > 1:  # 초기 메시지만 있는 경우는 제외
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("대화 초기화", key="clear_chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "안녕하세요! KAIST 규정에 대해 궁금한 점이 있으시면 무엇이든 물어보세요. 어떤 도움이 필요하신가요?"}
            ]
            # 대화 히스토리도 초기화
            st.session_state.chat_history = []
            st.rerun()

# 채팅 영역에 일관된 공간 제공
st.markdown('<div class="message-container">', unsafe_allow_html=True)

# 채팅 히스토리 표시 - 각 메시지는 정확히 한 번만 표시됨
for i, message in enumerate(st.session_state.messages):
    # 모든 메시지를 표시 (건너뛰는 메시지 없음)
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        if message["role"] == "assistant":
            # 어시스턴트 메시지인 경우 후속 질문 섹션 제거 후 표시
            clean_content = remove_follow_up_questions_section(message["content"])
            # HTML 렌더링 활성화
            st.markdown(clean_content, unsafe_allow_html=True)
            
            # 참고 문서가 있는 경우에만 표시 (상단에 배치)
            if message["role"] == "assistant" and "reference_docs" in message and message["reference_docs"]:
                with st.expander("📚 참고 문서"):
                    for doc_idx, doc in enumerate(message["reference_docs"]):
                        st.markdown(f"**문서 {doc_idx+1}**")
                        st.markdown(f"```\n{doc['content']}\n```")
                        if "metadata" in doc and doc["metadata"]:
                            st.markdown(f"*메타데이터:* {doc['metadata']}")
            
            # 첫 번째 환영 메시지(인덱스 0)가 아닌 경우에만 후속 질문 표시
            if i > 0:  
                # 후속 질문 추출 또는 기본 질문 사용
                follow_up_questions = ["다른 관련 규정에 대해 알려주세요", 
                                      "이 내용을 더 자세히 설명해주세요", 
                                      "이 규정의 예외사항이 있나요?"]
                
                # 메시지에 후속 질문이 저장되어 있으면 그것 사용
                if "follow_up_questions" in message and message["follow_up_questions"]:
                    follow_up_questions = message["follow_up_questions"]
                # 아니면 내용에서 추출
                else:
                    extracted = extract_follow_up_questions(message["content"])
                    if extracted:
                        follow_up_questions = extracted
                
                # 후속 질문 버튼 표시
                st.write("---")
                st.write("**더 질문해보세요:**")
                
                # 버튼을 예시 질문과 동일한 스타일로 적용
                st.markdown("""
                <style>
                /* 예시 질문 버튼과 일치하는 스타일 적용 */
                .stButton > button {
                    background-color: white !important;
                    color: #1e3a8a !important;
                    border: 1px solid #e5e7eb !important;
                    text-align: left !important;
                    justify-content: flex-start !important;
                    border-radius: 8px !important;
                    padding: 0.5rem 1rem !important;
                    margin-bottom: 0.5rem !important;
                    font-size: 0.85rem !important;
                    font-weight: 400 !important;
                    box-shadow: none !important;
                    transition: all 0.2s ease;
                }
                
                .stButton > button:hover {
                    border-color: #2563eb !important;
                    color: #1e40af !important;
                    background-color: #f8fafc !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # 각 질문을 버튼으로 표시 - Streamlit 버튼 사용
                for idx, question in enumerate(follow_up_questions):
                    # 질문 내용의 해시값을 포함하여 고유한 키 생성
                    unique_key = f"follow_up_{i}_{idx}_{abs(hash(question)) % 10000}"
                    if st.button(question, key=unique_key, use_container_width=True):
                        add_user_message(question)
                        st.rerun()
        else:
            # 사용자 메시지는 그대로 표시
            st.markdown(message["content"])

# 답변되지 않은 user 메시지가 있는지 확인
messages = st.session_state.messages
has_pending_user_message = (
    len(messages) > 1
    and messages[-1]["role"] == "user"
    and (len(messages) == 2 or messages[-2]["role"] == "assistant")
)

# 답변 생성 부분
if has_pending_user_message:
    # 마지막 질문 메시지는 채팅 히스토리에서 이미 표시됨, 여기서는 표시하지 않음
    
    # 답변 생성 (UI에 직접 표시하지 않고 st.session_state.messages에만 추가)
    if "retriever" not in st.session_state or "qa" not in st.session_state:
        st.error("시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "❌ 시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
            "reference_docs": [],
            "follow_up_questions": ["시스템 재시작하기", "도움말 보기", "문서 확인하기"]
        })
        st.rerun()
    else:
        with st.spinner('🤔 답변 생성 중...'):
            try:
                # 대화 히스토리 형식 변환 (ConversationalRetrievalChain 형식에 맞게)
                current_question = messages[-1]["content"]
                
                # 검색 결과
                search_docs = st.session_state.retriever.get_relevant_documents(current_question)
                
                # 답변 생성 - 대화 히스토리 활용
                result = st.session_state.qa({"question": current_question, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
                
                # 대화 히스토리에 현재 질문-답변 쌍 추가
                st.session_state.chat_history.append((current_question, answer))
                
                # 후속 질문 생성 - 답변에서 추출 또는 기본값 사용
                follow_up_questions = extract_follow_up_questions(answer)
                if not follow_up_questions:
                    # 기본 후속 질문
                    follow_up_questions = [
                        "이 내용을 더 자세히 설명해주세요",
                        "관련 규정의 예외사항이 있나요?",
                        "비슷한 다른 사례가 있을까요?"
                    ]
                
                # 검색 결과가 있는 경우
                if search_docs:
                    # 참고 문서 정보 저장을 위한 형식 변환
                    reference_docs = []
                    
                    # 검색 문서 정보 저장 (모든 문서 포함)
                    for doc in search_docs:
                        # 메타데이터 저장
                        metadata = {}
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata = doc.metadata
                        
                        # 참고 문서 정보 저장
                        reference_docs.append({
                            "content": doc.page_content,
                            "metadata": metadata
                        })
                    
                    # 메시지 저장 (참고 문서 정보 포함)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "reference_docs": reference_docs,
                        "follow_up_questions": follow_up_questions
                    })
                
                # 검색 결과가 없는 경우
                else:
                    # 참고 문서 없이 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,  # 경고 메시지 없이
                        "reference_docs": [],
                        "follow_up_questions": follow_up_questions
                    })
            
            except Exception as e:
                error_message = f"검색 및 답변 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ {error_message}",
                    "reference_docs": [],
                    "follow_up_questions": ["다시 질문하기", "시스템 재시작하기", "다른 방식으로 질문하기"]
                })
                if DEBUG_MODE:
                    with st.expander("🔍 디버그 정보"):
                        st.code(traceback.format_exc(), language="python")
            
            # 처리 후 페이지 새로고침
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 채팅 입력 처리 (마지막에 렌더링)
chat_input = st.chat_input("KAIST 규정에 대해 궁금한 점을 물어보세요")
if chat_input:
    # 시스템 초기화 확인
    if "qa" not in st.session_state or "retriever" not in st.session_state:
        st.error("시스템이 아직 초기화되지 않았습니다. 페이지를 새로고침 후 다시 시도해주세요.")
    else:
        if add_user_message(chat_input):
            st.rerun()

# 벡터DB 생성/로드 부분은 여기서 처리
try:
    # 벡터DB가 없으면 생성, 있으면 로드
    need_rebuild = force_rebuild or not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR)
    
    if need_rebuild:
        docs = []
        for filename in os.listdir(HWP_DIR):
            if filename.endswith('.hwp'):
                try:
                    file_path = os.path.join(HWP_DIR, filename)
                    loader = HWPLoader(file_path)
                    file_docs = loader.load()
                    docs.extend(file_docs)
                except Exception as e:
                    st.error(f"HWP 파일 로드 실패 ({filename}): {str(e)}")
                    continue
        
        if not docs:
            st.error("로드된 문서가 없습니다. HWP 파일이 올바른지 확인하세요.")
            st.stop()
        
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
        except Exception as e:
            st.error(f"문서 분할 실패: {str(e)}")
            st.stop()
        
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                model=OPENAI_EMBEDDING_MODEL
            )
            # 메타데이터에 임베딩 정보 추가
            embedding_info_str = json.dumps({"type": "openai", "provider": "standard", "model": OPENAI_EMBEDDING_MODEL})
            metadata = {"embedding_info": embedding_info_str}
            # 벡터DB 생성
            db = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings, 
                persist_directory=CHROMA_DIR,
                collection_metadata={"embedding_info": embedding_info_str}
            )
            db.persist()
            # 메타데이터 파일로 저장
            with open(os.path.join(CHROMA_DIR, "embedding_info.json"), "w") as f:
                json.dump({"type": "openai", "provider": "standard", "model": OPENAI_EMBEDDING_MODEL}, f)
            st.success(f"벡터DB 생성 완료! 총 {len(splits)}개 문서 조각이 임베딩되었습니다.")
        except Exception as e:
            st.error(f"벡터DB 생성 실패: {str(e)}")
            if DEBUG_MODE:
                import traceback
                st.code(traceback.format_exc(), language="python")
            st.stop()
    else:
        # 이전에 사용한 임베딩 정보 로드
        embedding_info_path = os.path.join(CHROMA_DIR, "embedding_info.json")
        if os.path.exists(embedding_info_path):
            try:
                with open(embedding_info_path, "r") as f:
                    saved_embedding_info = json.load(f)
                # 사용자에게 저장된 임베딩 정보 안내
                if saved_embedding_info["type"] != "openai":
                    st.warning(f"주의: 벡터DB는 {saved_embedding_info['type']} 임베딩으로 생성되었으나, 현재 openai 임베딩을 선택하셨습니다. 검색 결과가 정확하지 않을 수 있습니다.")
            except Exception as e:
                st.warning(f"임베딩 정보 로드 실패: {e}. 기본 임베딩을 사용합니다.")
                saved_embedding_info = None
        else:
            st.warning("임베딩 정보 파일을 찾을 수 없습니다. 벡터DB를 재생성하는 것이 좋습니다.")
            saved_embedding_info = None
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                model=OPENAI_EMBEDDING_MODEL
            )
            db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception as e:
            st.error(f"벡터DB 로드 실패: {str(e)}")
            if DEBUG_MODE:
                import traceback
                st.code(traceback.format_exc(), language="python")
            st.stop()
    try:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def create_llm_with_retry():
            return ChatOpenAI(
                temperature=0,
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                model_name=OPENAI_MODEL,
                request_timeout=60,
                http_client=httpx.Client(verify=False),
            )
        llm = create_llm_with_retry()
        # --- System Prompt 추가 ---
        system_message = SystemMessagePromptTemplate.from_template(
            "너는 KAIST 회계규정에 대한 질문 및 답변을 전문적으로 처리하는 챗봇이야. 항상 친절하고 정확하게 답변해줘. "
            "답변할 때는 반드시 참고한 규정 내용이나 조항을 명시적으로 언급하고, 가능한 경우 규정명이나 조항 번호도 함께 언급해줘. "
            "확실하지 않거나 규정에 명시되지 않은 내용에 대해서는 '이 부분은 규정에 명확히 명시되어 있지 않습니다'라고 솔직하게 답변해줘. 추측하지 말고 알고 있는 내용만 답변해. "
            "답변은 마크다운을 활용해 다음과 같이 구성해줘:\n\n"
            "질문에 대한 직접적인 답변을 여기에 작성해줘. 핵심을 간결하고 명확하게 설명해.\n\n"
            "관련 규정 설명과 출처를 여기에 간결하게 명시해줘. 규정명, 조항 번호 등을 구체적으로 포함해. 이 부분은 작게 작성하고 너무 길지 않게 해."
        )
        human_message = HumanMessagePromptTemplate.from_template(
            "다음 맥락 정보를 바탕으로 질문에 답변해주세요. 맥락 정보에서 참고한 규정 출처를 반드시 답변에 포함해주세요.\n\n"
            "맥락: {context}\n\n"
            "질문: {question}\n\n"
            "이전 대화: {chat_history}\n\n"
            "참고: \n"
            "- 답변은 마크다운을 활용해 만드세요. 중요 내용은 **볼드체**로 강조하세요.\n"
            "- 답변에 규정 출처(규정명, 조항 등)를 반드시 포함하세요.\n"
            "- 답변에 규정 출처(규정명, 조항 등)는 기울임채로, 작은 글씨로 답변하세요.\n"
            "- 확실하지 않은 내용은 추측하지 말고 명확히 모른다고 답변하세요.\n"
            "- 답변 마지막에는 반드시 다음 형식으로 사용자가 질문과 답변 내용과 관련된 후속 질문 3개를 제안해주세요:\n\n"
            "#### 추천 질문\n"
            "1. [첫 번째 관련 질문]\n"
            "2. [두 번째 관련 질문]\n"
            "3. [세 번째 관련 질문]"
        )
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # 대화 메모리 생성
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        
        # ConversationalRetrievalChain 생성
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0,
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                model_name=OPENAI_MODEL,
                request_timeout=60,
                http_client=httpx.Client(verify=False),
            ),
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            memory=memory,
            return_source_documents=True,
            return_generated_question=True,
            verbose=DEBUG_MODE
        )
        # qa 변수를 session_state에 할당
        st.session_state.qa = qa
        
        # 디버깅용 로그
        if DEBUG_MODE:
            st.write(f"[DEBUG] QA 모델이 초기화되었습니다.")
        
        # Retriever도 session_state에 할당
        st.session_state.retriever = retriever
    except Exception as e:
        st.error(f"RAG 파이프라인 생성 실패: {str(e)}")
        if DEBUG_MODE:
            import traceback
            st.code(traceback.format_exc(), language="python")
        st.stop()
except Exception as e:
    st.error(f"앱 실행 중 예상치 못한 오류가 발생했습니다: {str(e)}")
    if DEBUG_MODE:
        import traceback
        st.code(traceback.format_exc(), language="python")

# 사용자 질문과 답변을 기반으로 관련 후속 질문 생성하는 함수
def generate_follow_up_questions(question, answer):
    """사용자 질문과 답변을 기반으로 관련 후속 질문 생성"""
    try:
        # 후속 질문 생성을 위한 프롬프트
        prompt = f"""
        다음 질문과 답변을 참고하여, 사용자가 이어서 물어볼 만한 관련 질문 3개를 생성해주세요.
        질문과 답변의 맥락을 유지하고, 사용자가 더 알고 싶어할 만한 내용을 다루는 질문이어야 합니다.
        
        사용자 질문: {question}
        답변: {answer}
        
        관련 질문 3개 (간결하게):
        """
        
        # 모델 사용하여 질문 생성
        response = st.session_state.qa.combine_docs_chain.llm.predict(prompt)
        
        # 생성된 텍스트에서 질문 추출
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 번호 붙은 행만 추출 (1. 2. 3. 등)
            if re.match(r'^\d+\.?\s+', line):
                # 번호 제거하고 질문만 추출
                question = re.sub(r'^\d+\.?\s+', '', line).strip()
                if question and len(question) > 10:
                    questions.append(question)
            
        # 최대 3개 질문 반환
        return questions[:3]
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"후속 질문 생성 실패: {str(e)}")
        # 오류 시 빈 리스트 반환
        return []
