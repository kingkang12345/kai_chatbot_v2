import os
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import re

# LangChain 컴포넌트 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 디렉토리 설정
BACKDATA_DIR = os.path.join(os.path.dirname(__file__), 'backdata')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')
os.makedirs(BACKDATA_DIR, exist_ok=True)

# API 키 설정 - 기존 앱과 동일한 설정 사용
import openai
import httpx
import ssl
import urllib3

# 디버그 모드 활성화
DEBUG_MODE = False

# SSL 검증 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# API 키 설정
openai.api_key = "sk-k7ZAoJmlclL75pjwHgEcFw"
openai.api_base = "https://genai-sharedservice-americas.pwcinternal.com/"

# 페이지 설정
st.set_page_config(page_title="🔍 KAIST 미지급금명세서 검증 도구", layout="wide")

# CSS 스타일링
st.markdown("""
<style>
/* 전체 앱 스타일링 */
.stApp {
    background-color: white;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}

/* 헤더 스타일링 */
.header-container {
    padding: 1rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
}

/* 타이틀 스타일링 */
.main-title {
    color: #0a1c3e;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0;
    padding: 0;
    line-height: 1.2;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #4b5563 !important;
    font-size: 1.1rem !important;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* 카드 스타일링 */
.card {
    background-color: white;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
}

/* 테이블 스타일링 */
.dataframe {
    width: 100%;
    border-collapse: collapse;
}

.dataframe th {
    background-color: #f1f5f9;
    font-weight: 600;
    text-align: left;
    padding: 0.75rem 1rem;
    border-bottom: 2px solid #e2e8f0;
}

.dataframe td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #e2e8f0;
}

/* 위반 항목 강조 */
.violation {
    background-color: #fee2e2;
    color: #b91c1c;
    font-weight: 500;
}

/* 버튼 스타일링 */
.stButton button {
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: #1e40af;
}

/* 위젯 간격 조정 */
.stSelectbox, .stDateInput {
    margin-bottom: 1rem;
}

/* 로딩 스피너 */
.stSpinner {
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* 알림 메시지 */
.st-success {
    background-color: #ecfdf5;
    color: #065f46;
}

.st-error {
    background-color: #fef2f2;
    color: #b91c1c;
}

.st-info {
    background-color: #eff6ff;
    color: #1e40af;
}

/* 테이블 내 상태 표시 */
.status-ok {
    color: #047857;
    font-weight: 500;
}

.status-warning {
    color: #b45309;
    font-weight: 500;
}

.status-violation {
    color: #b91c1c;
    font-weight: 500;
}

/* 세부 정보 패널 */
.detail-panel {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid #e2e8f0;
}

/* 규정 인용 스타일 */
.regulation-citation {
    background-color: #f1f5f9;
    padding: 0.75rem;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# 헤더 컴포넌트
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">KAIST 미지급금명세서 검증 도구</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">회계 규정에 따른 미지급금명세서 항목 검증 및 위반 사항 식별</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 벡터DB 로드 함수
@st.cache_resource
def load_vector_db():
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            model="azure.text-embedding-3-large"
        )
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # LLM 설정
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            model_name="openai.gpt-4.1-mini-2025-04-14",
            request_timeout=60,
            http_client=httpx.Client(verify=False),
        )
        
        # 검증용 프롬프트 템플릿
        system_message = SystemMessagePromptTemplate.from_template(
            "너는 KAIST 회계규정에 따라 미지급금명세서 항목을 검증하는 도구야. "
            "각 항목이 규정을 준수하는지 검토하고, 위반 가능성이 있는 항목은 관련 규정과 함께 명확히 표시해줘."
        )
        human_message = HumanMessagePromptTemplate.from_template(
            "다음 미지급금명세서 항목이 KAIST 회계규정을 준수하는지 검토해줘:\n\n"
            "항목 정보: {item_info}\n\n"
            "규정 맥락: {context}\n\n"
            "이 항목이 규정을 위반하는지 여부와 그 이유를 JSON 형식으로 답변해줘:\n"
            "```json\n"
            "{{\n"
            "  \"violation\": true/false,\n"
            "  \"violation_type\": \"위반 유형 (예: 한도초과, 미승인지출, 증빙부족 등)\",\n"
            "  \"explanation\": \"위반 이유에 대한 상세 설명\",\n"
            "  \"regulation_reference\": \"관련 규정 및 조항 번호\"\n"
            "}}\n"
            "```\n"
            "규정을 위반하지 않는 경우에는 violation을 false로 설정하고, violation_type은 \"없음\"으로 설정해줘."
        )
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # QA 체인 생성
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa, retriever
    except Exception as e:
        st.error(f"벡터DB 로드 실패: {str(e)}")
        return None, None

# 자동 열 매핑 함수 추가
def auto_map_columns(df_columns):
    # 필수 필드와 가능한 매칭되는 컬럼명들의 매핑
    field_mappings = {
        # 핵심 금액 관련 필드
        '금액': ['결의금액', '소계', '공급가액'],
        '부가세': ['부가가치세'],
        '미지급금': ['미지급금'],
        
        # 거래 정보 필드
        '지출일자': ['거래일자', '지급예정일'],
        '지출내역': ['내용', '계정명'],
        '지출처': ['지급거래처', '거래처'],
        '증빙유형': ['증빙유형'],
        
        # 관리/추적 필드
        '문서번호': ['결의서번호', '문서번호'],
        '정산상태': ['정산대상상태', '결의상태'],
        '반납여부': ['반납여부'],
        
        # 부가 정보 필드
        '소득구분': ['소득유형'],
        '과세여부': ['과세사업여부'],
        '지급그룹': ['지급그룹'],
        '처리부서': ['기안자부서']
    }
    
    # 결과 매핑 저장
    mapping_result = {}
    
    # 각 필드에 대해 매칭되는 컬럼 찾기
    for field, possible_matches in field_mappings.items():
        # 우선순위 순서대로 매칭 시도
        for match in possible_matches:
            if match in df_columns:
                mapping_result[field] = match
                break
        
        # 매칭 실패시 None으로 설정
        if field not in mapping_result:
            mapping_result[field] = None
    
    return mapping_result

# 미지급금명세서 데이터 로드 함수 수정
def load_unpaid_data():
    files = []
    for file in os.listdir(BACKDATA_DIR):
        if file.endswith('.csv') or file.endswith('.xlsx') or file.endswith('.xls'):
            files.append(file)
    
    if not files:
        st.warning("backdata 폴더에 미지급금명세서 데이터 파일이 없습니다. CSV 또는 Excel 파일을 해당 폴더에 추가해주세요.")
        return None, None
    
    # 파일 선택 (여러 파일이 있는 경우)
    selected_file = st.selectbox("분석할 파일 선택", files)
    file_path = os.path.join(BACKDATA_DIR, selected_file)
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8', nrows=100)  # 상위 100줄만 읽기
        else:
            df = pd.read_excel(file_path, nrows=100)  # 상위 100줄만 읽기
        
        # 자동 열 매핑 수행
        auto_mapping = auto_map_columns(df.columns)
        
        # 전체 데이터 다시 읽기
        if file_path.endswith('.csv'):
            full_df = pd.read_csv(file_path, encoding='utf-8')
        else:
            full_df = pd.read_excel(file_path)
        
        return full_df, auto_mapping
    except Exception as e:
        st.error(f"파일 로드 실패: {str(e)}")
        return None, None

# 항목 검증 함수
def validate_item(item, qa):
    try:
        # 기본 검증 규칙 적용
        validation_checks = []
        
        # 1. 금액 관련 검증
        if '금액' in item and '부가세' in item:
            total = float(item['금액'])
            tax = float(item.get('부가세', 0))
            if tax > total * 0.1:  # 부가세가 공급가액의 10%를 초과하는 경우
                validation_checks.append("부가세 금액이 비정상적으로 높습니다.")
        
        # 2. 미지급금 검증
        if '미지급금' in item and float(item['미지급금']) > 0:
            validation_checks.append("미지급금이 존재합니다.")
        
        # 3. 증빙 검증
        if '증빙유형' in item and pd.isna(item['증빙유형']):
            validation_checks.append("증빙서류가 누락되었습니다.")
        
        # 4. 반납 여부 검증
        if '반납여부' in item and item['반납여부'] == 'Y':
            validation_checks.append("반납 처리된 항목입니다.")
        
        # 5. 과세 처리 검증
        if '과세여부' in item and '소득구분' in item:
            if item['과세여부'] == 'Y' and pd.isna(item['소득구분']):
                validation_checks.append("과세 대상이나 소득구분이 누락되었습니다.")
        
        # 항목 정보 문자열 구성
        basic_info = []
        for key, value in item.items():
            if not pd.isna(value):  # null이 아닌 값만 포함
                basic_info.append(f"{key}: {value}")
        
        item_info = "\n".join(basic_info)
        if validation_checks:
            item_info += "\n\n기본 검증 결과:\n" + "\n".join(f"- {check}" for check in validation_checks)
        
        # 규정 검색 및 검증
        result = qa({"query": f"미지급금명세서 항목 검증: {item_info}"})
        answer = result["result"]
        
        # JSON 응답 추출 및 기본 검증 결과 통합
        json_match = re.search(r'```json\s*(.*?)\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            response = json.loads(json_str)
            
            # 기본 검증 결과 통합
            if validation_checks:
                response["violation"] = True
                existing_explanation = response.get("explanation", "")
                checks_text = "\n- " + "\n- ".join(validation_checks)
                response["explanation"] = f"기본 검증 결과:{checks_text}\n\n규정 검증 결과:\n{existing_explanation}"
        else:
            # JSON을 찾을 수 없는 경우
            response = {
                "violation": bool(validation_checks),
                "violation_type": "기본 검증" if validation_checks else "분석 불가",
                "explanation": "\n".join(validation_checks) if validation_checks else "응답에서 JSON 형식을 찾을 수 없습니다.",
                "regulation_reference": "N/A"
            }
        
        return response
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"항목 검증 중 오류 발생: {str(e)}")
        return {
            "violation": True,
            "violation_type": "검증 오류",
            "explanation": f"검증 처리 중 오류가 발생했습니다: {str(e)}",
            "regulation_reference": "N/A"
        }

# 규정 위반 의심 항목에 대한 대시보드 표시
def display_dashboard(df, results, summary):
    """결과 대시보드 표시 - 최적화 버전"""
    # 1. 전체 요약
    st.subheader("📊 검증 결과 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 데이터", f"{summary['전체 데이터 수']:,}건")
    with col2:
        st.metric("기본 규칙 위반", f"{summary['기본 규칙 위반 수']:,}건")
    with col3:
        st.metric("GPT 검증 수행", f"{summary['GPT 검증 수행 수']:,}건")
    with col4:
        st.metric("GPT 검증 위반", f"{summary['GPT 검증 위반 수']:,}건")
    
    # 2. 데이터 필터링 옵션
    st.subheader("🔍 상세 결과")
    filter_option = st.radio(
        "표시 항목:",
        ["전체 항목", "규칙 위반 항목만", "GPT 검증 항목만"],
        horizontal=True
    )
    
    # 3. 필터링된 결과 표시
    filtered_df = df.copy()
    if filter_option == "규칙 위반 항목만":
        mask = results['기본규칙위반'] | (results['GPT검증결과'] == True)
        filtered_df = df[mask]
    elif filter_option == "GPT 검증 항목만":
        filtered_df = df[results['GPT검증수행']]
    
    # 결과 컬럼 추가
    filtered_df['위반여부'] = results['기본규칙위반']
    filtered_df['GPT검증'] = results['GPT검증수행']
    filtered_df['검증결과'] = results['GPT검증설명']
    
    # 스타일 적용
    st.dataframe(
        filtered_df.style.apply(lambda x: ['background-color: #fee2e2' if x['위반여부'] else '' for _ in x], axis=1),
        height=400
    )
    
    # 4. 결과 다운로드 버튼
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="검증 결과 CSV 다운로드",
        data=csv,
        file_name=f'미지급금명세서_검증결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )

def filter_by_included_prefixes(df):
    """특정 접두어를 가진 문서번호만 분석에 포함"""
    if '문서번호' not in df.columns:
        return df, 0
    
    # 포함할 접두어 목록
    included_prefixes = [
        'GEX', 'POD', 'MEX', 'TED', 'TEF', 'RRA', 'RSA'
    ]
    
    # 문서번호 접두어 확인을 위한 함수
    def has_included_prefix(doc_num):
        if pd.isna(doc_num):
            return False
        doc_str = str(doc_num).upper()
        for prefix in included_prefixes:
            if doc_str.startswith(prefix):
                return True
        return False
    
    # 포함할 항목 필터링
    mask = df['문서번호'].apply(has_included_prefix)
    filtered_df = df[mask]
    
    # 제외된 항목 수 반환
    excluded_count = len(df) - len(filtered_df)
    
    return filtered_df, excluded_count

def validate_basic_rules(df):
    """기본 규칙 기반 1차 검증 - 벡터화 연산으로 고속화"""
    violations = {}
    
    # 모든 검증을 한 번에 처리
    if '결의금액' in df.columns and '부가가치세' in df.columns:
        violations['부가세초과'] = (df['부가가치세'] > df['결의금액'] * 0.1)
    if '증빙유형' in df.columns:
        violations['증빙누락'] = df['증빙유형'].isna()
    if '미지급금' in df.columns:
        violations['미지급잔액'] = (df['미지급금'] > 0)
    if '과세사업여부' in df.columns and '소득유형' in df.columns:
        violations['과세누락'] = (df['과세사업여부'] == 'Y') & (df['소득유형'].isna())
    if '반납여부' in df.columns:
        violations['반납처리'] = (df['반납여부'] == 'Y')
    
    # 결과를 한 번에 DataFrame으로 변환
    return pd.DataFrame(violations, index=df.index)

def select_validation_targets(df, basic_violations, max_samples=100):
    """GPT 검증 대상 선정 - 최대 100개로 제한"""
    # 1. 기본 규칙 위반 심각도 계산
    violation_count = basic_violations.sum(axis=1)
    severe_violations = violation_count[violation_count >= 2]  # 2개 이상 규칙 위반
    
    # 2. 고액 거래 (상위 0.5%)
    if '결의금액' in df.columns:
        high_amount = df.nlargest(min(int(len(df) * 0.005), 50), '결의금액').index
    else:
        high_amount = pd.Index([])
    
    # 3. 우선순위 기반 대상 선정
    priority_items = pd.Index(list(set(severe_violations.index) | set(high_amount)))
    
    # 4. 추가 샘플링 (필요한 경우)
    remaining_count = max_samples - len(priority_items)
    if remaining_count > 0 and len(df) > len(priority_items):
        # 아직 선택되지 않은 항목에서 랜덤 샘플링
        remaining_items = df.index.difference(priority_items)
        random_samples = np.random.choice(remaining_items, 
                                        size=min(remaining_count, len(remaining_items)), 
                                        replace=False)
        final_targets = priority_items.union(random_samples)
    else:
        final_targets = priority_items
    
    return df.loc[final_targets]

def process_large_dataset(df, qa):
    """대량 데이터 처리 - 최적화 버전"""
    # 1. 기본 규칙 검증 (벡터화 연산)
    basic_violations = validate_basic_rules(df)
    
    # 2. GPT 검증 대상 선정 (최대 100개)
    target_items = select_validation_targets(df, basic_violations)
    total_targets = len(target_items)
    
    st.write(f"전체 {len(df):,}개 중 {total_targets}개 항목에 대해 상세 검증을 수행합니다.")
    
    # 3. 선정된 항목들에 대해 GPT 검증 수행
    gpt_results = []
    
    with st.spinner(f'GPT 검증 진행 중... (0/{total_targets})'):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, (_, row) in enumerate(target_items.iterrows(), 1):
            # GPT 검증
            result = validate_item(row.to_dict(), qa)
            gpt_results.append(result)
            
            # 진행상태 업데이트 (10개 단위로)
            if i % 10 == 0 or i == total_targets:
                progress = i / total_targets
                progress_bar.progress(progress)
                progress_text.text(f'GPT 검증 진행 중... ({i}/{total_targets})')
    
    # 4. 결과 종합 (벡터화 연산)
    final_results = pd.DataFrame(index=df.index)
    final_results['기본규칙위반'] = basic_violations.any(axis=1)
    final_results['위반규칙수'] = basic_violations.sum(axis=1)
    
    # GPT 검증 결과 매핑
    final_results['GPT검증수행'] = final_results.index.isin(target_items.index)
    
    # GPT 검증 결과를 데이터프레임으로 변환
    gpt_df = pd.DataFrame(gpt_results, index=target_items.index)
    final_results.loc[target_items.index, 'GPT검증결과'] = gpt_df['violation']
    final_results.loc[target_items.index, 'GPT검증설명'] = gpt_df['explanation']
    
    # 5. 위반 항목 요약
    violation_summary = {
        '전체 데이터 수': len(df),
        '기본 규칙 위반 수': basic_violations.any(axis=1).sum(),
        'GPT 검증 수행 수': len(target_items),
        'GPT 검증 위반 수': gpt_df['violation'].sum() if not gpt_df.empty else 0
    }
    
    return final_results, violation_summary

# 메인 앱 실행 흐름 수정
def main():
    # 사이드바에 앱 설명
    with st.sidebar:
        st.markdown("## 🔍 앱 사용법")
        st.markdown("""
        1. `backdata` 폴더에 미지급금명세서 CSV 또는 Excel 파일을 준비합니다.
        2. 앱이 자동으로 폴더 내 파일을 감지하여 표시합니다.
        3. 파일을 선택하고 '검증 시작' 버튼을 클릭합니다.
        4. 각 항목이 KAIST 회계 규정을 준수하는지 자동 검증합니다.
        5. 검증 결과를 확인하고 필요시 CSV로 다운로드합니다.
        """)
        
        st.markdown("## ⚠️ 주의사항")
        st.markdown("""
        - 이 도구는 규정 위반 가능성이 있는 항목을 식별하는 것이며, 최종 판단은 회계 담당자가 해야 합니다.
        - 규정 해석에 따라 결과가 달라질 수 있습니다.
        - 완전한 자동화가 아닌 검토 보조 도구로 사용하세요.
        """)
    
    # 벡터DB 로드
    qa, retriever = load_vector_db()
    if qa is None or retriever is None:
        st.error("벡터DB 로드에 실패했습니다. 'chroma_db' 폴더가 존재하고 올바른 형식인지 확인하세요.")
        return
    
    # 데이터 로드 (자동 매핑 포함)
    df, auto_mapping = load_unpaid_data()
    if df is None:
        return
    
    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df.head(), height=200)
    
    # 열 매핑 섹션
    st.subheader("데이터 열 매핑")
    
    # 자동 매핑 결과 표시
    st.markdown("### 🤖 자동 매핑 결과")
    mapping_status = pd.DataFrame({
        '필수 필드': auto_mapping.keys(),
        '매핑된 열': [auto_mapping[k] if auto_mapping[k] else "매핑 실패" for k in auto_mapping.keys()],
        '상태': ['✅ 성공' if auto_mapping[k] else '❌ 실패' for k in auto_mapping.keys()]
    })
    st.dataframe(mapping_status, height=250)
    
    # 매핑 실패한 필드가 있는지 확인
    failed_mappings = [field for field, value in auto_mapping.items() if not value]
    
    if failed_mappings:
        st.warning(f"다음 필드의 자동 매핑에 실패했습니다: {', '.join(failed_mappings)}")
        st.info("수동으로 매핑을 진행해주세요.")
        
        # 수동 매핑 UI
        columns = df.columns.tolist()
        col1, col2 = st.columns(2)
        
        with col1:
            for i, field in enumerate(failed_mappings[:len(failed_mappings)//2 + len(failed_mappings)%2]):
                auto_mapping[field] = st.selectbox(
                    f"{field} 열 선택",
                    options=["선택하지 않음"] + columns,
                    key=f"manual_mapping_{field}"
                )
        
        with col2:
            for i, field in enumerate(failed_mappings[len(failed_mappings)//2 + len(failed_mappings)%2:]):
                auto_mapping[field] = st.selectbox(
                    f"{field} 열 선택",
                    options=["선택하지 않음"] + columns,
                    key=f"manual_mapping_{field}"
                )
    
    # 검증 시작 버튼
    if st.button("검증 시작", use_container_width=True):
        # 필수 필드가 모두 매핑되었는지 확인
        unmapped_fields = [field for field, value in auto_mapping.items() if not value or value == "선택하지 않음"]
        if unmapped_fields:
            st.error(f"다음 필수 필드가 매핑되지 않았습니다: {', '.join(unmapped_fields)}")
            return
        
        # 매핑된 열만 포함하는 데이터프레임 생성
        mapped_df = pd.DataFrame()
        for field, column in auto_mapping.items():
            mapped_df[field] = df[column]
        
        # 1. 특정 접두어 문서번호 필터링 - 수정된 부분
        with st.spinner("데이터 필터링 중..."):
            filtered_df, excluded_count = filter_by_included_prefixes(mapped_df)
            st.info(f"문서번호 필터링: 전체 {len(mapped_df):,}개 중 {len(filtered_df):,}개가 포함되었습니다. (제외: {excluded_count:,}개)")
            
            if len(filtered_df) == 0:
                st.warning("필터링 후 분석할 데이터가 없습니다.")
                return
        
        # 2. 기본 규칙 검증 (벡터화 연산으로 빠르게 처리)
        with st.spinner("기본 규칙 검증 중..."):
            violations = validate_basic_rules(filtered_df)
            filtered_df['위반여부'] = violations.any(axis=1)
            filtered_df['위반규칙수'] = violations.sum(axis=1)
            
            # 위반 항목 개수
            violation_count = filtered_df['위반여부'].sum()
            total_count = len(filtered_df)
            
            # 메트릭 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 항목 수", f"{total_count:,}개")
            with col2:
                st.metric("규칙 위반 항목 수", f"{violation_count:,}개")
            with col3:
                violation_pct = (violation_count / total_count * 100) if total_count > 0 else 0
                st.metric("규칙 위반 비율", f"{violation_pct:.1f}%")
        
        # 3. 필터링 옵션
        filter_option = st.radio(
            "표시 항목:",
            ["전체 항목", "규칙 위반 항목만"],
            horizontal=True
        )
        
        # 4. 필터링된 결과 표시
        display_df = filtered_df if filter_option == "전체 항목" else filtered_df[filtered_df['위반여부']]
        
        # 5. 데이터프레임 표시
        st.dataframe(
            display_df.style.apply(
                lambda x: ['background-color: #fee2e2' if x['위반여부'] else '' for _ in x], 
                axis=1
            ),
            height=400
        )
        
        # 6. 결과 다운로드 버튼
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="검증 결과 CSV 다운로드",
            data=csv,
            file_name=f'미지급금명세서_검증결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )
        
        # 7. GPT 검증 옵션 (별도 버튼으로)
        st.subheader("🤖 GPT 규정 검증 (선택사항)")
        st.warning("GPT 검증은 시간이 오래 걸릴 수 있습니다. 필요한 경우에만 실행하세요.")
        
        # GPT 검증 버튼
        col1, col2 = st.columns([1, 2])
        with col1:
            sample_size = st.number_input(
                "검증할 항목 수", 
                min_value=1, 
                max_value=min(30, violation_count), 
                value=min(10, violation_count)
            )
        
        with col2:
            if st.button("선택한 개수만큼 GPT로 상세 검증", use_container_width=True):
                # 위반 항목 중에서 무작위 샘플링
                if violation_count > 0:
                    gpt_targets = filtered_df[filtered_df['위반여부']].sample(n=min(sample_size, violation_count))
                    
                    # GPT 검증 수행
                    with st.spinner(f'GPT 검증 진행 중... (0/{len(gpt_targets)})'):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        gpt_results = []
                        for i, (_, row) in enumerate(gpt_targets.iterrows(), 1):
                            # GPT 검증
                            result = validate_item(row.to_dict(), qa)
                            gpt_results.append(result)
                            
                            # 진행상태 업데이트
                            progress = i / len(gpt_targets)
                            progress_bar.progress(progress)
                            progress_text.text(f'GPT 검증 진행 중... ({i}/{len(gpt_targets)})')
                        
                        # GPT 검증 결과 표시
                        st.subheader("GPT 검증 결과")
                        for i, (_, row) in enumerate(gpt_targets.iterrows()):
                            with st.expander(f"항목 {i+1}: {row.get('지출내역', '')[:50]}"):
                                st.write("**항목 정보:**")
                                for key, val in row.items():
                                    if not pd.isna(val):
                                        st.write(f"- {key}: {val}")
                                
                                st.write("**검증 결과:**")
                                if gpt_results[i]["violation"]:
                                    st.error(gpt_results[i]["explanation"])
                                else:
                                    st.success("규정 준수: " + gpt_results[i]["explanation"])
                                
                                if gpt_results[i]["regulation_reference"] != "N/A":
                                    st.info(f"**관련 규정:** {gpt_results[i]['regulation_reference']}")
                else:
                    st.info("규칙 위반 항목이 없어 GPT 검증을 수행할 수 없습니다.")

if __name__ == "__main__":
    main() 