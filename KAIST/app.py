import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from io import BytesIO
import msoffcrypto

st.set_page_config(page_title="AI 기반 지출 이상 탐지", layout="wide")
st.title("💡 AI 기반 지출 이상거래 탐지 시스템")

uploaded_file = st.file_uploader("📤 Excel/CSV 파일 업로드 (지출 데이터)", type=["xlsx", "csv"])
password = st.text_input("엑셀 파일 암호 (필요시 입력)", type="password")

if uploaded_file:
    st.write("[DEBUG] 파일 업로드 완료: ", uploaded_file.name)
    try:
        if uploaded_file.name.endswith('.csv'):
            st.write("[DEBUG] CSV 파일로 인식, 읽기 시도")
            df = pd.read_csv(uploaded_file)
            st.write("[DEBUG] CSV DataFrame 미리보기:")
            st.write(df.head())
        elif uploaded_file.name.endswith('.xlsx'):
            if password:
                st.write("[DEBUG] 암호 입력됨, msoffcrypto로 해제 시도")
                try:
                    decrypted = BytesIO()
                    uploaded_file.seek(0)
                    office_file = msoffcrypto.OfficeFile(uploaded_file)
                    office_file.load_key(password=password)
                    office_file.decrypt(decrypted)
                    decrypted.seek(0)
                    st.write("[DEBUG] 암호 해제 성공, 엑셀 읽기 시도")
                    df = pd.read_excel(decrypted, engine='openpyxl')
                    st.write("[DEBUG] 암호 해제된 엑셀 DataFrame 미리보기:")
                    st.write(df.head())
                except Exception as e:
                    st.error(f"암호화된 파일을 해제하는 중 오류가 발생했습니다: {e}")
                    st.stop()
            else:
                st.write("[DEBUG] 암호 미입력, 일반 엑셀로 읽기 시도")
                try:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    st.write("[DEBUG] 엑셀 DataFrame 미리보기:")
                    st.write(df.head())
                except Exception:
                    uploaded_file.seek(0)
                    st.write("[DEBUG] 엑셀로 읽기 실패, CSV로 재시도")
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.write("[DEBUG] 엑셀->CSV DataFrame 미리보기:")
                        st.write(df.head())
                    except Exception:
                        st.error("파일을 읽을 수 없습니다. 파일이 손상되었거나 올바른 형식이 아닙니다.")
                        st.stop()
        else:
            st.error("지원하지 않는 파일 형식입니다. xlsx 또는 csv 파일을 업로드해주세요.")
            st.stop()
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
    st.write("[DEBUG] DataFrame 컬럼:", df.columns.tolist())

    # 날짜 컬럼 변환
    for col in ["거래일자", "생성일자"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    st.write("[DEBUG] 날짜 컬럼 변환 및 전처리 완료")

    # 중복 청구 탐지
    df['중복의심'] = df.duplicated(subset=['계좌번호', '예금주', '소계', '내용'], keep=False)

    # 사전지급 탐지
    df['사전지급의심'] = df['거래일자'] < df['생성일자']

    # 소득유형-계정 불일치
    def 소득유형_계정_불일치(row):
        if pd.isna(row['소득유형']) or pd.isna(row['계정']):
            return False
        if '갑근' in row['소득유형'] and '과세' not in str(row['계정']):
            return True
        if '비과세' in row['소득유형'] and '비과세' not in str(row['계정']):
            return True
        return False

    df['소득유형계정불일치'] = df.apply(소득유형_계정_불일치, axis=1)

    # 기안자 반복 청구
    df['기안자지급패턴'] = df.groupby(['기안자성명', '지급거래처'])['소계'].transform('count')
    df['기안자반복의심'] = df['기안자지급패턴'] > 1

    # 종합 의심 사유
    def 종합사유(row):
        사유 = []
        if row['중복의심']: 사유.append('중복청구')
        if row['사전지급의심']: 사유.append('사전지급')
        if row['소득유형계정불일치']: 사유.append('소득유형-계정 불일치')
        if row['기안자반복의심']: 사유.append('기안자 반복지급')
        return ', '.join(사유)

    df['의심사유'] = df.apply(종합사유, axis=1)

    st.markdown("### 🔍 분석 결과 요약")
    st.dataframe(df[['결의서번호', '문서번호', '기안자성명', '지급거래처', '소계', '거래일자', '생성일자', '의심사유']])
    st.write("[DEBUG] 분석 결과 DataFrame 미리보기:")
    st.write(df[['결의서번호', '문서번호', '기안자성명', '지급거래처', '소계', '거래일자', '생성일자', '의심사유']].head())

    # 다운로드 버튼
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button("📥 결과 다운로드 (Excel)", data=output.getvalue(), file_name="분석결과.xlsx")

