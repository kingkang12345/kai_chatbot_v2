import pandas as pd
import numpy as np

# Excel 파일 읽기
df = pd.read_excel('backdata/20250521_미지급명세서_요청자료-반납여부 추가.xlsx', 
                   sheet_name='SQL Results', 
                   nrows=100)  # 상위 100행 읽기

# 기본 정보 출력
print("\n=== 데이터 기본 정보 ===")
print(f"행 수: {len(df)}")
print(f"열 수: {len(df.columns)}")

# 컬럼 목록과 데이터 타입 출력
print("\n=== 컬럼 목록 및 데이터 타입 ===")
for i, col in enumerate(df.columns, 1):
    # null이 아닌 값의 개수
    non_null_count = df[col].count()
    # null 비율
    null_ratio = (len(df) - non_null_count) / len(df) * 100
    # 유니크 값 개수
    unique_count = df[col].nunique()
    
    print(f"{i}. {col}")
    print(f"   - 데이터 타입: {df[col].dtype}")
    print(f"   - Null 비율: {null_ratio:.1f}%")
    print(f"   - 유니크 값 수: {unique_count}")
    
    # 숫자형 컬럼인 경우 기본 통계
    if np.issubdtype(df[col].dtype, np.number):
        print(f"   - 최소값: {df[col].min()}")
        print(f"   - 최대값: {df[col].max()}")
        print(f"   - 평균값: {df[col].mean():.2f}")
    # 문자열 컬럼인 경우 샘플 값
    elif df[col].dtype == 'object':
        sample_values = df[col].dropna().unique()[:3]  # 최대 3개의 고유값 표시
        print(f"   - 샘플 값: {', '.join(str(x) for x in sample_values)}")
    print()

# 데이터 샘플 출력
print("\n=== 데이터 샘플 (처음 5행) ===")
print(df.head().to_string())

# 주요 통계 정보
print("\n=== 숫자형 컬럼 통계 정보 ===")
print(df.describe().to_string())

# 컬럼 간 관계 분석
print("\n=== 관련 컬럼 그룹 ===")
# 비슷한 이름을 가진 컬럼들을 그룹화
column_groups = {}
for col in df.columns:
    col_lower = col.lower()
    # 주요 키워드로 그룹화
    for keyword in ['금액', '일자', '번호', '코드', '명', '상태', '구분']:
        if keyword in col_lower:
            if keyword not in column_groups:
                column_groups[keyword] = []
            column_groups[keyword].append(col)

for keyword, cols in column_groups.items():
    if cols:
        print(f"\n{keyword} 관련 컬럼:")
        for col in cols:
            print(f"  - {col}") 