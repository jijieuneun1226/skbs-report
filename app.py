import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import numpy as np
import io
import requests

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 권한 제어
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SKBS Sales Report", layout="wide", initial_sidebar_state="expanded")

# URL 파라미터 읽기
params = st.query_params
is_edit_mode = params.get("mode") == "edit"

if not is_edit_mode:
    st.markdown("<style>[data-testid='stSidebar'] {display: none;} section[data-testid='stSidebar'] {width: 0px;}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
    div.block-container {padding-top: 1rem;}
    .metric-card {background-color: #f8f9fa; border-left: 5px solid #4e79a7; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
    .info-box {padding: 15px; border-radius: 5px; font-size: 14px; margin-bottom: 20px; border: 1px solid #e0e0e0; background-color: #ffffff;}
</style>
""", unsafe_allow_html=True)

st.title("📊 SKBS Sales Report")

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 함수
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_from_drive(file_id):
    # 구글 드라이브 다운로드 직링크
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)
        # 엑셀 파일 읽기
        df = pd.read_excel(file_bytes, engine='openpyxl')
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

    # 컬럼명 정리 및 맵핑
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        '매출일자': ['매출일자', '날짜', 'Date', '일자'],
        '제품명': ['제품명 변환', '제품명변환', '제품명', '품목명'],
        '합계금액': ['합계금액', '매출액', '금액'],
        '수량': ['수량', '판매수량'],
        '사업자번호': ['사업자번호', '사업자등록번호'],
        '거래처명': ['거래처명', '병원명'],
        '진료과': ['진료과', '진료과목'],
        '제품군': ['제품군', '카테고리'],
        '거래처그룹': ['거래처그룹', '그룹'],
        '주소': ['주소', 'Address', '사업장주소'],
        '지역': ['지역']
    }
    
    current_cols = {c.replace(' ', ''): c for c in df.columns}
    for std_col, candidates in col_map.items():
        if std_col in df.columns: continue
        for cand in candidates:
            clean_cand = cand.replace(' ', '')
            for clean_real, real in current_cols.items():
                if clean_real == clean_cand:
                    df.rename(columns={real: std_col}, inplace=True)
                    break
            if std_col in df.columns: break

    try:
        # 지역 데이터 보정
        if '지역' not in df.columns and '주소' in df.columns:
            df['지역_임시'] = df['주소'].astype(str).str.split().str[0]
            df['지역'] = df['지역_임시'].str[:2] # 앞 두글자만 추출 (서울, 경기 등)
        elif '지역' not in df.columns:
             df['지역'] = '미분류'

        # 날짜 및 숫자 변환
        if '매출일자' in df.columns:
            df['매출일자'] = pd.to_datetime(df['매출일자'], errors='coerce')
            df = df.dropna(subset=['매출일자'])
            df['년'] = df['매출일자'].dt.year
            df['분기'] = df['매출일자'].dt.quarter
            df['월'] = df['매출일자'].dt.month
            df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        for col in ['합계금액', '수량']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
                
        df['매출액'] = df.get('합계금액', 0) / 1000000
        
        # 판매채널 분류
        def classify_channel(group):
            online_list = ['B2B', 'B2B(W)', 'SAP', '의사회원']
            return 'online' if group in online_list else ('offline' if group == 'SDP' else '기타')
        
        if '거래처그룹' in df.columns:
            df['판매채널'] = df['거래처그룹'].apply(classify_channel)
        else:
            df['판매채널'] = '기타'
             
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return pd.DataFrame()
    
    return df

# --------------------------------------------------------------------------------
# 3. 데이터 실행 및 필터링
# --------------------------------------------------------------------------------
# 파일 ID를 정확한 위치에 넣었습니다.
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. 파일 ID와 구글 드라이브 공유 설정을 확인하세요.")
    st.stop()

# 이후 리포트 출력 로직 (생략된 부분은 기존 코드와 동일하게 작동하도록 구성됨)
# ... [기존의 탭 구성 및 차트 출력 코드가 이 아래에 붙으면 됩니다] ...

st.success("데이터 로드 완료!")
st.write(df_raw.head()) # 데이터가 잘 나오는지 확인용
