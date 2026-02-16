import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import numpy as np
import io
import requests

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SKBS Sales Report", layout="wide", initial_sidebar_state="expanded")

# 관리자 모드 체크
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
# 2. 데이터 로드 및 전처리 (ID 고정 버전)
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

    # 컬럼 표준화
    df.columns = df.columns.astype(str).str.strip()
    col_map = {
        '매출일자': ['매출일자', '날짜', 'Date', '일자'],
        '제품명': ['제품명', '제 품 명', '품목명', '제품명 변환'],
        '합계금액': ['합계금액', '매출액', '금액'],
        '수량': ['수량', '판매수량'],
        '사업자번호': ['사업자번호', '사업자등록번호', '거래처코드'],
        '거래처명': ['거래처명', '병원명'],
        '주소': ['주소', 'Address', '사업장주소'],
        '지역': ['지역']
    }
    
    for std_col, candidates in col_map.items():
        if std_col in df.columns: continue
        for cand in candidates:
            if cand in df.columns:
                df.rename(columns={cand: std_col}, inplace=True)
                break

    try:
        # 날짜 및 숫자 처리
        df['매출일자'] = pd.to_datetime(df['매출일자'], errors='coerce')
        df = df.dropna(subset=['매출일자'])
        df['년'] = df['매출일자'].dt.year
        df['월'] = df['매출일자'].dt.month
        df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        for col in ['합계금액', '수량']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['매출액'] = df['합계금액'] / 1000000
        
        # 지역 추출 (주소의 앞 두 글자)
        if '지역' not in df.columns and '주소' in df.columns:
            df['지역'] = df['주소'].astype(str).str[:2]
        elif '지역' not in df.columns:
            df['지역'] = '미분류'
            
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return pd.DataFrame()
    return df

# --------------------------------------------------------------------------------
# 3. 데이터 실행 및 필터링
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. 파일 구조를 확인하세요.")
    st.stop()

# 사이드바 필터 (관리자 모드일 때만 보임)
if is_edit_mode:
    with st.sidebar:
        st.header("⚙️ Filter")
        years = sorted(df_raw['년'].unique(), reverse=True)
        sel_years = st.multiselect("년도 선택", years, default=years[:1])
else:
    sel_years = [df_raw['년'].max()]

df_final = df_raw[df_raw['년'].isin(sel_years)]

# --------------------------------------------------------------------------------
# 4. 분석 리포트 화면 구성 (탭)
# --------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 성과 요약", "🏥 거래처 분석", "📦 제품 분석"])

with tab1:
    st.subheader(f"🚀 {sel_years}년 성과 현황")
    c1, c2, c3 = st.columns(3)
    total_sales = df_final['매출액'].sum()
    total_hospitals = df_final['거래처명'].nunique()
    
    c1.metric("총 매출액", f"{total_sales:,.1f} M")
    c2.metric("거래처 수", f"{total_hospitals:,} 곳")
    c3.metric("판매 제품 수", f"{df_final['제품명'].nunique():,} 종")

    st.markdown("---")
    st.markdown("#### 월별 매출 추이")
    monthly_sales = df_final.groupby('년월')['매출액'].sum().reset_index()
    fig = px.line(monthly_sales, x='년월', y='매출액', markers=True, text=monthly_sales['매출액'].round(1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🏆 상위 매출 거래처 (Top 20)")
    top_cust = df_final.groupby('거래처명')['매출액'].sum().sort_values(ascending=False).head(20).reset_index()
    fig_cust = px.bar(top_cust, x='매출액', y='거래처명', orientation='h', color='매출액')
    fig_cust.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cust, use_container_width=True)

with tab3:
    st.subheader("📦 제품별 매출 비중")
    prod_sales = df_final.groupby('제품명')['매출액'].sum().reset_index()
    fig_pie = px.pie(prod_sales, values='매출액', names='제품명', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# 하단 데이터 미리보기 (확인용)
with st.expander("🔍 데이터 원본 보기"):
    st.dataframe(df_final.head(100))
