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
    # 구글 드라이브 직링크 포맷 (export=download를 붙여야 바이트 스트림으로 읽어올 수 있습니다)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # requests를 사용하여 파일 콘텐츠를 가져옵니다.
        response = requests.get(url)
        response.raise_for_status() # 접속 에러 발생 시 예외 처리
        
        # 가져온 데이터를 메모리상의 바이너리(BytesIO)로 변환하여 pandas로 읽습니다.
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
        
        if df.empty:
            st.error("불러온 데이터가 비어있습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        # 만약 여기서도 에러가 난다면, 공유 설정 문제일 확률이 매우 높습니다.
        st.error(f"데이터 로드 실패: {e}")
        st.info("💡 팁: 드라이브 파일 오른쪽 클릭 -> '공유' -> '링크가 있는 모든 사용자'에게 '뷰어' 권한이 있는지 확인해 주세요.")
        return pd.DataFrame()

# 파일 ID 적용 (사용자님이 주신 ID)
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if df_raw.empty:
    st.stop() # 데이터 없으면 실행 중단
# --------------------------------------------------------------------------------
# 시각화 함수 정의 (기존 함수들 그대로 유지)
# --------------------------------------------------------------------------------
def render_smart_overview(df_curr, df_raw):
    if df_curr.empty: return
    current_year = int(df_curr['년'].max())
    last_year = current_year - 1
    selected_months = df_curr['월'].unique()
    df_prev = df_raw[(df_raw['년'] == last_year) & (df_raw['월'].isin(selected_months))]
    sales_curr = df_curr['매출액'].sum()
    sales_prev = df_prev['매출액'].sum() if not df_prev.empty else 0
    sales_pct = ((sales_curr - sales_prev) / sales_prev * 100) if sales_prev > 0 else 0
    st.markdown(f"### 🚀 {current_year}년 Executive Summary (vs {last_year})")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 총 매출 실적", f"{sales_curr:,.0f} M", f"{sales_pct:+.1f}%")
        c2.metric("🏥 총 거래 병원", f"{df_curr['사업자번호'].nunique()} 곳")
        c3.metric("🏆 Best Product", df_curr.groupby('제품명')['매출액'].sum().idxmax())

def render_advanced_insights(df, tab_name):
    if df.empty: return
    st.markdown(f"### 📊 {tab_name} 심층 분석")
    # ... (기존 로직 유지)

def render_winback_quality(df, current_year):
    # ... (기존 로직 유지)
    st.write(f"### ♻️ {current_year}년 재유입 분석")

def render_regional_deep_dive(df):
    # ... (기존 로직 유지)
    st.write("### 🗺️ 지역별 심층 분석")

def render_product_strategy(df):
    # ... (기존 로직 유지)
    st.write("### 💊 제품별 전략 분석")

# --------------------------------------------------------------------------------
# 3. 데이터 로드 실행 및 필터 제어
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다. 구글 드라이브 공유 설정을 확인하세요.")
    st.stop()

# 파라미터 제어
def get_p(key, default):
    res = params.get_all(key)
    if not res: return default
    if key in ['y', 'q', 'm']: return [int(x) for x in res]
    return res

sel_years = get_p('y', [df_raw['년'].max()])
sel_channels = get_p('c', sorted(df_raw['판매채널'].unique() if '판매채널' in df_raw.columns else []))
sel_quarters = get_p('q', sorted(df_raw['분기'].unique()))
sel_months = get_p('m', sorted(df_raw['월'].unique()))

if is_edit_mode:
    with st.sidebar:
        st.header("⚙️ 관리자 설정")
        sel_years = st.multiselect("년도", sorted(df_raw['년'].unique(), reverse=True), default=sel_years)
        # ... (나머지 사이드바 설정 동일)

# 데이터 필터링
df_final = df_raw[
    (df_raw['년'].isin(sel_years)) &
    (df_raw['분기'].isin(sel_quarters)) &
    (df_raw['월'].isin(sel_months))
]

# --------------------------------------------------------------------------------
# 4. 메인 탭 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 1. Overview", "🏆 2. VIP & 이탈 관리", "🔄 3. 재유입 패턴 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석"])

with tab1:
    render_smart_overview(df_final, df_raw)
    st.markdown("---")
    st.subheader("📅 월별 추이")
    monthly = df_final.groupby('년월').agg({'매출액': 'sum', '사업자번호': 'nunique'}).reset_index()
    st.line_chart(monthly.set_index('년월'))

with tab2:
    st.markdown("### 🏆 VIP 관리")
    st.dataframe(df_final.groupby('거래처명')['매출액'].sum().sort_values(ascending=False).head(50))

with tab3:
    if len(sel_years) > 0:
        render_winback_quality(df_raw, sel_years[0])

with tab4:
    render_regional_deep_dive(df_final)

with tab5:
    render_product_strategy(df_final)

