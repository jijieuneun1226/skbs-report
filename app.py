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
    # 구글 드라이브 직링크 포맷 (ZIP 에러 방지용)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

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
        if '지역' not in df.columns and '주소' in df.columns:
            df['지역_임시'] = df['주소'].astype(str).str.split().str[0]
            addr_map = {
                '서울': '서울', '서울시': '서울', '서울특별시': '서울',
                '경기': '경기', '경기도': '경기', '부산': '부산', '부산광역시': '부산',
                '대구': '대구', '대구광역시': '대구', '인천': '인천', '인천광역시': '인천',
                '광주': '광주', '광주광역시': '광주', '대전': '대전', '대전광역시': '대전',
                '울산': '울산', '울산광역시': '울산', '세종': '세종', '세종특별자치시': '세종',
                '강원': '강원', '강원도': '강원', '충북': '충북', '충청북도': '충북',
                '충남': '충남', '충청남도': '충남', '전북': '전북', '전라북도': '전북',
                '전남': '전남', '전라남도': '전남', '경북': '경북', '경상북도': '경북',
                '경남': '경남', '경상남도': '경남', '제주': '제주', '제주도': '제주'
            }
            df['지역'] = df['지역_임시'].map(addr_map).fillna('기타')
        elif '지역' not in df.columns:
             df['지역'] = '미분류'

        df['매출일자'] = pd.to_datetime(df['매출일자'])
        df = df.sort_values('매출일자')
        df['년'] = df['매출일자'].dt.year
        df['분기'] = df['매출일자'].dt.quarter
        df['월'] = df['매출일자'].dt.month
        df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        if '제품명' in df.columns:
            df['제품명'] = df['제품명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
        
        for col in ['합계금액', '수량']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['매출액'] = df['합계금액'] / 1000000
        
        def classify_channel(group):
            online_list = ['B2B', 'B2B(W)', 'SAP', '의사회원']
            return 'online' if group in online_list else ('offline' if group == 'SDP' else '기타')
        if '거래처그룹' in df.columns:
            df['판매채널'] = df['거래처그룹'].apply(classify_channel)
             
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return pd.DataFrame()
    return df

@st.cache_data
def classify_customers(df, target_year):
    cust_year = df.groupby(['사업자번호', '년']).size().unstack(fill_value=0)
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg({
        '거래처명': 'last', '진료과': 'last', '지역': 'last', '매출일자': 'max'
    }).rename(columns={'매출일자': '최근구매일'})
    sales_ty = df[df['년'] == target_year].groupby('사업자번호')['매출액'].sum()
    base_info['해당년도_매출'] = base_info.index.map(sales_ty).fillna(0)
    
    classification = {}
    for biz_no in base_info.index:
        has_ty = (target_year in cust_year.columns) and (cust_year.loc[biz_no, target_year] > 0)
        has_t1 = (target_year - 1 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 1] > 0)
        past_years = [y for y in cust_year.columns if y < target_year - 1]
        has_history = cust_year.loc[biz_no, past_years].sum() > 0 if past_years else False
        
        if has_ty:
            if has_t1: status = "✅ 기존 (유지)"
            else: status = "🔄 재유입 (복귀)" if has_history else "🆕 신규 (New)"
        else:
            status = "📉 이탈 관리"
        classification[biz_no] = status
    base_info['상태'] = base_info.index.map(classification)
    return base_info

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
