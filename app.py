import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import numpy as np
import requests
import io
import re
from datetime import timedelta

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 권한 제어
# --------------------------------------------------------------------------------
st.set_page_config(page_title="SKBS Sales Report", layout="wide", initial_sidebar_state="expanded")

params = st.query_params
is_edit_mode = params.get("mode") == "edit"

if not is_edit_mode:
    st.markdown("<style>[data-testid='stSidebar'] {display: none;} section[data-testid='stSidebar'] {width: 0px;}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
    div.block-container {padding-top: 1rem;}
    .metric-card {background-color: #f8f9fa; border-left: 5px solid #4e79a7; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
    .info-box {padding: 10px; border-radius: 5px; font-size: 13px; margin-bottom: 15px; border: 1px solid #e0e0e0; line-height: 1.6;}
    .guide-text {color: #FF4B4B; font-size: 13px; font-weight: 600; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 SKBS Sales Report")

def get_p(key, default, df_full=None, col=None):
    res = params.get_all(key)
    if not res: return default
    if 'all' in res and df_full is not None and col is not None:
        return sorted(df_full[col].unique())
    if key in ['y', 'q', 'm']: return [int(x) for x in res]
    return res

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (오류 수정: 브랜드 시트 클리닝 및 복수 반환)
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=2)
def load_data_from_drive(file_id):
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    brand_data = {}
    try:
        response = session.get(initial_url, stream=True)
        if "text/html" in response.headers.get("Content-Type", "").lower():
            html_content = response.text
            match_action = re.search(r'action="([^"]+)"', html_content)
            inputs = re.findall(r'name="([^"]+)"\s+value="([^"]+)"', html_content)
            params_dict = {name: value for name, value in inputs}
            if match_action:
                real_download_url = match_action.group(1).replace("&amp;", "&")
                response = session.get(real_download_url, params=params_dict, stream=True)
        
        if response.status_code != 200: return pd.DataFrame(), {}
        file_bytes = io.BytesIO(response.content)
        
        xls = pd.ExcelFile(file_bytes, engine='openpyxl')
        sheets = xls.sheet_names
        
        # 매출 데이터 (시트명 'SKBS' 반영)
        df = pd.read_excel(xls, sheet_name='SKBS' if 'SKBS' in sheets else 0)
        
        # 브랜드 분석 데이터 로드 및 컬럼 전처리
        for sn in ['Brand_Monthly', 'Brand_Total', 'Brand_Direct_Sales', 'Brand_Competitor']:
            if sn in sheets:
                b_df = pd.read_excel(xls, sheet_name=sn)
                b_df.columns = [re.sub(r'\s+', '', str(c)) for c in b_df.columns]
                brand_data[sn] = b_df
            else:
                brand_data[sn] = pd.DataFrame()
                
    except Exception as e:
        st.error(f"❌ 로드 오류: {e}"); return pd.DataFrame(), {}

    # 메인 데이터 전처리 (기존 로직 유지)
    df.columns = [re.sub(r'\s+', '', str(c)) for c in df.columns]
    if "매출일자" not in df.columns:
        for idx, row in df.head(10).iterrows():
            if any("매출일자" in str(x) for x in row):
                df.columns = df.iloc[idx].astype(str).str.replace(r'\s+', '', regex=True)
                df = df.iloc[idx+1:].reset_index(drop=True)
                break
    col_map = {'매출일자':['매출일자','날짜','Date'], '제품명':['제품명변환','제품명'], '합계금액':['합계금액','금액','매출액'], '수량':['수량','Qty'], '사업자번호':['사업자번호','BizNo'], '거래처명':['거래처명','병원명']}
    for std, cands in col_map.items():
        for cand in cands:
            if cand in df.columns: df.rename(columns={cand: std}, inplace=True); break
    try:
        if '매출일자' in df.columns:
            df['매출일자'] = pd.to_datetime(df['매출일자'], errors='coerce')
            df = df.dropna(subset=['매출일자'])
            df['년'], df['분기'], df['월'] = df['매출일자'].dt.year, df['매출일자'].dt.quarter, df['매출일자'].dt.month
        df['매출액'] = (pd.to_numeric(df.get('합계금액', 0), errors='coerce').fillna(0) / 1000000).astype(np.float32)
        def classify_channel(group): return 'online' if group in ['B2B', 'B2B(W)', 'SAP', '의사회원'] else 'offline'
        if '거래처그룹' in df.columns: df['판매채널'] = df['거래처그룹'].apply(classify_channel)
        if '제품명' in df.columns: df['제품명'] = df['제품명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    except: pass
    return df, brand_data

# --------------------------------------------------------------------------------
# 3. [SK분석 기본 폼] 분석 함수 정의 (기존 1~5 로직 완전 보존)
# --------------------------------------------------------------------------------
def render_smart_overview(df_curr, df_raw):
    if df_curr.empty: return
    current_year = int(df_curr['년'].max())
    last_year = current_year - 1
    selected_months = df_curr['월'].unique()
    df_prev = df_raw[(df_raw['년'] == last_year) & (df_raw['월'].isin(selected_months))]
    sales_curr, sales_prev = df_curr['매출액'].sum(), df_prev['매출액'].sum()
    sales_gap = sales_curr - sales_prev
    sales_pct = (sales_gap / (sales_prev if sales_prev > 0 else 1) * 100)
    cust_curr, cust_prev = set(df_curr['사업자번호']), set(df_prev['사업자번호'])
    new_cust, lost_cust, retained_cust = len(cust_curr - cust_prev), len(cust_prev - cust_curr), len(cust_curr & cust_prev)
    st.markdown(f"### 🚀 {current_year}년 Summary (vs {last_year})")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        with c1: st.metric("💰 총 매출 실적", f"{sales_curr:,.0f} 백만원", f"{sales_pct:+.1f}% (YoY)")
        with c2: st.metric("🏥 총 거래 병원", f"{len(cust_curr)} 처")
        with c3: st.metric("🏆 Best Product", df_curr.groupby('제품명')['매출액'].sum().idxmax())

def render_winback_quality(df_final, df_raw, current_year):
    st.markdown(f"### ♻️ {current_year}년 재유입 현황 분석")
    df_h = df_raw.sort_values(['사업자번호', '매출일자']).copy()
    df_h['구매간격'] = (df_h['매출일자'] - df_h.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    wb_data = df_h[(df_h['사업자번호'].isin(df_final['사업자번호'])) & (df_h['구매간격'] >= 180)].copy()
    if wb_data.empty: st.info("재유입 데이터 없음"); return
    st.metric("재유입 거래처", f"{wb_data['사업자번호'].nunique()} 처")
    st.dataframe(wb_data[['거래처명', '제품명', '매출액', '구매간격']], use_container_width=True)

def render_regional_deep_dive(df):
    if df.empty: return
    reg_stats = df.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    st.plotly_chart(px.scatter(reg_stats, x='Count', y='Sales', size='Sales', color='지역', text='지역'), use_container_width=True)

def render_product_strategy(df):
    if df.empty: return
    p_stats = df.groupby('제품명').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    st.plotly_chart(px.bar(p_stats.sort_values('Sales'), x='Sales', y='제품명', orientation='h'), use_container_width=True)

@st.cache_data
def classify_customers(df, target_year):
    cust_year = df.groupby(['사업자번호', '년']).size().unstack(fill_value=0)
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg({'거래처명': 'last', '매출일자': 'max'})
    return base_info

# --------------------------------------------------------------------------------
# [신규 및 보완] 🏠 6. 브랜드관 성과 분석 함수
# --------------------------------------------------------------------------------
def render_brand_store_analysis(brand_data, sel_years):
    st.markdown("### 🏠 브랜드관 성과 및 마케팅 효용성 분석")
    
    # 년도 필터 적용 (2026 선택 시 2025를 보여주라는 요청 등 사용자 로직 반영)
    target_year = sel_years[0] if sel_years else 2025
    
    # 데이터가 없을 경우 처리
    if not brand_data or brand_data['Brand_Total'].empty:
        st.warning("⚠️ 브랜드관 분석 시트가 데이터에 존재하지 않습니다."); return

    # 1. 데이터 필터링
    df_total = brand_data['Brand_Total']
    df_total = df_total[df_total['년도'] == target_year]
    
    df_direct = brand_data['Brand_Direct_Sales'].copy()
    if not df_direct.empty:
        df_direct['년'] = pd.to_datetime(df_direct['구매일']).dt.year
        df_direct = df_direct[df_direct['년'] == target_year]

    df_monthly = brand_data['Brand_Monthly'].copy()
    if not df_monthly.empty:
        df_monthly['년'] = df_monthly['월'].str[:4].astype(int)
        df_monthly = df_monthly[df_monthly['년'] == target_year]

    # 2. 총괄 성과 지표 계산
    uv = df_total['UV'].sum() if not df_total.empty else 0
    pv = df_total['PV'].sum() if not df_total.empty else 0
    conv_sales = df_direct['매출'].sum() if not df_direct.empty else 0
    conv_count = df_direct['사업자번호'].nunique() if not df_direct.empty else 0
    atv = conv_sales / conv_count if conv_count > 0 else 0

    # 3. 상단 데이터 요약 및 인사이트
    st.subheader("✔️ 성과 요약 및 인사이트")
    with st.container(border=True):
        col_summary, col_insight = st.columns([1, 1.5])
        with col_summary:
            st.write(f"• **기준 년도:** {target_year}년")
            st.write(f"• **총 방문자:** {uv:,}명 (PV: {pv:,}회)")
            st.write(f"• **전환 매출:** {conv_sales:,.0f}원")
        with col_insight:
            st.write(f"• **성과 분석:** 브랜드관 방문자 중 약 **{(conv_count/uv*100 if uv>0 else 0):.1f}%**가 실제 구매로 전환되었습니다.")
            st.write(f"• **영업 기회:** 객단가는 약 **{atv:,.0f}원**으로, 브랜드관 유입 고객의 구매력이 높게 나타납니다.")

    # 4. 운영 총괄 성과 표
    st.markdown("#### 📊 브랜드관 운영 총괄 성과")
    perf_df = pd.DataFrame({
        "항목": ["UV (방문자수)", "브랜드관 전환 매출액", "구매 전환 처수", "객단가(ATV)"],
        "성과": [f"{uv:,}명", f"{conv_sales:,.0f}원", f"{conv_count:,}처", f"{atv:,.0f}원"]
    })
    st.table(perf_df)

    # 5. 월별 유입 추이 (오류 방지 로직 포함)
    st.markdown("#### 📅 월별 브랜드관 유입 및 관심도 추이")
    if not df_monthly.empty:
        fig_monthly = px.line(df_monthly, x='월', y=['UV', 'PV'], markers=True, title=f"{target_year}년 방문 지표")
        st.plotly_chart(fig_monthly, use_container_width=True)
    else: st.info("월별 추이 데이터가 없습니다.")

    # 6. 매출 기여도 및 타사 구매 분석
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("#### 🛍️ 브랜드관 구매 전환 매출 기여도 (Top 5)")
        if not df_direct.empty:
            top5 = df_direct.groupby('상품명').agg({'매출':'sum', '수량':'sum'}).sort_values('매출', ascending=False).head(5).reset_index()
            st.dataframe(top5.rename(columns={'매출':'매출액(원)', '수량':'구매수량'}), use_container_width=True, hide_index=True)
        else: st.info("당일 구매 데이터가 없습니다.")
    
    with c_r:
        st.markdown("#### 🛡️ 경쟁사 방어 분석 (타 브랜드 구매)")
        df_c = brand_data['Brand_Competitor']
        if not df_c.empty:
            st.plotly_chart(px.pie(df_c, values='매출', names='상품명', hole=0.4), use_container_width=True)
        else: st.info("타 브랜드 구매 데이터가 없습니다.")

# --------------------------------------------------------------------------------
# 4. 필터 및 실행
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw, brand_data_dict = load_data_from_drive(DRIVE_FILE_ID)
if df_raw.empty: st.stop()

sel_years = get_p('y', [df_raw['년'].max()])
sel_channels = get_p('c', sorted(df_raw['판매채널'].unique()))
sel_quarters = get_p('q', sorted(df_raw['분기'].unique()))
sel_months = get_p('m', sorted(df_raw['월'].unique()))
sel_cats = get_p('cat', sorted(df_raw['제품군'].unique()), df_raw, '제품군')
sel_products = get_p('prod', sorted(df_raw['제품명'].unique()), df_raw, '제품명')

df_final = df_raw[(df_raw['년'].isin(sel_years)) & (df_raw['판매채널'].isin(sel_channels)) & (df_raw['분기'].isin(sel_quarters)) & (df_raw['월'].isin(sel_months)) & (df_raw['제품군'].isin(sel_cats)) & (df_raw['제품명'].isin(sel_products))]

# --------------------------------------------------------------------------------
# 5. 메인 탭 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 1. Overview", "🏆 2. 매출 상위 거래처 & 이탈 관리", "🔄 3. 재유입 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석", "🏠 6. 브랜드관 성과 분석"])

with tab1:
    render_smart_overview(df_final, df_raw)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 매출액 (년도)", f"{df_raw[df_raw['년'].isin(sel_years)]['매출액'].sum():,.0f} 백만원")
    c2.metric("총 구매처수 (년도)", f"{df_raw[df_raw['년'].isin(sel_years)]['사업자번호'].nunique():,} 처")
    c3.metric("분기 매출액", f"{df_final['매출액'].sum():,.0f} 백만원")
    c4.metric("분기 구매처수", f"{df_final['사업자번호'].nunique():,} 처")

with tab2:
    if not df_final.empty:
        ranking_v = df_final.groupby(['거래처명']).agg({'매출액': 'sum'}).sort_values('매출액', ascending=False).head(100)
        st.subheader("🥇 매출 상위 거래처 Top 100")
        st.dataframe(ranking_v, use_container_width=True)

with tab3: render_winback_quality(df_final, df_raw, sel_years[0])
with tab4: render_regional_deep_dive(df_final)
with tab5: render_product_strategy(df_final)
with tab6: render_brand_store_analysis(brand_data_dict, sel_years) # 수정된 호출 방식
