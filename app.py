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

# [수정 1] URL 축축약 대응을 위한 파라미터 로드 함수 개선
def get_p(key, default, df_full=None, col=None):
    res = params.get_all(key)
    if not res: return default
    if 'all' in res and df_full is not None and col is not None:
        return sorted(df_full[col].unique())
    if key in ['y', 'q', 'm']: return [int(x) for x in res]
    return res

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=2)
def load_data_from_drive(file_id):
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
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
        
        if response.status_code != 200: return pd.DataFrame()
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
    except Exception as e:
        st.error(f"❌ 로드 오류: {e}"); return pd.DataFrame()

    df.columns = [re.sub(r'\s+', '', str(c)) for c in df.columns]
    
    if "매출일자" not in df.columns:
        for idx, row in df.head(10).iterrows():
            if any("매출일자" in str(x) for x in row):
                df.columns = df.iloc[idx].astype(str).str.replace(r'\s+', '', regex=True)
                df = df.iloc[idx+1:].reset_index(drop=True)
                break

    col_map = {
        '매출일자': ['매출일자', '날짜', 'Date'],
        '제품명': ['제품명변환', '제품명'],
        '합계금액': ['합계금액', '공급가액', '매출액'],
        '수량': ['수량', 'Qty'],
        '사업자번호': ['사업자번호', 'BizNo'],
        '거래처명': ['거래처명', '병원명'],
        '진료과': ['진료과'],
        '제품군': ['제품군'],
        '지역': ['지역']
    }
    for std_col, candidates in col_map.items():
        if std_col in df.columns: continue
        for cand in candidates:
            if cand in df.columns: df.rename(columns={cand: std_col}, inplace=True); break

    try:
        if '주소' in df.columns: df['지역'] = df['주소'].astype(str).str.split().str[0]
        if '매출일자' in df.columns:
            df['매출일자'] = pd.to_datetime(df['매출일자'], errors='coerce')
            df = df.dropna(subset=['매출일자'])
            df['년'] = df['매출일자'].dt.year.astype(np.int16)
            df['분기'] = df['매출일자'].dt.quarter.astype(np.int8)
            df['월'] = df['매출일자'].dt.month.astype(np.int8)
            df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        df['매출액'] = (pd.to_numeric(df.get('합계금액', 0), errors='coerce').fillna(0) / 1000000).astype(np.float32)
        df['수량'] = pd.to_numeric(df.get('수량', 0), errors='coerce').fillna(0).astype(np.int32)
        
        def classify_channel(group):
            online_list = ['B2B', 'B2B(W)', 'SAP', '의사회원']
            return 'online' if group in online_list else ('offline' if group == 'SDP' else '기타')
        if '거래처그룹' in df.columns: df['판매채널'] = df['거래처그룹'].apply(classify_channel)
        
        for col in ['거래처명', '제품명', '제품군', '진료과', '지역']:
            if col in df.columns: df[col] = df[col].astype(str).replace('nan', '미분류')
        if '사업자번호' not in df.columns: df['사업자번호'] = df['거래처명']
        if '제품명' in df.columns:
            df['제품명'] = df['제품명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    except Exception as e:
        st.error(f"❌ 전처리 오류: {e}"); return pd.DataFrame()
    return df

# --------------------------------------------------------------------------------
# 3. [SK분석 기본 폼] 분석 함수 정의
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
    st.markdown("""<div class="info-box"><b>💡 분석 지표 기준:</b> 신규(전년무→올해유), 이탈(전년유→올해무), 유지율(전년유→올해유 비율)</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        with c1:
            st.metric("💰 총 매출 실적", f"{sales_curr:,.0f} 백만원", f"{sales_pct:+.1f}% (YoY)")
            st.area_chart(df_curr.groupby('월')['매출액'].sum(), height=50, color="#FF4B4B")
        with c2:
            st.metric("🏥 총 거래 병원", f"{len(cust_curr)} 처")
            st.markdown(f"- ✨신규: <span style='color:blue'>+{new_cust}</span> / 💔이탈: <span style='color:red'>-{lost_cust}</span>", unsafe_allow_html=True)
            if len(cust_curr) > 0: st.progress(retained_cust / len(cust_curr), text=f"고객 유지율 {(retained_cust/len(cust_curr))*100:.1f}%")
        with c3:
            top_p = df_curr.groupby('제품명')['매출액'].sum().idxmax()
            st.metric("🏆 Best Product", top_p)
            st.write(f"기여: **{df_curr.groupby('제품명')['매출액'].sum().max():,.0f} 백만원**")

def render_winback_quality(df_final, df_raw, current_year):
    st.markdown(f"### ♻️ {current_year}년 재유입 현황 분석")
    df_history = df_raw.sort_values(['사업자번호', '매출일자']).copy()
    df_history['구매간격'] = (df_history['매출일자'] - df_history.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    winback_data = df_history[(df_history['사업자번호'].isin(df_final['사업자번호'])) & (df_history['구매간격'] >= 180)].copy()
    winback_ids = winback_data['사업자번호'].unique()
    
    if len(winback_ids) == 0:
        st.info("♻️ 해당 조건 내 재유입 데이터(6개월 공백 기준)가 없습니다."); return

    sales_curr = df_final[df_final['사업자번호'].isin(winback_ids)].groupby(['사업자번호', '거래처명', '지역'])['매출액'].sum()
    sales_history = df_raw[df_raw['사업자번호'].isin(winback_ids)].groupby(['사업자번호', '거래처명', '지역'])['매출액'].max()
    
    df_wb = pd.DataFrame(index=sales_curr.index)
    df_wb['올해매출'] = sales_curr
    df_wb['과거최고'] = sales_history
    df_wb['회복률'] = (df_wb['올해매출'] / df_wb['과거최고'].replace(0,1) * 100).fillna(0)
    df_wb['상태'] = df_wb['회복률'].apply(lambda x: "완전 회복" if x>=80 else ("회복 중" if x>=20 else "재진입 초기"))
    df_wb = df_wb.reset_index().sort_values('올해매출', ascending=False)
    df_wb['Bubble_Size'] = df_wb['올해매출'].apply(lambda x: max(x, 0.1))

    c1, c2, c3 = st.columns(3)
    c1.metric("재유입 거래처", f"{len(df_wb)} 처")
    c2.metric("확보 매출", f"{df_wb['올해매출'].sum():,.0f} 백만원")
    c3.metric("평균 회복률", f"{df_wb['회복률'].mean():.1f}%")
    
    col_ch, col_li = st.columns([1, 1])
    with col_ch:
        try:
            fig = px.scatter(df_wb, x='과거최고', y='올해매출', color='상태', hover_name='거래처명', size='Bubble_Size',
                             color_discrete_map={"완전 회복": "green", "회복 중": "orange", "재진입 초기": "red"})
            st.plotly_chart(fig, use_container_width=True)
        except: st.warning("차트 생성 불가")
    with col_li:
        st.markdown('<p class="guide-text">💡 행 클릭 시 하단에서 상세 공백 이력 확인 가능</p>', unsafe_allow_html=True)
        event_wb = st.dataframe(df_wb[['상태', '거래처명', '올해매출', '회복률']], hide_index=True, use_container_width=True,
                               on_select="rerun", selection_mode="single-row")

    if len(event_wb.selection.rows) > 0:
        sel_idx = event_wb.selection.rows[0]
        sel_biz_no = df_wb.iloc[sel_idx]['사업자번호']
        detail_hist = df_history[df_history['사업자번호'] == sel_biz_no].sort_values('매출일자', ascending=False).copy()
        detail_hist['매출일자_str'] = detail_hist['매출일자'].dt.strftime('%Y-%m-%d')
        st.dataframe(detail_hist[['매출일자_str', '제품명', '매출액', '수량', '구매간격']].rename(columns={'매출일자_str':'매출일자', '구매간격':'공백(일)'})
                     .style.applymap(lambda v: 'background-color: #ffcccc;' if isinstance(v, (int, float)) and v >= 180 else '', subset=['공백(일)'])
                     .format({'매출액': '{:,.1f} 백만원'}), use_container_width=True)

def render_regional_deep_dive(df):
    if df.empty: return
    reg_stats = df.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    reg_stats['Per'] = reg_stats['Sales'] / reg_stats['Count']
    st.markdown("### 🗺️ 지역별 심층 분석")
    fig = px.scatter(reg_stats, x='Count', y='Per', size='Sales', color='지역', text='지역', labels={'Count': '거래처 수', 'Per': '평균 객단가'})
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(reg_stats, values='Sales', names='지역', hole=0.3, title="지역별 매출 기여도"), use_container_width=True)
    with c2:
        risk = []
        for r in df['지역'].unique():
            r_df = df[df['지역'] == r]
            risk.append({'지역': r, '의존도': (r_df.groupby('거래처명')['매출액'].sum().max() / r_df['매출액'].sum() * 100)})
        st.plotly_chart(px.bar(pd.DataFrame(risk).sort_values('의존도', ascending=False), x='의존도', y='지역', orientation='h', color='의존도', color_continuous_scale='Reds', title="핵심 거점 매출 의존도"), use_container_width=True)

def render_product_strategy(df):
    if df.empty: return
    st.markdown("### 💊 제품별 전략 분석 (BCG & Seasonality)")
    p_stats = df.groupby('제품명').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.scatter(p_stats, x='Count', y='Sales', size='Sales', color='제품명', text='제품명', title="BCG 매트릭스"), use_container_width=True)
    with c2:
        pivot = df.pivot_table(index='제품명', columns='월', values='매출액', aggfunc='sum', fill_value=0)
        st.plotly_chart(px.imshow(pivot.div(pivot.max(axis=1), axis=0), color_continuous_scale="Reds", title="제품별 판매 시즌 집중도"), use_container_width=True)
    with st.expander("🧩 **함께 팔기(Cross-selling) 기회 분석기**", expanded=True):
        all_prods = sorted(df['제품명'].unique())
        col1, col2 = st.columns(2)
        b_p = col1.selectbox("Anchor 제품 (이미 쓰는 것)", all_prods, index=0)
        t_p = col2.selectbox("Target 제품 (팔고 싶은 것)", all_prods, index=min(1, len(all_prods)-1))
        if b_p != t_p:
            acc_A, acc_B = set(df[df['제품명']==b_p]['거래처명'].unique()), set(df[df['제품명']==t_p]['거래처명'].unique())
            targets = list(acc_A - acc_B)
            st.metric("🎯 추가 영업 기회", f"{len(targets)} 처")
            if targets:
                t_info = df[(df['거래처명'].isin(targets)) & (df['제품명']==b_p)].groupby(['거래처명','지역'])['매출액'].sum().reset_index().sort_values('매출액', ascending=False)
                st.dataframe(t_info.head(50), use_container_width=True)

@st.cache_data
def classify_customers(df, target_year):
    cols_to_agg = {'거래처명':'last','매출일자':'max'}
    if '진료과' in df.columns: cols_to_agg['진료과'] = 'last'
    if '지역' in df.columns: cols_to_agg['지역'] = 'last'
    cust_year = df.groupby(['사업자번호', '년']).size().unstack(fill_value=0)
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg(cols_to_agg).rename(columns={'매출일자':'최근구매일'})
    sales_ty = df[df['년']==target_year].groupby('사업자번호')['매출액'].sum()
    base_info['해당년도_매출'] = base_info.index.map(sales_ty).fillna(0)
    classification = {}
    for biz in base_info.index:
        has_ty = (target_year in cust_year.columns) and (cust_year.loc[biz, target_year] > 0)
        has_t1 = (target_year-1 in cust_year.columns) and (cust_year.loc[biz, target_year-1] > 0)
        has_h = cust_year.loc[biz, [y for y in cust_year.columns if y < target_year-1]].sum() > 0 if len(cust_year.columns)>1 else False
        status = "✅ 기존 (유지)" if has_ty and has_t1 else ("🔄 재유입 (복귀)" if has_ty and has_h else ("🆕 신규 (New)" if has_ty else ("📉 1년 이탈" if has_t1 else "💤 장기 이탈")))
        classification[biz] = status
    base_info['상태'] = base_info.index.map(classification)
    return base_info

# --------------------------------------------------------------------------------
# 4. 데이터 로드 및 필터링
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)
if df_raw.empty: st.stop()

# [수정 1 반영] URL 축약 지원 필터 설정
sel_years = get_p('y', [df_raw['년'].max()])
sel_channels = get_p('c', sorted(df_raw['판매채널'].unique()))
sel_quarters = get_p('q', sorted(df_raw['분기'].unique()))
sel_months = get_p('m', sorted(df_raw['월'].unique()))
sel_cats = get_p('cat', sorted(df_raw['제품군'].unique()), df_raw, '제품군')
sel_products = get_p('prod', sorted(df_raw['제품명'].unique()), df_raw, '제품명')

if is_edit_mode:
    with st.sidebar:
        sel_channels = st.multiselect("판매채널", sorted(df_raw['판매채널'].unique()), default=sel_channels)
        sel_years = st.multiselect("년도", sorted(df_raw['년'].unique(), reverse=True), default=sel_years)
        sel_quarters = st.multiselect("분기", sorted(df_raw['분기'].unique()), default=sel_quarters)
        sel_months = st.multiselect("월", sorted(df_raw['월'].unique()), default=sel_months)
        sel_cats = st.multiselect("제품군", sorted(df_raw['제품군'].unique()), default=sel_cats)
        sel_products = st.multiselect("제품명", sorted(df_raw['제품명'].unique()), default=sel_products)
        if st.button("🔗 축약 공유 링크 생성"):
            base_url = "https://skbs-sales-2026-cbktkdtxsyrfzfrihefs2h.streamlit.app/"
            cat_p = "all" if len(sel_cats)==len(df_raw['제품군'].unique()) else "&cat=".join([urllib.parse.quote(x) for x in sel_cats])
            prod_p = "all" if len(sel_products)==len(df_raw['제품명'].unique()) else "&prod=".join([urllib.parse.quote(x) for x in sel_products])
            p_str = f"?y={'&y='.join(map(str, sel_years))}&c={'&c='.join(sel_channels)}&q={'&q='.join(map(str, sel_quarters))}&m={'&m='.join(map(str, sel_months))}&cat={cat_p}&prod={prod_p}"
            st.code(base_url + p_str)

df_final = df_raw[(df_raw['년'].isin(sel_years)) & (df_raw['판매채널'].isin(sel_channels)) & (df_raw['분기'].isin(sel_quarters)) & (df_raw['월'].isin(sel_months)) & (df_raw['제품군'].isin(sel_cats)) & (df_raw['제품명'].isin(sel_products))]

# --------------------------------------------------------------------------------
# 5. 메인 탭 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 1. Overview", "🏆 2. 매출 상위 & 이탈 관리", "🔄 3. 재유입 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석"])

with tab1:
    render_smart_overview(df_final, df_raw)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 매출 (년도)", f"{df_raw[df_raw['년'].isin(sel_years)]['매출액'].sum():,.0f}M")
    c2.metric("총 구매처 (년도)", f"{df_raw[df_raw['년'].isin(sel_years)]['사업자번호'].nunique():,}처")
    c3.metric("기간 매출액", f"{df_final['매출액'].sum():,.0f}M")
    c4.metric("기간 구매처수", f"{df_final['사업자번호'].nunique():,}처")
    col_a, col_b = st.columns([1, 1.5])
    col_a.plotly_chart(px.pie(df_final, values='매출액', names='진료과', hole=0.4, title="진료과 비중"), use_container_width=True)
    col_b.plotly_chart(px.bar(df_final.groupby('년월')['매출액'].sum().reset_index(), x='년월', y='매출액', title="월별 매출 추이"), use_container_width=True)

with tab2:
    if not df_final.empty:
        total_s = df_final['매출액'].sum()
        top100 = df_final.groupby(['사업자번호', '거래처명', '진료과']).agg({'매출액': 'sum'}).sort_values('매출액', ascending=False).head(100).reset_index()
        cls_d = classify_customers(df_raw, sel_years[0]); st_c = cls_d['상태'].value_counts()
        last_p = df_raw.groupby('사업자번호')['매출일자'].max(); top100['최근구매일'] = top100['사업자번호'].map(last_p)
        cur_d = df_raw['매출일자'].max(); risk_cnt = len(top100[(cur_d - top100['최근구매일']).dt.days >= 90])
        
        # [수정 2 반영] 글자색 해결: st.write 표준 방식 사용 (시스템 테마 대응)
        st.subheader("📊 데이터 요약")
        st.write(f"• 상위 100대 매출 합계: **{top100['매출액'].sum():,.0f} 백만원** (비중 {(top100['매출액'].sum()/total_s*100):.1f}%)")
        st.write(f"• 상태 분포: **기존({st_c.get('✅ 기존 (유지)',0)}처), 신규({st_c.get('🆕 신규 (New)',0)}처), 재유입({st_c.get('🔄 재유입 (복귀)',0)}처), 이탈({st_c.get('📉 1년 이탈',0)}처)**")
        st.subheader("💡 스마트 인사이트")
        st.write(f"• **VIP 이탈 위험:** 상위 100대 중 **{risk_cnt}처**가 90일 이상 공백 상태입니다. 관리가 필요합니다.")

    st.markdown("---")
    st.markdown('<p class="guide-text">💡 아래 표에서 행을 클릭하면 하단에 상세 실적이 표시됩니다.</p>', unsafe_allow_html=True)
    top100['상태'] = (cur_d - top100['최근구매일']).dt.days.apply(lambda x: '🚨 이탈위험' if x >= 90 else '✅ 정상')
    top100['최근구매일_str'] = top100['최근구매일'].dt.strftime('%Y-%m-%d')
    event_v = st.dataframe(top100[['상태', '거래처명', '진료과', '매출액', '최근구매일_str']], use_container_width=True, on_select="rerun", selection_mode="single-row")
    if len(event_v.selection.rows) > 0:
        sel_biz = top100.iloc[event_v.selection.rows[0]]['사업자번호']
        st.dataframe(df_raw[df_raw['사업자번호'] == sel_biz].groupby('제품명').agg({'매출액':'sum'}).sort_values('매출액', ascending=False).style.format({'매출액':'{:,.1f} 백만원'}), use_container_width=True)
    
    st.markdown("---")
    c_s1, c_s2 = st.columns([1, 2])
    with c_s1:
        st.dataframe(st_c.reset_index().rename(columns={'count':'거래처수'}), use_container_width=True)
        sel_st = st.selectbox("👇 그룹 선택", sorted(cls_d['상태'].unique()))
    with c_s2: st.plotly_chart(px.pie(cls_d[cls_d['상태']==sel_st], names='진료과', title="진료과 분포"), use_container_width=True)
    display_cls = cls_d[cls_d['상태']==sel_st].sort_values('해당년도_매출', ascending=False).copy()
    display_cls['최근구매일'] = display_cls['최근구매일'].dt.strftime('%Y-%m-%d')
    ev_cls = st.dataframe(display_cls[['거래처명','진료과','최근구매일','해당년도_매출']], use_container_width=True, on_select="rerun", selection_mode="single-row")
    if len(ev_cls.selection.rows) > 0:
        row_biz = display_cls.index[ev_cls.selection.rows[0]]
        st.dataframe(df_raw[df_raw['사업자번호']==row_biz].sort_values('매출일자', ascending=False).head(20), use_container_width=True)

with tab3:
    df_h_sum = df_raw.sort_values(['사업자번호', '매출일자']).copy()
    df_h_sum['구매간격'] = (df_h_sum['매출일자'] - df_h_sum.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    wb_v_sum = df_h_sum[(df_h_sum['사업자번호'].isin(df_final['사업자번호'])) & (df_h_sum['구매간격'] >= 180)].copy()
    if not wb_v_sum.empty:
        wb_ids = wb_v_sum['사업자번호'].unique()
        df_wb_f = pd.DataFrame({'올해': df_final[df_final['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].sum(), '최고': df_raw[df_raw['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].max()}).fillna(0)
        df_wb_f['회복률'] = (df_wb_f['올해'] / df_wb_f['최고'] * 100).replace([np.inf, -np.inf], 0)
        # [수정 2 반영] 글자색 해결
        st.subheader("📊 데이터 요약")
        st.write(f"• 총 재유입 거래처: **{len(wb_ids)} 처** (6개월 공백 기준) / 평균 회복률: **{df_wb_f['회복률'].mean():.1f}%**")
    render_winback_quality(df_final, df_raw, sel_years[0])

with tab4:
    reg_v = df_final.groupby('지역').agg(Sales=('매출액','sum'), Count=('사업자번호','nunique')).reset_index().sort_values('Sales', ascending=False)
    if not reg_v.empty:
        # [수정 2 반영] 글자색 해결
        st.subheader("📊 데이터 요약")
        st.write(f"• 최고 매출 지역: **{reg_v.iloc[0]['지역']}** / 최다 거래 지역: **{reg_v.sort_values('Count', ascending=False).iloc[0]['지역']}**")
    render_regional_deep_dive(df_final)
    st.markdown("---")
    st.markdown("### 🗺️ 지역별 상세 리스트")
    sel_r = st.selectbox("🔎 지역 선택", reg_v['지역'].unique(), key="p4_sel")
    col_r1, col_r2 = st.columns([1, 1.5])
    with col_r1: st.dataframe(reg_v.rename(columns={'Count':'구매처수'}), use_container_width=True)
    with col_r2: st.plotly_chart(px.pie(df_final[df_final['지역']==sel_r], values='매출액', names='제품명', hole=0.3, title="지역 제품 비중"), use_container_width=True)
    st.dataframe(df_final[df_final['지역']==sel_r].groupby(['거래처명','제품명']).agg({'매출액':'sum','수량':'sum'}).sort_values('매출액', ascending=False).head(50), use_container_width=True)

with tab5:
    p_v = df_final.groupby('제품명').agg(Sales=('매출액','sum'), Count=('사업자번호','nunique')).reset_index().sort_values('Sales', ascending=False)
    cat_v = df_final.groupby('제품군')['매출액'].sum().reset_index().sort_values('매출액', ascending=False)
    if not p_v.empty:
        # [수정 2 반영] 글자색 해결
        st.subheader("📊 데이터 요약")
        st.write(f"• 최다 판매: **{p_v.iloc[0]['제품명']}** / 최대 제품군: **{cat_v.iloc[0]['제품군']}**")
    
    # [수정 3 반영] 📦 제목 아래에 그래프 2개 5:5 배치
    st.markdown("### 📦 제품별 판매 현황")
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.bar(p_v.head(10), x='Sales', y='제품명', orientation='h', title="제품 매출 Top 10"), use_container_width=True)
    with g2: st.plotly_chart(px.pie(cat_v, values='매출액', names='제품군', hole=0.3, title="제품군 매출 비중"), use_container_width=True)
    
    st.markdown('<p class="guide-text">💡 행 클릭 시 상세 현황 표시</p>', unsafe_allow_html=True)
    ev_p = st.dataframe(p_v.rename(columns={'Count':'구매처수'}), use_container_width=True, on_select="rerun", selection_mode="single-row", height=300)
    if len(ev_p.selection.rows) > 0:
        sel_p = p_v.iloc[ev_p.selection.rows[0]]['제품명']
        st.dataframe(df_final[df_final['제품명']==sel_p].groupby('거래처명').agg({'매출액':'sum'}).sort_values('매출액', ascending=False).style.format({'매출액':'{:,.1f} 백만원'}), use_container_width=True)
    
    st.markdown("---")
    render_product_strategy(df_final)
