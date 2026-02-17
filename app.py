import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import numpy as np
import requests
import io
import re
import gc # 메모리 관리를 위한 모듈
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
# 2. 데이터 로드 및 전처리 (메모리 최적화)
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=1)
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
                real_url = match_action.group(1).replace("&amp;", "&")
                response = session.get(real_url, params=params_dict, stream=True)
        if response.status_code != 200: return pd.DataFrame()
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
        del file_bytes; gc.collect() # 메모리 즉시 회수
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}"); return pd.DataFrame()

    df.columns = [re.sub(r'\s+', '', str(c)) for c in df.columns]
    if "매출일자" not in df.columns:
        for idx, row in df.head(10).iterrows():
            if any("매출일자" in str(x) for x in row):
                df.columns = df.iloc[idx].astype(str).str.replace(r'\s+', '', regex=True)
                df = df.iloc[idx+1:].reset_index(drop=True)
                break
    col_map = {'매출일자':['매출일자','날짜','Date'],'제품명':['제품명변환','제품명'],'합계금액':['합계금액','매출액'],'수량':['수량','Qty'],'사업자번호':['사업자번호','BizNo'],'거래처명':['거래처명','병원명'],'진료과':['진료과'],'제품군':['제품군'],'지역':['지역']}
    for std, cands in col_map.items():
        if std in df.columns: continue
        for cand in cands:
            if cand in df.columns: df.rename(columns={cand:std}, inplace=True); break
    try:
        # 데이터 타입 최적화 (메모리 절약)
        df['매출일자'] = pd.to_datetime(df['매출일자'], errors='coerce')
        df = df.dropna(subset=['매출일자'])
        df['년'] = df['매출일자'].dt.year.astype(np.int16)
        df['분기'] = df['매출일자'].dt.quarter.astype(np.int8)
        df['월'] = df['매출일자'].dt.month.astype(np.int8)
        df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        df['매출액'] = (pd.to_numeric(df.get('합계금액',0), errors='coerce').fillna(0)/1000000).astype(np.float32)
        df['수량'] = pd.to_numeric(df.get('수량',0), errors='coerce').fillna(0).astype(np.int32)
        def classify_channel(group): return 'online' if str(group) in ['B2B','SAP','의사회원'] else 'offline'
        df['판매채널'] = df.get('거래처그룹','기타').apply(classify_channel)
        for col in ['거래처명','제품명','제품군','진료과','지역']:
            if col not in df.columns: df[col] = '미분류'
            else: df[col] = df[col].astype(str).replace('nan','미분류')
        df['제품명'] = df['제품명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    except Exception as e:
        st.error(f"전처리 오류: {e}"); return pd.DataFrame()
    return df

@st.cache_data
def classify_customers(df, target_year):
    # [최적화] groupby.unstack 대신 Set 연산 사용 (메모리 사용량 대폭 감소)
    cols_to_agg = {'거래처명':'last','매출일자':'max'}
    if '진료과' in df.columns: cols_to_agg['진료과'] = 'last'
    
    # 기본 정보 집계
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg(cols_to_agg).rename(columns={'매출일자':'최근구매일'})
    
    # 매출 집계
    sales_ty = df[df['년']==target_year].groupby('사업자번호')['매출액'].sum()
    base_info['해당년도_매출'] = base_info.index.map(sales_ty).fillna(0)
    
    # Set 기반 상태 분류
    s_curr = set(df[df['년'] == target_year]['사업자번호'])
    s_prev = set(df[df['년'] == target_year - 1]['사업자번호'])
    s_past = set(df[df['년'] < target_year - 1]['사업자번호'])
    
    def get_status(biz):
        is_curr = biz in s_curr
        is_prev = biz in s_prev
        is_past = biz in s_past
        
        if is_curr:
            if is_prev: return "✅ 기존 (유지)"
            elif is_past: return "🔄 재유입 (복귀)"
            else: return "🆕 신규 (New)"
        else:
            if is_prev: return "📉 1년 이탈"
            else: return "💤 장기 이탈"

    base_info['상태'] = base_info.index.map(get_status)
    gc.collect()
    return base_info

# --------------------------------------------------------------------------------
# 3. 필터링 및 데이터 로드
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)
if df_raw.empty: st.stop()

sel_years = get_p('y', [df_raw['년'].max()])
sel_channels = get_p('c', sorted(df_raw['판매채널'].unique()))
sel_quarters = get_p('q', sorted(df_raw['분기'].unique()))
sel_months = get_p('m', sorted(df_raw['월'].unique()))
sel_cats = get_p('cat', sorted(df_raw['제품군'].unique()), df_raw, '제품군')
sel_products = get_p('prod', sorted(df_raw['제품명'].unique()), df_raw, '제품명')

if is_edit_mode:
    with st.sidebar:
        st.header("⚙️ 관리자 필터 설정")
        sel_channels = st.multiselect("판매채널", sorted(df_raw['판매채널'].unique()), default=sel_channels)
        sel_years = st.multiselect("년도", sorted(df_raw['년'].unique(), reverse=True), default=sel_years)
        sel_quarters = st.multiselect("분기", sorted(df_raw['분기'].unique()), default=sel_quarters)
        q_to_m = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}
        avail_m = sorted([m for q in sel_quarters for m in q_to_m.get(q, [])])
        sel_months = st.multiselect("월", avail_m, default=[m for m in sel_months if m in avail_m])
        sel_cats = st.multiselect("제품군", sorted(df_raw['제품군'].unique()), default=sel_cats)
        sel_products = st.multiselect("제품명", sorted(df_raw['제품명'].unique()), default=sel_products)
        if st.button("🔗 축약 공유 링크 생성"):
            base_url = "https://skbs-sales-2026-cbktkdtxsyrfzfrihefs2h.streamlit.app/"
            cat_p = "all" if len(sel_cats) == len(df_raw['제품군'].unique()) else "&cat=".join([urllib.parse.quote(x) for x in sel_cats])
            prod_p = "all" if len(sel_products) == len(df_raw['제품명'].unique()) else "&prod=".join([urllib.parse.quote(x) for x in sel_products])
            p_str = f"?y={'&y='.join(map(str, sel_years))}&c={'&c='.join(sel_channels)}&q={'&q='.join(map(str, sel_quarters))}&m={'&m='.join(map(str, sel_months))}&cat={cat_p}&prod={prod_p}"
            st.code(base_url + p_str)

df_final = df_raw[(df_raw['년'].isin(sel_years)) & (df_raw['판매채널'].isin(sel_channels)) & (df_raw['분기'].isin(sel_quarters)) & (df_raw['월'].isin(sel_months)) & (df_raw['제품군'].isin(sel_cats)) & (df_raw['제품명'].isin(sel_products))]
gc.collect()

# --------------------------------------------------------------------------------
# 4. 분석 모듈 정의
# --------------------------------------------------------------------------------
def render_smart_overview(df_curr, df_raw_full):
    if df_curr.empty: return
    cur_y = int(df_curr['년'].max()); last_y = cur_y - 1
    df_prev = df_raw_full[(df_raw_full['년'] == last_y) & (df_raw_full['월'].isin(df_curr['월'].unique()))]
    sales_curr, sales_prev = df_curr['매출액'].sum(), df_prev['매출액'].sum()
    sales_pct = ((sales_curr - sales_prev) / (sales_prev if sales_prev > 0 else 1) * 100)
    cust_curr, cust_prev = set(df_curr['사업자번호']), set(df_prev['사업자번호'])
    new_c, lost_c = len(cust_curr - cust_prev), len(cust_prev - cust_curr)

    st.markdown(f"### 🚀 {cur_y}년 Summary")
    st.info("💡 **지표 정의:** 신규(전년무→올해유), 이탈(전년유→올해무), 유지율(전년유→올해유)")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 총 매출 실적", f"{sales_curr:,.0f}M", f"{sales_pct:+.1f}%")
        c2.metric("🏥 총 거래 병원", f"{len(cust_curr):,} 처")
        c3.metric("🏆 Best Product", df_curr.groupby('제품명')['매출액'].sum().idxmax())
        col_a, col_b = st.columns([1, 1.5])
        col_a.plotly_chart(px.pie(df_curr, values='매출액', names='진료과', hole=0.4, title="진료과 비중"), use_container_width=True)
        col_b.plotly_chart(px.bar(df_curr.groupby('년월')['매출액'].sum().reset_index(), x='년월', y='매출액', text_auto='.1f', title="월별 매출 추이"), use_container_width=True)

def render_winback_quality(df_v, df_r, current_year):
    st.markdown(f"### ♻️ {current_year}년 재유입 현황 분석")
    st.info("🔍 **재유입 정의:** 직전 구매일로부터 **최소 180일(6개월) 이상 공백기** 이후 다시 구매가 발생한 거래처")
    df_h = df_r.sort_values(['사업자번호', '매출일자']).copy()
    df_h['구매간격'] = (df_h['매출일자'] - df_h.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    wb_v = df_h[(df_h['사업자번호'].isin(df_v['사업자번호'])) & (df_h['구매간격'] >= 180)].copy()
    if wb_v.empty: st.warning("해당 조건의 재유입 데이터가 없습니다."); return
    
    wb_ids = wb_v['사업자번호'].unique()
    s_c = df_v[df_v['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].sum()
    s_m = df_r[df_r['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].max()
    df_wb_l = pd.DataFrame({'올해': s_c, '최고': s_m}).fillna(0)
    df_wb_l['회복률'] = (df_wb_l['올해'] / df_wb_l['최고'] * 100).replace([np.inf, -np.inf], 0)
    df_wb_l['상태'] = df_wb_l['회복률'].apply(lambda x: "완전 회복" if x>=80 else ("회복 중" if x>=20 else "재진입 초기"))
    df_wb_l['거래처명'] = df_v[df_v['사업자번호'].isin(wb_ids)].groupby('사업자번호')['거래처명'].last()
    df_wb_l['bubble'] = df_wb_l['올해'].apply(lambda x: max(x, 0.1))
    
    col_a, col_b = st.columns([1, 1])
    with col_a: st.plotly_chart(px.scatter(df_wb_l, x='최고', y='올해', color='상태', hover_name='거래처명', size='bubble', color_discrete_map={"완전 회복":"green","회복 중":"orange","재진입 초기":"red"}), use_container_width=True)
    with col_b:
        st.markdown('<p class="guide-text">💡 행 클릭 시 상세 공백 이력 확인 가능</p>', unsafe_allow_html=True)
        ev_wb = st.dataframe(df_wb_l[['상태', '거래처명', '올해', '회복률']], use_container_width=True, on_select="rerun", selection_mode="single-row")
    if len(ev_wb.selection.rows) > 0:
        sel_biz = df_wb_l.index[ev_wb.selection.rows[0]]
        st.dataframe(df_h[df_h['사업자번호']==sel_biz].sort_values('매출일자', ascending=False), use_container_width=True)

def render_regional_deep_dive(df):
    if df.empty: return
    reg_stats = df.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    reg_stats['Per'] = reg_stats['Sales'] / reg_stats['Count']
    st.markdown("### 🗺️ 지역별 심층 분석")
    fig = px.scatter(reg_stats, x='Count', y='Per', size='Sales', color='지역', text='지역', labels={'Count': '거래처 수', 'Per': '평균 객단가'})
    st.plotly_chart(fig, use_container_width=True)

def render_product_strategy(df):
    if df.empty: return
    st.markdown("### 💊 제품별 전략 분석")
    p_stats = df.groupby('제품명').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    p_stats['Bubble_Size'] = p_stats['Sales'].apply(lambda x: max(x, 0.1))
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.scatter(p_stats, x='Count', y='Sales', size='Bubble_Size', color='제품명', text='제품명', title="BCG 매트릭스 (병원수 vs 매출)"), use_container_width=True)
    with c2:
        pivot = df.pivot_table(index='제품명', columns='월', values='매출액', aggfunc='sum', fill_value=0)
        st.plotly_chart(px.imshow(pivot.div(pivot.max(axis=1), axis=0), color_continuous_scale="Reds", title="제품별 판매 시즌 집중도"), use_container_width=True)

    with st.expander("🧩 **함께 팔기(Cross-selling) 기회 분석기**", expanded=True):
        all_p = sorted(df['제품명'].unique())
        if len(all_p) > 1:
            col_s1, col_s2 = st.columns(2)
            b_p = col_s1.selectbox("Anchor(이미 쓰는것)", all_p, index=0, key='bp_sel')
            t_p = col_s2.selectbox("Target(안쓰는것)", all_p, index=min(1, len(all_p)-1), key='tp_sel')
            if b_p != t_p:
                targets = list(set(df[df['제품명']==b_p]['거래처명'].unique()) - set(df[df['제품명']==t_p]['거래처명'].unique()))
                st.metric("🎯 추가 영업 기회", f"{len(targets)} 처")
                if targets:
                    st.dataframe(df[(df['거래처명'].isin(targets)) & (df['제품명']==b_p)].groupby(['거래처명','지역'])['매출액'].sum().reset_index().sort_values('매출액', ascending=False).head(50), use_container_width=True)

# --------------------------------------------------------------------------------
# 5. 메인 탭 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 1. Overview", "🏆 2. 매출 상위 & 이탈 관리", "🔄 3. 재유입 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석"])

with tab1: render_smart_overview(df_final, df_raw)

with tab2:
    if not df_final.empty:
        total_s = df_final['매출액'].sum()
        top100 = df_final.groupby(['사업자번호', '거래처명', '진료과']).agg({'매출액': 'sum'}).sort_values('매출액', ascending=False).head(100).reset_index()
        cls_d = classify_customers(df_raw, sel_years[0]); st_c = cls_d['상태'].value_counts()
        last_p = df_raw.groupby('사업자번호')['매출일자'].max(); top100['최근구매일'] = top100['사업자번호'].map(last_p)
        cur_d = df_raw['매출일자'].max(); risk_cnt = len(top100[(cur_d - top100['최근구매일']).dt.days >= 90])
        
        st.info(f"**📊 데이터 요약:** 상위 100대 매출 **{top100['매출액'].sum():,.0f}M** ({(top100['매출액'].sum()/total_s*100):.1f}%) | 상태 분포: **기존({st_c.get('✅ 기존 (유지)',0)}), 신규({st_c.get('🆕 신규 (New)',0)}), 재유입({st_c.get('🔄 재유입 (복귀)',0)}), 이탈({st_c.get('📉 1년 이탈',0)})**")
        st.warning(f"**💡 스마트 인사이트:** 상위 100대 중 **{risk_cnt}처**가 90일 이상 공백으로 **이탈 위험** 상태입니다.")
        
    st.markdown('<p class="guide-text">💡 행 클릭 시 상세 실적 확인</p>', unsafe_allow_html=True)
    top100['상태'] = (cur_d - top100['최근구매일']).dt.days.apply(lambda x: '🚨 이탈위험' if x >= 90 else '✅ 정상')
    top100['최근구매일_str'] = top100['최근구매일'].dt.strftime('%Y-%m-%d')
    event_v = st.dataframe(top100[['상태', '거래처명', '진료과', '매출액', '최근구매일_str']], use_container_width=True, on_select="rerun", selection_mode="single-row")
    if len(event_v.selection.rows) > 0:
        sel_biz = top100.iloc[event_v.selection.rows[0]]['사업자번호']
        st.dataframe(df_raw[df_raw['사업자번호'] == sel_biz].groupby('제품명').agg({'매출액':'sum'}).sort_values('매출액', ascending=False), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 거래처 분류별 조회")
    c_s1, c_s2 = st.columns([1, 2])
    with c_s1:
        sel_st = st.selectbox("👇 그룹 선택", sorted(cls_d['상태'].unique()), key='status_sel')
        st.dataframe(st_c.reset_index().rename(columns={'count':'거래처수'}), use_container_width=True)
    with c_s2: st.plotly_chart(px.pie(cls_d[cls_d['상태']==sel_st], names='진료과', title="진료과 분포"), use_container_width=True)
    display_cls = cls_d[cls_d['상태']==sel_st].sort_values('해당년도_매출', ascending=False).copy()
    display_cls['최근구매일'] = display_cls['최근구매일'].dt.strftime('%Y-%m-%d')
    ev_cls = st.dataframe(display_cls[['거래처명','진료과','최근구매일','해당년도_매출']], use_container_width=True, on_select="rerun", selection_mode="single-row")
    if len(ev_cls.selection.rows) > 0:
        row_biz = display_cls.index[ev_cls.selection.rows[0]]
        st.dataframe(df_raw[df_raw['사업자번호']==row_biz].sort_values('매출일자', ascending=False).head(20), use_container_width=True)

with tab3:
    df_h_v = df_raw.sort_values(['사업자번호', '매출일자']).copy()
    df_h_v['구매간격'] = (df_h_v['매출일자'] - df_h_v.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    wb_base = df_h_v[(df_h_v['사업자번호'].isin(df_final['사업자번호'])) & (df_h_v['구매간격'] >= 180)].copy()
    if not wb_base.empty:
        wb_ids_v = wb_base['사업자번호'].unique()
        df_wb_f = pd.DataFrame({'올해': df_final[df_final['사업자번호'].isin(wb_ids_v)].groupby('사업자번호')['매출액'].sum(), '최고': df_raw[df_raw['사업자번호'].isin(wb_ids_v)].groupby('사업자번호')['매출액'].max()}).fillna(0)
        df_wb_f['회복률'] = (df_wb_f['올해'] / df_wb_f['최고'] * 100).replace([np.inf, -np.inf], 0)
        st.info(f"**📊 데이터 요약:** 총 재유입 **{len(wb_ids_v)}처** / 평균 회복률 **{df_wb_f['회복률'].mean():.1f}%**")
        st.warning(f"**💡 스마트 인사이트:** 이탈 후 복귀 시 첫 구매 제품은 **{wb_base.groupby('제품명').size().idxmax()}**입니다.")
    render_winback_quality(df_final, df_raw, sel_years[0])

with tab4:
    reg_v = df_final.groupby('지역').agg(Sales=('매출액','sum'), Count=('사업자번호','nunique')).reset_index().sort_values('Sales', ascending=False)
    if not reg_v.empty:
        st.info(f"**📊 데이터 요약:** 최다 거래 **{reg_v.sort_values('Count', ascending=False).iloc[0]['지역']}** / 최고 매출 **{reg_v.iloc[0]['지역']}**")
        st.warning(f"**💡 스마트 인사이트:** **{reg_v.iloc[0]['지역']}** 지역의 객단가 및 거점 병원 의존도를 점검하세요.")
    render_regional_deep_dive(df_final)
    st.markdown("### 🗺️ 지역별 상세 실적")
    st.markdown('<p class="guide-text">💡 지역 선택 시 상세 리스트 연동</p>', unsafe_allow_html=True)
    sel_r = st.selectbox("🔎 지역 선택", reg_v['지역'].unique(), key="p4_sel")
    col_r1, col_r2 = st.columns([1, 1.5])
    with col_r1: st.dataframe(reg_v.rename(columns={'Count':'구매처수'}), use_container_width=True)
    with col_r2: st.plotly_chart(px.pie(df_final[df_final['지역']==sel_r], values='매출액', names='제품명', hole=0.3, title="지역 제품 비중"), use_container_width=True)
    st.dataframe(df_final[df_final['지역']==sel_r].groupby(['거래처명','제품명']).agg({'매출액':'sum','수량':'sum'}).sort_values('매출액', ascending=False).head(50), use_container_width=True)

with tab5:
    p_v = df_final.groupby('제품명').agg(Sales=('매출액','sum'), Qty=('수량', 'sum'), Count=('사업자번호','nunique')).reset_index().sort_values('Sales', ascending=False)
    cat_v = df_final.groupby('제품군')['매출액'].sum().reset_index().sort_values('매출액', ascending=False)
    if not p_v.empty:
        st.info(f"**📊 데이터 요약:** 최다 판매 **{p_v.iloc[0]['제품명']}** ({int(p_v.iloc[0]['Qty']):,}개) / 최대 제품군 **{cat_v.iloc[0]['제품군']}**")
        st.warning(f"**💡 스마트 인사이트:** **{p_v.iloc[0]['제품명']}**이 성장을 주도 중입니다. 하단 크로스셀링 분석을 활용하세요.")
    
    st.markdown("### 📦 제품별 판매 현황")
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.bar(p_v.head(10), x='Sales', y='제품명', orientation='h', title="제품 매출 Top 10"), use_container_width=True)
    with g2: st.plotly_chart(px.pie(cat_v, values='매출액', names='제품군', hole=0.3, title="제품군 매출 비중"), use_container_width=True)
    
    st.markdown('<p class="guide-text">💡 행 클릭 시 상세 병원 리스트 표시</p>', unsafe_allow_html=True)
    ev_p = st.dataframe(p_v.rename(columns={'Count':'구매처수'}), use_container_width=True, on_select="rerun", selection_mode="single-row", height=300)
    if len(ev_p.selection.rows) > 0:
        sel_p = p_v.iloc[ev_p.selection.rows[0]]['제품명']
        st.dataframe(df_final[df_final['제품명']==sel_p].groupby('거래처명').agg({'매출액':'sum'}).sort_values('매출액', ascending=False), use_container_width=True)
    
    st.markdown("---")
    render_product_strategy(df_final)
