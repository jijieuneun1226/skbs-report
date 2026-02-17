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
# 1. 페이지 설정 및 권한 제어 (SK분석 기본 폼 유지)
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
    /* 글자색 강제 고정 스타일 */
    .fix-text { color: #000000 !important; font-size: 15px; line-height: 1.8; margin-bottom: 5px; }
    .fix-title-blue { color: #0044cc !important; font-weight: 800; font-size: 18px; margin-top: 10px; margin-bottom: 10px; }
    .fix-title-orange { color: #cc5500 !important; font-weight: 800; font-size: 18px; margin-top: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 SKBS Sales Report")

def get_p(key, default):
    res = params.get_all(key)
    if not res: return default
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
        '제품명': ['제품명변환', '제 품 명', '제품명'],
        '합계금액': ['합계금액', '공급가액', '금액', '매출액'],
        '수량': ['수량', 'Qty', '판매수량'],
        '사업자번호': ['사업자번호', 'BizNo'],
        '거래처명': ['거래처명', '병원명'],
        '진료과': ['진료과', '진료과목'],
        '제품군': ['제품군', '카테고리'],
        '거래처그룹': ['거래처그룹', '그룹'],
        '주소': ['도로명주소', '주소'],
        '지역': ['지역', '시도']
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
    st.markdown("""<div class="info-box">
    <b>🔍 재유입 정의:</b> 직전 구매일로부터 <b>최소 180일(6개월) 이상 공백기</b> 이후 다시 구매가 발생한 거래처 (선택 기간 내 첫 구매 기준)<br>
    <b>🚦 회복 퀄리티:</b> 과거 전성기(최고 매출) 대비 현재 매출 수준<br>
    - 🟢 <b>완전 회복:</b> 80% 이상 / 🟡 <b>회복 중:</b> 20~80% / 🔴 <b>재진입 초기:</b> 20% 미만<br>
    <b>📈 평균 회복률 뜻:</b> 재유입된 거래처들이 과거 가장 많이 구매했던 시기 대비 현재 평균적으로 몇 %까지 구매력이 회복되었는지를 나타냄
    </div>""", unsafe_allow_html=True)

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
                             category_orders={"상태": ["완전 회복", "회복 중", "재진입 초기"]},
                             color_discrete_map={"완전 회복": "green", "회복 중": "orange", "재진입 초기": "red"})
            fig.add_shape(type="line", x0=0, y0=0, x1=df_wb['과거최고'].max(), y1=df_wb['과거최고'].max(), line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        except: st.warning("차트 생성 불가")
    with col_li:
        st.markdown('<p class="guide-text">💡 리스트의 행을 클릭하면 하단에서 실제 공백 기간과 구매 이력을 확인할 수 있습니다.</p>', unsafe_allow_html=True)
        event_wb = st.dataframe(df_wb[['상태', '거래처명', '올해매출', '회복률']], hide_index=True, use_container_width=True,
                               on_select="rerun", selection_mode="single-row",
                               column_config={"회복률": st.column_config.ProgressColumn("회복도", format="%.1f%%", min_value=0, max_value=100), "올해매출": st.column_config.NumberColumn(format="%.1f 백만원")})

    if len(event_wb.selection.rows) > 0:
        sel_idx = event_wb.selection.rows[0]
        sel_biz_no = df_wb.iloc[sel_idx]['사업자번호']
        sel_name = df_wb.iloc[sel_idx]['거래처명']
        st.markdown(f"#### 🔍 [{sel_name}] 실제 구매 간격 및 상세 내역")
        detail_hist = df_history[df_history['사업자번호'] == sel_biz_no].sort_values('매출일자', ascending=False).copy()
        detail_hist['매출일자_str'] = detail_hist['매출일자'].dt.strftime('%Y-%m-%d')
        st.dataframe(detail_hist[['매출일자_str', '제품명', '매출액', '수량', '구매간격']].rename(columns={'매출일자_str':'매출일자', '구매간격':'직전구매후공백(일)'})
                     .style.applymap(lambda v: 'background-color: #ffcccc; font-weight: bold;' if isinstance(v, (int, float)) and v >= 180 else '', subset=['직전구매후공백(일)'])
                     .format({'매출액': '{:,.1f} 백만원', '직전구매후공백(일)': '{:,.0f} 일'}), 
                     use_container_width=True)

def render_regional_deep_dive(df):
    if df.empty: return
    reg_stats = df.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    reg_stats['Per'] = reg_stats['Sales'] / reg_stats['Count']
    
    st.markdown("### 🗺️ 지역별 심층 효율성 및 거점 영향력 분석")
    st.markdown(f"""<div class="info-box">
    <b>📈 지역 전략 요약:</b><br>
    - <b>최고 매출 지역:</b> 기간 내 전체 합산 매출액이 가장 큰 지역<br>
    - <b>영업 효율 1위:</b> 거래처 1처당 평균 매출(객단가)이 가장 높은 지역<br>
    - <b>활성 지역 수:</b> 기간 내 단 1건이라도 매출이 발생한 총 행정 구역 수<br>
    - <b>핵심 거점 의존도:</b> 지역 내 1위 거래처가 차지하는 매출 비중. 높을수록 해당 거래처 이탈 시 리스크가 큼
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("최고 매출 지역", reg_stats.loc[reg_stats['Sales'].idxmax(), '지역'])
        c2.metric("영업 효율 1위", reg_stats.loc[reg_stats['Per'].idxmax(), '지역'], f"{reg_stats['Per'].max():.1f} 백만원/처")
        c3.metric("활성 지역 수", f"{len(reg_stats)} 개")

    fig = px.scatter(reg_stats, x='Count', y='Per', size='Sales', color='지역', text='지역', 
                     labels={'Count': '거래처 수', 'Per': '평균 객단가 (백만원)'})
    fig.add_hline(y=reg_stats['Per'].mean(), line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(reg_stats, values='Sales', names='지역', hole=0.3, title="지역별 매출 기여도 (%)"), use_container_width=True)
    with c2:
        risk = []
        for r in df['지역'].unique():
            r_df = df[df['지역'] == r]
            risk.append({'지역': r, '의존도': (r_df.groupby('거래처명')['매출액'].sum().max() / r_df['매출액'].sum() * 100)})
        st.plotly_chart(px.bar(pd.DataFrame(risk).sort_values('의존도', ascending=False), x='의존도', y='지역', orientation='h', color='의존도', color_continuous_scale='Reds', title="핵심 거점 매출 의존도 (%)"), use_container_width=True)

def render_product_strategy(df):
    if df.empty: return
    st.markdown("### 💊 제품별 전략 심층 분석")
    p_stats = df.groupby('제품명').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    p_stats['Bubble_Size'] = p_stats['Sales'].apply(lambda x: max(x, 0.1))
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(p_stats, x='Count', y='Sales', size='Bubble_Size', color='제품명', text='제품명', title="제품 BCG 매트릭스 (병원수 vs 매출)"), use_container_width=True)
    with c2:
        total_acc = df['사업자번호'].nunique()
        p_stats['Penetration'] = (p_stats['Count'] / total_acc) * 100
        st.plotly_chart(px.bar(p_stats.sort_values('Penetration'), x='Penetration', y='제품명', orientation='h', text_auto='.1f', title=f"시장 침투율 (%)"), use_container_width=True)
    
    st.markdown("#### 📅 제품별 판매 시즌 집중도 (Seasonality)")
    st.markdown("""<div class="info-box">
    <b>💡 분석 가이드:</b> 연간 최대 매출 월을 1.0으로 환산. 🟥 진할수록 성수기임을 의미합니다.
    </div>""", unsafe_allow_html=True)
    season_pivot = df.pivot_table(index='제품명', columns='월', values='매출액', aggfunc='sum', fill_value=0)
    st.plotly_chart(px.imshow(season_pivot.div(season_pivot.max(axis=1), axis=0), color_continuous_scale="Reds", aspect="auto"), use_container_width=True)

    with st.expander("🧩 **함께 팔기(Cross-selling) 기회 분석기**", expanded=True):
        st.markdown("""<div class="info-box">
        <b>🎯 추출 기준:</b> Anchor 제품(기존 사용중) 구매처 중, Target 제품(패키지 제안)을 아직 한 번도 구매하지 않은 병원 리스트를 추출합니다.
        </div>""", unsafe_allow_html=True)
        col_sel1, col_sel2 = st.columns(2)
        all_prods = sorted(df['제품명'].unique())
        with col_sel1: base_p = st.selectbox("Anchor 제품 (이미 쓰는 것)", all_prods, index=0)
        with col_sel2: target_p = st.selectbox("Target 제품 (팔고 싶은 것)", all_prods, index=min(1, len(all_prods)-1))
        if base_p != target_p:
            acc_A, acc_B = set(df[df['제품명'] == base_p]['거래처명'].unique()), set(df[df['제품명'] == target_p]['거래처명'].unique())
            targets = list(acc_A - acc_B)
            st.metric("🎯 추가 영업 기회", f"{len(targets)} 처")
            if targets:
                t_info = df[(df['거래처명'].isin(targets)) & (df['제품명'] == base_p)].groupby(['거래처명', '지역'])['매출액'].sum().reset_index().sort_values('매출액', ascending=False)
                st.dataframe(t_info.head(50), column_config={"매출액": st.column_config.NumberColumn("구매액(백만원)", format="%.1f")}, hide_index=True)

@st.cache_data
def classify_customers(df, target_year):
    cust_year = df.groupby(['사업자번호', '년']).size().unstack(fill_value=0)
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg({'거래처명': 'last', '진료과': 'last', '지역': 'last', '매출일자': 'max'}).rename(columns={'매출일자': '최근구매일'})
    sales_ty = df[df['년'] == target_year].groupby('사업자번호')['매출액'].sum()
    base_info['해당년도_매출'] = base_info.index.map(sales_ty).fillna(0)
    classification = {}
    for biz_no in base_info.index:
        has_ty = (target_year in cust_year.columns) and (cust_year.loc[biz_no, target_year] > 0)
        has_t1 = (target_year - 1 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 1] ... [truncated]
