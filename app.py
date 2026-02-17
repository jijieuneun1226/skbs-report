import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import numpy as np
import requests
import io
import re

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

def get_p(key, default):
    res = params.get_all(key)
    if not res: return default
    if key in ['y', 'q', 'm']: return [int(x) for x in res]
    return res

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (메모리 최적화 필수)
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
                real_url = match_action.group(1).replace("&amp;", "&")
                response = session.get(real_url, params=params_dict, stream=True)
        
        if response.status_code != 200: return pd.DataFrame()
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패. 메모리 또는 링크 확인 필요: {e}"); return pd.DataFrame()

    df.columns = [re.sub(r'\s+', '', str(c)) for c in df.columns]
    
    # 헤더 자동 탐색
    if "매출일자" not in df.columns:
        for idx, row in df.head(10).iterrows():
            if any("매출일자" in str(x) for x in row):
                df.columns = df.iloc[idx].astype(str).str.replace(r'\s+', '', regex=True)
                df = df.iloc[idx+1:].reset_index(drop=True)
                break

    col_map = {
        '매출일자': ['매출일자', '날짜', 'Date'],
        '제품명': ['제품명변환', '제 품 명', '제품명'],
        '합계금액': ['합계금액', '공급가액', '매출액'],
        '수량': ['수량', 'Qty'],
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
        st.error(f"❌ 전처리 중 오류: {e}"); return pd.DataFrame()
    return df

@st.cache_data
def classify_customers(df, target_year):
    cust_year = df.groupby(['사업자번호', '년']).size().unstack(fill_value=0)
    base_info = df.sort_values('매출일자').groupby('사업자번호').agg({'거래처명': 'last', '진료과': 'last', '지역': 'last', '매출일자': 'max'}).rename(columns={'매출일자': '최근구매일'})
    sales_ty = df[df['년'] == target_year].groupby('사업자번호')['매출액'].sum()
    base_info['해당년도_매출'] = base_info.index.map(sales_ty).fillna(0)
    classification = {}
    for biz_no in base_info.index:
        has_ty = (target_year in cust_year.columns) and (cust_year.loc[biz_no, target_year] > 0)
        has_t1 = (target_year - 1 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 1] > 0)
        has_t2 = (target_year - 2 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 2] > 0)
        has_t3 = (target_year - 3 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 3] > 0)
        past_y = [y for y in cust_year.columns if y < target_year - 1]
        has_h = cust_year.loc[biz_no, past_y].sum() > 0 if past_y else False
        if has_ty:
            if has_t1: status = "✅ 기존 (유지)"
            else: status = "🔄 재유입 (복귀)" if has_h else "🆕 신규 (New)"
        else:
            if has_t1: status = "📉 1년 이탈"
            elif has_t2: status = "📉 2년 이탈"
            elif has_t3: status = "📉 3년 이탈"
            else: status = "💤 장기 이탈"
        classification[biz_no] = status
    base_info['상태'] = base_info.index.map(classification)
    return base_info

# --------------------------------------------------------------------------------
# 4. 필터 및 실행
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)
if df_raw.empty: st.stop()

sel_years = get_p('y', [df_raw['년'].max()])
sel_channels = get_p('c', sorted(df_raw['판매채널'].unique()))
sel_quarters = get_p('q', sorted(df_raw['분기'].unique()))
sel_months = get_p('m', sorted(df_raw['월'].unique()))
sel_cats = get_p('cat', sorted(df_raw['제품군'].unique()))
sel_products = get_p('prod', sorted(df_raw['제품명'].unique()))

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
        
        st.markdown("---")
        if st.button("🔗 뷰어용 공유 링크 생성"):
            base_url = "https://skbs-sales-2026-cbktkdtxsyrfzfrihefs2h.streamlit.app/" 
            c_encoded = [urllib.parse.quote(val) for val in sel_channels]
            cat_encoded = [urllib.parse.quote(val) for val in sel_cats]
            prod_encoded = [urllib.parse.quote(val) for val in sel_products]
            p_string = (f"?y={'&y='.join(map(str, sel_years))}&c={'&c='.join(c_encoded)}&q={'&q='.join(map(str, sel_quarters))}"
                        f"&m={'&m='.join(map(str, sel_months))}&cat={'&cat='.join(cat_encoded)}&prod={'&prod='.join(prod_encoded)}")
            st.success("공유 링크 생성 완료!"); st.code(base_url + p_string, language="text")

df_final = df_raw[
    (df_raw['년'].isin(sel_years)) & (df_raw['판매채널'].isin(sel_channels)) &
    (df_raw['분기'].isin(sel_quarters)) & (df_raw['월'].isin(sel_months)) &
    (df_raw['제품군'].isin(sel_cats)) & (df_raw['제품명'].isin(sel_products))
]

# --------------------------------------------------------------------------------
# 5. 메인 탭 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 1. Overview", "🏆 2. 상위 거래처 & 이탈 관리", "🔄 3. 재유입 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석"])

with tab1:
    curr_y = int(df_final['년'].max()) if not df_final.empty else 2026
    df_prev = df_raw[(df_raw['년'] == (curr_y-1)) & (df_raw['월'].isin(df_final['월'].unique()))]
    sales_curr, sales_prev = df_final['매출액'].sum(), df_prev['매출액'].sum()
    sales_pct = ((sales_curr - sales_prev) / (sales_prev if sales_prev > 0 else 1) * 100)
    cust_curr, cust_prev = set(df_final['사업자번호']), set(df_prev['사업자번호'])
    new_cust, lost_cust = len(cust_curr - cust_prev), len(cust_prev - cust_curr)

    st.markdown(f"### 🚀 {curr_y}년 Executive Summary (vs {curr_y-1})")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        c1.metric("💰 총 매출 실적", f"{sales_curr:,.0f} 백만원", f"{sales_pct:+.1f}% (YoY)")
        c1.area_chart(df_final.groupby('월')['매출액'].sum(), height=50, color="#FF4B4B")
        c2.metric("🏥 총 거래 병원", f"{len(cust_curr)} 처")
        c2.markdown(f"- ✨신규: <span style='color:blue'>+{new_cust}</span> / 💔이탈: <span style='color:red'>-{lost_cust}</span>", unsafe_allow_html=True)
        top_p = df_final.groupby('제품명')['매출액'].sum().idxmax() if not df_final.empty else "N/A"
        c3.metric("🏆 Best Product", top_p)
        c3.write(f"기여: **{df_final.groupby('제품명')['매출액'].sum().max():,.0f} 백만원**")
    
    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 📈 년도/분기 현황 요약")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 매출액(년)", f"{df_raw[df_raw['년'].isin(sel_years)]['매출액'].sum():,.0f}M")
        c2.metric("총 구매처수(년)", f"{df_raw[df_raw['년'].isin(sel_years)]['사업자번호'].nunique():,}처")
        c3.metric("선택기간 매출액", f"{sales_curr:,.0f}M")
        c4.metric("선택기간 구매처수", f"{len(cust_curr):,}처")
        col_a, col_b = st.columns([1, 1.5])
        col_a.plotly_chart(px.pie(df_final, values='매출액', names='진료과', hole=0.4, title="진료과 비중"), use_container_width=True)
        col_b.plotly_chart(px.bar(df_final.groupby('년월')['매출액'].sum().reset_index(), x='년월', y='매출액', text_auto='.1f', title="월별 매출 추이", color_discrete_sequence=['#a8dadc']), use_container_width=True)

with tab2:
    st.markdown("### 🏆 상위 거래처 및 거래처 분류 상세 분석")
    if not df_final.empty:
        ranking = df_final.groupby(['사업자번호', '거래처명', '진료과']).agg({'매출액': 'sum'}).reset_index()
        top100 = ranking.sort_values('매출액', ascending=False).head(100).copy()
        last_p_map = df_raw.groupby('사업자번호')['매출일자'].max()
        top100['최근구매일_dt'] = top100['사업자번호'].map(last_p_map)
        cur_date = df_raw['매출일자'].max()
        top100['공백일수'] = (cur_date - top100['최근구매일_dt']).dt.days
        risk_cnt = len(top100[top100['공백일수'] >= 90])
        top100_ratio = (top100['매출액'].sum() / df_final['매출액'].sum() * 100)
        
        st.markdown(f"**[📊 데이터 요약]**\n* 상위 100대 매출 합계: **{top100['매출액'].sum():,.0f} 백만원** ({top100_ratio:.1f}%)\n* 최고 매출 거래처: **{top100.iloc[0]['거래처명']}**")
        insight = f"현재 상위 100대 거래처 중 **{risk_cnt}처({risk_cnt}%)**가 90일 이상 구매가 없는 **이탈 위험** 상태입니다."
        if risk_cnt > 20: st.error(insight)
        else: st.info(insight)

    st.markdown("""<div class="info-box">🆕신규: 최초구매 / ✅기존: 연속구매 / 🔄재유입: 공백 후 복귀 / 📉이탈: 기간 내 구매 부재<br>※ <b>정상:</b> 90일 이내 구매 / <b>이탈위험:</b> 90일 초과 공백</div>""", unsafe_allow_html=True)
    with st.expander("🥇 매출 상위 거래처 Top 100 리스트", expanded=True):
        st.markdown('<p class="guide-text">💡 행 클릭 시 하단 상세 실적 표시</p>', unsafe_allow_html=True)
        top100['최근구매일'] = top100['최근구매일_dt'].dt.strftime('%Y-%m-%d')
        top100['상태'] = top100['공백일수'].apply(lambda x: '🚨 이탈위험' if x >= 90 else '✅ 정상')
        event_vip = st.dataframe(top100[['상태', '거래처명', '진료과', '매출액', '최근구매일']].style.format({'매출액': '{:,.1f} 백만원'}), use_container_width=True, on_select="rerun", selection_mode="single-row", height=350)
        if len(event_vip.selection.rows) > 0:
            v_idx = top100.index[event_vip.selection.rows[0]]
            v_detail = df_raw[df_raw['사업자번호'] == top100.loc[v_idx, '사업자번호']].groupby('제품명').agg({'매출액': 'sum'}).reset_index()
            st.dataframe(v_detail.sort_values('매출액', ascending=False).style.format({'매출액': '{:,.1f} 백만원'}), use_container_width=True)

with tab3:
    st.markdown("### ♻️ 재유입(180일 공백 기준) 현황 분석")
    df_raw_sorted = df_raw.sort_values(['사업자번호', '매출일자'])
    df_raw_sorted['구매간격'] = (df_raw_sorted['매출일자'] - df_raw_sorted.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    wb_ids = df_raw_sorted[(df_raw_sorted['사업자번호'].isin(df_final['사업자번호'])) & (df_raw_sorted['구매간격'] >= 180)]['사업자번호'].unique()
    
    if len(wb_ids) > 0:
        sales_wb = df_final[df_final['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].sum()
        sales_hist = df_raw[df_raw['사업자번호'].isin(wb_ids)].groupby('사업자번호')['매출액'].max()
        df_wb = pd.DataFrame({'올해매출': sales_wb, '과거최고': sales_hist, '거래처명': df_final[df_final['사업자번호'].isin(wb_ids)].groupby('사업자번호')['거래처명'].last()}).reset_index()
        df_wb['회복률'] = (df_wb['올해매출'] / df_wb['과거최고'] * 100).fillna(0)
        df_wb['상태'] = df_wb['회복률'].apply(lambda x: "완전 회복" if x>=80 else ("회복 중" if x>=20 else "재진입 초기"))
        
        st.markdown(f"**[📊 데이터 요약]**\n* 총 재유입 거래처: **{len(wb_ids)}처**\n* 평균 회복률: **{df_wb['회복률'].mean():.1f}%**")
        st.info(f"복귀 고객의 평균 회복률은 {df_wb['회복률'].mean():.1f}%이며, 완전 회복 그룹은 {len(df_wb[df_wb['회복률']>=80])}처입니다.")

        st.markdown('<p class="guide-text">💡 행 클릭 시 상세 공백일수 및 이력 확인</p>', unsafe_allow_html=True)
        event_wb = st.dataframe(df_wb[['상태', '거래처명', '올해매출', '회복률']].style.format({'올해매출': '{:,.1f} 백만원'}), use_container_width=True, on_select="rerun", selection_mode="single-row")
        
        if len(event_wb.selection.rows) > 0:
            sel_biz = df_wb.iloc[event_wb.selection.rows[0]]['사업자번호']
            dt_hist = df_raw_sorted[df_raw_sorted['사업자번호'] == sel_biz].sort_values('매출일자', ascending=False).copy()
            dt_hist['매출일자'] = dt_hist['매출일자'].dt.strftime('%Y-%m-%d')
            st.dataframe(dt_hist[['매출일자', '제품명', '매출액', '수량', '구매간격']].rename(columns={'구매간격':'공백일수'}).style.applymap(lambda v: 'background-color: #ffcccc;' if isinstance(v, (int, float)) and v >= 180 else '', subset=['공백일수']).format({'매출액': '{:,.1f} 백만원'}), use_container_width=True)

with tab4:
    st.markdown("### 🗺️ 지역별 심층 효율성 및 거점 분석")
    reg_s = df_final.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
    reg_s = reg_s[reg_s['Sales'] > 0]
    reg_s['Per'] = reg_s['Sales'] / reg_s['Count']
    
    st.markdown(f"**[📊 데이터 요약]**\n* 최고 매출: **{reg_s.loc[reg_s['Sales'].idxmax(), '지역']}** ({reg_s['Sales'].max():,.0f}백만원)\n* 최고 효율: **{reg_s.loc[reg_s['Per'].idxmax(), '지역']}**")
    st.warning("핵심 거점 의존도: 지역 내 1위처의 매출 비중. 높을수록 해당처 이탈 시 타격 큼")
    
    st.plotly_chart(px.scatter(reg_s, x='Count', y='Per', size='Sales', color='지역', text='지역', labels={'Count': '거래처 수', 'Per': '평균 객단가(백만원)'}), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.pie(reg_s, values='Sales', names='지역', hole=0.3, title="지역별 매출 기여도"), use_container_width=True)
    risk = []
    for r in df_final['지역'].unique():
        r_df = df_final[df_final['지역'] == r]
        risk.append({'지역': r, '의존도': (r_df.groupby('거래처명')['매출액'].sum().max() / r_df['매출액'].sum() * 100)})
    col_b.plotly_chart(px.bar(pd.DataFrame(risk).sort_values('의존도', ascending=False), x='의존도', y='지역', orientation='h', color='의존도', color_continuous_scale='Reds', title="거점 의존도(%)"), use_container_width=True)

with tab5:
    st.markdown("### 📦 제품별 판매 현황 및 시장 침투")
    p_main = df_final.groupby('제품명').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index().sort_values('Sales', ascending=False)
    
    st.markdown(f"**[📊 데이터 요약]**\n* 최다 판매: **{p_main.iloc[0]['제품명']}**\n* 시장 침투율: **{(p_main['Count'].sum() / df_final['사업자번호'].nunique() * 100):.1f}%**")
    st.info("Seasonality: 색이 진할수록 해당 월에 판매가 집중되는 성수기입니다.")

    st.plotly_chart(px.scatter(p_main, x='Count', y='Sales', size='Sales', color='제품명', text='제품명', title="BCG 매트릭스 (병원수 vs 매출)"), use_container_width=True)
    
    st.markdown("#### 📅 제품별 판매 시즌 집중도")
    pivot = df_final.pivot_table(index='제품명', columns='월', values='매출액', aggfunc='sum', fill_value=0)
    st.plotly_chart(px.imshow(pivot.div(pivot.max(axis=1), axis=0), color_continuous_scale="Reds", aspect="auto"), use_container_width=True)

    with st.expander("🧩 함께 팔기(Cross-selling) 기회 분석", expanded=True):
        c_sel1, c_sel2 = st.columns(2)
        all_p = sorted(df_final['제품명'].unique())
        b_p = c_sel1.selectbox("Anchor(쓰는것)", all_p, index=0)
        t_p = c_sel2.selectbox("Target(안쓰는것)", all_p, index=min(1, len(all_p)-1))
        if b_p != t_p:
            acc_a, acc_b = set(df_final[df_final['제품명']==b_p]['거래처명'].unique()), set(df_final[df_final['제품명']==t_p]['거래처명'].unique())
            targets = list(acc_a - acc_b)
            st.metric("🎯 추가 영업 기회", f"{len(targets)} 처")
            if targets:
                st.dataframe(df_final[(df_final['거래처명'].isin(targets))&(df_final['제품명']==b_p)].groupby('거래처명')['매출액'].sum().reset_index().sort_values('매출액', ascending=False), use_container_width=True)
