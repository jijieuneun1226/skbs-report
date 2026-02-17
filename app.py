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
    .info-box {padding: 10px; border-radius: 5px; font-size: 13px; margin-bottom: 15px; border: 1px solid #e0e0e0; line-height: 1.6; background-color: #fcfcfc;}
    .summary-box {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; margin-bottom: 20px;}
    .insight-box {background-color: #fff9e6; padding: 15px; border-radius: 10px; border-left: 5px solid #ffcc00; margin-bottom: 20px;}
    .guide-text {color: #FF4B4B; font-size: 13px; font-weight: 600; margin-bottom: 10px;}
    h4 {margin-top: 0px; margin-bottom: 10px; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 SKBS Sales Report")

def get_p(key, default):
    res = params.get_all(key)
    if not res: return default
    if key in ['y', 'q', 'm']: return [int(x) for x in res]
    return res

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (기존 로직 유지)
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
# 3. 분석 화면 구성 (인사이트 추가 버전)
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

df_final = df_raw[
    (df_raw['년'].isin(sel_years)) & 
    (df_raw['판매채널'].isin(sel_channels)) &
    (df_raw['분기'].isin(sel_quarters)) & 
    (df_raw['월'].isin(sel_months)) &
    (df_raw['제품군'].isin(sel_cats)) &
    (df_raw['제품명'].isin(sel_products))
]

# --------------------------------------------------------------------------------
# 메인 탭
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 1. Overview", "🏆 2. 매출 상위 & 이탈 관리", "🔄 3. 재유입 분석", "🗺️ 4. 지역 분석", "📦 5. 제품 분석"])

# --- 탭 1. Overview ---
with tab1:
    if not df_final.empty:
        current_year = int(df_final['년'].max())
        st.markdown(f"### 🚀 {current_year}년 통합 대시보드")
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        with c1:
            st.metric("💰 총 매출 실적", f"{df_final['매출액'].sum():,.0f} 백만원")
            st.area_chart(df_final.groupby('월')['매출액'].sum(), height=100, color="#FF4B4B")
        with c2:
            st.metric("🏥 총 거래 병원", f"{df_final['사업자번호'].nunique():,} 처")
        with c3:
            top_p = df_final.groupby('제품명')['매출액'].sum().idxmax()
            st.metric("🏆 Best Product", top_p)

        st.markdown("---")
        col_a, col_b = st.columns([1, 1.5])
        with col_a: st.plotly_chart(px.pie(df_final, values='매출액', names='진료과', hole=0.4, title="진료과별 매출 비중"), use_container_width=True)
        with col_b:
            monthly_b = df_final.groupby('년월')['매출액'].sum().reset_index()
            st.plotly_chart(px.bar(monthly_b, x='년월', y='매출액', text_auto='.1f', title="월별 매출 추이", color_discrete_sequence=['#a8dadc']), use_container_width=True)

# --- 탭 2. 상위 거래처 & 이탈 관리 ---
with tab2:
    st.markdown("### 🏆 상위 거래처 & 이탈 관리 분석")
    
    if not df_final.empty:
        total_sales = df_final['매출액'].sum()
        ranking = df_final.groupby(['사업자번호', '거래처명', '진료과']).agg({'매출액': 'sum'}).reset_index()
        top100 = ranking.sort_values('매출액', ascending=False).head(100)
        top100_sales = top100['매출액'].sum()
        top100_ratio = (top100_sales / total_sales * 100) if total_sales > 0 else 0
        
        cls_df = classify_customers(df_raw, sel_years[0])
        st_counts = cls_df['상태'].value_counts()
        
        cur_date = df_raw['매출일자'].max()
        top100['최근구매일'] = top100['사업자번호'].map(df_raw.groupby('사업자번호')['매출일자'].max())
        top100['위험상태'] = top100['최근구매일'].apply(lambda x: '🚨 이탈위험' if (cur_date - x).days >= 90 else '✅ 정상')
        risk_count = len(top100[top100['위험상태'] == '🚨 이탈위험'])
        highest_cust = top100.iloc[0]

        # [📊 데이터 요약]
        st.markdown(f"""
        <div class="summary-box">
            <h4>📊 데이터 요약</h4>
            <ul>
                <li><b>상위 100대 매출 합계:</b> {top100_sales:,.0f} 백만원 (전체 매출의 {top100_ratio:.1f}% 차지)</li>
                <li><b>거래처 상태 분포:</b> 기존({st_counts.get('✅ 기존 (유지)', 0)}처), 신규({st_counts.get('🆕 신규 (New)', 0)}처), 재유입({st_counts.get('🔄 재유입 (복귀)', 0)}처), 이탈({st_counts.get('📉 1년 이탈', 0)}처)</li>
                <li><b>평균 객단가:</b> 처당 약 { (total_sales / df_final['사업자번호'].nunique() * 100).round(0) if df_final['사업자번호'].nunique() > 0 else 0:,.0f} 만원</li>
                <li><b>최고 매출 거래처:</b> {highest_cust['거래처명']} ({highest_cust['매출액']:,.0f} 백만원)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # [💡 스마트 인사이트]
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 스마트 인사이트</h4>
            <ul>
                <li><b>VIP 이탈 위험 알림:</b> 현재 상위 100대 거래처 중 <b>{risk_count}처({risk_count}%)</b>가 90일 이상 구매가 없는 이탈 위험 상태입니다. 특히 매출 기여도가 가장 높은 <b>{highest_cust['거래처명']}</b>의 공백이 {(cur_date - highest_cust['최근구매일']).days}일째 지속되고 있어 즉각적인 관리가 필요합니다.</li>
                <li><b>신규 vs 이탈 밸런스:</b> 금기 신규 유입 거래처는 {st_counts.get('🆕 신규 (New)', 0)}처인 반면, 이탈(1년 기준) 거래처는 {st_counts.get('📉 1년 이탈', 0)}처입니다. 유입 대비 이탈이 많으므로 신환 유입보다 기존 고객 유지를 위한 프로모션 강화가 우선시됩니다.</li>
                <li><b>진료과 집중도 리스크:</b> 현재 매출의 {(df_final.groupby('진료과')['매출액'].sum().max() / total_sales * 100):.1f}%가 <b>{df_final.groupby('진료과')['매출액'].sum().idxmax()}</b>에 편중되어 있습니다. 해당 과의 정책 변화나 경쟁사 침투 시 타격이 클 수 있으므로, 타 진료과로의 제품 라인업 확장이 권장됩니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🥇 매출 상위 거래처 Top 100 리스트", expanded=True):
            st.dataframe(top100[['위험상태', '거래처명', '진료과', '매출액', '최근구매일']].style.format({'매출액': '{:,.1f} 백만원'}), use_container_width=True)

# --- 탭 3. 재유입 분석 ---
with tab3:
    st.markdown("### 🔄 재유입 심층 분석")
    df_history = df_raw.sort_values(['사업자번호', '매출일자']).copy()
    df_history['구매간격'] = (df_history['매출일자'] - df_history.groupby('사업자번호')['매출일자'].shift(1)).dt.days
    winback_data = df_history[(df_history['사업자번호'].isin(df_final['사업자번호'])) & (df_history['구매간격'] >= 180)].copy()
    
    if not winback_data.empty:
        winback_ids = winback_data['사업자번호'].unique()
        sales_curr = df_final[df_final['사업자번호'].isin(winback_ids)].groupby('사업자번호')['매출액'].sum()
        sales_max = df_raw[df_raw['사업자번호'].isin(winback_ids)].groupby('사업자번호')['매출액'].max()
        df_wb = pd.DataFrame({'올해매출': sales_curr, '과거최고': sales_max})
        df_wb['회복률'] = (df_wb['올해매출'] / df_wb['과거최고'] * 100).fillna(0)
        
        wb_full = len(df_wb[df_wb['회복률'] >= 80])
        wb_mid = len(df_wb[(df_wb['회복률'] < 80) & (df_wb['회복률'] >= 20)])
        wb_low = len(df_wb[df_wb['회복률'] < 20])
        trigger_p = winback_data.groupby('제품명').size().idxmax()

        # [📊 데이터 요약]
        st.markdown(f"""
        <div class="summary-box">
            <h4>📊 데이터 요약</h4>
            <ul>
                <li><b>총 재유입 거래처:</b> {len(winback_ids)} 처 (6개월 공백 후 복귀 기준)</li>
                <li><b>재유입 발생 총 매출:</b> {df_wb['올해매출'].sum():,.0f} 백만원</li>
                <li><b>평균 회복률:</b> {df_wb['회복률'].mean():.1f}% (과거 최고 매출 대비 현재 매출 비율)</li>
                <li><b>그룹별 분포:</b> 완전 회복({wb_full}처), 회복 중({wb_mid}처), 재진입 초기({wb_low}처)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # [💡 스마트 인사이트]
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 스마트 인사이트</h4>
            <ul>
                <li><b>회복 퀄리티 요약:</b> 올해 재유입된 거래처의 평균 회복률은 <b>{df_wb['회복률'].mean():.1f}%</b>입니다. 완전 회복 그룹이 {wb_full}처로, 복귀 고객들의 충성도가 빠르게 정상화되고 있습니다.</li>
                <li><b>복귀 트리거(Trigger) 제품:</b> 이탈 고객들이 복귀 시 가장 먼저 주문하는 제품은 <b>{trigger_p}</b>입니다. 휴면 업체 대상 마케팅 시 이 제품을 전면에 내세우는 것이 효과적입니다.</li>
                <li><b>재유입 매출 기여도:</b> 전체 매출 중 재유입 거래처가 기여하는 비중은 <b>{(df_wb['올해매출'].sum()/df_final['매출액'].sum()*100):.1f}%</b>입니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(px.scatter(df_wb.reset_index(), x='과거최고', y='올해매출', size='올해매출', hover_name='사업자번호', title="재유입 거래처 회복 퀄리티 (과거 최고치 vs 현재)"), use_container_width=True)

# --- 탭 4. 지역 분석 ---
with tab4:
    st.markdown("### 🗺️ 지역별 거점 및 효율성 분석")
    if not df_final.empty:
        reg_stats = df_final.groupby('지역').agg(Sales=('매출액', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
        top_sales_reg = reg_stats.loc[reg_stats['Sales'].idxmax()]
        top_count_reg = reg_stats.loc[reg_stats['Count'].idxmax()]
        
        # 의존도 계산
        risk = []
        for r in df_final['지역'].unique():
            r_df = df_final[df_final['지역'] == r]
            risk.append({'지역': r, '의존도': (r_df.groupby('거래처명')['매출액'].sum().max() / r_df['매출액'].sum() * 100)})
        df_risk = pd.DataFrame(risk).sort_values('의존도', ascending=False)

        # [📊 데이터 요약]
        st.markdown(f"""
        <div class="summary-box">
            <h4>📊 데이터 요약</h4>
            <ul>
                <li><b>최다 거래 지역:</b> {top_count_reg['지역']} ({top_count_reg['Count']:,}처)</li>
                <li><b>최고 매출 지역:</b> {top_sales_reg['지역']} ({top_sales_reg['Sales']:,.0f} 백만원)</li>
                <li><b>활성 지역 수:</b> {len(reg_stats)} 개 지역</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # [💡 스마트 인사이트]
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 스마트 인사이트</h4>
            <ul>
                <li><b>커버리지 분석:</b> 현재 <b>{top_count_reg['지역']}</b> 지역이 가장 많은 거래처를 보유하며 핵심 영업 지역으로 기능하고 있습니다.</li>
                <li><b>지역별 핵심 거점 의존도:</b> <b>{df_risk.iloc[0]['지역']}</b>은 상위 1개 병원의 매출 비중이 <b>{df_risk.iloc[0]['의존도']:.1f}%</b>에 달합니다. 거점 병원 의존도를 낮추기 위해 인근 중소 병원 대상의 크로스셀링 전략이 필요합니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(px.bar(reg_stats.sort_values('Sales', ascending=False), x='지역', y='Sales', color='Sales', title="지역별 매출 합계"), use_container_width=True)

# --- 탭 5. 제품 분석 ---
with tab5:
    st.markdown("### 📦 제품별 전략 및 크로스셀링 분석")
    if not df_final.empty:
        p_stats = df_final.groupby('제품명').agg(Sales=('매출액', 'sum'), Qty=('수량', 'sum'), Count=('사업자번호', 'nunique')).reset_index()
        top_p = p_stats.loc[p_stats['Sales'].idxmax()]
        penetration = (top_p['Count'] / df_final['사업자번호'].nunique() * 100)
        
        # [📊 데이터 요약]
        st.markdown(f"""
        <div class="summary-box">
            <h4>📊 데이터 요약</h4>
            <ul>
                <li><b>최다 판매 제품:</b> {top_p['제품명']} ({top_p['Qty']:,}개 / {top_p['Sales']:,.0f} 백만원)</li>
                <li><b>최대 매출 제품군:</b> {df_final.groupby('제품군')['매출액'].sum().idxmax()} (전체 매출의 {(df_final.groupby('제품군')['매출액'].sum().max()/df_final['매출액'].sum()*100):.1f}%)</li>
                <li><b>시장 침투율:</b> 전체 거래처 중 {penetration:.1f}%가 <b>{top_p['제품명']}</b>을 구매 중</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # [💡 스마트 인사이트]
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 스마트 인사이트</h4>
            <ul>
                <li><b>카테고리 기여도:</b> {df_final.groupby('제품군')['매출액'].sum().idxmax()} 제품군이 실적을 견인하고 있으며, 성숙기 단계에 진입한 주력 제품 외에 신규 라인업 확장이 필요합니다.</li>
                <li><b>교차 판매 기회:</b> 특정 제품군 내의 구매 편중이 확인되므로, 패키지 제안을 통한 <b>크로스셀링(Cross-selling)</b> 영업 타겟 선정이 유효할 것으로 보입니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(px.scatter(p_stats, x='Count', y='Sales', size='Sales', color='제품명', text='제품명', title="제품 포트폴리오 (병원 수 vs 매출)"), use_container_width=True)

        with st.expander("🧩 함께 팔기(Cross-selling) 기회 분석기", expanded=True):
            all_prods = sorted(df_final['제품명'].unique())
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1: base_p = st.selectbox("Anchor 제품 (이미 쓰는 것)", all_prods, index=0)
            with c_sel2: target_p = st.selectbox("Target 제품 (팔고 싶은 것)", all_prods, index=min(1, len(all_prods)-1))
            if base_p != target_p:
                acc_A = set(df_final[df_final['제품명'] == base_p]['거래처명'].unique())
                acc_B = set(df_final[df_final['제품명'] == target_p]['거래처명'].unique())
                targets = list(acc_A - acc_B)
                st.metric("🎯 추가 영업 기회", f"{len(targets)} 처")
                if targets:
                    st.dataframe(df_final[df_final['거래처명'].isin(targets[:50])][['거래처명', '지역']].drop_duplicates(), use_container_width=True)
