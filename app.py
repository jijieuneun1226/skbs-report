import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 권한 제어 (URL 파라미터)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="SKBS Sales Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL 파라미터 확인 (?mode=edit 인 경우에만 수정 가능)
query_params = st.query_params
is_edit_mode = query_params.get("mode") == "edit"

# 보는 사람(일반 접속자)에게는 사이드바를 숨기는 CSS
if not is_edit_mode:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
            section[data-testid="stSidebar"] {width: 0px;}
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    div.block-container {padding-top: 1rem;}
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #4e79a7;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .guide-text {
        color: #007BFF;
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .status-desc {
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 SKBS Sales Report")

# --------------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        df = pd.read_excel(url, engine='openpyxl')
    except Exception as e:
        st.error(f"구글 드라이브 연결 실패: {e}")
        return pd.DataFrame()

    df.columns = df.columns.astype(str).str.strip()
    
    # 컬럼 매핑 로직
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
        df['매출일자'] = pd.to_datetime(df['매출일자'])
        df = df.sort_values('매출일자') # 날짜순 정렬
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
        
        # 채널 구분
        def classify_channel(group):
            online_list = ['B2B', 'B2B(W)', 'SAP', '의사회원']
            if group in online_list: return '🌐 온라인'
            elif group == 'SDP': return '🏢 오프라인'
            else: return '기타'

        if '거래처그룹' in df.columns:
            df['판매채널'] = df['거래처그룹'].apply(classify_channel)
        
        str_cols = ['거래처그룹', '제품명', '제품군', '진료과', '지역']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '미분류')
    except Exception as e:
        st.error(f"데이터 전처리 오류: {e}")
        return pd.DataFrame()
    return df

# --------------------------------------------------------------------------------
# 3. 데이터 로드 및 사이드바 제어
# --------------------------------------------------------------------------------
DRIVE_FILE_ID = '1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr'
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if df_raw.empty:
    st.warning("데이터를 불러오지 못했습니다.")
    st.stop()

# 기본 필터 값 초기화
sel_years = [df_raw['년'].max()]
sel_channels = sorted(df_raw['판매채널'].unique())
sel_quarters = sorted(df_raw['분기'].unique())
sel_months = sorted(df_raw['월'].unique())
sel_cats = []
sel_products = []

# [관리자 모드] 사이드바 노출
if is_edit_mode:
    with st.sidebar:
        st.header("⚙️ 관리자 설정")
        st.success("수정 모드 활성화")
        sel_channels = st.multiselect("0️⃣ 판매채널 선택", sorted(df_raw['판매채널'].unique()), default=sel_channels)
        df_s1 = df_raw[df_raw['판매채널'].isin(sel_channels)]
        
        sel_years = st.multiselect("1️⃣ 년도 선택", sorted(df_s1['년'].unique(), reverse=True), default=sel_years)
        df_s2 = df_s1[df_s1['년'].isin(sel_years)]
        
        sel_quarters = st.multiselect("2️⃣ 분기 선택", sorted(df_s2['분기'].unique()), default=sel_quarters)
        sel_months = st.multiselect("3️⃣ 월 선택", sorted(df_s2['월'].unique()), default=sel_months)
        
        avail_cats = sorted(df_s2['제품군'].unique())
        sel_cats = st.multiselect("4️⃣ 제품군 선택", avail_cats)
        
        df_s3 = df_s2[df_s2['제품군'].isin(sel_cats)] if sel_cats else df_s2
        avail_products = sorted(df_s3['제품명'].unique())
        sel_products = st.multiselect("5️⃣ 제품명 선택", avail_products)

# 최종 데이터 필터링 로직
df_year_filtered = df_raw[df_raw['년'].isin(sel_years)]
df_final = df_year_filtered[df_year_filtered['판매채널'].isin(sel_channels)]
if sel_quarters: df_final = df_final[df_final['분기'].isin(sel_quarters)]
if sel_months: df_final = df_final[df_final['월'].isin(sel_months)]
if sel_cats: df_final = df_final[df_final['제품군'].isin(sel_cats)]
if sel_products: df_final = df_final[df_final['제품명'].isin(sel_products)]

# --------------------------------------------------------------------------------
# 4. 메인 화면 구성
# --------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏆 VIP & 이탈", "🔄 재유입 분석", "🗺️ 지역 분석", "📦 제품 분석"])

with tab1:
    st.markdown("### 📈 성과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 매출액 (선택년도)", f"{df_year_filtered['매출액'].sum():,.0f}M")
    c2.metric("총 구매처수 (선택년도)", f"{df_year_filtered['사업자번호'].nunique():,}처")
    c3.metric("필터조건 매출액", f"{df_final['매출액'].sum():,.1f}M")
    c4.metric("필터조건 구매처수", f"{df_final['사업자번호'].nunique():,}처")
    
    col_a, col_b = st.columns([1, 1.5])
    with col_a:
        st.subheader("🏥 진료과별 매출")
        st.plotly_chart(px.pie(df_final, values='매출액', names='진료과', hole=0.4), use_container_width=True)
    with col_b:
        st.subheader("📅 월별 매출/처수 추이")
        monthly = df_final.groupby('년월').agg({'매출액': 'sum', '사업자번호': 'nunique'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly['년월'], y=monthly['매출액'], name='매출(M)', yaxis='y1', marker_color='#a8dadc'))
        fig.add_trace(go.Scatter(x=monthly['년월'], y=monthly['사업자번호'], name='처수', yaxis='y2', line=dict(color='#e63946', width=3)))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🏆 VIP 고객 관리")
    st.info("💡 **이탈 위험 기준:** 최근 구매일로부터 **90일 이상** 경과된 거래처입니다.")
    
    ranking = df_final.groupby(['사업자번호', '거래처명', '진료과']).agg({'매출액': 'sum', '수량': 'sum'}).reset_index()
    top100 = ranking.sort_values('매출액', ascending=False).head(100).copy()
    
    last_p = df_raw.groupby('사업자번호')['매출일자'].max()
    cur_date = df_raw['매출일자'].max()
    top100['최근구매일'] = top100['사업자번호'].map(last_p)
    top100['상태'] = top100['최근구매일'].apply(lambda x: '🚨 이탈위험' if (cur_date - x).days >= 90 else '✅ 정상')
    
    st.markdown('<p class="guide-text">행을 선택하면 하단에 상세 구매 품목이 나타납니다.</p>', unsafe_allow_html=True)
    sel_event = st.dataframe(top100[['상태', '거래처명', '진료과', '매출액', '수량']].style.format({'매출액': '{:,.1f}M'}), 
                             use_container_width=True, on_select="rerun", selection_mode="single-row")

    if len(sel_event.selection.rows) > 0:
        idx = top100.index[sel_event.selection.rows[0]]
        bz_no = top100.loc[idx, '사업자번호']
        st.subheader(f"🔍 {top100.loc[idx, '거래처명']} 상세 품목")
        dtl = df_raw[df_raw['사업자번호'] == bz_no].groupby('제품명').agg({'수량': 'sum', '매출액': 'sum'}).reset_index()
        st.table(dtl.sort_values('매출액', ascending=False).style.format({'매출액': '{:,.1f}M'}))

    st.markdown("---")
    st.markdown(f"**※ 거래처 상태 분류 ({sel_years[0]}년 기준)**")
    st.write("🆕 신규: 올해 첫 거래 | ✅ 기존: 작년/올해 연속 | 🔄 재유입: 이탈 후 복귀 | 📉 이탈: 올해 거래 없음")

with tab4:
    st.markdown("### 🗺️ 지역별 현황")
    reg_data = df_final.groupby('지역').agg({'매출액': 'sum', '사업자번호': 'nunique'}).reset_index().sort_values('매출액', ascending=False)
    st.dataframe(reg_data.rename(columns={'사업자번호': '구매처수'}).style.format({'매출액': '{:,.1f}M'}), use_container_width=True)
    st.plotly_chart(px.bar(reg_data, x='지역', y='매출액', color='지역', title="지역별 매출 규모"), use_container_width=True)

with tab5:
    st.markdown("### 📦 제품별 판매 현황")
    prod_data = df_final.groupby('제품명').agg({'매출액': 'sum', '수량': 'sum', '사업자번호': 'nunique'}).reset_index().sort_values('매출액', ascending=False)
    st.dataframe(prod_data.rename(columns={'사업자번호': '구매처수'}).style.format({'매출액': '{:,.1f}M'}), use_container_width=True)
