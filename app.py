import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 스타일
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="SKBS Sales Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. 구글 드라이브 데이터 로드 함수 (수정됨)
# --------------------------------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간 동안 데이터 유지
def load_data_from_drive(file_id):
    # 구글 드라이브 직펌 링크 생성
    url = f'https://drive.google.com/uc?id={'1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr'}'
    
    try:
        # 대용량 엑셀 처리를 위해 engine='openpyxl' 명시
        df = pd.read_excel(url, engine='openpyxl')
    except Exception as e:
        st.error(f"구글 드라이브에서 데이터를 불러오지 못했습니다. 권한 설정을 확인해주세요. ({e})")
        return pd.DataFrame()

    # 데이터 전처리 (기존 로직 유지)
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
        df['년'] = df['매출일자'].dt.year
        df['분기'] = df['매출일자'].dt.quarter
        df['월'] = df['매출일자'].dt.month
        df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        for col in ['합계금액', '수량']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['매출액'] = df['합계금액'] / 1000000
        
        str_cols = ['거래처그룹', '제품명', '제품군', '진료과', '지역']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '미분류')
                
        df = df.sort_values(by=['사업자번호', '매출일자'])
    except Exception as e:
        st.error(f"데이터 전처리 오류: {e}")
        return pd.DataFrame()
    
    return df

# --------------------------------------------------------------------------------
# 3. 거래처 분류 함수 (기존 로직 유지)
# --------------------------------------------------------------------------------
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
        has_t2 = (target_year - 2 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 2] > 0)
        has_t3 = (target_year - 3 in cust_year.columns) and (cust_year.loc[biz_no, target_year - 3] > 0)
        past_years = [y for y in cust_year.columns if y < target_year - 1]
        has_history = cust_year.loc[biz_no, past_years].sum() > 0 if past_years else False
            
        if has_ty:
            if has_t1: status = "✅ 기존 (유지)"
            else: status = "🔄 재유입 (복귀)" if (has_history or has_t2 or has_t3) else "🆕 신규 (New)"
        else:
            if has_t1: status = "📉 1년 이탈 (최근)"
            elif has_t2: status = "📉 2년 연속 이탈"
            elif has_t3: status = "📉 3년 연속 이탈"
            else: status = "💤 장기 이탈 (4년+)"
        classification[biz_no] = status
    base_info['상태'] = base_info.index.map(classification)
    return base_info

# --------------------------------------------------------------------------------
# 4. 실행 및 사이드바 (수정됨)
# --------------------------------------------------------------------------------
st.title("📊 SKBS Sales Report")

# 구글 드라이브 아이디로 데이터 자동 로드
DRIVE_FILE_ID = '1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr'
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if not df_raw.empty:
    st.subheader("🔍 데이터 로드 점검")
    st.write(f"전체 데이터 행 개수: {len(df_raw)}개")
    st.write("데이터 샘플 상위 5줄:", df_raw.head())

if df_raw.empty:
    st.warning("데이터가 로드되지 않았습니다. 구글 드라이브 링크가 '링크가 있는 모든 사용자에게 공개'되어 있는지 확인해 주세요.")
    st.stop()

with st.sidebar:
    st.header("🔍 데이터 필터링")
    st.success("✅ 구글 드라이브 데이터가 연결되었습니다.")
    
    # 년도 필터
    all_years = sorted(df_raw['년'].unique(), reverse=True)
    sel_years = st.multiselect("1️⃣ 년도 선택", all_years, default=all_years[:1])
    
    # (이하 필터링 로직은 기존과 동일하되 df_raw 기반으로 흐르도록 구성)
    df_step1 = df_raw[df_raw['년'].isin(sel_years)] if sel_years else df_raw
    
    avail_quarters = sorted(df_step1['분기'].unique())
    sel_quarters = st.multiselect("2️⃣ 분기 선택", avail_quarters, default=avail_quarters)
    df_step2 = df_step1[df_step1['분기'].isin(sel_quarters)] if sel_quarters else df_step1
    
    avail_months = sorted(df_step2['월'].unique())
    sel_months = st.multiselect("3️⃣ 월 선택", avail_months, default=avail_months)
    df_step3 = df_step2[df_step2['월'].isin(sel_months)] if sel_months else df_step2

    if '거래처그룹' in df_raw.columns:
        avail_groups = sorted(df_step3['거래처그룹'].unique())
        sel_groups = st.multiselect("4️⃣ 거래처그룹 선택", avail_groups, default=avail_groups)
        df_step4 = df_step3[df_step3['거래처그룹'].isin(sel_groups)] if sel_groups else df_step3
    else:
        sel_groups = []; df_step4 = df_step3

    if '제품군' in df_raw.columns:
        avail_cats = sorted(df_step4['제품군'].unique())
        sel_cats = st.multiselect("5️⃣ 제품군 선택", avail_cats, default=avail_cats)
        df_step5 = df_step4[df_step4['제품군'].isin(sel_cats)] if sel_cats else df_step4
    else:
        sel_cats = []; df_step5 = df_step4

    if '제품명' in df_raw.columns:
        avail_products = sorted(df_step5['제품명'].unique())
        sel_products = st.multiselect("6️⃣ 제품명 선택", avail_products, default=avail_products)
    else:
        sel_products = []

    # 최종 필터링 적용
    df_year_filtered = df_raw[df_raw['년'].isin(sel_years)] if sel_years else df_raw
    df_final = df_year_filtered.copy()
    if sel_quarters: df_final = df_final[df_final['분기'].isin(sel_quarters)]
    if sel_months: df_final = df_final[df_final['월'].isin(sel_months)]
    if sel_groups: df_final = df_final[df_final['거래처그룹'].isin(sel_groups)]
    if sel_cats: df_final = df_final[df_final['제품군'].isin(sel_cats)]
    if sel_products: df_final = df_final[df_final['제품명'].isin(sel_products)]

# --------------------------------------------------------------------------------
# 5. 메인 탭 (기존 탭 구성 유지)
# --------------------------------------------------------------------------------
# (사용자가 올린 기존 탭 로직 Tab 1 ~ Tab 5 그대로 유지)
# ... [생략: 제공해주신 탭 코드를 그대로 하단에 붙여넣으시면 됩니다] ...


