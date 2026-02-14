import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="SKBS Sales Report", layout="wide")

# 2. 데이터 로드 함수
@st.cache_data(ttl=3600)
def load_data_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        df = pd.read_excel(url, engine='openpyxl')
        # 컬럼명 정리
        df.columns = df.columns.astype(str).str.strip()
        
        # 날짜 및 숫자 변환
        df['매출일자'] = pd.to_datetime(df['매출일자'])
        df['년'] = df['매출일자'].dt.year
        df['월'] = df['매출일자'].dt.month
        df['년월'] = df['매출일자'].dt.strftime('%Y-%m')
        
        # 합계금액 및 수량 숫자화
        for col in ['합계금액', '수량']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['매출액'] = df['합계금액'] / 1000000
        
        # 제품명 정제 (접두사 제거)
        if '제품명' in df.columns:
            df['제품명'] = df['제품명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
            
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

# 3. 메인 실행
st.title("📊 SKBS Sales Report")
DRIVE_FILE_ID = '1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr'
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if not df_raw.empty:
    with st.sidebar:
        st.header("🔍 필터 설정")
        # 데이터에 있는 년도만 표시
        available_years = sorted(df_raw['년'].unique(), reverse=True)
        sel_years = st.multiselect("년도 선택", available_years, default=available_years)
        
        # 제품명 필터
        available_p = sorted(df_raw['제품명'].unique())
        sel_p = st.multiselect("제품명 선택", available_p, default=available_p[:5])

    # 필터링 적용
    df_final = df_raw[df_raw['년'].isin(sel_years) & df_raw['제품명'].isin(sel_p)]

    # 리포트 출력
    c1, c2 = st.columns(2)
    with c1:
        st.metric("총 매출액", f"{df_final['매출액'].sum():,.1f} 백만원")
    with c2:
        st.metric("총 거래처 수", f"{df_final['거래처명'].nunique():,} 처")

    st.subheader("📅 월별 매출 추이")
    monthly = df_final.groupby('년월')['매출액'].sum().reset_index()
    st.plotly_chart(px.line(monthly, x='년월', y='매출액', markers=True), use_container_width=True)

    st.subheader("🏆 제품별 매출 비중")
    p_sales = df_final.groupby('제품명')['매출액'].sum().reset_index()
    st.plotly_chart(px.pie(p_sales, values='매출액', names='제품명', hole=0.4), use_container_width=True)

    st.subheader("🏥 거래처별 매출 Top 10")
    top_cust = df_final.groupby('거래처명')['매출액'].sum().nlargest(10).reset_index()
    st.bar_chart(top_cust.set_index('거래처명'))
else:
    st.info("데이터를 불러오는 중입니다...")
