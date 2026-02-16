import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# 1. 페이지 설정
st.set_page_config(page_title="SKBS Sales Report", layout="wide")
st.title("📊 SKBS Sales Report")

# 2. 데이터 로드 함수 (새 버전 업로드 시 발생하는 ZIP 에러 방지)
@st.cache_data(ttl=60) # 데이터 수정이 잦으므로 캐시 유효 시간을 1분으로 단축
def load_data_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)
        # 로드 시점에 데이터 형식을 유연하게 읽도록 설정
        df = pd.read_excel(file_bytes, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

# 3. 데이터 실행 및 필터링
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if not df_raw.empty:
    # [핵심 수정] 새 버전 업로드 시 변할 수 있는 컬럼명 정리
    # 모든 공백을 제거하여 '제 품 명'과 '제품명'을 동일하게 인식하도록 합니다.
    df_raw.columns = df_raw.columns.astype(str).str.replace(' ', '').str.strip()

    try:
        # 필수 컬럼 존재 여부 확인 및 날짜 변환
        if '매출일자' in df_raw.columns:
            df_raw['매출일자'] = pd.to_datetime(df_raw['매출일자'], errors='coerce')
            df_raw = df_raw.dropna(subset=['매출일자'])
            df_raw['년'] = df_raw['매출일자'].dt.year
            df_raw['월'] = df_raw['매출일자'].dt.month
            df_raw['년월'] = df_raw['매출일자'].dt.strftime('%Y-%m')
        
        # 숫자형 변환 (합계금액, 수량 열이 수정 중 삭제되었을 경우를 대비)
        for col in ['합계금액', '수량']:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
            else:
                df_raw[col] = 0 # 열이 사라졌다면 0으로 기본값 설정
        
        df_raw['매출액(M)'] = df_raw['합계금액'] / 1000000

        st.success("✅ 새 버전 데이터 로드 성공!")

        # 4. 시각화 대시보드
        tab1, tab2, tab3 = st.tabs(["📈 종합 현황", "🏥 거래처 분석", "📦 제품 상세"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매출액", f"{df_raw['매출액(M)'].sum():,.1f} M")
            c2.metric("총 거래처 수", f"{df_raw['거래처명'].nunique():,} 곳")
            c3.metric("총 판매량", f"{df_raw['수량'].sum():,.0f} 개")
            
            monthly = df_raw.groupby('년월')['매출액(M)'].sum().reset_index()
            st.plotly_chart(px.line(monthly, x='년월', y='매출액(M)', markers=True, title="월별 매출 추이"), use_container_width=True)

        with tab2:
            st.subheader("🏥 상위 매출 거래처 (Top 15)")
            top_h = df_raw.groupby('거래처명')['매출액(M)'].sum().sort_values(ascending=False).head(15).reset_index()
            st.plotly_chart(px.bar(top_h, x='매출액(M)', y='거래처명', orientation='h', color='매출액(M)'), use_container_width=True)

        with tab3:
            st.subheader("📦 제품별 점유율")
            # 컬럼명이 '제품명'인지 '제 품 명'인지 상관없이 처리됨
            prod_s = df_raw.groupby('제품명')['매출액(M)'].sum().reset_index()
            st.plotly_chart(px.pie(prod_s, values='매출액(M)', names='제품명', hole=0.4), use_container_width=True)

        with st.expander("🔍 업로드된 데이터 확인"):
            st.dataframe(df_raw.head(100))

    except Exception as e:
        st.error(f"⚠️ 데이터 처리 중 오류: {e}")
        st.info("파일 수정 시 '매출일자', '거래처명', '제품명', '합계금액' 열 이름이 유지되었는지 확인해주세요.")
else:
    st.warning("데이터를 불러올 수 없습니다. 파일 ID나 공유 설정을 확인하세요.")
