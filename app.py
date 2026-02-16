import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# 1. 페이지 설정
st.set_page_config(page_title="SKBS Sales Report", layout="wide")
st.title("📊 SKBS Sales Report")

# 2. 데이터 로드 함수 (구글 드라이브 보안 안내 페이지 우회 로직 추가)
@st.cache_data(ttl=3600)
def load_data_from_drive(file_id):
    # 구글 드라이브에서 대용량 파일 다운로드 시 안내 페이지 우회용 세션 생성
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    
    try:
        # ZIP 에러 방지를 위해 바이트 스트림으로 로드
        file_bytes = io.BytesIO(response.content)
        df = pd.read_excel(file_bytes, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return pd.DataFrame()

# 3. 데이터 실행 및 전처리
DRIVE_FILE_ID = "1lFGcQST27rBuUaXcuOJ7yRnMlQWGyxfr"
df_raw = load_data_from_drive(DRIVE_FILE_ID)

if not df_raw.empty:
    # 컬럼명 표준화 (이미지에 맞춰 공백 제거 및 별칭 지정)
    df_raw.columns = df_raw.columns.astype(str).str.replace(' ', '').str.strip()
    
    # 필수 컬럼 맵핑
    col_map = {
        '매출일자': '매출일자',
        '제품명': '제품명', # 이미지의 '제 품 명' 공백 제거됨
        '합계금액': '합계금액',
        '수량': '수량',
        '거래처명': '거래처명'
    }

    try:
        # 전처리: 날짜 및 숫자 변환
        df_raw['매출일자'] = pd.to_datetime(df_raw['매출일자'], errors='coerce')
        df_raw = df_raw.dropna(subset=['매출일자'])
        df_raw['년월'] = df_raw['매출일자'].dt.strftime('%Y-%m')
        
        for col in ['합계금액', '수량']:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        
        # 금액 단위 변환 (백만원)
        df_raw['매출액(M)'] = df_raw['합계금액'] / 1000000

        st.success("✅ 데이터 로드 및 전처리 완료!")

        # 4. 시각화 대시보드
        tab1, tab2, tab3 = st.tabs(["📈 종합 요약", "🏥 거래처/지역", "📦 제품 상세"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매출액", f"{df_raw['매출액(M)'].sum():,.1f} M")
            c2.metric("총 거래처", f"{df_raw['거래처명'].nunique():,} 곳")
            c3.metric("총 판매량", f"{df_raw['수량'].sum():,.0f} 개")
            
            # 월별 추이 그래프
            monthly = df_raw.groupby('년월')['매출액(M)'].sum().reset_index()
            fig_line = px.line(monthly, x='년월', y='매출액(M)', title="월별 매출 추이 (단위: 백만원)", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        with tab2:
            st.subheader("🏆 상위 매출 거래처 Top 15")
            top_hospitals = df_raw.groupby('거래처명')['매출액(M)'].sum().sort_values(ascending=False).head(15).reset_index()
            fig_bar = px.bar(top_hospitals, x='매출액(M)', y='거래처명', orientation='h', color='매출액(M)')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("📦 제품별 매출 비중")
            prod_sales = df_raw.groupby('제품명')['매출액(M)'].sum().reset_index()
            fig_pie = px.pie(prod_sales, values='매출액(M)', names='제품명', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        # 데이터 미리보기
        with st.expander("🔍 전체 데이터 보기"):
            st.dataframe(df_raw)

    except Exception as e:
        st.error(f"⚠️ 전처리 중 오류 발생: {e}")
else:
    st.warning("데이터를 불러올 수 없습니다. 구글 드라이브 파일 ID와 공유 설정을 다시 확인해주세요.")
