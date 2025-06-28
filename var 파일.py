import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 데이터 불러오기 =====
data1 = pd.read_csv(r"C:\Users\USER\Downloads\환율_yyyymm.csv").rename(columns={"환율": "data1"})
data2 = pd.read_csv(r"C:\Users\USER\Downloads\소매 yyyymm.csv", encoding="cp949").rename(columns={"소매판매": "data2"})
data3 = pd.read_csv(r"C:\Users\USER\Downloads\경제심리지수.csv", encoding="cp949").rename(columns={"변동폭": "data3"})
data4 = pd.read_csv(r"C:\Users\USER\Downloads\1st_Difference_of_CPI_Data.csv").rename(columns={"소비자물가지수": "data4"})

# 날짜 컬럼 정리 (data1의 날짜가 다를 경우 처리)
data1 = data1.rename(columns={"날짜": "yyyymm"}) if "날짜" in data1.columns else data1
for df in [data1, data2, data3, data4]:
    df['yyyymm'] = df['yyyymm'].astype(str)

# 병합
merged_df = data1.merge(data2, on='yyyymm') \
                 .merge(data3, on='yyyymm') \
                 .merge(data4, on='yyyymm')
merged_df = merged_df.set_index('yyyymm').dropna()

# ===== 학습용, 테스트용 분할 =====
train_size = 200
train_df = merged_df.iloc[:train_size]
test_df = merged_df.iloc[train_size:]

# ===== VAR 학습 =====
model = VAR(train_df)
lag_order = model.select_order(12)
best_lag = lag_order.aic
results = model.fit(best_lag)

# ===== 예측 =====
forecast_input = train_df.values[-best_lag:]
forecast = results.forecast(y=forecast_input, steps=len(test_df))

# ===== 예측 결과를 DataFrame으로 정리 =====
forecast_df = pd.DataFrame(forecast, index=test_df.index, columns=merged_df.columns)

# ===== 변수별 예측 vs 실제 시각화 =====
for col in merged_df.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(test_df.index, test_df[col], label='실제값', marker='o')
    plt.plot(forecast_df.index, forecast_df[col], label='예측값', marker='x')
    plt.title(f'{col} 예측 vs 실제')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
