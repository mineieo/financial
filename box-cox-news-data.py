import pandas as pd
import matplotlib.pyplot as plt

# 파일 불러오기
file_path = "C:\\Users\\USER\\Downloads\\Box-Cox_______.csv"
data = pd.read_csv(file_path, encoding='cp949')

# Box-Cox 열 이름 자동 탐색
boxcox_column = [col for col in data.columns if 'BoxCox' in col or 'Box-Cox' in col][0]
boxcox_series = data[boxcox_column].dropna()

# Z-score 계산
z_scores = (boxcox_series - boxcox_series.mean()) / boxcox_series.std()

# 이상치 인덱스: |Z| >= 1.5
outlier_idx = z_scores[abs(z_scores) >= 1.5].index

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(boxcox_series.index, boxcox_series.values, label='Box-Cox 변환 시계열')
plt.scatter(outlier_idx, boxcox_series.loc[outlier_idx], color='red', label='|Z| ≥ 1.96 이상치', zorder=5)
plt.title("Box-Cox 변환 시계열 및 이상치 표시 (|Z| ≥ 1.96)")
plt.xlabel("시점")
plt.ylabel("Box-Cox 값")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
