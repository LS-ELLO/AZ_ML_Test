#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("https://github.com/data-labs/data/raw/main/weight-height.csv")

# 단위 변환 (인치, 파운드 --> cm, kg)
data.Height = data.Height*2.54 # cm
data.Weight = data.Weight*0.453592 # kg

# 사본 사용 (원본 백업)
df = data.copy()

#%%
from sklearn.model_selection import train_test_split
x = df.Height.values
y = df.Weight.values

x_train, x_test, y_train, y_test = train_test_split(x, y)

#%%
# 학습 및 검증 데이터 크기 확인
print(x_train.shape, x_test.shape, y_train.shape,y_test.shape)

#%%
# 모델 생성, 학습, 사용 후 성능 보기 (R-Squared)
from sklearn.linear_model import LinearRegression
model = LinearRegression() # (1) 모델 생성
model.fit(x_train.reshape(-1,1), y_train) # (2) 학습
print(model.score(x_test.reshape(-1,1), y_test)) # (3) 이용, 성능평가

#%%
# 학습한 계수(파라미터)
a, b = model.coef_, model.intercept_
print(a, b)

#%%
# 훈련과 검증 데이터에 대한 회귀 직선 보기
plt.figure(figsize=(10,4))
xs = np.linspace(140,200,2)
ys = a*xs + b

plt.subplot(1,2,1)  
plt.title('Train Data')
plt.plot(xs,ys, c='r')
plt.scatter(x_train,y_train, s=0.05)

plt.subplot(1,2,2)  
plt.title('Test Data')
plt.plot(xs,ys, c='r')
plt.scatter(x_test,y_test, s=0.05)