import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from statsmodels.tools.sm_exceptions import ConvergenceWarning

output = 'outputs'
if not os.path.exists(output):
    os.makedirs(output)
'''
DATA PREPROCESSING
'''
df = pd.read_csv('daily-website-visitors.csv')

# Ép kiểu dữ liệu
cols_fix = ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']
for col in cols_fix:
    df[col] = df[col].str.replace(',','').astype(int)
print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Kiểm tra giá trị thiếu và đảm bảo tính liên tục của dữ liệu
print(f'Number of missing values:\n{df.isnull().sum()}')
df = df.asfreq('D')

df.info()
print(df.head())

'''
DATA VISUALIZATION
'''
plt.figure(figsize=(15, 7))
plt.plot(df['Unique.Visits'], label='Unique Visits', linewidth=1)

plt.title('Daily Unique Visits (2014-2020)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Number of Unique Visitors')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/unique_visitors_trend.png')

'''
EDA
'''
# 1. Time Series Decomposition
results = seasonal_decompose(df['Unique.Visits'], model='additive', period=7)
fig = results.plot()

plt.suptitle('Time Series Decomposition of Unique Visits', fontsize=16, y=0.95)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.savefig('outputs/time_series_decomposition.png')

# 2. Kiểm định tính dừng (Stationary Test)
def adf_test(series, title=''):
    results_adf = adfuller(series)
    print(f'{title}\n p-value: {results_adf[1]:.4f}')
    if results_adf[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
adf_test(df['Unique.Visits'], "ADF Test on Original Series")

# 3. Sai phân (Differencing)
df_diff = df['Unique.Visits'].diff().dropna()   # Lấy sai phân bậc 1 (Ngày t trừ ngày t-1)
adf_test(df_diff, "ADF Test on Differenced Series")

# 4. ACF and PACF plots
# Tìm giá trị tới hạn (Critical Value)
n = len(df_diff)
critical_value = 1.96 / np.sqrt(n)
print(f'Critical value for 95% confidence: ±{critical_value:.4f}')

# Vẽ biểu đồ ACF và PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plot_acf(df_diff, ax=axes[0], lags=40)  # Vẽ biểu đồ ACF (xác định q)
axes[0].set_title('ACF of Differenced Unique Visits')

plot_pacf(df_diff, ax=axes[1], lags=40) # Vẽ biểu đồ PACF (xác định p)
axes[1].set_title('PACF of Differenced Unique Visits')  
plt.tight_layout()
plt.savefig('outputs/acf_pacf_plots.png')

'''
FEATURE ENGINEERING
'''
# 1. Các đặc trưng thời gian (Time-based features)
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# Vẽ biểu đồ các đặc trưng thời gian mới tạo
# Biểu đồ theo thứ
weekly_avg = df.groupby('day_of_week')['Unique.Visits'].mean()
plt.figure(figsize=(10, 5))
colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'orange', 'orange'] 
weekly_avg.plot(kind='bar', color=colors)

plt.title('Average of Unique Visits by Day (0=Monday, 6=Sunday)')
plt.xlabel('Day of week')
plt.ylabel('Number of visits')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('outputs/daily_visits.png')

# Biểu đồ theo tháng
plt.figure(figsize=(10, 5))
df.groupby('month')['Unique.Visits'].mean().plot(kind='line', marker='o')
plt.title('Average of Unqiue Visits by Month')
plt.xlabel('Month')
plt.ylabel('Number of Visits')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/monthly_visits.png')

# 2. Đặc trưng độ trễ (Lag features)
df['Lag_1'] = df['Unique.Visits'].shift(1)  # Giá trị ngày hôm trước
df['Lag_7'] = df['Unique.Visits'].shift(7)  # Giá trị đúng ngày này tuần trước

# 3. Đặc trưng trung bình trượt (Rolling features)
df['rolling_mean_7'] = df['Unique.Visits'].rolling(window=7).mean()
df['rolling_std_7'] = df['Unique.Visits'].rolling(window=7).std()

# 4. Xử lý giá trị thiếu (NaN) do Shift/Rolling tạo
df.dropna(inplace=True)

# 5. Độ tương quan (Correlation)
# Giữ lại các cột quan trọng
important_cols = [
    'Unique.Visits', 
    'day_of_week', 
    'is_weekend', 
    'month',
    'Lag_1', 
    'Lag_7', 
    'rolling_mean_7'
]
df_final = df[important_cols]

correlation_matrix = df_final.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Website Traffic Features')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')

'''
MODEL DEVELOPMENT & EVALUATION
'''
# 5.1. Tinh chỉnh danh sách biến 
# Chuẩn bị dữ liệu
target = 'Unique.Visits'
exog_cols = ['Lag_7', 'rolling_mean_7']

df_p = df.reset_index()[['Date', 'Unique.Visits']]
df_p.columns = ['ds', 'y']

# Chia Train/Test (Tách 30 ngày cuối)
train_data_s = df.iloc[:-30]
test_data_s = df.iloc[-30:]

train_data_p = df_p.iloc[:-30]
test_data_p = df_p.iloc[-30:]

# 5.2. Huấn luyện mô hình 
# 5.2.1. Mô hình SARIMA
model_sarima = SARIMAX(train_data_s['Unique.Visits'],
                        exog=train_data_s[exog_cols],
                        order=(1,1,1),
                        seasonal_order=(1,1,1,7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

model_sarima_fit = model_sarima.fit(disp=False)
print(model_sarima_fit.summary())

# Dự báo 30 ngày (SARIMA)
sarima_forecast = model_sarima_fit.get_forecast(steps=30,
                                                exog=test_data_s[exog_cols])
sarima_pred = sarima_forecast.predicted_mean

# 5.2.2. Mô hình Prophet
model_prophet = Prophet(yearly_seasonality=True, 
                        weekly_seasonality=True, 
                        daily_seasonality=False)
model_prophet_fit = model_prophet.fit(train_data_p)

# Dự báo 30 ngày (Prophet)
future = model_prophet.make_future_dataframe(periods=30)
prophet_forecast = model_prophet_fit.predict(future)
prophet_pred = prophet_forecast.iloc[-30:]['yhat'].values

# 5.3. So sánh (Comparison) và Đánh giá (Evaluation)
def evaluate_model (y_actual, y_pred, model_name):
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual-y_pred)/y_actual))*100
    r2 = r2_score(y_actual, y_pred)

    print(f'{model_name} Evaluation')
    print(f'MAE: {mae:.4}')
    print(f'MAPE: {mape:.4}')
    print(f'RMSE: {rmse:.4}')
    print(f'R2: {r2:.4}')

# 5.4 Time Series Cross-Validation (SARIMA)
tscv = TimeSeriesSplit(n_splits=5, test_size=30)
mae_scores = []
mape_scores = []
rmse_scores = []
r2_scores = []

warnings.filterwarnings('ignore', category=ConvergenceWarning)
fold = 1
for train_index, test_index in tscv.split(df):
    cv_train = df.iloc[train_index]
    cv_test = df.iloc[test_index]

    # Huấn luyện lại trên từng Fold
    model_cv = SARIMAX(cv_train['Unique.Visits'],
                       exog=cv_train[exog_cols],
                       order=(1,1,1),
                       seasonal_order=(1,1,1,7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
    model_cv_fit = model_cv.fit(disp=False)

    # Dự báo trên tập Test của từng Fold
    pred_cv = model_cv_fit.get_forecast(steps=len(cv_test),
                                        exog=cv_test[exog_cols]).predicted_mean
    
    # Tính các chỉ số cho từng Fold
    mae = mean_absolute_error(cv_test['Unique.Visits'], pred_cv)
    mape = np.mean(np.abs((cv_test['Unique.Visits']-pred_cv)/cv_test['Unique.Visits']))*100
    rmse = np.sqrt(mean_absolute_error(cv_test['Unique.Visits'], pred_cv))
    r2 = r2_score(cv_test['Unique.Visits'], pred_cv)

    mae_scores.append(mae)
    mape_scores.append(mape)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f'Fold {fold}: MAE = {mae:.4f} | MAPE = {mape:.4f}% | RMSE = {rmse:.4f} | R2 = {r2:.4f}')
    fold += 1
warnings.filterwarnings('default', category=ConvergenceWarning)

print(f'MAE trung bình: {np.mean(mae_scores):.4f}')
print(f'MAPE trung bình: {np.mean(mape_scores):.4f}%')
print(f'RMSE trung bình: {np.mean(rmse_scores):.4f}')
print(f'R2 trung bình: {np.mean(r2_scores):.4f}')

# Đánh giá từng mô hình
sarima_eval = evaluate_model(test_data_s['Unique.Visits'], sarima_pred, 'SARIMA')
prophet_eval = evaluate_model(test_data_p['y'], prophet_pred, 'Prophet')

'''
CONCLUSION
'''
plt.figure(figsize=(12, 6))
plt.plot(test_data_s.index, test_data_s['Unique.Visits'],
         label ='Actual', color='black', alpha=0.6)
plt.plot(test_data_s.index, sarima_pred,
         label='SARIMA', marker='o')
plt.plot(test_data_s.index, prophet_pred,
         label ='Prophet', linestyle='--')
plt.title('Comparison: Prediction by SARIMA and Prophet (Last 30 days)')
plt.xlabel('Date')
plt.ylabel('Unique Visits')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6) 
plt.tight_layout()
plt.savefig('outputs/comparison_sarima_prophet.png')

