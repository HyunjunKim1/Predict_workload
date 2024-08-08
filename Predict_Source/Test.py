import pandas as pd
import prophet as ph
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# DataFrame 생성
all_forecasts = pd.DataFrame()

# 엑셀파일에서 데이터 가져오기.
file_path = 'D:\Git\Predict_workload\Dataset.xlsx'
columns_to_select = ['ITEM_CODE', 'PRODUCT_WPNL', 'CLOCK_ACCEPT', 'RECIPE']

df = pd.read_excel(file_path, usecols=columns_to_select)

print(df)

item_codes = df['ITEM_CODE'].unique()

model = ph.Prophet()
        
for item in item_codes:
    # 학습된 모델 불러오기
    with open('D:\Git\Predict_workload\prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
        # 다음날 하루 (1일) 생산량 예측
        future = model.make_future_dataframe(periods=30, freq='D')
        
        # 예측
        forecast = model.predict(future)
        
        # 예측 데이터에 ITEM 정보 추가
        forecast['ITEM_CODE'] = item
        all_forecasts = pd.concat([all_forecasts, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'ITEM_CODE']]], ignore_index=True)
        
# yhat = trend + daily + other components(weekly, lower, Hight 등등)
output_file_path = 'D:\Git\Predict_workload\daily_predicted_production_times.xlsx'
all_forecasts.to_excel(output_file_path, index=False)
print(f'Predictions saved to {output_file_path}')

