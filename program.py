import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# นำเข้าข้อมูล AQI
data = pd.read_csv('C:\\Users\\User\\Desktop\\Project\\predictAQI\\Day_graph_data_AQI.csv')

# แปลงปีในคอลัมน์ datebegin เป็น ค.ศ.
data['datebegin'] = data['datebegin'].str.replace('2562', '2019')
data['datebegin'] = data['datebegin'].str.replace('2563', '2020')
data['datebegin'] = data['datebegin'].str.replace('2564', '2021')
data['datebegin'] = data['datebegin'].str.replace('2565', '2022')
data['datebegin'] = data['datebegin'].str.replace('2566', '2023')

# แปลงคอลัมน์ datebegin เป็นรูปแบบของวันที่ใน Python
data['datebegin'] = pd.to_datetime(data['datebegin'], format='%d/%m/%Y')

# แปลง timestamp เป็นตัวเลข
data['datebegin'] = data['datebegin'].apply(lambda x: x.timestamp())

# เลือก features (คอลัมน์ยกเว้น AQI) และ target (ค่า AQI)
X = data.drop(columns=['City', 'Country', 'Station area', 'Station type', 'AQI'])  # เลือกทุกคอลัมน์ยกเว้น AQI และคอลัมน์ที่ไม่เหมาะสม
y = data['AQI']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# ทำนายค่า AQI ด้วยชุดทดสอบ
y_pred_gb = gb_model.predict(X_test)
# ปัดเศษของค่า AQI ที่ทำนายได้
y_pred_rounded = [round(pred) for pred in y_pred_gb]

# สร้างวันที่ในอนาคตที่ต้องการ
future_dates = pd.date_range(start='2024-01-01', end='2024-01-31')  # ตั้งแต่วันที่ 1 มกราคม 2024 ถึง วันที่ 31 มกราคม 2024

# สร้าง DataFrame ของวันที่ในอนาคตพร้อมกับค่า AQI ที่ทำนายได้
future_predicted_data = pd.DataFrame({
    'Date': future_dates,
    'Predicted_AQI': y_pred_rounded[:31]
})

# คำนวณค่า MSE กับ R2 ของ Gradient Boosting Model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

import matplotlib.pyplot as plt

# สร้างกราฟแท่ง
plt.figure(figsize=(10, 6))
plt.bar(future_predicted_data['Date'], future_predicted_data['Predicted_AQI'], color='skyblue')
plt.title('Predicted AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# แสดงกราฟ
plt.show()

# บันทึก DataFrame เป็นไฟล์ CSV
# future_predicted_data.to_csv('future_predicted_data.csv', index=False)

# print('Mean Squared Error (Gradient Boosting):', mse_gb)
# print('R-squared (Gradient Boosting):', r2_gb)
# print(future_predicted_data)