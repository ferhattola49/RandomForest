import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Veriyi yükle
data_url = r"C:\Users\Administrator\Desktop\veri.xlsx"
dataFrame = pd.read_excel(data_url, index_col='Gün', parse_dates=True)

# Sadece 'Net Satış Miktarı_MA' sütununu kullan
dataFrame = dataFrame[['Net Satış Miktarı_MA']]

# Dönemsel özellikler ekleme
dataFrame['Month'] = dataFrame.index.month
dataFrame['Week'] = dataFrame.index.isocalendar().week
dataFrame['Day'] = dataFrame.index.day
dataFrame['DayOfWeek'] = dataFrame.index.dayofweek

# Eğitim ve test verisi olarak ayırma
train = dataFrame.iloc[:877]
test = dataFrame.iloc[877:]

# Sadece hedef sütunu (Net Satış Miktarı_MA) ölçekleme
scaler_target = MinMaxScaler()
scaled_train_target = scaler_target.fit_transform(train[['Net Satış Miktarı_MA']])
scaled_test_target = scaler_target.transform(test[['Net Satış Miktarı_MA']])

# Diğer özellikleri ölçekleme
scaler_features = MinMaxScaler()
scaled_train_features = scaler_features.fit_transform(train.drop(columns=['Net Satış Miktarı_MA']))
scaled_test_features = scaler_features.transform(test.drop(columns=['Net Satış Miktarı_MA']))

# Özellik oluşturma (geçmiş n veri noktasını kullanarak)
def create_features(data, n_lags=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    return df

n_lags = 12  # geçmiş 12 veri noktasını kullan
train_features = create_features(scaled_train_target, n_lags)
test_features = create_features(scaled_test_target, n_lags)

# Eğitim ve test verilerini ayırma
X_train = train_features.iloc[:, :-1].values
y_train = train_features.iloc[:, -1].values
X_test = test_features.iloc[:, :-1].values
y_test = test_features.iloc[:, -1].values

# Random Forest modelini tanımlama
rf_model = RandomForestRegressor(random_state=42)

# Hiperparametre optimizasyonu
param_grid = {
    'n_estimators': [300],
    'max_depth': [30],
    'min_samples_split': [10],
    'min_samples_leaf': [4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# En iyi modeli seçme
best_rf_model = grid_search.best_estimator_

# Test seti üzerinde tahmin yapma
test_pred = best_rf_model.predict(X_test)

# Tahminleri ölçekten geri çevirme
test_pred_inverse = scaler_target.inverse_transform(test_pred.reshape(-1, 1))
y_test_inverse = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(train.index[-len(X_train):], scaler_target.inverse_transform(y_train.reshape(-1, 1)), label='Eğitim Seti')
plt.plot(test.index[-len(X_test):], y_test_inverse, label='Gerçek Değerler')
plt.plot(test.index[-len(X_test):], test_pred_inverse, label='Random Forest Tahminleri')
plt.legend()
plt.xlabel('Tarih')
plt.ylabel('Net Satış Miktarı_MA')
plt.title('Random Forest ile Zaman Serisi Tahmini')
plt.show()

# Performans metrikleri
mse = mean_squared_error(y_test_inverse, test_pred_inverse)
mae = mean_absolute_error(y_test_inverse, test_pred_inverse)
r2 = r2_score(y_test_inverse, test_pred_inverse)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Accuracy hesaplama
threshold = 0.1  # %10 hata payı
accuracy = np.mean(np.abs((y_test_inverse - test_pred_inverse) / y_test_inverse) < threshold)
print(f'Accuracy: {accuracy}')

# Gelecek tahminleri
def forecast_future(model, data, n_steps, n_lags, scaler_target):
    future_predictions = []
    current_data = data.copy()

    for _ in range(n_steps):
        features = np.array(current_data[-n_lags:]).reshape(1, -1)
        prediction = model.predict(features)
        future_predictions.append(prediction[0])

        # Yeni tahmini veriye ekle
        current_data = np.append(current_data, prediction)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_inverse = scaler_target.inverse_transform(future_predictions)
    return future_predictions_inverse

n_steps = 100  # Tahmin yapmak istediğiniz gün sayısı
future_predictions = forecast_future(best_rf_model, scaled_test_target.flatten(), n_steps, n_lags, scaler_target)

# Gelecek tahminlerini görselleştirme
future_dates = pd.date_range(start=test.index[-1], periods=n_steps)  # Burada düzelttim
future_predictions_flat = future_predictions.flatten()  # Tahminleri düzleştir

plt.figure(figsize=(10, 6))
plt.plot(dataFrame.index, dataFrame['Net Satış Miktarı_MA'], label='Geçmiş Değerler')
plt.plot(future_dates, future_predictions_flat, label='Gelecek Tahminleri')
plt.legend()
plt.xlabel('Tarih')
plt.ylabel('Net Satış Miktarı_MA')
plt.title('Random Forest ile Gelecek Tahminleri')
plt.show()

 # Gelecek tahminlerini Excel'e kaydetme
future_df = pd.DataFrame({'Tarih': future_dates, 'Tahminler': future_predictions_flat})
future_df.to_excel(r"C:\Users\Administrator\Desktop\gelecek_tahminleri(gercek).xlsx", index=False)
print("Gelecek tahminleri Excel dosyasına kaydedildi.")