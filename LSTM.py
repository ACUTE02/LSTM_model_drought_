import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# 1. Load Dataset
summer_yearly = pd.read_csv(r"E:\LSTM\0-2017-yearly.csv")
summer_yearly['Date_MOD'] = summer_yearly['DATE'].str.split('_').str[1].astype(int)

# 2. Decompose Series
def decompose_series(df, var, period):
    analysis = df[var]
    result = seasonal_decompose(analysis, model="additive", period=period, extrapolate_trend='freq')
    return result.trend, result.seasonal, result.resid

tr, sea, resid = decompose_series(summer_yearly, 'D0', 50)

# 3. Handle NaN values
tr = tr.dropna().values.reshape(-1,1)

# 4. Normalize Data
scaler = MinMaxScaler(feature_range=(0,1))
tr_scaled = scaler.fit_transform(tr)

# 5. Create Training Data
train_size = int(len(tr_scaled) * 0.8)
train_data = tr_scaled[:train_size]

features_set = []
labels = []

for i in range(50, len(train_data)):
    features_set.append(train_data[i-50:i])
    labels.append(train_data[i])

features_set, labels = np.array(features_set), np.array(labels)

# 6. Build Model
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(features_set.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train Model
history = model.fit(features_set, labels,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

# 8. Prepare Test Data
test_data = tr_scaled[train_size - 50:]
test_features = []
test_labels = tr_scaled[train_size:]

for i in range(50, len(test_data)):
    test_features.append(test_data[i-50:i])

test_features = np.array(test_features)

# 9. Predictions
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(test_labels)

# 10. Save Model (Correct Way)
model.save("LSTM_model.h5")

# 11. Plot Accuracy/Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()

# 12. Plot Predictions
plt.figure(figsize=(16,6))
plt.plot(actual, color='blue', label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('Drought Trend Prediction')
plt.legend()
plt.show()