from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math

app = Flask(__name__)

def train_and_predict(csv_data, days_to_predict):
    # Load and preprocess the data
    df = pd.read_csv(csv_data)
    df_ori = df.reset_index()[['Date', 'Amount']]
    df_ori_test_size = int(len(df_ori) * 0.70)
    df_ori_test = df_ori.loc[df_ori_test_size:, ['Date', 'Amount']]
    
    df2 = df.reset_index()['Amount']
    
    Q1 = df2.quantile(0.25)
    Q3 = df2.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df['Category'] = np.where((df2 < lower_bound) | (df2 > upper_bound),
    'Unexpected Expense', 'Regular Expense')

    regular_expenses = df[df['Category'] == 'Regular Expense']
    df2_regular = regular_expenses['Amount']
    
    df2_regular_with_date = regular_expenses[['Date', 'Amount']]
    df2_regular_with_date_size = int(len(df2_regular_with_date) * 0.70)
    df_regular_with_date = df2_regular_with_date.loc[df2_regular_with_date_size:, ['Date', 'Amount']]

    unexpected_expenses = df[df['Category'] == 'Unexpected Expense']
    df2_unexpected = unexpected_expenses['Amount']
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')

    df = df.dropna(subset=['Date'])

    unexpected_expenses = df[df['Category'] == 'Unexpected Expense']

    unexpected_expenses_by_month = unexpected_expenses.groupby(unexpected_expenses['Date'].dt.to_period('M')).agg(
        count=('Amount', 'size'),
        amounts=('Amount', list)
    ).reset_index()

    unexpected_expenses_by_month.columns = ['Month', 'Count', 'Amounts']
    
    unexpected_expenses_by_month['Month'] = unexpected_expenses_by_month['Month'].astype(str)

    result_unexpected_expenses = unexpected_expenses_by_month.to_dict(orient='records')
    
    df2_log = np.log(df2_regular + 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df2_scaled = scaler.fit_transform(np.array(df2_log).reshape(-1, 1))
    
    train_size = int(len(df2_scaled) * 0.70)
    train_data, test_data = df2_scaled[0:train_size, :], df2_scaled[train_size:len(df2_scaled), :1]
    
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    time_step = 1
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    data_length = len(df2_scaled)
    if data_length <= 100:
        learning_rate = 0.001
    else:
        learning_rate = 0.0001
    
    
    # LSTM Model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=64, verbose=1, callbacks=[early_stopping])
    
    
    # Predictions on test data
    test_predict = model.predict(X_test, verbose=0)
    test_predict_original = np.exp(scaler.inverse_transform(test_predict)) - 1
    
    # Predict future values
    def predict_future_days(model, all_test_data, days_to_predict):
        temp_input = list(all_test_data.flatten())
        future_predictions = []
        for _ in range(days_to_predict):
            current_input = np.array(temp_input[-1:]).reshape(1, 1, 1)
            next_prediction = model.predict(current_input, verbose=0)[0, 0]
            future_predictions.append(next_prediction)
            temp_input.append(next_prediction)
        return np.array(future_predictions)
    
    future_predictions = predict_future_days(model, test_data, days_to_predict)
    future_predictions_original = np.exp(scaler.inverse_transform(future_predictions.reshape(-1, 1))) - 1
    
    return {
        "original_data": df_ori.to_json(orient='records'),
        "original_data_test": df_ori_test.to_json(orient='records'),
        "df_regular_with_date": df_regular_with_date.to_json(orient='records'),
        "test_predictions": test_predict_original.flatten().tolist(),
        "future_predictions": future_predictions_original.flatten().tolist(),
        "unexpected_expenses": result_unexpected_expenses
    }


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    days_to_predict = int(request.form.get('days', 7))
    
    predictions = train_and_predict(file, days_to_predict)
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)