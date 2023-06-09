import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
# import tensorflow as tf
from psx import stocks, tickers


def get_training_data():
    fname = "live_stream_training_data.csv"
    try:
        # getting the live data from the stocks of the given banks till 28/02/2023
        data = stocks(["SILK", '''"UBL", "HBL", "AKBL"'''],
                      start=datetime.date(2015, 1, 1), end=datetime.date(2023, 2, 28))
        # removing the null data
        print('type(data) = ', type(data))
        data.dropna()
        # saving the data in .csv files
        data.to_csv(fname, mode='w')
        # print("data_frame describe = ", data_frame.describe)
    except:
        print("Unable to get the data atm!")


def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i+n_steps, :-1])
        y.append(data[i+n_steps-1, -1])
    return tf.convert_to_tensor(np.array(X).astype('float32')), tf.convert_to_tensor(np.array(y).astype('float32'))


def train_lstm():

    # preprocessing our data using scikit-learn
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import TimeSeriesSplit

    # building our LSTM models using Tensorflow Keras
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # from tensorflow.keras.layers import LSTM
    # from tensorflow.keras.layers import Dropout
    # from tensorflow.keras.layers import *
    # from tensorflow.keras.callbacks import EarlyStopping

    # Data Preprocessing
    # features
    # target_y = df['Close']
    # print('dataframe - \n', df)
    # print('df describe() = ', df.describe())
    df = pd.read_csv('live_stream_training_data.csv')
    # feature_X = df.iloc[:, 2:6]
    # sc = StandardScaler()
    # X_ft = sc.fit_transform(feature_X.values)
    # X_ft = pd.DataFrame(columns=feature_X.columns,
    #                     data=X_ft, index=feature_X.index)

    # # Train and Test Sets for Stock Price Prediction
    # X1, y1 = lstm_split(X_ft.values, n_steps=2)

    # train_split = 0.8
    # split_idx = int(np.ceil(len(X1)*train_split))
    # date_index = df.index

    # X_train, X_test = X1[:split_idx], X1[split_idx:]
    # y_train, y_test = y1[:split_idx], y1[split_idx:]
    # # x_train_date, x_test_date = date_index[:split_idx], date_index[split_idx:]

    # print(X1.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # # Building the LSTM model

    # lstm = Sequential()
    # lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]),
    #               activation='relu', return_sequences=True))
    # lstm.add(Dense(1))
    # lstm.compile(loss='mean_squared_error', optimizer='adam')

    # print('lstm_summary:\n', lstm.summary())

    # # fit model to training data
    # history = lstm.fit(X_train, y_train, epochs=10,
    #                    batch_size=4, verbose=2, shuffle=False)
    # print(history)
    # y_pred = lstm.predict(X_test)
    print('df columns = ', df.columns)
    df = df.drop(columns=['Ticker'])
    print('New df columns = ', df.columns)
    # df['Date'] = pd.to_datetime(df.Date)
    df.describe()
    df['Date'] = pd.to_datetime(df.Date)
    print(df.describe())
    x = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Linear regression Model for stock prediction
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.15, shuffle=False, random_state=0)

    # Check dim
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
    regression = LinearRegression()
    regression.fit(train_x, train_y)
    print("regression coefficient", regression.coef_)
    print("regression intercept", regression.intercept_)

    # the coefficient of determination R²
    regression_confidence = regression.score(test_x, test_y)
    print("linear regression confidence: ", regression_confidence)

    # Prediction
    predicted = regression.predict(test_x)
    print('Close Predictions:')
    print(predicted)
    print(predicted.shape)

    # print("Predicted Close Values:", y_pred)
    # print('X_test type and shape: ', type(X_test), X_test.shape)
    # print('y_pred type and shape: ', type(y_pred), y_pred.shape)

    # # form pkl
    import pickle
    # from keras.models import load_model
    # lstm.save('stock_close_model.h5')
    # # save the model to a .pkl file using pickle
    with open('stock_close_model_lr.pkl', 'wb') as f:
        pickle.dump(regression, f)


get_training_data()
train_lstm()
