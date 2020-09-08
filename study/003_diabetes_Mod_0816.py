
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 당뇨병 데이터셋 가져오기
x, y = load_diabetes(True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))


def diabetes_DNN(x_train, y_train, x_test, y_test):

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(10,))
    x = Dense(150, activation='elu')(inputs)
    x = Dense(130, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(23, activation='elu')(x)
    x = Dense(14, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation='elu')(x)
    x = Dense(1)(x)

    model = Model(inputs, x)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30, epochs=150)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


diabetes_DNN(x_train, y_train, x_test, y_test)


def diabetes_CNN(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], 5, 2, 1)
    x_test = x_test.reshape(x_test.shape[0], 5, 2, 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(5, 2, 1))
    x = Conv2D(50, (5, 2), input_shape=(5, 2, 1), activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(30, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(15, activation='elu')(x)
    x = Dense(5, activation='elu')(x)
    x = Dense(1)(x)

    model = Model(inputs, x)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30, epochs=150)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


diabetes_CNN(x_train, y_train, x_test, y_test)


def diabetes_LSTM(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(10, 1))
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(55, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(30)(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='elu')(x)
    x = Dense(7, activation='elu')(x)
    x = Dense(1)(x)

    model = Model(inputs, x)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30, epochs=150)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


diabetes_LSTM(x_train, y_train, x_test, y_test)
