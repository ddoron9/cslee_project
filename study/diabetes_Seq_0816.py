
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
    model = Sequential()

    model.add(Dense(150, input_dim=10, activation='elu'))
    model.add(Dense(130, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='elu'))
    model.add(Dense(23, activation='elu'))
    model.add(Dense(14, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='elu'))
    model.add(Dense(1))

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

    model = Sequential()
    model.add(Conv2D(50, (5, 2), input_shape=(5, 2, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(30, activation='elu'))
    model.add(Dense(15, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='elu'))
    model.add(Dense(1))

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

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(55, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(30))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30, epochs=150)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


diabetes_LSTM(x_train, y_train, x_test, y_test)
