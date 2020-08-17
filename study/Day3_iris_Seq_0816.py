from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.model_selection import train_test_split

# 클래스 객체
data = load_iris()
print(data.keys())
# data 각행은 꽃의 feature (150,4)
# 그와 일치하는 타겟의 행은 꽃 label을 알려줌 (150,)
#target_names : 라벨
# feature_names : 데이터의 특성이 의미하는 것
# filename : 데이터셋이 저장된 위치
# 데이터 정보 알려줌
# print(data.DESCR)
# stratify=target 클래스 비율 유지
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=34)


def iris_DNN(x_train, y_train, x_test, y_test):

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model = Sequential()

    model.add(Dense(20, input_dim=4, activation='elu'))
    model.add(Dense(26, activation='elu'))
    model.add(Dense(30, activation='elu'))
    model.add(Dense(23, activation='elu'))
    model.add(Dense(14, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='elu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30,
              epochs=150, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


iris_DNN(x_train, y_train, x_test, y_test)


def iris_CNN(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], 2, 2, 1)
    x_test = x_test.reshape(x_test.shape[0], 2, 2, 1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model = Sequential()
    model.add(Conv2D(50, (2, 2), input_shape=(2, 2, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(30, activation='elu'))
    model.add(Dense(15, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
              epochs=100,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


iris_CNN(x_train, y_train, x_test, y_test)


def iris_LSTM(x_train, y_train, x_test, y_test):

    model = Sequential()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.add(LSTM(50, return_sequences=True, input_shape=(4, 1)))

    model.add(Dropout(0.2))
    model.add(LSTM(55, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(30))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='elu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30,
              epochs=150, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


iris_LSTM(x_train, y_train, x_test, y_test)
