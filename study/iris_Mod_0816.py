from sklearn.datasets import load_iris
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, LSTM
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

    inputs = Input(shape=(4,))
    x = Dense(20, activation='elu')(inputs)
    x = Dense(30, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(26, activation='elu')(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(23, activation='elu')(x)
    x = Dense(14, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation='elu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs, x)
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
    inputs = Input(shape=(2, 2, 1))
    x = Conv2D(50, (2, 2), activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(15, activation='elu')(x)
    x = Dense(5, activation='elu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
              epochs=100,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


iris_CNN(x_train, y_train, x_test, y_test)


def iris_LSTM(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(4, 1))
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(55, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(30)(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='elu')(x)
    x = Dense(7, activation='elu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30,
              epochs=150, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


iris_LSTM(x_train, y_train, x_test, y_test)
