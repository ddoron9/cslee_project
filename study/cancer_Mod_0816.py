from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()
cancer.keys()

# 라벨 0 악성 / 1 양성
print(cancer.target_names)

# data column
print(cancer.feature_names)


x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, stratify=cancer.target, random_state=34)


def cancer_DNN(x_train, y_train, x_test, y_test):

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(30,))
    x = Dense(20, activation='elu')(inputs)
    x = Dense(30, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(26, activation='elu')(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(23, activation='elu')(x)
    x = Dense(14, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation='elu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30,
              epochs=150, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


cancer_DNN(x_train, y_train, x_test, y_test)


def cancer_CNN(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], 6, 5, 1)
    x_test = x_test.reshape(x_test.shape[0], 6, 5, 1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    inputs = Input(shape=(6, 5, 1))
    x = Conv2D(50, (6, 5), activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(15, activation='elu')(x)
    x = Dense(5, activation='elu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
              epochs=100,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


cancer_CNN(x_train, y_train, x_test, y_test)


def cancer_LSTM(x_train, y_train, x_test, y_test):

    model = Sequential()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(30, 1))
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(55, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(30)(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='elu')(x)
    x = Dense(7, activation='elu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=30,
              epochs=150, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


cancer_LSTM(x_train, y_train, x_test, y_test)
