
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, LSTM
from sklearn.model_selection import train_test_split
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def mnist_DNN(x_train, y_train, x_test, y_test):
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    inputs = Input(shape=(784,))
    x = Dense(20, activation='elu')(inputs)
    x = Dense(30, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(15, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(5, activation='elu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs, x)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=32,
              epochs=50,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


mnist_DNN(x_train, y_train, x_test, y_test)


def mnist_CNN(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(50, (3, 3))(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(50, (3, 3))(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(30, activation='elu')(x)
    x = Dense(15, activation='elu')(x)
    x = Dense(5, activation='elu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs, x)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=32,
              epochs=50,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


mnist_CNN(x_train, y_train, x_test, y_test)


def mnist_LSTM(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1]*x_train.shape[2], 1)
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1]*x_test.shape[2], 1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    inputs = Input(shape=(784, 1))
    x = LSTM(50, return_sequences=True, input_shape=(784, 1))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(55, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(30)(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='elu')(x)
    x = Dense(7, activation='elu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=500,
              epochs=15,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


mnist_LSTM(x_train, y_train, x_test, y_test)
