
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def mnist_DNN(x_train, y_train, x_test, y_test):
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = Sequential()

    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(30, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
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
    model = Sequential()
    model.add(Conv2D(50, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(30, activation='elu'))
    model.add(Dense(15, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
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
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(784, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(55, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(30))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(7, activation='elu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32,
              epochs=15,   validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


mnist_LSTM(x_train, y_train, x_test, y_test)
