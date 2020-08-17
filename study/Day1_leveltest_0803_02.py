import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np

# 1번


def one():
    x = np.array(list(range(1, 11)))
    y = np.array(list(range(1, 11)))
    x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(1, input_dim=1))

    model.compile(optimizer='SGD', loss='mse')

    model.fit(x_tr, y_tr, batch_size=10, epochs=35)

    print(y_test, model.predict(x_test))

# 2번


def two():
    x = np.array(list(range(1, 11)))
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    y_hot = to_categorical(y)

    x_tr, x_test, y_tr, y_test = train_test_split(x, y_hot, test_size=0.2)

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    model.fit(x_tr, y_tr, epochs=200, batch_size=10, validation_split=0.2)
    pre = model.predict(x_test)
    print(pre)
    print(f'predict : {tf.argmax(pre,1)} real : {tf.argmax(y_test,1)}')


# 3번
def thr():
    x = np.array(list(range(1, 11)))
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])

    model.fit(x_tr, y_tr, epochs=300, batch_size=10, validation_split=0.2)
    print(f'predict : {model.predict(x_test).flatten()} real : {y_test}')

# 4번


def four():
    x = [[i, i + 1, i + 2, i + 3] for i in range(1, 7)]
    y = np.array([5, 6,7,8,9, 10])

    tf.compat.v1.reset_default_graph()

    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(20, input_shape=(4, 1)))
    model.add(Dense(7))
    model.add(Dense(4))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=1, batch_size=1)
    print(f'predict : {model.predict(x)} real : {y}')


one()
two()
thr()
four()
