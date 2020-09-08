import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

# 1번


def one():
    x = np.array(list(range(1, 9)))
    y = np.array(list(range(1, 9)))

    x_test = np.array([9, 10])
    y_test = np.array([9, 10])

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    model.fit(x, y, epochs=1, batch_size=10, validation_split=0.2)

    _, acc = model.evaluate(x_test, y_test)
    print(acc)

 # 2번


def two():
    x = np.array(list(range(1, 9)))
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3])

    x_test = np.array([9, 10])
    y_test = np.array([4, 5])

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    model.fit(x, y, epochs=1, batch_size=10, validation_split=0.2)

    _, acc = model.evaluate(x_test, y_test)
    print(acc)


# 3번
def thr():
    x = np.array(list(range(1, 9)))
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    x_test = np.array([9, 10])
    y_test = np.array([1, 0])

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='sigmoid'))
    model.add(Dense(4))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])

    model.fit(x, y, epochs=1, batch_size=10, validation_split=0.2)

    loss, acc = model.evaluate(x_test, y_test)
    print(loss, acc)

# 4번


def four():
    x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9]])
    y = np.array([5, 6, 10])

    tf.compat.v1.reset_default_graph()

    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(20, input_shape=(4, 1)))
    model.add(Dense(7))
    model.add(Dense(4))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=1, batch_size=1)


one()
two()
thr()
four()
