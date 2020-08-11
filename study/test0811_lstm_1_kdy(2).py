from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def LSTM_input(x_data_num, time_step):
    a = np.empty((x_data_num, 1), dtype=float)
    dataset = np.array(range(1, x_data_num+10))
    for i in range(10):
        a = np.append(
            a, dataset[i:i+x_data_num].reshape(x_data_num, 1), axis=1)
    a = a[:, 1:]
    x = a[:, :time_step]
    y = a[:, time_step:]
    return x.reshape(x.shape[0], x.shape[1], 1), y.reshape(y.shape[0], y.shape[1], 1)


x, y = LSTM_input(91, 7)
print(x.shape, y.shape)
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2,   random_state=0)

model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(x.shape[1], 1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y,
          epochs=800,
          batch_size=1,
          validation_split=0.2,
          callbacks=[
              # verbose 1 언제 멈추는지 알려줌 /monitor 어느 성능을 기준으로
              # patience 성능 증가하지 않는 에폭을 얼마나 놔둘지
              EarlyStopping(monitor='val_loss', patience=100, verbose=1),
              ModelCheckpoint('./best_model.h5',
                              monitor='val_loss', save_best_only=True)
          ])

loss_met = model.evaluate(x_test, y_test, batch_size=1)
print('loss_met : ', loss_met)

x_pre = np.array([[i] for i in range(111, 118)])
x_pre = x_pre.reshape(1, x_pre.shape[0], 1)
y_real = np.array([118, 119, 120])

y_pre = model.predict(x_pre)


print(f'predict : {y_pre}')
print(f'real : {y_real}')
