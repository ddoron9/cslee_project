from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def LSTM_input_mtm(x_data_num, time_step, feature):
    a = np.empty((1, time_step), dtype=float)
    a.shape
    dataset = np.array(range(1, x_data_num+9))
    for i in range(0, x_data_num):
        for j in range(3):
            a = np.append(a, dataset[np.newaxis, i+j:i+j+time_step], axis=0)
    a = np.append(a, (a[:, -1]+feature).reshape(len(a), 1), axis=1)
    a = a[1:, :]
    return a[:, :time_step].reshape(x_data_num, time_step, feature), a[:, -1].reshape(x_data_num, 1, feature)


x, y = LSTM_input_mtm(109, 7, 3)
x_pre = x[-1].reshape(1, x[-1].shape[0], x[-1].shape[1])
y_real = y[-1].reshape(1, y[-1].shape[0], y[-1].shape[1])

x, x_test, y, y_test = train_test_split(
    x[:-1], y[:-1], test_size=0.2,   random_state=0)

model = Sequential()
# return_sequences = true 모든 입력에 대해 출력을 내놓음
model.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(7, 3)))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=False))
model.add(Dense(14, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=500, batch_size=1,
          validation_split=0.2,
          callbacks=[
              EarlyStopping(monitor='val_loss', patience=150, verbose=1),
              ModelCheckpoint('./best_model_mtm.h5',
                              monitor='val_loss', save_best_only=True)
          ])

loss_met = model.evaluate(x_test, y_test, batch_size=1)
print('loss_met : ', loss_met)

y_pre = model.predict(x_pre)

print(f'predict : {y_pre}')
print(f'real : {y_real}')
