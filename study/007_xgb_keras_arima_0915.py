import pandas as pd
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Flatten, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random
from sklearn.decomposition import PCA
import tensorflow as tf
# seed
RS = 20200916
random.seed(RS)
np.random.seed(RS)


# 전처리 클래스
class Data_Pre():
    def __init__(self, days):

        # 삼성 주가
        df = pd.read_csv('./samsung.csv', thousands=',', index_col='일자')
        # df.head()
        df = df.rename(columns={'시가': 'samsung market', '고가': 'samsung high', '저가': 'samsung low', '종가': 'samsung end',
                                '거래량': 'samsung volumn', '등락률': 'samsung fluct',
                                '외국계': 'samsung foreign', '외인비': 'samsung for rate'})

        # 결측치 제거 / 시간 역순 배열
        df = df.dropna().loc[::-1]

        # TARGET은 종가
        self.end = df['samsung end']

        # 금 시세
        gold = pd.read_csv('./gold.csv', thousands=',', index_col='일자')
        gold = gold.rename(columns={'시가': 'gold market', '고가': 'gold high', '저가': 'gold low', '종가': 'gold end', '거래량': 'gold volumn', '등락률': 'gold fluct',
                                    '외국계': 'gold foreign', '외인비': 'gold for rate'})

        gold = gold.loc[:'2018-05-04'] . loc[::-1]
        # 코스닥
        kosdaq = pd.read_csv('./kosdaq150.csv', thousands=',', index_col='일자')

        kosdaq = kosdaq.rename(columns={'시가': 'kosdaq market', '고가': 'kosdaq high', '저가': 'kosdaq low', '종가': 'kosdaq end',
                                        '등락률': 'kosdaq fluct', '거래대금': 'kosdaq volumn'})
        kosdaq = kosdaq.dropna().loc[::-1]
        # 코스피지수
        kospi = pd.read_csv('./kospi200.csv', thousands=',', index_col='일자')

        kospi = kospi.rename(columns={'시가': 'kospi market', '고가': 'kospi high', '저가': 'kospi low', '종가': 'kospi end',
                                      '거래대금': 'kospi volumn'})
        kospi = kospi.dropna().loc[::-1]

        # 하나로 합침
        df = pd.concat([df, gold, kosdaq, kospi], axis=1)

        # x
        self.df = df

        #(n, days)
        label = np.empty((1, days))

        for i in range(len(self.end)-days):

            label = np.append(label, np.array(
                list(self.end.iloc[i+1:i+days+1])).reshape(1, days), axis=0)

        # y_train
        self.y = label[1:, :]

    # 종가 하나로만 예측
    def arima(self):
        return self.end

    def pca(self):
        pca = PCA(n_components=0.999999999)
        df_new = pca.fit_transform(self.df)
        return df_new, self.y

    # feature selection 할 것
    def feature(self):
        x_train = np.array(self.df)
        model = XGBRegressor().fit(x_train[:-1, :], self.df['samsung end'][1:])
        importance = model.feature_importances_
        # plt.bar([x for x in range(len(importance))], importance)
        # plt.show()

        # 내림차순으로 importance column index
        ind = np.argsort(importance)[::-1]
        sum = 0
        lst = []

        # 98.5 % 까지의 누적 importance 가지는 feature 만 남길 것
        for i in ind:
            lst.append(self.df.columns[i])
            sum += importance[i]
            if sum > 0.985:
                break

        x = self.df.loc[:, lst]
        return np.array(x), self.y


class Modeling():

    def __init__(self, days):

        # window size
        self.feat = 20

        d = Data_Pre(days)

        # arima x 값 (samsung end)
        self.ari = d.arima()

        x_train, y_train = d.pca()
        # x_train, y_train = d.feature()

        # predict할 날짜의 실제 종가
        self.r = x_train[-(days):, 0]

        x_pre = x_train[-self.feat:, :]

        self.x = x_train[:-days, :]
        self.y = y_train
        # print(x_train.shape,y_train.shape)

        self.pre = x_pre
        self.d = days

    def xgbr(self):
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(XGBRegressor(
            objective='reg:squarederror', booster='gblinear')).fit(self.x, self.y)
        # 그리드 서치용
        hyperparameter_grid = [
            {
                'estimator__n_estimators': [5430, 5400],
                'estimator__eta': [0.229, 0.228],
                'estimator__lambda': [0.001, 0.01, 0.02, 0.03]
            }
        ]
        # grid = GridSearchCV(model,hyperparameter_grid,scoring='neg_root_mean_squared_error' ,verbose=2, cv=3,n_jobs=5)
        # grid.fit(x_train,self.y)

        # print(f'best params : {grid.best_params_ }')

        random_grid = [
            {
                'estimator__n_estimators': [5430, 5400],
                'estimator__eta': [0.229, 0.228],
                'estimator__lambda': [0.001, 0.01, 0.02, 0.03]
            }
        ]
        rand = RandomizedSearchCV(
            model, random_grid, scoring='neg_root_mean_squared_error', n_iter=150, verbose=2, cv=3, n_jobs=-1)
        rand.fit(self.x, self.y)
        predict = rand.best_estimator_.predict(self.pre[-self.d:, :])[-1]

        print('xgb ', f'{self.d} 일 간의 데이터 예측값 : {predict}')
        return list(map(int, predict))

    def keras(self):
        def split():
            x = np.empty((1, self.x.shape[1]))
            for i in range(len(self.x)-self.feat+1):
                for j in range(self.feat):
                    x = np.append(x, np.array(
                        self.x[i+j, :]).reshape(-1, self.x.shape[1]), axis=0)
            return x[1:, :].reshape(-1, self.feat, self.x.shape[1]), self.y[(self.feat-1):]

        x, y = split()

        scaler = MinMaxScaler()
        x = scaler.fit_transform(
            x.reshape(-1, x.shape[1]*x.shape[2])) .reshape(-1, self.feat, x.shape[2])
        y = scaler.fit_transform(y)

        from tensorflow.keras import backend as K

        K.clear_session()
        from tensorflow.keras.regularizers import l2

        batch_size = 16
        inputs = Input(shape=(x.shape[1], x.shape[2]))
        z = LSTM(self.feat, return_sequences=True,
                 kernel_initializer='random_uniform')(inputs)
        z = Dropout(0.4)(z)
        z = LSTM(30)(z)
        z = Dropout(0.4)(z)
        z = Flatten()(z)
        z = Dense(64, activation='relu')(z)
        z = Dropout(0.2)(z)
        z = Dense(32, activation='selu')(z)
        z = Dropout(0.2)(z)
        z = Dense(16, activation='selu')(z)
        z = Dense(8, activation='selu')(z)
        z = Dense(self.d)(z)

        model = Model(inputs=inputs, outputs=z)

        model.compile(loss='mse',  optimizer='adam')
        model.fit(x, y, batch_size=batch_size,  shuffle=False, epochs=400)
        predict = scaler.inverse_transform(model.predict(
            self.pre.reshape(1, self.pre.shape[0], self.pre.shape[1])))

        print('keras ', f'{self.d} 일 간의 데이터 예측값 : {predict}')
        return predict.reshape(-1)

    def arima(self):
        from statsmodels.tsa.arima_model import ARIMA

        # 종가만 이용
        model = ARIMA(self.ari, order=(1, 1, 1))

        model_fit = model.fit(trend='nc', full_output=True, disp=1)
        print(model_fit.summary())
        forecast_data = model_fit.forecast(steps=self.d)
        # 예측값, stderr, upper bound, lower bound
        print('arima ', f'{self.d} 일 간의 데이터 예측값 : {forecast_data[0]}')
        return forecast_data[0]


col = ['9/11', '9/14', '9/15', '9/16']
days = 2
m = Modeling(days)
data = pd.DataFrame(columns=col[-days:])
data.loc['lstm'] = m.keras()
data.loc['xgb'] = m.xgbr()
data.loc['arima'] = list(m.arima())
data.round(-1).to_csv(f'./samsung_output{RS}.csv', sep=',')
