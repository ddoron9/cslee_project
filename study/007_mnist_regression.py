#!/usr/bin/env python
# coding: utf-8

# In[72]:


#mnist 회귀
import tensorflow as tf
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow. keras.callbacks import EarlyStopping
 
ss =StandardScaler()
x_train = ss.fit_transform(x_train.reshape(-1,28*28))
x_test = ss.transform(x_test.reshape(-1,28*28))
pca = PCA(n_components=140)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test) 
print(sum(pca.explained_variance_ratio_))


# In[80]:


from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Model
inputs = Input(shape=(x_train.shape[1]))
d = Dense(256, activation='relu',kernel_initializer='he_normal')(inputs)
d = Dropout(0.4)(d)
d = Dense(160, activation='relu',kernel_initializer='he_normal')(d)
d = Dropout(0.4)(d)
d = Dense(100, activation='relu',kernel_initializer='he_normal')(d)
d = Dropout(0.4)(d)
d = Dense(40, activation='relu',kernel_initializer='he_normal')(d)
d = Dropout(0.2)(d)
d = Dense(10, activation='relu',kernel_initializer='he_normal')(d)
d = Dropout(0.3)(d)
d = Dense(1)(d)

model = Model(inputs=inputs,outputs=d)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train,batch_size=64,epochs=400, validation_split=0.2, 
          callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)])
 


# In[81]:


pred = list(map(int,model.predict(x_test).reshape(-1).round(0))) 


# In[82]:



from sklearn.metrics import mean_squared_error ,r2_score
MSE = mean_squared_error(list(y_test), pred )
r2 = r2_score(list(y_test), pred )


# In[83]:


print(f'mse = {MSE} \nr2 = {r2}')


# In[84]:


cnt = 0
for i in range(len(y_test)):
    if y_test[i] == pred[i]:
        cnt+=1
print(f'test accuracy : {cnt/len(y_test)*100}')


# In[ ]:




