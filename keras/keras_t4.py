
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
np.random.seed(1337)  # for reproducibility

# 按顺序建立model
from keras.models import Sequential
# dense表示全连接层
from keras.layers import Dense

import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()


X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
# 建立模型
model = Sequential()


model.add(Dense(units=1, input_dim=1))

# choose loss function and optimizing method
# sgd表示随机梯度下降法
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    # 使用已一批批数据
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
# 获得学习的到的参数
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()