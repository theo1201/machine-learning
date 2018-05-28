#三大件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = 'data' + os.sep + 'LogiReg_data.txt'
print(path)
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# pdData.head()

# The logistic regression

def sigmoid(z):
	return  1/(1+np.exp(-z))
def model(x,theta):
	return  sigmoid(np.dot(x,theta.T))

pdData.insert(0,'Ones',1)
orig_data = pdData.as_matrix()
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]

theta = np.zeros([1,3])

# D(hθ(x),y)=−ylog(hθ(x))−(1−y)log(1−hθ(x))
# 定义一个损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))

# 计算梯度

def gradient(X, y, theta):
	grad = np.zeros(theta.shape)
	error = (model(X, theta) - y).ravel()
	for j in range(len(theta.ravel())):  # for each parmeter
		term = np.multiply(error, X[:, j])
		grad[0, j] = np.sum(term) / len(X)

	return grad
# 定义三种梯度
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold

import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

import time

def descent(data, theta, batchSize, stopType, thresh, alpha):
	init_time = time.time()
	# 迭代次数
	i = 0
	# batch
	k = 0
	X,y = shuffleData(data=data)
	# 梯度
	grad = np.zeros(theta.shape)
	# 损失值
	costs = [cost(X,y,theta)]
	while True:
		grad = gradient(X[k:k+batchSize],y[k:k+batchSize],theta)
		k +=batchSize
		# 如果k>=n,表示一次循环结束
		if k>= n:
			k = 0
			X,y = shuffleData(data)
		# 参数更新
		theta = theta-alpha*grad
		# 计算新的损失
		costs.append(cost(X,y,theta))
		i+=1

		if stopType ==STOP_ITER: value = i
		elif stopType == STOP_COST: value = costs
		elif stopType == STOP_GRAD: value = grad
		if stopCriterion(stopType,value,thresh):break
	return  theta,i-1,costs,grad,time.time()-init_time

# 设置打印的信息
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


#选择的梯度下降方法是基于所有样本的
# 设定设定的批次数
n=100
# 根据迭代次数停止

# ***Original data - learning rate: 1e-06 - Gradient descent - Stop: 5000 iterations
# Theta: [[-0.00027127  0.00705232  0.00376711]] - Iter: 5000 - Last cost: 0.63 - Duration: 2.24s
# ***Original data - learning rate: 0.001 - Gradient descent - Stop: costs change < 1e-06
# Theta: [[-5.13364014  0.04771429  0.04072397]] - Iter: 109901 - Last cost: 0.38 - Duration: 50.10s
# ***Original data - learning rate: 0.001 - Gradient descent - Stop: gradient norm < 0.05
# Theta: [[-2.37033409  0.02721692  0.01899456]] - Iter: 40045 - Last cost: 0.49 - Duration: 19.12s

# runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
# # 根据损失值停止
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
# # 根据梯度变化停止
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

# 用sklearn计算
from  sklearn import  preprocessing as pp
scaled_data = orig_data.copy()
scaled_data[:,1:3] = pp.scale(orig_data[:,1:3])

# runExpe(scaled_data,theta,n,STOP_ITER,thresh=5000,alpha=0.001)

# runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

# theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
# 随机梯度下降更快，但是我们需要迭代的次数也需要更多，所以还是用batch的比较合适！！！
runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)


# 设定阀值
#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]
# 计算精度
scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
