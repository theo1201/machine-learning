import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
data_train = pd.read_csv("Train.csv")
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())

import  matplotlib.pyplot as plt
# fig = plt.figure()
# fig.set(alpha = 0.2)
#
# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not.
# plt.title(u"获救情况 (1为获救)") # puts a title on our graph
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # sets the y axis lable
# plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
# plt.title(u"按年龄看获救分布 (1为获救)")
#
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.
#
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()
#
# #看看各乘客等级的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
#
# plt.show()
#
# #看看各登录港口的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登录港口乘客的获救情况")
# plt.xlabel(u"登录港口")
# plt.ylabel(u"人数")
#
# plt.show()

# #看看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
# plt.show()
#
# #然后我们再来看看各种舱级别情况下各性别的获救情况
# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
# ax1.legend([u"女性/高级舱"], loc='best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"女性/低级舱"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/高级舱"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/低级舱"], loc='best')
#
# plt.show()
# # 那堂兄弟和父母呢？
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
#
# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
#
# #ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，不纳入考虑的特征范畴
# #cabin只有204个乘客有值，我们先看看它的一个分布
# data_train.Cabin.value_counts()
#
# #cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效
# #那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"按Cabin有无看获救情况")
# plt.xlabel(u"Cabin有无")
# plt.ylabel(u"人数")
# plt.show()

# 通常遇到缺值的情况，我们会有几种常见的处理方式
#
# 如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# 如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
# 如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
# 有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
# 本例中，后两种处理方式应该都是可行的，我们先试试拟合补全吧(虽然说没有特别多的背景可供我们拟合，这不一定是一个多么好的选择)

# 我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据
from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
	# 把已有的数值型特征取出来丢进Random Forest Regressor中
	age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

	# 乘客分成已知年龄和未知年龄两部分
	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()

	# y即目标年龄
	y = known_age[:, 0]

	# X即特征属性值
	X = known_age[:, 1:]

	# fit到RandomForestRegressor之中
	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	rfr.fit(X, y)

	# 用得到的模型进行未知年龄结果预测
	predictedAges = rfr.predict(unknown_age[:, 1::])

	# 用得到的预测结果填补原缺失数据
	df.loc[(df.Age.isnull()), 'Age'] = predictedAges

	return df, rfr

# cabin只有204个乘客有值,船仓位置的处理
def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
	df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
	return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
# print(data_train.head())
# 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化/one-hot编码。
# 以Embarked为例，原本一个属性维度，因为其取值可以是[‘S’,’C’,’Q‘]，而将其平展开为’Embarked_C’,’Embarked_S’, ‘Embarked_Q’三个属性
#
# 原本Embarked取值为S的，在此处的”Embarked_S”下取值为1，在’Embarked_C’, ‘Embarked_Q’下取值为0
# 原本Embarked取值为C的，在此处的”Embarked_C”下取值为1，在’Embarked_S’, ‘Embarked_Q’下取值为0
# 原本Embarked取值为Q的，在此处的”Embarked_Q”下取值为1，在’Embarked_C’, ‘Embarked_S’下取值为0
# 我们使用pandas的”get_dummies”来完成这个工作，并拼接在原来的”data_train”之上，如下所示。

# 因为逻辑回归建模时，需要输入的特征都是数值型特征
# 我们先对类目型的特征离散/因子化
# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

# 对于单独的one-hot，下面的方式
# Replace all the occurences of male with the number 0.
# data_train["Sex"].unique()先参看下一个取值，然后手动进行取值编码
# data_train.loc[titanic["Sex"] == "male", "Sex"] = 0
# data_train.loc[titanic["Sex"] == "female", "Sex"] = 1
# 添加属性
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df.head())

# 我们还得做一些处理，仔细看看Age和Fare两个属性，乘客的数值幅度变化，也忒大了吧！！如果大家了解逻辑回归与梯度下降的话，
# 会知道，各属性值之间scale差距太大，将对收敛速度造成几万点伤害值

# -------------------------对连续值特征进行标准化------------------------
# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
# 这样可以加速logistic regression的收敛
# 引入标准化模块
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
# fit(x,y)传两个参数的是有监督学习的算法，
# fit(x)传一个参数的是无监督学习的算法，比如降维、特征提取、标准化
# age = df['Age']
# age = np.array(age).reshape(1,-1)
# age_scaler_param=scaler.fit(df['Age'])
# transform函数是一定可以替换为fit_transform函数的
# fit_transform函数不能替换为transform函数！！！
# sklearn里的封装好的各种算法都要fit、然后调用各种API方法，transform只是其中一个API方法，
# 所以当你调用除transform之外的方法，必须要先fit，为了通用的写代码，还是分开写比较好
# 单变量特征时需要进行转换
# age_scaler_param=scaler.fit(df['Age'].reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1,1))
# fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1,1))

# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模。

# -----------------------向量化，转为numpy格式矩阵------------------------
from sklearn.linear_model import LogisticRegression
# from sklearn import cross_validation即将废弃
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 转成numpy格式
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
# penalty正则化选择参数，alty参数可选择的值为"l1"和"l2".分别对应L1的正则化和L2的正则化，默认是L2的正则化。
# 机器学习或者统计机器学习常见的损失函数如下：
#
# 1.0-1损失函数 （0-1 loss function）
# 2.平方损失函数（quadratic loss function)
# 3.绝对值损失函数(absolute loss function)
# 4.对数损失函数（logarithmic loss function) 或对数似然损失函数(log-likehood loss function)
# 逻辑回归中，采用的则是对数损失函数。如果损失函数越小，表示模型越好。
# C：float，默认值：1.0
# 正规化强度的反转; 必须是积极的浮动。与支持向量机一样，较小的值指定更强的正则化。
 #L1正则，惩罚因子1，最终误差在1e-6下
clf = LogisticRegression(penalty='l1', tol=1e-6)
# scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
clf.fit(X, y)

# 接下来咱们对训练集和测试集做一样的操作
data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
# 这里因为X是全局变量所以不能被改变
# X=null_age[:, 1:]
X_data = null_age[:, 1:]
predictedAges = rfr.predict(X_data)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape(-1,1))


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

# result.to_csv("logistic_regression_predictions.csv", index=False)
# pd.read_csv("logistic_regression_predictions.csv")

# 著名的learning curve可以帮我们判定我们的模型现在所处的状态。我们以样本数为横坐标，
# 训练和交叉验证集上的错误率作为纵坐标，两种状态分别如下两张图所示：
# 过拟合(overfitting/high variace)，欠拟合(underfitting/high bias)
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.learning_curve import learning_curve
from sklearn.model_selection import learning_curve
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
# train_sizes训练参数
# n_jobs：整数，可选# 并行运行的作业数（默认值为1）。
# cv：int，交叉验证生成器或可迭代的，可选的默认为3折交叉验证，
# estimator估计器，传入model
# verbose详细，控制详细程度：越高，消息越多，学习精度。
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
						train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
	"""
	画出data在某模型上的learning curve.
	参数解释
	----------
	estimator : 你用的分类器。
	title : 表格的标题。
	X : 输入的feature，numpy类型
	y : 输入的target vector
	ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
	cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
	n_jobs : 并行的的任务数(默认1)
	"""
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)

	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	if plot:
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel(u"训练样本数")
		plt.ylabel(u"得分")
		# PyPlot中反转Y轴目前，Y轴从0开始并达到最大值。我希望Y轴从最大值开始并上升到0。
		plt.gca().invert_yaxis()
		# 开启网格
		plt.grid()
		# 填充均值+-标准差上下
		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
						 alpha=0.1, color="b")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
						 alpha=0.1, color="r")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

		plt.legend(loc="best")

		plt.draw()
		plt.gca().invert_yaxis()
		plt.show()
	# 求平均水平
	midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
	# 测试数据和训练数据的预测差异
	diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
	return midpoint, diff


midpoint,diff = plot_learning_curve(clf, u"学习曲线", X, y)
print(midpoint,diff)
# 我们先看看那些权重绝对值非常大的feature，在我们的模型上：
print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))
# Sex属性，如果是female会极大提高最后获救的概率，而male会很大程度拉低这个概率。
# Pclass属性，1等舱乘客最后获救的概率会上升，而乘客等级为3会极大地拉低这个概率。
# 有Cabin值会很大程度拉升最后获救概率(这里似乎能看到了一点端倪，事实上从最上面的有无Cabin记录的Survived分布图上看出，即使有Cabin记录的乘客也有一部分遇难了，估计这个属性上我们挖掘还不够)
# Age是一个负相关，意味着在我们的模型里，年龄越小，越有获救的优先权(还得回原数据看看这个是否合理）
# 有一个登船港口S会很大程度拉低获救的概率，另外俩港口压根就没啥作用(这个实际上非常奇怪，因为我们从之前的统计图上并没有看到S港口的获救率非常低，所以也许可以考虑把登船港口这个feature去掉试试)。
# 船票Fare有小幅度的正相关(并不意味着这个feature作用不大，有可能是我们细化的程度还不够，举个例子，说不定我们得对它离散化，再分至各个乘客等级上？)

# 交叉验证
# from sklearn import cross_validation
# from sklearn.model_selection import cross_val_score
from  sklearn import  model_selection as ms
from  sklearn import linear_model
# 简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print(ms.cross_val_score(clf, X, y, cv=5))

# 分割数据
split_train, split_cv = ms.train_test_split(df, test_size=0.3, random_state=0)
# 处理训练数据中的训练部分
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])

# 处理训练数据中的测试部分
# 对cross validation数据进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# cv_df的size= 4020*1

predictions = clf.predict(cv_df.as_matrix()[:,1:])
# ValueError: Need to specify at least one of 'labels', 'index' or 'columns'
# split_cv[ predictions != cv_df.as_matrix()[:,0] ].drop()需要将这句分开写，不然会报错
index =split_cv[ predictions != cv_df.as_matrix()[:,0] ].index
split_cv.drop(index = index)
# 去除预测错误的case看原始dataframe数据
#split_cv['PredictResult'] = predictions
origin_data_train = pd.read_csv("Train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].
		isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
# Age属性不使用现在的拟合方式，而是根据名称中的『Mr』『Mrs』『Miss』等的平均值进行填充。
# Age不做成一个连续值属性，而是使用一个步长进行离散化，变成离散的类目feature。
# Cabin再细化一些，对于有记录的Cabin属性，我们将其分为前面的字母部分(我猜是位置和船层之类的信息) 和 后面的数字部分(应该是房间号，有意思的事情是，如果你仔细看看原始数据，你会发现，这个值大的情况下，似乎获救的可能性高一些)。
# Pclass和Sex俩太重要了，我们试着用它们去组出一个组合属性来试试，这也是另外一种程度的细化。
# 单加一个Child字段，Age<=12的，设为1，其余为0(你去看看数据，确实小盆友优先程度很高啊)
# 如果名字里面有『Mrs』，而Parch>1的，我们猜测她可能是一个母亲，应该获救的概率也会提高，因此可以多加一个Mother字段，此种情况下设为1，其余情况下设为0
# 登船港口可以考虑先去掉试试(Q和C本来就没权重，S有点诡异)
# 把堂兄弟/兄妹 和 Parch 还有自己 个数加在一起组一个Family_size字段(考虑到大家族可能对最后的结果有影响)
# Name是一个我们一直没有触碰的属性，我们可以做一些简单的处理，比如说男性中带某些字眼的(‘Capt’, ‘Don’, ‘Major’, ‘Sir’)可以统一到一个Title，女性也一样。
data_train[data_train['Name'].str.contains("Major")]
data_train = pd.read_csv("Train.csv")
data_train['Sex_Pclass'] = data_train.Sex + "_" + data_train.Pclass.map(str)

from sklearn.ensemble import RandomForestRegressor


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
	# 把已有的数值型特征取出来丢进Random Forest Regressor中
	age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

	# 乘客分成已知年龄和未知年龄两部分
	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()

	# y即目标年龄
	y = known_age[:, 0]

	# X即特征属性值
	X = known_age[:, 1:]

	# fit到RandomForestRegressor之中
	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	rfr.fit(X, y)

	# 用得到的模型进行未知年龄结果预测
	predictedAges = rfr.predict(unknown_age[:, 1::])

	# 用得到的预测结果填补原缺失数据
	df.loc[(df.Age.isnull()), 'Age'] = predictedAges

	return df, rfr


def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
	df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
	return df

# 处理新特征
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_train['Sex_Pclass'], prefix='Sex_Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)

# 标准化
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1,1))
# fare_scale_param = scaler.fit(df['Fare'])
# df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1,1))

from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
 # 处理预测数据
data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
data_test['Sex_Pclass'] = data_test.Sex + "_" + data_test.Pclass.map(str)
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_test['Sex_Pclass'], prefix= 'Sex_Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape(-1,1))

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions2.csv", index=False)
# 这个就是预测的结果
print(result)

# 一般做到后期，咱们要进行模型优化的方法就是模型融合啦
# 最简单的模型融合大概就是这么个意思，比如分类问题，当我们手头上有一堆在
# 同一份数据集上训练得到的分类器(比如logistic regression，SVM，KNN，random forest，神经网络)，
# 那我们让他们都分别去做判定，然后对结果做投票统计，取票数最多的结果为最后结果。
# 既然这个时候模型没得选，那咱们就在数据上动动手脚咯。大家想想，如果模型出现过拟合现在，
# 一定是在我们的训练上出现拟合过度造成的对吧。
# 那我们干脆就不要用全部的训练集，每次取训练集的一个subset，做训练，这样，
# 我们虽然用的是同一个机器学习算法，但是得到的模型却是不一样的；同时，因为我们没有任何一份子数据集是全的，因此即使出现过拟合，
# 也是在子训练集上出现过拟合，而不是全体数据上，这样做一个融合，可能对最后的结果有一定的帮助。对，这就是常用的Bagging。
from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]
# 训练集
# 数据抽取特征
# 建立模型
# 单独训练数据（训练集预测），综合评估
# 预测集，数据抽取特征，建立模型，预测数据
# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# class sklearn.ensemble.BaggingRegressor（base_estimator = None，n_estimators = 10，max_samples = 1.0，max_features = 1.0，bootstrap = True，
# bootstrap_features = False，oob_score = False，warm_start = False，n_jobs = 1，random_state = None，verbose = 0 ）
# bootstrap_features功能是否被替换
# n_estimators集合中的基本估计量的数量
bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
# as_matrix从版本0.23.0开始弃用：DataFrame.values()改为使用。
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
# result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions2.csv", index=False)
print(result)


