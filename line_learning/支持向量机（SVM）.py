import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()
#随机来点数据
# scikit中的make_blobs方法常被用来生成聚类算法的测试数据，直观地说，
# make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
from sklearn.datasets.samples_generator import make_blobs
# sklearn.datasets.make_blobs
# (n_samples=100, n_features=2,centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)[source]
# n_samples是待生成的样本的总数。
# n_features是每个样本的特征数。
# centers表示类别数。
# cluster_std表示每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
# x为数据集，y为标签集
# c为标签c=y表示一共三种颜色标签
# s为点的大小
# cmap 为颜色分类，autumn自动
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)

# 训练一个SVM
from sklearn.svm import SVC
# 建立一个model
model= SVC(kernel="linear")
model.fit(X,y)

# 绘制函数
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x)
    # 首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()
    # 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()
    # 返回的是视图（view，也颇有几分C / C + +引用reference的意味），会影响（reflects）原始矩阵。
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    # 因为线性核函数最终预测的法平面为y = linear.coef_ * X + linear.intercept_
    # linear.coef_ * X + linear.intercept_ | / | | linear.coef_ | |, 决策函数并没有采用距离计算，而是，直接:
	#
    # decision_function = linear.coef_ * X + linear.intercept_，其若大于0则label预测为1，否则预测为0。
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)

print(model.support_vectors_)
# 接下来我们尝试一下，用不同多的数据点，看看效果会不会发生变化
#
# 分别使用60个和120个数据点

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]

    # l
    # C：C - SVC的惩罚参数C?默认值是1
    # .0
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，
    # 但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
    # sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
	#
    #                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
    #                 decision_function_shape=None, random_state=None)
    # https: // blog.csdn.net / szlcw1 / article / details / 52336824
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    # 获取当前的句柄
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# 调整子区域布局。
# subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=None, hspace=None)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))

from sklearn.datasets.samples_generator import make_circles
# sklearn.datasets.make_circles（n_samples = 100，shuffle = True，noise = None，random_state = None，factor = 0.8 ）[source]
# factor：double <1（默认值= .8）
#
# 内圈和外圈之间的比例因子。noise为噪声
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)


# 加入高维和变换
#加入了新的维度r
from mpl_toolkits import mplot3d
r = np.exp(-(X ** 2).sum(1))
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

plot_3D(elev=45, azim=45, X=X, y=y)

#加入径向基函数
# RBF Network 通常只有三层。输入层、中间层计算输入 x 矢量
# 与样本矢量 c 欧式距离的 Radial Basis Function (RBF) 的值，输出层算它们的线性组合。
# 间层采用 RBF Kernel 对输入作非线性变换，以便输出层训练线性分类器。
clf = SVC(kernel='rbf', C=1E6)

clf.fit(X, y)

# 绘图
#这回牛逼了！
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');

# X, y = make_blobs(n_samples=100, centers=2,
#                   random_state=0, cluster_std=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
# 调节C参数
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)


X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.1)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)


for axi, gamma in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='rbf', gamma=gamma).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('gamma = {0:.1f}'.format(gamma), size=14)


plt.show()