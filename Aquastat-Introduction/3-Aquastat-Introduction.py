# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium

# system packages
import os, sys
import warnings
warnings.filterwarnings('ignore')

# basic wrangling
import numpy as np
import pandas as pd

# eda tools
import pivottablejs
import missingno as msno
import pandas_profiling

# interactive
import ipywidgets as widgets

# more technical eda
import sklearn
import scipy

from scripts.aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice

mpl_update = {'font.size':16,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'figure.figsize':[12.0,8.0],
              # 'axes.color_cycle':['#0055A7', '#2C3E4F', '#26C5ED', '#00cc66', '#D34100', '#FF9700','#091D32'],
              'axes.labelsize':16,
              'axes.labelcolor':'#677385',
              'axes.titlesize':20,
              'lines.color':'#0055A7',
              'lines.linewidth':3,
              'text.color':'#677385'}
mpl.rcParams.update(mpl_update)

data = pd.read_csv('aquastat.csv.gzip', compression='gzip')

# simplify regions
data.region = data.region.apply(lambda x: simple_regions[x])

# remove exploitable fields and national rainfall index
data = data.loc[~data.variable.str.contains('exploitable'),:]
data = data.loc[~(data.variable=='national_rainfall_index')]

# 观察数据
recent = time_slice(data, '2013-2017')
recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int)
recent.sort_values('rural_pop')[['total_pop','urban_pop','rural_pop']].head()
# Rural population = Total population - urban population
time_series(data, 'Qatar', 'total_pop').join(time_series(data, 'Qatar', 'urban_pop')).\
    join(time_series(data, 'Qatar', 'rural_pop'))

# Shape of the data
# 数据分布是倾斜的吗？
# 有异常值吗？它们可行吗？
# 有不连续的吗?
recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int)
# 让我们尝试计算偏度和峰度和绘制直方图显示。
# 左偏：均值<中位数
# Scipy的stats模块包含了多种概率分布的随机变量，随机变量分为连续的和离散的两种。
# 查看偏度，正态分布的偏度应为零。负偏度表示偏左，正偏表示右偏。
recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.skew)
# 查看峰度，峰度也是一个正态分布和零只能是积极的。我们肯定有一些异常值！
recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.kurtosis)

fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(recent.total_pop.values, bins=50);
ax.set_xlabel('Total population');
ax.set_ylabel('Number of countries');
ax.set_title('Distribution of population of countries 2013-2017');

# 问题，数据偏度大，峰值高，不符合正太分布，为了让预测更加准确，需要先对数据进行变换
# 解决方法是什么？通常，使用LOG转换将使变量更为正常。
# 对数变换是数据变换的一种常用方式，数据变换的目的在于使数据的呈现方式接近我们所希望的前提假设，从而更好的进行统计推断。
# 另外一个列子
# 左边是正常数据，可以看到随着时间推进，电力生产也变得方差越来越大，即越来越不稳定。 这种情况下常有的分析假设经常就不会满足
# （误差服从独立同分布的正态分布，时间序列要求平稳）。
# 理论上，我们将这类问题抽象成这种模型，即分布的标准差与其均值线性相
# 先做log变换，然后观察偏度
recent[['total_pop']].apply(np.log).apply(scipy.stats.skew)
# 先做log变换，然后观察峰度
recent[['total_pop']].apply(np.log).apply(scipy.stats.kurtosis)
# 只能是减少，但是并没有摆脱峰度
# 画图展示
def plot_hist(df, variable, bins=20, xlabel=None, by=None,
              ylabel=None, title=None, logx=False, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
    if logx:
        if df[variable].min() <= 0:
            df[variable] = df[variable] - df[variable].min() + 1
            print('Warning: data <=0 exists, data transformed by %0.2g before plotting' % (- df[variable].min() + 1))

        bins = np.logspace(np.log10(df[variable].min()),
                           np.log10(df[variable].max()), bins)
        ax.set_xscale("log")

    ax.hist(df[variable].dropna().values, bins=bins);

    if xlabel:
        ax.set_xlabel(xlabel);
    if ylabel:
        ax.set_ylabel(ylabel);
    if title:
        ax.set_title(title);

    return ax
plot_hist(recent, 'total_pop', bins=25, logx=True,
          xlabel='Log of total population', ylabel='Number of countries',
          title='Distribution of total population of countries 2013-2017')
plt.show()

# Normalization 正太分布
# 计算人口密度
recent['population_density'] = recent.total_pop.divide(recent.total_area)
# 绘制人口变化曲线
plt.plot(time_series(data, 'United States of America', 'total_pop'));
plt.xlabel('Year');
plt.ylabel('Population');
plt.title('United States population over time');
# One region一个地区，很多国家的人口变化
with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    north_america = time_slice(subregion(data, 'North America'), '1958-1962').sort_values('total_pop').index.tolist()
    for country in north_america:
        plt.plot(time_series(data, country, 'total_pop'), label=country);
        plt.xlabel('Year');
        plt.ylabel('Population');
        plt.title('North American populations over time');
    plt.legend(loc=2,prop={'size':10});
# 这除了北美洲是最大的国家之外，什么也没有告诉我们。我们想了解每个国家的人口是如何随着时间的推移而变化的，
# 主要是参照自身的变化。
# 我们应该通过什么标准化？我们可以选择一个国家的最小、平均、中位数、最大值…或任何其他位置
# 让我们选择最小值，这样我们就能看到每个国家在起始人口上的增长。
# 这个图显示的要比上图更加明确，因为取最小值可以看清变化
with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    for country in north_america:
        ts = time_series(data, country, 'total_pop')
        ts['norm_pop'] = ts.total_pop/ts.total_pop.min()*100
        plt.plot(ts['norm_pop'], label=country);
        plt.xlabel('Year');
        plt.ylabel('Percent increase in population');
        plt.title('Percent increase in population from 1960 in North American countries');
    plt.legend(loc=2,prop={'size':10});

north_america_pop = variable_slice(subregion(data, 'North America'), 'total_pop')
north_america_norm_pop = north_america_pop.div(north_america_pop.min(axis=1), axis=0)*100
north_america_norm_pop = north_america_norm_pop.loc[north_america]
# 绘制热力图，来反映人口变化的快慢程度
fig, ax = plt.subplots(figsize=(16, 12));
sns.heatmap(north_america_norm_pop, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True));
plt.xticks(rotation=45);
plt.xlabel('Time period');
plt.ylabel('Country, ordered by population in 1960 (<- greatest to least ->)');
plt.title('Percent increase in population from 1960');
plt.show()

# Exploring total renewable water resources
# 水资源的数据探索参考人口变化
plot_hist(recent, 'total_renewable', bins=50,
          xlabel='Total renewable water resources ($10^9 m^3/yr$)',
          ylabel='Number of countries',
          title='Distribution of total renewable water resources, 2013-2017');
# logx=True,就是log变换
plot_hist(recent, 'total_renewable', bins=50,
          xlabel='Total renewable water resources ($10^9 m^3/yr$)',
          ylabel='Number of countries', logx=True,
          title='Distribution of total renewable water resources, 2013-2017');

north_america_renew = variable_slice(subregion(data, 'North America'), 'total_renewable')
# 绘制热力图，来反映水资源变化的快慢程度
fig, ax = plt.subplots(figsize=(16, 12));
sns.heatmap(north_america_renew, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True));
plt.xticks(rotation=45);
plt.xlabel('Time period');
plt.ylabel('Country, ordered by Total renewable water resources in 1960 (<- greatest to least ->)');
plt.title('Total renewable water resources increase in population from 1960');

# 多变量对比图
# Assessing many variables¶
def two_hist(df, variable, bins=50,
              ylabel='Number of countries', title=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,8))
    ax1 = plot_hist(df, variable, bins=bins,
                    xlabel=variable, ylabel=ylabel,
                    ax=ax1, title=variable if not title else title)
    ax2 = plot_hist(df, variable, bins=bins,
                    xlabel='Log of '+ variable, ylabel=ylabel,
                    logx=True, ax=ax2,
                    title='Log of '+ variable if not title else title)
    plt.close()
    return fig


def hist_over_var(df, variables, bins=50,
                  ylabel='Number of countries', title=None):
    variable_slider = widgets.Dropdown(options=variables.tolist(),
                                       value=variables[0],
                                       description='Variable:',
                                       disabled=False,
                                       button_style='')
    widgets.interact(two_hist, df=widgets.fixed(df),
                     variable=variable_slider, ylabel=widgets.fixed(ylabel),
                     title=widgets.fixed(title), bins=widgets.fixed(bins))

hist_over_var(recent, recent.columns, bins=20)
plt.show()
