import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
import warnings
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

data = pd.read_csv('aquastat.csv.gzip', compression='gzip')
# print(data.head())
# Research the variables
print(data[['variable','variable_full']].drop_duplicates())
# 观察数据
# Describe the panel
data.country.nunique()
countries = data.country.unique()
data.time_period.nunique()

time_periods = data.time_period.unique()
mid_periods = range(1960,2017,5)
# Dataset is unbalanced because there is not data for
# 	every country at every time period (more on missing data in the next notebook).
data[data.variable=='total_area'].value.isnull().sum()

# 探索的维度
# 横截面：一个时期内所有国家
# 时间序列：一个国家随着时间的推移
# 面板数据：所有国家随着时间的推移（作为数据给出）
# 地理空间：所有地理上相互联系的国家

# 时间维度的切片
def time_slice(df, time_period):
	# Only take data for time period of interest
	df = df[df.time_period == time_period]

	# Pivot table 绘制图表
	df = df.pivot(index='country', columns='variable', values='value')

	df.columns.name = time_period

	return df
print(time_slice(data, time_periods[0]).head())

# 对于给定的一个国家
def country_slice(df, country):
	# Only take data for country of interest
	df = df[df.country == country]

	# Pivot table
	df = df.pivot(index='variable', columns='time_period', values='value')

	df.index.name = country
	return df
print(country_slice(data, countries[40]).head())

# 对于变化的因素
# By variable
def variable_slice(df, variable):
	# Only data for that variable
	df = df[df.variable == variable]

	# Get variable for each country over the time periods
	df = df.pivot(index='country', columns='time_period', values='value')
	return df
print(variable_slice(data, 'total_pop').head())

# Time series for given country and variable
# 对于给定国家和变量在时间系列的变化
def time_series(df, country, variable):
	# Only take data for country/variable combo
	series = df[(df.country == country) & (df.variable == variable)]

	# Drop years with no data
	series = series.dropna()[['year_measured', 'value']]

	# Change years to int and set as index
	series.year_measured = series.year_measured.astype(int)
	series.set_index('year_measured', inplace=True)
	series.columns = [variable]
	return series
print(time_series(data, 'Belarus', 'total_pop'))


# 我们可能需要查看某些评估数据的子集。区域是一种直观的数据细分方式。
print(data.region.unique())
# 创建一个字典来查找新的、更简单的区域（亚洲、北美洲、南美洲、非洲、欧洲、大洋洲）
simple_regions ={
    'World | Asia':'Asia',
    'Americas | Central America and Caribbean | Central America': 'North America',
    'Americas | Central America and Caribbean | Greater Antilles': 'North America',
    'Americas | Central America and Caribbean | Lesser Antilles and Bahamas': 'North America',
    'Americas | Northern America | Northern America': 'North America',
    'Americas | Northern America | Mexico': 'North America',
    'Americas | Southern America | Guyana':'South America',
    'Americas | Southern America | Andean':'South America',
    'Americas | Southern America | Brazil':'South America',
    'Americas | Southern America | Southern America':'South America',
    'World | Africa':'Africa',
    'World | Europe':'Europe',
    'World | Oceania':'Oceania'
}
# 替换
data.region = data.region.apply(lambda x: simple_regions[x])
print(data.region.unique())
# 提取单个区域的函数
def subregion(data, region):
    return data[data.region==region]
