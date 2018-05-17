# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium
#pip install folium

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

# File with functions from prior notebook(s)
# sys.path.append('../../scripts/')
from scripts.aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice

# Update matplotlib defaults to something nicer
mpl_update = {'font.size':16,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'figure.figsize':[12.0,8.0],
              # 'axes.color_cycle':['#0055A7', '#2C3E4F', '#26C5ED', '#00cc66', '#D34100', '#FF9700','#091D32'],
              'axes.labelsize':20,
              'axes.labelcolor':'#677385',
              'axes.titlesize':20,
              'lines.color':'#0055A7',
              'lines.linewidth':3,
              'text.color':'#677385'}
mpl.rcParams.update(mpl_update)

data = pd.read_csv('aquastat.csv.gzip', compression='gzip')
data.region = data.region.apply(lambda x: simple_regions[x])
recent = time_slice(data, '2013-2017')
msno.matrix(recent, labels=True)
# plt.show()

#Total exploitable water resources 水资源总量
msno.matrix(variable_slice(data, 'exploitable_total'), inline=False, sort='descending');
# plt.xlabel('Time period');
# plt.ylabel('Country');
# plt.title('Missing total exploitable water resources data across countries and time periods \n \n \n \n')
# plt.show()
# ~5 = -6
# 解释： 将二进制数 + 1之后乘以 - 1，即 ~x = -(x + 1)，-(101 + 1) = -110
# 我们将删除该变量，因为这么少的数据点会导致很多问题。
data = data.loc[~data.variable.str.contains('exploitable'),:]

#national_rainfall_index 全国降水指数（NRI）（毫米/年)
msno.matrix(variable_slice(data, 'national_rainfall_index'),
            inline=False, sort='descending');
# plt.xlabel('Time period');
# plt.ylabel('Country');
# plt.title('Missing national rainfall index data across countries and time periods \n \n \n \n')

data = data.loc[~(data.variable=='national_rainfall_index')]

north_america = subregion(data, 'North America')
#指数完整性
msno.matrix(msno.nullity_sort(time_slice(north_america, '2013-2017'), sort='descending').T, inline=False)
#plt.title('Fraction of fields complete by country for North America \n \n');
# 抽查巴哈马缺少哪些数据以获得更多的了解
msno.nullity_filter(country_slice(data, 'Bahamas').T, filter='bottom', p=0.1)

# JSON with coordinates for country boundaries
geo = r'world.json'

null_data = recent['agg_to_gdp'].notnull()*1
map = folium.Map(location=[48, -102], zoom_start=2)
map.choropleth(geo_data=geo,
               data=null_data,
               columns=['country', 'agg_to_gdp'],
               key_on='feature.properties.name', reset=True,
               fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
               legend_name='Missing agricultural contribution to GDP data 2013-2017')
# file_path = r"test.html"
# map.save(file_path)     # 保存为html文件
# import webbrowser
# webbrowser.open(os.path.abspath())  # 默认浏览器打开
# 封装地图函数
def plot_null_map(df, time_period, variable,
				  legend_name=None):
	geo = r'world.json'

	ts = time_slice(df, time_period).reset_index().copy()
	ts[variable] = ts[variable].notnull() * 1
	map = folium.Map(location=[48, -102], zoom_start=2)
	map.choropleth(geo_data=geo,
				   data=ts,
				   columns=['country', variable],
				   key_on='feature.properties.name', reset=True,
				   fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
				   legend_name=legend_name if legend_name else variable)
	return map
# plot_null_map(data, '2013-2017', 'number_undernourished', 'Number undernourished is missing')

# Are there any patterns in missing data? Any questions that come to mind for further investigation?
fig, ax = plt.subplots(figsize=(16, 16));
sns.heatmap(data.groupby(['time_period','variable']).value.count().unstack().T , ax=ax);
plt.xticks(rotation=45);
plt.xlabel('Time period');
plt.ylabel('Variable');
plt.title('Number of countries with data reported for each variable over time')


# # 在试图了解数据中哪些信息之前，请务必了解数据代表什么。     This probably means that you are not using fork to start your
#         child processes and you have forgotten to use the proper idiom
#         in the main module:
# 这是对数据的Overview
pandas_profiling.ProfileReport(time_slice(data, '2013-2017'))



