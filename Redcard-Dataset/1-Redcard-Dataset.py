from __future__ import absolute_import, division, print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno
import pandas_profiling

from sklearn.datasets import make_blobs
import time

# Uncomment one of the following lines and run the cell:
# 读入数据
df = pd.read_csv("redcard.csv.gz", compression='gzip')
# 查看数据
# df.shape
# df.head()
print(df.describe().T)
print(df.dtypes)
all_columns = df.columns.tolist()
# 探索方向
# Exploration/hypotheses:
# Distribution of games played
# red cards vs games played
# Reds per game played vs total cards per game played by skin color
# Distribution of # red, # yellow, total cards, and fraction red per game played for all players by avg skin color
# How many refs did players encounter?
# Do some clubs play more aggresively and get carded more? Or are more reserved and get less?
# Does carding vary by leagueCountry?
# Do high scorers get more slack (fewer cards) for the same position?
# Are there some referees that give more red/yellow cards than others?
# how consistent are raters? Check with Cohen's kappa.
# how do red cards vary by position? e.g. defenders get more?
# Do players with more games get more cards, and is there difference across skin color?
# indication of bias depending on refCountry?
# Understand how the data's organized¶
df['height'].mean()
df['height'].mean()
np.mean(df.groupby('playerShort').height.mean())
# 这个三个有什么区别，最后 一个会更加准确
# Tidy Data 数据划分
df2 = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
     'key2':['one', 'two', 'one', 'two', 'one'],
     'data1':np.random.randn(5),
     'data2':np.random.randn(5)})
grouped = df2['data1'].groupby(df['key1'])
grouped.mean()
player_index = 'playerShort'
player_cols = [#'player', # drop player name, we have unique identifier
               'birthday',
               'height',
               'weight',
               'position',
               'photoID',
               'rater1',
               'rater2',
              ]
# Count the unique variables (if we got different weight values,
# for example, then we should get more than one unique value in this groupby)
all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})
print(all_cols_unique_players.head())

# If all values are the same per player then this should be empty (and it is!)
all_cols_unique_players[all_cols_unique_players > 1].dropna().head()
# A slightly more elegant way to test the uniqueness
all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0] == 0


def get_subgroup(dataframe, g_index, g_columns):
	"""Helper function that creates a sub-table from the columns and runs a quick uniqueness test."""
	g = dataframe.groupby(g_index).agg({col: 'nunique' for col in g_columns})
	if g[g > 1].dropna().shape[0] != 0:
		print("Warning: you probably assumed this had all unique values but it doesn't.")
	return dataframe.groupby(g_index).agg({col: 'max' for col in g_columns})
players = get_subgroup(df, player_index, player_cols)
print(players.head())
