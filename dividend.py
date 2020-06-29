import os
# import tarfile
# import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix

DIVIDEND_PATH = os.path.join("datasets","dividend")

def load_dividend_data(path=DIVIDEND_PATH):
	csv_path = os.path.join(path,"dividend.csv")
	# returns a pandas DataFrame object containing all data
	return pd.read_csv(csv_path)

def split_train_test(data,test_ratio):
	np.random.seed(42) # hard-coded seed to generate the same shuffled indices
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data)*test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices],data.iloc[test_indices]

## manual verification that data is loading
dividend = load_dividend_data()
# dividend.info()

## visualize data
# dividend.hist(bins=50,figsize=(20,15))
# plt.show()

# establish training set & test set
train_set, test_set = split_train_test(dividend,0.2) # test size if 20% of training data

# create an income category attribute with five categories
dividend["growth_cat"] = pd.cut(dividend["years_of_growth"],
	bins=[0., 5.0, 10.0, 15.0, 20.0, np.inf],
	labels=[1, 2, 3, 4, 5])
dividend["growth_cat"].hist()
# plt.show()

# do stratified sampling based on income category
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(dividend,dividend["growth_cat"]):
	strat_train_set = dividend.loc[train_index]
	strat_test_set = dividend.loc[test_index]

# examine income category proportions in test set to see if this worked as expected
# print(strat_test_set["growth_cat"].value_counts()/len(strat_test_set))
# ---> confirmed stratification matches growth_cat split

# clean-up by removing "growth_cat" attribute
for set_ in (strat_train_set,strat_test_set):
	set_.drop("growth_cat",axis=1,inplace=True)
# copy training set to avoid damage during analysis
dividend = strat_train_set.copy()

### Visualize the data to gain insights

# visualization of geographical information of all districts
# add predefined color map (called "jet") for population and price
dividend.plot(kind="scatter",x="years_of_growth",y="yield_pct",alpha=0.5,
	label="Yield vs Years of Growth",figsize=(10,7))
plt.legend()
plt.show()
