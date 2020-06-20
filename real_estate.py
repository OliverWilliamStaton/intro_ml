import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
	# creates datasets/housing directory in workspace
	os.makedirs(housing_path,exist_ok=True)
	# download housing.tgz
	tgz_path = os.path.join(housing_path,"housing.tgz")
	urllib.request.urlretrieve(housing_url,tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	# extract housing.csv
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path,"housing.csv")
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
housing = load_housing_data()
# housing.head()
# housing.info()
# housing.describe()

## visualize data
## %matplotlib inline # when running in jupyter notebook only
# housing.hist(bins=50,figsize=(20,15))
# plt.show()

# establish training set & test set
train_set, test_set = split_train_test(housing,0.2) # test size if 20% of training data

### NOTE: random sampling may not be representative of the target demographic.
# In this example, analysts have observed that median income is a significant driver
# Exploring a stratified sampling based off of median income

# create an income category attribute with five categories
housing["income_cat"] = pd.cut(housing["median_income"],
	bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
	labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()
plt.show()

# do stratified sampling based on income category
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

# examine income category proportions in test set to see if this worked as expected
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

### Now, compare the test set generated with stratified sampling, and in a test
# set generated using purely random sampling. It is observed that the test set
# generated using stratified sampling has income category proportions almost identical
# to those in the full dataset, whereas the test set generated using purely random
# sampling is skewed.

# finally, remove the "income_cat" attribute
for set_ in (strat_train_set,strat_test_set):
	set_.drop("income_cat",axis=1,inplace=True)


