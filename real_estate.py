import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

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
# plt.show()

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

# copy training set to avoid damage during analysis
housing = strat_train_set.copy()

### Now, discover and visualize the data to gain insights
# this includes visualizing overall data, analysing linear and scatter plot correlations
# additionally, experiment with attribute combinations

# visualization of geographical information of all districts
# add predefined color map (called "jet") for population and price
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,
	s=housing["population"]/100,label="population",figsize=(10,7),
	c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()

# check for linear correlations by computing the standard correlation coefficient (Pearson's r)
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# check for other scatter plot correlations between several promising attributes
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

# the most promising attribute to predict median house value is the median income
# zoom in on their correction scatterplot
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
# plt.show()

# experiment with attribute combinations
# combine to determine rooms/household
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# combine to determine bedrooms/room
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# combine to determine population/household
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
# check for linear correlations with these new combined attributes
corr_matrix["median_house_value"].sort_values(ascending=False)

### In conclusion from the above analysis, the following observations were made:
# - identified a few data quirks that may need to be cleaned up before feeding into ML algorithm
# - found interesting correlations between attributes, in particular with the target attribute
# - noticed some attributes have tail-heavy distribution
# - bedrooms_per_room attribute is much more correlated than total number of rooms or bedrooms

### Prepare the data for ML algorithms
# TBD...


