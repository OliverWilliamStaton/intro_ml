import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import transformations as tf
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

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

# revert to a clean training set
# separate predictors and labels
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# ML algorithms cannot work with missing features. Here are 3 options to handle such a case.
# Option 1: get rid of corresponding districts
housing.dropna(subset=["total_bedrooms"])
# Option 2: get rid of the whole attribute
housing.drop("total_bedrooms",axis=1)
# Option 3: set the values to some value (zero, the mean, the median, etc.)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

from sklearn.impute import SimpleImputer
# replace each attribute's missing values with the median of that attribute
imputer = SimpleImputer(strategy="median")
# create copy of data without text attribute "ocean_proximity"
housing_num = housing.drop("ocean_proximity",axis=1)
# fit the imputer instance to training data
imputer.fit(housing_num)
# use the "trained" imputer to transform training set; replace missing values with learned medians
x = imputer.transform(housing_num)
# put it back into pandas DataFrame
housing_tr = pd.DataFrame(x,columns=housing_num.columns,index=housing_num.index)

### Handling text and categorical attributes

# convert categories from text to numbers
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# see 1D array of categories for each categorical attribute
ordinal_encoder.categories_

# apply one-hot encoding to break the assumption of nearby similarities
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# see 1HOT encoding array
housing_cat_1hot.toarray()

### Custom Transformers

# combine attributes
attr_addr = tf.CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_addr.transform(housing.values)

### Feature scaling

# create a transformation pipeline for standard scaler estimation
# TODO - move to common transformation.py module
num_pipeline = Pipeline([
		('imputer', SimpleImputer(strategy="median")),
		('attribs_adder', tf.CombinedAttributesAdder()),
		('std_scaler', StandardScaler()),
	])
# housing_num_tr = num_pipeline.fit_transform(housing_num)

# constructor requires a list of tuples: [name,transformer,list of indices]
full_pipeline = ColumnTransformer([
		("num",num_pipeline,list(housing_num)),
		("cat",OneHotEncoder(),["ocean_proximity"]),
	])
# apply the full pipeline to the housing data
# ColumnTransformer applies each transformer to the appropriate columns and
# 	concatenates the outputs along the second axis
housing_prepared = full_pipeline.fit_transform(housing)

### Traning and evaluating on the Training Set

# model 1: linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

# test out the data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_date_prepared = full_pipeline.transform(some_data)
print("Predictions:",lin_reg.predict(some_date_prepared))
print("Labels:",list(some_labels))

# measure RMSE on the whole training set
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# lin_rmse # answer = 68628.19819848922 (underfitting)

# model 2: decision tree regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

# test out the data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_date_prepared = full_pipeline.transform(some_data)
print("Predictions:",tree_reg.predict(some_date_prepared))
print("Labels:",list(some_labels))

# test out the data
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # answer = 0.0 (overfitting)

# model 3: random forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

# test out the data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_date_prepared = full_pipeline.transform(some_data)
print("Predictions:",forest_reg.predict(some_date_prepared))
print("Labels:",list(some_labels))

# test out the data
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse # answer = 18680.2942402

def display_scores(scores):
	print("Scores:",scores)
	print("Mean:",scores.mean())
	print("Standard deviation:",scores.std())

# better evaluation using cross-validation (k-fold cross-validation)
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores) # mean score: 71154 +/- 3231

scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores) # mean score: 69052 +/- 2731

scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores) # mean score: 50150 +/- 1902

### fine-tune your model

# automatically explore hyperparameter combinations
# GridSearchCV searches and cross-validates possible combinations
from sklearn.model_selection import GridSearchCV
param_grid = [
	{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
	{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)

# determine best hyperparameters & estimator
grid_search.best_params_
grid_search.best_estimator_

# determine evaluation scores
cvres = grid_search.cv_results_
for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
	print(np.sqrt(-mean_score),params)

### evaluate system on Test Set

final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
display_scores(final_rmse) #48760
