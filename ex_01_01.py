import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import sklearn.linear_model

# Load the data
print(">>> Loading data")
oecd_bli = pd.read_csv("oecd_bli_2020.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")

# Prepare the data
print(">>> Preparing data")
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
print(">>> Visualizing data")
country_stats.plot(kind='scatter',x='GDP per capita',y='Life satisfaction')
plt.show()

# Select a model
print(">>> Model being selected")
model = sklearn.linear_model.LinearRegression()
# model = sklearn.linear_model.KNeighborsRegression()

# Train model
print(">>> Model under training")
model.fit(x,y)

# Make a prediction for Cyprus
x_new = [[22587]] # Cyprus's GDP per capita
print(model.predict(x_new))

print(">>> Complete")

