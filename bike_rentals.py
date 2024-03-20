# Bike Rental Project
# Objective: Predict the number of bike rentals exploring which of the given variables mainly affects the amount of bike rentals.

#!pip install scikeras[tensorflow]
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV, GridSearchCV, cross_val_predict
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from plotly.subplots import make_subplots
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, boxcox, skew, norm, jarque_bera
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf, pacf
from yellowbrick.regressor import ResidualsPlot
from statsmodels.stats.diagnostic import het_breuschpagan
from pprint import pprint

# Data Understanding
df_1 = pd.read_csv('bike_rental_hour.csv')
df_2 = df_1.copy()
df_1.head()
df_1.dtypes


# Renaming Columns
datasets = [df_1,df_2]
for i in datasets:
  i.rename(columns={'instant':'rental_id', 'dteday':'date', 'yr':'year', 'mnth':'month', 'hr':'hour', 'weathersit':'weather_type', 'temp':'temperature', 'atemp':'adjusted_temperature', 'hum':'humidity',
                   'casual':'casual_rider', 'registered':'registered_rider', 'cnt':'count'}, inplace=True)

# Decoding Data
# Decoding Windspeed
y = [ i for i in df_1['windspeed']*67]
df_2['windspeed_d'] = y

# Decoding Normalized Temperature
t_min = -8
t_max = 35
normalized_temps = df_1['temperature'].values

decoded_temps = [temp * (t_max - t_min) + t_min for temp in normalized_temps]
df_2['temp_d'] = decoded_temps

# Decoding Normalized Adjusted Temperature
t_min = -16
t_max = 50
normalized_adjtemps = df_1['adjusted_temperature'].values

decoded_adjtemps = [adjtemp * (t_max - t_min) + t_min for adjtemp in normalized_adjtemps]
df_2['adjtemp_d'] = decoded_adjtemps

# Decoding Humidity
z = [ i for i in df_1['humidity']]
df_2['humidity_d'] = z


normalized_humidity = df_1['humidity'].values

decoded_humidity = [hum * 100 for hum in normalized_humidity]
df_2['humidity_d'] = decoded_humidity

# Checking for Duplicates
if df_1.duplicated().any():
    print("\nThe DataFrame has duplicates.")
else:
    print("\nThe DataFrame does not have duplicates.")

# Checking for Missing Values
df_1.isnull().sum(axis=0)

# Exploratory Data Analysis

# Bike Rental vs Rider Type
# First, we take the sum of casual riders and registered riders, grouped by year, and then we plot the respective pie charts

# Taking the sum of casual and registered riders grouped by year (returns a Series)
casual_riders = df_2.groupby('year')['casual_rider'].sum()
registered_riders = df_2.groupby('year')['registered_rider'].sum()

# Taking the sum of casual and registered riders combined for 2011 and 2012
casual_riders_c = df_2["casual_rider"].sum()
registered_riders_c = df_2["registered_rider"].sum()

# Sorting casual rider sums by year and storing the values in the respective lists, "11" for the year 2011 and "12" for the year 2012
casual_riders_11 = casual_riders[0]
casual_riders_12 = casual_riders[1]

# Sorting registered rider sums by year and storing the values in the respective lists, "11" for the year 2011 and "12" for the year 2012
registered_riders_11 = registered_riders[0]
registered_riders_12 = registered_riders[1]

# Printing the numbers to verify correctness
print("Combined: ", casual_riders_c, registered_riders_c)
print("2011: ", casual_riders_11, registered_riders_11)
print("2012: ", casual_riders_12, registered_riders_12)

# Defining data for the pie charts, "11" for the year 2011 and "12" for the year 2012
labels = ['Casual Riders', 'Registered Riders']
sizes = [casual_riders_c, registered_riders_c]
sizes_11 = [casual_riders_11, registered_riders_11]
sizes_12 = [casual_riders_12, registered_riders_12]

# Pie chart for 2011
plt.pie(sizes_11, labels=labels, autopct='%1.1f%%', startangle=90)
fig1 = plt.gcf()
plt.axis('equal')
plt.title('Distribution of Riders in 2011')
plt.show()

# Pie chart for 2012
plt.pie(sizes_12, labels=labels, autopct='%1.1f%%', startangle=90)
fig2 = plt.gcf()
plt.axis('equal')
plt.title('Distribution of Riders in 2012')
plt.show()

# Combined pie chart for 2011 and 2012
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
fig3 = plt.gcf()
plt.axis('equal')
plt.title('Combined distribution of Riders in 2011 and 2012')
plt.show()
