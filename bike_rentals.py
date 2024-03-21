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

# Bike Rental at Different Time Intervals
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
months_t = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = px.box(df_2, x='month', y='count', title='Bike rental count per month',
             labels={'count': 'Total Rentals', 'month': 'Month'})
fig.update_layout(boxgroupgap= 0.1, boxgap= 0.2, width = 900, font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.5,
                  xaxis = dict(tickmode = 'array', tickvals = months, ticktext = months_t))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, tick0=1, dtick=1)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/month.html")

# Bike rental per weekday
weekday_names = ['Saturday', 'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday' ]
fig = px.box(df_2, x='weekday', y='count', color = 'weekday', title='Bike rental count per weekday',
             labels={'count': 'Total Rentals', 'weekday': 'Weekdays'})

fig.data[0].name = 'Friday'
fig.data[1].name = 'Saturday'
fig.data[2].name = 'Sunday'
fig.data[3].name = 'Monday'
fig.data[4].name = 'Tuesday'
fig.data[5].name = 'Wednesday'
fig.data[6].name = 'Thursday'


fig.update_xaxes(categoryorder='array', categoryarray=weekday_names)
fig.update_layout(boxgroupgap= 0.1, boxgap= 0.2, width = 900, font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.5,
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4,5,6],
        ticktext = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    ))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/weekday.html")

# Bike rental per hour
df_weekday = df_2.copy()
df_weekday['weekday'] = df_weekday['weekday'].replace({0:'Sunday', 1:'Monday',2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'})
hour_day_df = df_weekday.groupby(["hour", "weekday"])["count"].mean().to_frame().reset_index()

df_season = df_2.copy()
df_season['season'] = df_season['season'].replace({1:'Winter', 2:'Spring', 3:'Summer', 4:'Autumn'})
season_df = df_season.groupby(["hour", "season"])["count"].mean().to_frame().reset_index()

df_weather = df_2.copy()
df_weather['weather_type'] = df_weather['weather_type'].replace({1:'Clear', 2:'Mist', 3:'Light Rain', 4:'Heavy Rain'})
weather_df = df_weather.groupby(["hour", "weather_type"])["count"].mean().to_frame().reset_index()

fig = px.line(hour_day_df, x='hour', y='count', color='weekday',
              title="Bike Rental Count vs Hour-Weekday comparison",
              labels={
                     "hour": "Time of the day (Hrs)",
                     "count": "Total Bike Rentals (Average)",
                     "weekday": "Weekday"
                 },
              category_orders={"weekday": ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']})
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, range=[0, 600])
fig.show()
# fig.write_html("/content/drive/MyDrive/Imagenes/uno.html")

fig = px.line(season_df, x='hour', y='count', color='season',
              title="Rental Bikes vs Hour-Season comparison",
              labels={
                     "hour": "Time of the day (Hrs)",
                     "count": "Total Bike Rentals (Average)",
                     "season": "Season"
                 },
              category_orders={"season": ['Winter', 'Spring', 'Summer', 'Autumn']})
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10, range=[0, 600])
fig.show()
# fig.write_html("/content/drive/MyDrive/Imagenes/dos.html")

fig = px.line(weather_df, x='hour', y='count', color='weather_type',
              title="Bike Rental Count vs Hour-Weather comparison",
              labels={
                     "hour": "Time of the day (Hrs)",
                     "count": "Total Bike Rentals (Average)",
                     "weather_type": "Weather"
                 },
              category_orders={"weather_type": ['Clear', 'Mist', 'Light Rain', 'Heavy Rain']})
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, range=[0, 550])
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/tres.html")

# Bike rental per day over 2 years
df_d = df_2.copy()
df_d['year'] = df_d['year'].replace({0:'2011', 1:'2012'})
df_d['date'] = df_d['date'].astype('datetime64[ns]')

h_d = df_d.groupby(['date', 'year'])['count'].sum()
df_d2 = pd.DataFrame({'suma' : h_d}).reset_index()

#df_d2

fig = px.bar(df_d2, x="date", y="suma", color = 'year',
            title="Bike rental per day during all 2 years",
            labels={
                     "date": "Day",
                     "suma": "Total Bike Rentals",
                     "year": "Year"
                 })
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/perday.png")

# Bike rental at different weather situations
# Bike rental vs windspeed
df_w = df_2.copy()

df_ws = df_w.groupby(['windspeed_d'])['count'].sum()
df_ws = pd.DataFrame({'suma' : df_ws}).reset_index()

fig = px.area(df_ws, x="windspeed_d", y="suma",
            title="Bike rental vs Windspeed",
            labels={
                     "windspeed_d": "Windspeed",
                     "suma": "Total Bike Rentals"
                 })
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()

# Bike rental vs decoded temperature
fig = px.box(df_2, x='temp_d', y="count", width=1300, height=700,
             title='Box Plot of Total Rentals vs. Temperature',
             labels={"temp_d": "Temperature [C]",
                     "count": "Bike Rentals"
                 })
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.5)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10, tick0=-7, dtick=5)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()

# Bike rental vs humidity
f = df_2.groupby(['humidity_d'])['count'].sum()
df_h = pd.DataFrame({'suma' : f}).reset_index()

fig = px.bar(df_h, x="humidity_d", y="suma",
            title="Bike rental vs Humidity",
            labels={
                     "humidity_d": "Humidity",
                     "suma": "Total Bike Rentals"
                 })
fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()

# Bike rental vs season
df_s = df_2.copy()
df_s['season'] = df_s['season'].replace({1:'Winter', 2:'Spring', 3:'Summer', 4:'Autumn'})

g = df_s.groupby(['season'])['count'].sum()
df_s2 = pd.DataFrame({'suma' : g}).reset_index()

fig = px.bar(df_s2, x="season", y="suma", color = 'season', text_auto = '.2s',
            title="Bike rental per Season",
            labels={
                     "season": "Season",
                     "suma": "Total Bike Rentals"
                 },
             category_orders={"season": ['Winter', 'Spring', 'Summer', 'Autumn']})

fig.update_layout(font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.47, bargap=0.5)
fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, range=[0, 1200000])
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/season.html")

# VII. Bike rental at different day types
