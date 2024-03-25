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

# a. Bike rental vs. holiday
fig = px.box(df_1, x='holiday', y='count', title='Bike rental count vs. type of day: holiday',
             labels={'count': 'Total Rentals', 'holiday': 'Type of day'})
fig.update_layout(boxgroupgap= 0.1, boxgap= 0.1, width = 700, font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.5,
                  xaxis = dict(tickmode = 'array', tickvals = [0,1], ticktext = ['Non-Holiday', 'Holiday']))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/holiday.html")

# b. Bike rental vs. working day
fig = px.box(df_1, x='workingday', y='count', title='Bike rental count vs. type of day: working day',
             labels={'count': 'Total Rentals', 'workingday': 'Type of day'})
fig.update_layout(boxgroupgap= 0.1, boxgap= 0.1, width = 700, font_color="#007CD8", title=dict(font=dict(size=25)), title_x=0.5,
                  xaxis = dict(tickmode = 'array', tickvals = [0,1], ticktext = ['Non-Workingday', 'Workingday']))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/working_day.html")

# F. Dealing with Outliers
df_3 = df_1.copy()

cont_vars = ['temperature', 'adjusted_temperature', 'humidity', 'windspeed']

plt.figure(figsize=(15, 40))
i = 0
for col in cont_vars:
  i += 1
  plt.subplot(9, 4, i)
  plt.boxplot(df_3[col])
  plt.title('{} boxplot'.format(col))
#plt.savefig('/content/drive/MyDrive/Imagenes/outliers.png')
plt.show()

wins_dict = {}
df_3 = df_1.copy()

winsorization_limits = {
    'variable3': (0.002, 0.0),
    'variable4': (0, 0.02)
}
plt.figure(figsize=(10, 5))

for i, (variable_key, limits) in enumerate(winsorization_limits.items()):
    col = cont_vars[i + 2]  # We will skip temperature variables as they don't have outliers
    wins_data = winsorize(df_3[col], limits=limits)
    wins_dict[col] = wins_data

    df_3[col] = wins_data

    plt.subplot(1, 2, i + 1)
    plt.boxplot(df_3[col])
    plt.title('{} boxplot (Winsorized: {} - {})'.format(col, limits[0], limits[1]))

plt.tight_layout()
#plt.savefig('/content/drive/MyDrive/Imagenes/no_outliers.png')
plt.show()

# Changes will be embedded in df_3 dataset
for col in cont_vars:
    if col in wins_dict:
        df_3[col] = wins_dict[col]

# 1. Correlation Matrix
relevant_columns= df_3.iloc[:,2:17]
correlation_matrix = relevant_columns.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
#plt.title('Correlation Matrix')
plt.show()

# 2. VIF Calculation
df_4 = df_3.copy()

def calc_vif(X):
    vif=pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

VIF = df_4.drop(columns = ['rental_id','date','count', 'casual_rider', 'registered_rider'])

calc_vif(VIF)

# Second Iteration
VIF = VIF.drop(columns = ['adjusted_temperature'])
calc_vif(VIF)

# Third Iteration
VIF = VIF.drop(columns = ['season'])
calc_vif(VIF)

t = [i for i in VIF]
print(t)

df_5 = df_3.copy()
Y = df_5['count']
SFS = df_5.drop(columns = ['rental_id','date','count', 'casual_rider', 'registered_rider'])
lin_reg = LinearRegression()
sfs1 = sfs(lin_reg, k_features = 10, forward = False, verbose = 1, scoring = 'neg_mean_squared_error')
sfs1 = sfs1.fit(SFS,Y)
fin_names = list(sfs1.k_feature_names_)
print(fin_names)

# 3. Building Models
# A. Multivariable Linear Regression Model
ret = df_3['count']

baby_blue = '#448EE4'
plt.figure(figsize=(10, 6))
ax = sns.histplot(ret, kde=True, color=baby_blue, bins=30, alpha=0.3)
ax.set_title('Distribution of Count Data')
ax.set_xlabel('Count')
ax.set_ylabel('Frequency')

# Descriptive Statistics
print('STATISTICS:')
print('Sample size: ', len(ret))
print('Mean: ', ret.mean())
print('Variance: ', ret.var())
print('Skewness: ', ret.skew())
print('Kurtosis: ', ret.kurt())
print('Maximum: ', ret.max())
print('Minimum: ', ret.min())
print('Jarque-Bera Test Results:', jarque_bera(ret))
print('Augmented Dickey-Fuller Test Results:', adf(ret, maxlag=1)[0:2])
print('Autocorrelation:', acf(ret, nlags=5))
print('Partial-Autocorrelation:', pacf(ret, nlags=5))

# plt.savefig('/content/drive/MyDrive/Imagenes/skewness.png')
plt.show()

# I. Logarithmic Scale Transformation
plt.hist((np.log(df_3.loc[:,'count'])))
plt.show()
skewness = stats.skew(df_3['count'])
print(f"Skewness after log transformation: {skewness}")

# II. Square Root Transformation
plt.hist((np.sqrt(df_3.loc[:,'count'])))
plt.show()

df_sqrt = np.sqrt(df_3['count'])
skewness = stats.skew(df_sqrt)
print(f"Skewness after sqrt transformation: {skewness}")

# III. Box-Cox Transformation
df_4 = df_3.copy()

df_4['transformed_count'], lambda_value = boxcox(df_3['count'] + 1)

sns.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'white'})

fig, ax = plt.subplots()
sns.distplot(df_4['transformed_count'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
# plt.savefig('/content/drive/MyDrive/Imagenes/noskewness.png')
fig.show()
skewness_after_boxcox = skew(df_4['transformed_count'], nan_policy='omit')
print(f"Skewness after Box-Cox transformation: {skewness_after_boxcox}")

# Regression Model
X = df_4.drop(columns = ['rental_id','date','count', 'transformed_count', 'casual_rider', 'registered_rider','month', 'temperature'])
y = df_4['transformed_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)


y_pred = linear_reg_model.predict(X_test)

mae_mv = mean_absolute_error(y_test, y_pred)
mse_mv = mean_squared_error(y_test, y_pred, squared=False)
r2_mv = r2_score(y_test, y_pred)

# Evaluation of the model
print("Mean Absolute Error:", mae_mv)
print("Root Mean Squared Error:", sqrt(mse_mv))
print("R-squared Value:", r2_mv)

comparison_df = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})

# Using Features from VIF selection
X = df_4.drop(columns=['rental_id', 'date','transformed_count', 'count', 'casual_rider', 'registered_rider', 'month', 'temperature'])
y = df_4['transformed_count']

# Backward elimination with a 5% significance level
def backward_elimination(X, y, feature_names, significance_level=0.05):
    while True:
        X_with_intercept = sm.add_constant(X)
        model = sm.OLS(y, X_with_intercept).fit()

        # Getting the feature with the highest p-value
        max_p_value = model.pvalues.drop('const').max()
        if max_p_value > significance_level:
            max_p_value_feature = model.pvalues.drop('const').idxmax()
            X = X.drop(columns=[max_p_value_feature])
            feature_names = feature_names.drop(max_p_value_feature)
            print(f'Removing feature: {max_p_value_feature}, P-value: {max_p_value:.4f}')
        else:
            break

    return X, feature_names

selected_feature_names = X.columns
X_selected, selected_feature_names = backward_elimination(X, y, selected_feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Fitting our Linear Regression model with Backward Elimination
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

y_pred = linear_reg_model.predict(X_test)

# Evaluation of the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", sqrt(mean_squared_error(y_test, y_pred, squared=False)))
print("R-squared Value:", r2_score(y_test, y_pred))

print("Selected Features:", selected_feature_names)
comparison_df.head(10)

# IV. Residual Analysis
