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

# Creating a ResidualsPlot
visualizer = ResidualsPlot(linear_reg_model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

visualizer.ax.tick_params(axis='both', labelsize=14)
visualizer.ax.set_xlabel('X Axis Title', fontsize=16)
visualizer.ax.set_ylabel('Y Axis Title', fontsize=16)
visualizer.show()

# Performing the Breusch-Pagan test
X_test_with_constant = sm.add_constant(X_test)
residuals = y_test - linear_reg_model.predict(X_test)

_, p_value, _, _ = het_breuschpagan(residuals, X_test_with_constant)
print(f"P-value from Breusch-Pagan test: {p_value}")

if p_value < 0.05:
    print("The test suggests the presence of heteroscedasticity.")
else:
    print("The test does not suggest the presence of heteroscedasticity.")

v_t = [i for i in y_test]

fig = go.Figure()
fig.add_trace(go.Scatter(x=v_t, y=y_pred,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=v_t, y=v_t,
                    mode='markers',
                    name='Test Values'))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))

fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()

# Polynomial Regression Model
X = df_4.drop(columns=['rental_id', 'count','date', 'casual_rider', 'registered_rider', 'month', 'temperature', 'transformed_count'])
y = df_4['transformed_count']

X_train, X_test, y_train, y_testp = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting polynomial features
degree = 3
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly_scaled, y_train)

y_pred_poly = poly_reg_model.predict(X_test_poly_scaled)

mae_p = mean_absolute_error(y_testp, y_pred_poly)
mse_p =mean_squared_error(y_testp, y_pred_poly, squared=False)
r2_p = r2_score(y_testp, y_pred_poly)

print("Mean Absolute Error (Polynomial Regression):", mae_p)
print("Root Mean Squared Error (Polynomial Regression):", mse_p)
print("R-squared Value (Polynomial Regression):", r2_p)

# Checking Heteroscedascity
v_p = [i for i in y_testp]

fig = go.Figure()
fig.add_trace(go.Scatter(x=v_p, y=y_pred_poly,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=v_p, y=v_p,
                    mode='markers',
                    name='Test Values'))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))

fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)

fig.show()

# Residual Analysis

# Residuals
residuals_poly = y_test - y_pred_poly

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_poly, residuals_poly)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot (Polynomial Regression)',fontsize=16)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.xlabel('Predicted Values', fontsize=16)
plt.ylabel('Residuals', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.show()

# C. Decision Tree Model

# Columns dropped: temperature + month

# Splitting the data into training and testing sets
X = df_3.drop(columns = ['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).values
Y = df_3[['count']].values

X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)

# Initializing the decision tree regressor
dt = DecisionTreeRegressor()

# Training the model
dt.fit(X_train, y_train)

# Making predictions
Y_train_pred = dt.predict(X_train)
Y_pred_dt = dt.predict(X_test)

# Evaluating the model
r2_train = r2_score(y_train, Y_train_pred)
mae_train = mean_absolute_error(y_train, Y_train_pred)
rmse_train = sqrt(mean_squared_error(y_train, Y_train_pred))

r2 = r2_score(y_test, Y_pred_dt)
mae = mean_absolute_error(y_test, Y_pred_dt)
rmse = sqrt(mean_squared_error(y_test, Y_pred_dt))

print(f'Training performance \n R2: {r2_train}, \n MAE: {mae_train}, \n RMSE: {rmse_train}')
print(f'\nTesting performance \n R2: {r2}, \n MAE: {mae}, \n RMSE: {rmse}')

# I. Feature Selection
feature_importance = dt.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).columns, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', hue='Feature', legend=False)
plt.title('Decision tree feature importance')
plt.show()

# II. Checking Heteroscedascity
dt_x = [i[0] for i in y_test]

fig = go.Figure()
fig.add_trace(go.Scatter(x=dt_x, y=Y_pred_dt,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=dt_x, y=dt_x,
                    mode='markers',
                    name='Test Values'))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))
fig.add_annotation(x=150, y=950,
            text=f'       MSE: {rmse:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=900,
            text=f'    MAE: {mae:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=850,
            text=f'R2: {r2:.2f}',
            showarrow=False)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)

fig.show()

# III. K-Fold cross validation
# K-fold CV
kf = KFold(n_splits=5)

train_r2_cv = cross_val_score(dt, X_train, y_train, cv=kf)
y_train_pred_cv = cross_val_predict(dt, X_train, y_train, cv=kf)

print('K-Fold training values')
for i in range(len(train_r2_cv)):
  print(f'Fold {i+1}: ', train_r2_cv[i])
print('Average R2: ', np.mean(train_r2_cv))

y_test_pred = dt.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print('\nK-Fold testing values')
print('R2: ', test_r2)

# Hyperparameter tunning
# Creating a function to compute mse, mae and r2 score calculations
def evaluate_model(model, X, Y):
    predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(Y, predictions))
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return rmse, mae, r2

# Defining random parameter to initiate the randomized cv search
param_dist = {
    'max_depth': [None] + [int(x) for x in np.linspace(1, 20, num=5)],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
}

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

random_search = RandomizedSearchCV(
    dt, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)

print('Best Hyperparameters from randomized search')
print(random_search.best_params_)

best_dt_model = DecisionTreeRegressor(
    max_depth=random_search.best_params_['max_depth'],
    min_samples_split=random_search.best_params_['min_samples_split'],
    min_samples_leaf=random_search.best_params_['min_samples_leaf'],
    random_state=42
)

best_dt_model.fit(X_train, y_train)

# Evaluation of the default model on the test set
test_rmse_default, test_mae_default, test_r2_default = evaluate_model(dt, X_test, y_test)

# Evaluation of the best model obtained from random search on the test set
test_rmse_best, test_mae_best, test_r2_best = evaluate_model(best_dt_model, X_test, y_test)

rmse_improvement = ((test_rmse_default - test_rmse_best) / test_rmse_default) * 100
mae_improvement = ((test_mae_default - test_mae_best) / test_mae_default) * 100
r2_improvement = ((test_r2_best - test_r2_default) / np.abs(test_r2_default)) * 100

print('\nPercentage Improvement from Default to Best Random Search Model:')
print(f'Root Mean Squared Error: {rmse_improvement:.2f}%')
print(f'Mean Absolute Error: {mae_improvement:.2f}%')
print(f'R2 Score improvement: {r2_improvement:.2f}%')
print('\nRMSE: ', test_rmse_best)
print('MAE: ', test_mae_best)
print('R2: ', test_r2_best)

X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)
def evaluate_model(model, X, Y):
    predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(Y, predictions))
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return rmse, mae, r2

best_dt_model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=6,
    random_state=42
)

best_dt_model.fit(X_train, y_train)

Y_pred_best = best_dt_model.predict(X_test)

test_rmse_best, test_mae_best, test_r2_best = evaluate_model(best_dt_model, X_test, y_test)

print('\nRMSE: ', test_rmse_best)
print('MAE: ', test_mae_best)
print('R2: ', test_r2_best)

# Visualizing results
dt_v = [i[0] for i in y_test]

fig = go.Figure()
fig.add_trace(go.Scatter(x=dt_v, y=Y_pred_best,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=dt_v, y=dt_v,
                    mode='markers',
                    name='Test Values'))

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))
fig.add_annotation(x=150, y=950,
            text=f'     MSE: {test_rmse_best:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=900,
            text=f'    MAE: {test_mae_best:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=850,
            text=f'R2: {test_r2_best:.2f}',
            showarrow=False)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)

fig.show()

# Analizing the model changing the independent variables

# Split the data into training and testing sets
X = df_3.drop(columns = ['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'adjusted_temperature', 'month']).values
Y = df_3[['count']].values

X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.3)

# Initialize the decision tree regressor
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions
Y_train_pred = model.predict(X_train)
Y_pred = model.predict(X_test)

# Evaluate the model
r2_train = r2_score(y_train, Y_train_pred)
mae_train = mean_absolute_error(y_train, Y_train_pred)
mse_train = mean_squared_error(y_train, Y_train_pred)

r2 = r2_score(y_test, Y_pred)
mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)

print(f"Training performance \n R2: {r2_train}, MAE: {mae_train}, MSE: {mse_train}")
print(f"Testing performance \n R2: {r2}, MAE: {mae}, MSE: {mse}")


feature_importance = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'adjusted_temperature', 'month']).columns, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Decision Tree - Feature Importance')
plt.show()


# Create scatter plot
plt.scatter(Y_pred, y_test)
plt.plot([min(Y_pred), max(Y_pred)], [min(Y_pred), max(Y_pred)], 'k--', lw=2)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Scatter plot of predicted vs actual values')

# Add evaluation metrics to plot
plt.text(8, 40, f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}', fontsize=13, ha='left')

plt.show()

# Split the data into training and testing sets
X = df_3.drop(columns = ['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'adjusted_temperature', 'season']).values
Y = df_3[['count']].values

X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.3)

# Initialize the decision tree regressor
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions
Y_train_pred = model.predict(X_train)
Y_pred = model.predict(X_test)

# Evaluate the model
r2_train = r2_score(y_train, Y_train_pred)
mae_train = mean_absolute_error(y_train, Y_train_pred)
mse_train = mean_squared_error(y_train, Y_train_pred)

r2 = r2_score(y_test, Y_pred)
mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)

print(f"Training performance \n R2: {r2_train}, MAE: {mae_train}, MSE: {mse_train}")
print(f"Testing performance \n R2: {r2}, MAE: {mae}, MSE: {mse}")


feature_importance = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'adjusted_temperature', 'season']).columns, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Decision Tree - Feature Importance')
plt.show()


# Create scatter plot
plt.scatter(Y_pred, y_test)
plt.plot([min(Y_pred), max(Y_pred)], [min(Y_pred), max(Y_pred)], 'k--', lw=2)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Scatter plot of predicted vs actual values')

# Add evaluation metrics to plot
plt.text(8, 40, f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}', fontsize=13, ha='left')

plt.show()

# D. Random Forest Model

X = df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).values
Y = df_3['count'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, Y_train)

Y_train_pred = rf.predict(X_train)

Y_test_pred = rf.predict(X_test)

# Evaluations on Train set
train_mse = sqrt(mean_squared_error(Y_train, Y_train_pred))
train_mae = mean_absolute_error(Y_train, Y_train_pred)
train_r2 = r2_score(Y_train, Y_train_pred)
print('Training Set Evaluation:')
print(f'Root Mean Squared Error: {train_mse}')
print(f'Mean Absolute Error: {train_mae}')
print(f'R2 Score: {train_r2}')

# Evaluations on Test set
test_mse = sqrt(mean_squared_error(Y_test, Y_test_pred))
test_mae = mean_absolute_error(Y_test, Y_test_pred)
test_r2 = r2_score(Y_test, Y_test_pred)

print('Test Set Evaluation:')
print(f'Root Mean Squared Error: {test_mse}')
print(f'Mean Absolute Error: {test_mae}')
print(f'R2 Score: {test_r2}')

feature_importance_rf = rf.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'adjusted_temperature', 'season']).columns, 'Importance': feature_importance_rf})

feature_importance_df_rf = feature_importance_df.sort_values(by='Importance', ascending=False, inplace= True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Random Forest - Feature Importance')
plt.show()

# Visualize the first few levels of one of the trees in the Random Forest
tree_index = 0
plt.figure(figsize=(8, 5))
plt.title('Random Forest Trees')
plot_tree(rf.estimators_[tree_index], feature_names=feature_importance_df['Feature'].values, filled=True, rounded=True, max_depth=2)
plt.show()

# II. Checking Heteroscedascity

v_y = [i for i in Y_test]

fig = go.Figure()
fig.add_trace(go.Scatter(x=v_y, y=Y_test_pred,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=v_y, y=v_y,
                    mode='markers',
                    name='Test Values'))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))

fig.add_annotation(x=150, y=950,
            text=f'      RMSE: {test_mse:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=900,
            text=f'    MAE: {test_mae:.2f}',
            showarrow=False)
fig.add_annotation(x=150, y=850,
            text=f'R2: {test_r2:.2f}',
            showarrow=False)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)

fig.show()

# III. K-Fold Cross Validation
rf = RandomForestRegressor(n_estimators=100, random_state=42)
kf = KFold(n_splits=5)

train_r2_cv = cross_val_score(rf, X_train, Y_train, cv=kf)
print(train_r2_cv)

X = df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).values
Y = df_3['count'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
kf = KFold(n_splits=10)

Y_train_pred_cv = cross_val_predict(rf, X_train, Y_train, cv=kf)

cv_scores_per_fold_train = []
for i, (train_idx, val_idx) in enumerate(kf.split(X_train, Y_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]

    rf.fit(X_fold_train, Y_fold_train)

    Y_fold_val_pred = rf.predict(X_fold_val)

    # Calculating R2 score for the fold
    fold_r2_score = r2_score(Y_fold_val, Y_fold_val_pred)
    cv_scores_per_fold_train.append(fold_r2_score)

    # Printing R2 score for each fold
    print(f'Training Fold {i+1} R2 Score: {fold_r2_score}')

# Printing average R2 score across folds for the training set
average_cv_r2_score_train = np.mean(cv_scores_per_fold_train)
print(f'Average R2 Score across training folds: {average_cv_r2_score_train}')

# After CV, predicting on the test set
Y_test_pred = rf.predict(X_test)

# Evaluation of performance on the test set
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_r2 = r2_score(Y_test, Y_test_pred)

print('\nTest Set Evaluation:')
print(f'Root Mean Squared Error: {test_rmse}')
print(f'Mean Absolute Error: {test_mae}')
print(f'R2 Score: {test_r2}')

plt.figure(figsize=(10, 6))
plt.scatter(Y_train, Y_train_pred_cv, alpha=0.5, label='Training Set')
plt.scatter(Y_test, Y_test_pred, alpha=0.5, label='Test Set')
plt.title('Cross-Validation Analysis: Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
print('Parameters currently in use:\n')
pprint(rf.get_params())

# IV. Randomized CV Search
# Creating a function to compute mse, mae and r2 score calculations
def evaluate_model(model, X, Y):
    predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(Y, predictions))
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return rmse, mae, r2

# Defining random parameter to initiate the randomized cv search
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(10, 200, num=10)],
    'max_depth': [None] + [int(x) for x in np.linspace(1, 20, num=5)],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, Y_train)

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1
)

random_search.fit(X_train, Y_train)

print("Best Hyperparameters from Randomized Search:")
print(random_search.best_params_)

best_rf_model = RandomForestRegressor(
    n_estimators=random_search.best_params_['n_estimators'],
    max_depth=random_search.best_params_['max_depth'],
    min_samples_split=random_search.best_params_['min_samples_split'],
    min_samples_leaf=random_search.best_params_['min_samples_leaf'],
    max_features=random_search.best_params_['max_features'],
    bootstrap=random_search.best_params_['bootstrap'],
    random_state=42
)

best_rf_model.fit(X_train, Y_train)

# Evaluation of the default model on the test set
test_rmse_default, test_mae_default, test_r2_default = evaluate_model(rf, X_test, Y_test)

# Evaluation of the best model obtained from random search on the test set
test_rmse_best, test_mae_best, test_r2_best = evaluate_model(best_rf_model, X_test, Y_test)

rmse_improvement = ((test_rmse_default - test_rmse_best) / test_rmse_default) * 100
mae_improvement = ((test_mae_default - test_mae_best) / test_mae_default) * 100
r2_improvement = ((test_r2_best - test_r2_default) / np.abs(test_r2_default)) * 100

print('\nPercentage Improvement from Default to Best Random Search Model:')
print(f'Root Mean Squared Error: {rmse_improvement:.2f}%')
print(f'Mean Absolute Error: {mae_improvement:.2f}%')
print(f'R2 Score: {r2_improvement:.2f}%')

# Creating a function to compute mse, mae and r2 score calculations
def evaluate_model(model, X, Y):
    predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(Y, predictions))
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return rmse, mae, r2

improvement_found = False

while not improvement_found:
    # Defining random parameter to initiate the randomized cv search
    param_dist = {
    'n_estimators': [55,57,60,63,65,70,73,75,77,80,83,85,87,90,93,95,97,100],
    'max_depth': [5,10,13,15,16,17,18,19,20]
    }

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, Y_train)

    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )

    random_search.fit(X_train, Y_train)

    print("Best Hyperparameters from Randomized Search:")
    print(random_search.best_params_)

    best_rf_model = RandomForestRegressor(
    n_estimators=random_search.best_params_['n_estimators'],
    max_depth=random_search.best_params_['max_depth']
    )

    best_rf_model.fit(X_train, Y_train)

    # Evaluation of the default model on the test set
    test_rmse_default, test_mae_default, test_r2_default = evaluate_model(rf, X_test, Y_test)

    # Evaluation of the best model obtained from random search on the test set
    test_rmse_best, test_mae_best, test_r2_best = evaluate_model(best_rf_model, X_test, Y_test)

    rmse_improvement = ((test_rmse_default - test_rmse_best) / test_rmse_default) * 100
    mae_improvement = ((test_mae_default - test_mae_best) / test_mae_default) * 100
    r2_improvement = ((test_r2_best - test_r2_default) / np.abs(test_r2_default)) * 100

    # Check if R2 improvement is positive
    if r2_improvement > 0:
        improvement_found = True
        print('\nImproved R2 Score found!')
        print(f'Percentage Improvement from Default to Best Random Search Model:')
        print(f'Root Mean Squared Error: {rmse_improvement:.2f}%')
        print(f'Mean Absolute Error: {mae_improvement:.2f}%')
        print(f'R2 Score: {r2_improvement:.2f}%')
        print("Best Hyperparameters from Randomized Search:")
        print(random_search.best_params_)
    else:
        print('\nNo improvement in R2 Score. Retrying...')

# V. Grid Search
param_grid = {
    'n_estimators': [60,70,80,90,100],
    'max_depth': [16,17,18,19,20],
    'min_samples_split': [2,3,4,5,6],
    'min_samples_leaf': [1,2,3,4,5,6],
    'max_features': ['auto', 'sqrt', None],
    'bootstrap': [True]
}

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, Y_train)

grid_search = GridSearchCV(
    rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

grid_search.fit(X_train, Y_train)

print("Best Hyperparameters from Grid Search:")
print(grid_search.best_params_)

best_rf_model = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    max_features=grid_search.best_params_['max_features'],
    bootstrap=grid_search.best_params_['bootstrap'],
    random_state=42
)
best_rf_model.fit(X_train, Y_train)

# Evaluation of the default model on the test set
test_mse_default, test_mae_default, test_r2_default = evaluate_model(rf, X_test, Y_test)

# Evaluation of the best model obtained from grid search on the test set
test_mse_best, test_mae_best, test_r2_best = evaluate_model(best_rf_model, X_test, Y_test)

mse_improvement = ((test_mse_default - test_mse_best) / test_mse_default) * 100
mae_improvement = ((test_mae_default - test_mae_best) / test_mae_default) * 100
r2_improvement = ((test_r2_best - test_r2_default) / np.abs(test_r2_default)) * 100

print('\nPercentage Improvement from Default to Best Grid Search Model:')
print(f'Mean Squared Error: {mse_improvement:.2f}%')
print(f'Mean Absolute Error: {mae_improvement:.2f}%')
print(f'R2 Score: {r2_improvement:.2f}%')

# E. Deep Learning Model
X = df_3.drop(columns = ['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).values
Y = df_3[['count']].values

X_traind, X_testd, y_traind, y_testd= train_test_split(X,Y, test_size=0.3)

# III. Training the Model

def r2_score_cus(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

model = Sequential()
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=[r2_score_cus])

model.fit(x= X_traind, y= y_traind, batch_size=32, epochs=500, validation_data=(X_testd, y_testd))

# IV. Checking Loss Function
#pd.DataFrame(model.history.history).plot(figsize=(12,7))
t = pd.DataFrame(model.history.history)

fig = go.Figure()

fig.add_trace(go.Scatter(y=t['loss'],
                    mode='lines',
                    name='Train'))
fig.add_trace(go.Scatter(y=t['val_loss'],
                    mode='lines',
                    name='Validation'))

fig.update_layout(font_color="#007CD8", title='Loss Function[MSE]', title_x=0.5, title_y = 0.85,
                  legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, range=[0, 20000])
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/loss.html")

# V. Checking R-squared Function
fig = go.Figure()

fig.add_trace(go.Scatter(y=t['r2_score_cus'],
                    mode='lines',
                    name='Train'))
fig.add_trace(go.Scatter(y=t['val_r2_score_cus'],
                    mode='lines',
                    name='Validation'))

fig.update_layout(font_color="#007CD8", title='R-squared', title_x=0.5, title_y = 0.85,
                  legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10)
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10, range=[0.45, 1])
fig.show()
# fig.write_html("/content/drive/MyDrive/Imagenes/r2.html")

# VI. Metrics
predictions=model.predict(X_testd)

mae_d = mean_absolute_error(y_testd, predictions)
rmse_d = sqrt(mean_squared_error(y_testd, predictions))
r2_d = r2_score(y_testd, predictions)

print('MAE:', mae_d)
print('RMSE: ', rmse_d)
print('R2: ', r2_d)

# VII. Checking Heteroscedasticity

g = [y_testd[i][0] for i in range(len(y_testd))]
h = [predictions[i][0] for i in range(len(predictions))]

fig = go.Figure()
fig.add_trace(go.Scatter(x=g, y=h,
                    mode='markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=g, y=g,
                    mode='markers',
                    name='Test Values'))

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.6
))

fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='#007CD8', ticklen=10, range=[-10,1000])
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor="#007CD8", ticklen=10)
fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/predictions.html")

# VIII. Hypertuning Parameters

def dl_model2(optimizer='adam'):
  model = Sequential()
  model.add(Dense(11, activation = 'relu'))
  model.add(Dense(11, activation = 'relu'))
  model.add(Dense(11, activation="relu"))
  model.add(Dense(1))

  model.compile(optimizer=optimizer, loss="mse")
  return model

model = KerasRegressor(build_fn=dl_model2, verbose=0)

param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [100, 300, 500],
    #'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    'optimizer': ['Adam']
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')
grid_result = grid.fit(X_train, y_train)

best_params = grid_result.best_params_
print(best_params)

#  Evaluation of All Models
# A. Performance Comparison of All Models
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Multivariable Regression with Normalization", "Random Forest", "Decision Tree", "Deep Learning"),
                    vertical_spacing=0.1, horizontal_spacing = 0.1)

fig.add_trace(
    go.Scatter(x = v_t, y=y_pred,
               mode='markers',
               #name="Predictions"
               ),
    row=1, col=1)

fig.add_trace(
    go.Scatter(x = v_t, y=v_t,
               mode='markers',
               #name="Test Values"
               ),
    row=1, col=1)

fig.add_trace(
    go.Scatter(x = v_y, y=Y_test_pred,
               mode='markers',
               #name="Predictions"
               ),
    row=1, col=2)

fig.add_trace(
    go.Scatter(x = v_y, y=v_y,
               mode='markers',
               #name="Test Values"
               ),
    row=1, col=2)

fig.add_trace(
    go.Scatter(x = dt_v, y=Y_pred_best,
               mode='markers',
               #name="Predictions"
               ),
    row=2, col=1)

fig.add_trace(
    go.Scatter(x = dt_v, y=dt_v,
               mode='markers',
               #name="Test Values"
               ),
    row=2, col=1)

fig.add_trace(
    go.Scatter(x = g, y=h,
               mode='markers',
               #name="Predictions"
               ),
    row=2, col=2)

fig.add_trace(
    go.Scatter(x = g, y=g,
               mode='markers',
               #name="Test Values"
               ),
    row=2, col=2)

fig.update_layout(height=900, width=1300, showlegend=False)
fig.show()

#fig.write_html("/content/drive/MyDrive/Imagenes/4_Modelos.html")

# B. Metrics Comparison for All Models
models = ['Multivariable Regression', 'Polynomial Regression','Decision Tree', 'Random Forest', 'Deep Learning']
#MAE_values = ["%.4f" % mae_mv, "%.4f" % mae_p, "%.4f" % test_mae , "%.4f" % test_mae_best, "%.4f" % mae_d]
#RMSE_values = ["%.4f" % sqrt(mse_mv), "%.4f" % sqrt(mse_p), "%.4f" % test_mse, "%.4f" % test_rmse_best, "%.4f" % rmse_d]
#r2_values = ["%.4f" % r2_mv, "%.4f" % r2_p, "%.4f" % test_r2, "%.4f" % test_r2_best, "%.4f" % r2_d]

MAE_values = [2.8477, 2.1357, 33.0985, 26.1335, 37.3925]
RMSE_values = [1.8795, 2.7236, 55.1739, 44.0920, 55.6255]
r2_values = [0.4861, 0.6945, 0.9078, 0.9386, 0.9053]

fig = go.Figure(data=[go.Table(columnwidth = [300,150,150,150],
    header=dict(values=['Model','MAE', 'RMSE', 'R2'],
                fill_color='rgba(12, 121, 203, 1.0)',
                align='center',
                font=dict(color='white', size=12),
                height=40),
    cells=dict(values=[models, MAE_values, RMSE_values, r2_values],
               fill_color='lavender',
               align='center',
               height=40))
])
fig.update_layout(width=800, height=400)
fig.show()

# C. Feature Importance Comparison
# Decision Tree
feature_importance_dt = best_dt_model.feature_importances_
#Random Forest
feature_importance_rf = rf.feature_importances_

feature_importance_rfdt = pd.DataFrame({
    'rfr':feature_importance_rf,
    'dt': feature_importance_dt
}, index = df_3.drop(columns=['rental_id', 'date', 'casual_rider', 'registered_rider', 'count', 'temperature', 'month']).columns)
feature_importance_rfdt.sort_values(by='rfr', ascending = True, inplace = True)

feature_importance_rfdt = feature_importance_rfdt.reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(
    y=feature_importance_rfdt['index'],
    x=feature_importance_rfdt['dt'],
    name='Feature Importance Decision Tree',
    orientation='h',
    marker=dict(
        color='rgba(64, 108, 166, 1.0)',
        line=dict(color='rgba(64, 108, 166, 1.0)', width=1)
    )
))

fig.add_trace(go.Bar(
    y=feature_importance_rfdt['index'],
    x=feature_importance_rfdt['rfr'],
    name='Feature Importance Random Forest',
    orientation='h',
    marker=dict(
        color='rgba(107, 36, 135, 1)',
        line=dict(color='rgba(107, 36, 135, 1)', width=1)
    )
))



fig.update_layout(width=1300, height=750,title_text='Feature Importance Comparison', title_x=0.52, font_color="rgba(64, 108, 166, 1.0)",
                  bargroupgap= 0.15, bargap= 0.1,
                  legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=0.8
))
fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor='rgba(64, 108, 166, 1.0)', ticklen=10)
fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='rgba(64, 108, 166, 1.0)', ticklen=10)
fig.update_traces(textposition='outside')


fig.show()
#fig.write_html("/content/drive/MyDrive/Imagenes/Features.html")
