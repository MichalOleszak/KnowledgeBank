# Data ingestion & inspection ------------------------------------------------------------------------------------------
import pandas as pd
airquality = pd.read_csv('data/airquality.csv')

# Head & tail
airquality.head()
airquality.tail()

# Data types
airquality.info()

# Pandas and NumPy working together
import numpy as np
# Create numpy array of DataFrame values
np_vals = airquality.values
# Create new array of base 10 logarithm values
np_vals_log10 = np.log10(np_vals)
# Create numpy array of new DataFrame
df_log10 = np.log10(airquality)
# Print original and new data containers
print(type(np_vals), type(np_vals_log10))
print(type(airquality), type(df_log10))
# Numpy methods can be used on padas dataframes!


# Creating DataFrames from scratch -------------------------------------------------------------------------------------
# Zip lists to build a DataFrame
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]
zipped = list(zip(list_keys, list_values))    # Zips the 2 lists together into one list of (key,value) tuples
data = dict(zipped)                           # Builds a dictionary with the zipped list
df = pd.DataFrame(data)                       # Builds a DataFrame from the dictionary
# Labelling columns
df.columns = ["col_A", "col_B"]
# Building DataFrames with broadcasting
cities = ['Manheim',
          'Preston park',
          'Biglerville']
data = {'state': 'PA', 'city': cities}
df = pd.DataFrame(data)


# Importing & Exporting data -------------------------------------------------------------------------------------------
new_labels = ['year', 'population']
df = pd.read_csv('data/world_population.csv', header=0, names=new_labels)

# Messy data
file_messy = 'data/messy_stock_data.txt'
df1 = pd.read_csv(file_messy)
print(df1.head())
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')
print(df2.head())

# Export
df2.to_csv(file_clean, index=False)
df2.to_excel('file_clean.xlsx', index=False)


# Plotting with pandas -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
df = pd.DataFrame(df2.iloc[0,].convert_objects(convert_numeric=True))[1:]
df.columns = ['Temperature']
df.plot(color='red')
plt.title('Temperature in Austin')
plt.xlabel('Hours since midnight August 1, 2010')
plt.ylabel('Temperature (degrees F)')
plt.show()

# Plot entire dataframe
df = pd.read_csv('data/weather_data_austin_2010.csv').iloc[0:1000, 0:3]
df.plot()
plt.show()

# Plot each column separately
df.plot(subplots=True)
plt.show()

# Plot one column
column_list1 = ['DewPoint']
df[column_list1].plot()
plt.show()

# Plot selected columns only
column_list2 = ['Temperature', 'DewPoint']
df[column_list2].plot()
plt.show()

# Line plots
df = pd.read_csv('data/stock.csv', delimiter=';')
df.plot(x='Month', y=['AAPL', 'IBM'])
plt.title('Monthly stock prices')
plt.ylabel('Price ($US)')
plt.show()

# Scatter plots
df = pd.read_csv('data/auto-mpg.csv')
df.plot(kind='scatter', x='hp', y='mpg')
plt.title('Fuel efficiency vs Horse-power')
plt.xlabel('Horse-power')
plt.ylabel('Fuel efficiency (mpg)')
plt.show()

# Box plots
cols = ['weight', 'mpg']
df[cols].plot(kind='box', subplots=True)
plt.show()

# Histogram, pdf and cdf
df = pd.read_csv('data/tips.csv')
df['fraction'] = df.tip / df.total_bill
fig, axes = plt.subplots(nrows=2, ncols=1)     # This formats the plots such that they appear on separate rows
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()
# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', normed=True, cumulative=True, bins=30, range=(0,.3))
plt.show()


# Summary statistics ---------------------------------------------------------------------------------------------------
df = pd.read_csv('data/percent-bachelors-degrees-women-usa.csv')
print(df.Engineering.min())
print(df.Engineering.max())
# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')
# Plot the average percentage per year
mean.plot()
plt.show()

# Check statistics used by the boxplot
df = pd.read_csv("data/titanic.csv")
print(df.Fare.describe())
df.Fare.plot(kind='box')
# Quantiles, Std
print(df.quantile([0.05, 0.95]))
print(df.std())

# Separating populations with Boolean indexing
df[df['Sex'] == 'male'].count()       # how many males there were?
# Ticket fare vs class
fig, axes = plt.subplots(nrows=1, ncols=3)
df.loc[df['Pclass'] == 1].plot(ax=axes[0], y='Fare', kind='box')
df.loc[df['Pclass'] == 2].plot(ax=axes[1], y='Fare', kind='box')
df.loc[df['Pclass'] == 3].plot(ax=axes[2], y='Fare', kind='box')
plt.show()


# Pandas Time Series ---------------------------------------------------------------------------------------------------
# Using DatetimeIndex
date_list = ['20100101 00:00',
             '20110201 01:00',
             '20120301 02:00']
temperature_list = [20.1, 23.4, 22.2]
my_datetimes = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M')
time_series = pd.Series(temperature_list, index=my_datetimes)

# Partial string indexing and slicing
time_series.loc['2010-01-01 00:00:00']
time_series.loc['February 1st, 2011']
time_series.loc['02/01/2011':'03/01/2012']

# Reindexing the index - joining time series with differing time indices
ts_all_dates = pd.to_datetime(['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04', '2016-07-05', '2016-07-06',
                               '2016-07-07', '2016-07-08', '2016-07-09', '2016-07-10', '2016-07-11', '2016-07-12',
                               '2016-07-13', '2016-07-14', '2016-07-15', '2016-07-16', '2016-07-17'])
ts_weekdays_dates = pd.to_datetime(['2016-07-01', '2016-07-04', '2016-07-05', '2016-07-06', '2016-07-07', '2016-07-08',
                                    '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14', '2016-07-15'])
ts_all = pd.Series(range(0, 17), index=ts_all_dates)
ts_weekdays = pd.Series(range(0, 11), index=ts_weekdays_dates)

ts3 = ts_weekdays.reindex(ts_all.index)
ts4 = ts_weekdays.reindex(ts_all.index, method="ffill")   # forward fill (last observed value)
ts5 = ts_weekdays.reindex(ts_all.index, method="bfill")   # forward fill (next observed value)
ts5 = ts_weekdays.reindex(ts_all.index, method="linear")  # linear fill (linear interpolation)
ts_all + ts_weekdays
ts_all + ts3
ts_all + ts4

# Resampling pandas time series (statistical methods computed over different time intervals)
# Resampling frequencies:
# min, T - minute
# H - hour
# D - day
# B - business day
# W - week
# M - month
# Q - quarter
# A - year
# eg. 2W - biweekly
df = pd.read_csv('data/weather_data_austin_2010.csv', parse_dates=True, index_col='Date')
df["Temperature"].resample('6H').mean()    # Downsample to 6 hour data and aggregate by mean
df["Temperature"].resample('D').count()    # Downsample to daily data and count the number of data points
august = df["Temperature"].loc["August 2010"]    # Extract temperature data for August
august_highs = august.resample("D").max()        # Downsample to obtain only the daily highest temperatures in August

# Smoothing with moving average
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']
smoothed = unsmoothed.rolling(window=24).mean()
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})
august.plot()
plt.show()

# Method chaining and filtering
df = pd.read_csv("data/austin_airport_departure_data_2015_july.csv", parse_dates=True, index_col='Date (MM/DD/YYYY)')
df.columns = df.columns.str.strip()                     # Strip extra whitespace from the column names
dallas = df['Destination Airport'].str.contains("DAL")  # Extract data for which the destination airport is Dallas
daily_departures = dallas.resample("D").sum()           # Compute the total number of Dallas departures each day
stats = daily_departures.describe()                     # Generate the summary statistics for daily Dallas departures

# Time zone conversion
df = pd.read_csv("data/austin_airport_departure_data_2015_july.csv")
df.columns = df.columns.str.strip()
mask = df['Destination Airport'] == "LAX"  # Buid a Boolean mask to filter out all the 'LAX' departure flights
la = df[mask]  # Use the mask to subset the data
times_tz_none = pd.to_datetime(la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'])   # Create a datetime series
times_tz_central = times_tz_none.dt.tz_localize('US/Central')     # Localize the time to US/Central
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')   # Convert the datetimes from US/Central to US/Pacific

# Visualizing pandas time series
# Plot styles:
# colors: b - blue, k - black, g - green, r - red, c - cyan
# markers: . - dot, o - circle, * - star, s - square, + - plus
# lines: - - solid, : - dotted, -: - dashed,
# Raw data
df = pd.read_csv('data/weather_data_austin_2010.csv').iloc[0:746, [0, 3]]
df.plot()
# With datetime indexing
df.Date = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.plot()
# Plotting date ranges with partial indexing
df = pd.read_csv('data/weather_data_austin_2010.csv')
df.Date = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()


# Indexing, slicing, filtering, transforming ---------------------------------------------------------------------------
df.loc['row_label', 'col_label']              # index with labels
df.iloc[2,3 ]                                 # index with positions
df[['col1', 'col2']]                          # separate df with column subset

df['col']                                     # pandas series (1D array with labeled index)
df[['col']]                                   # 1D data frame
df['col'][1:4]                                # part of the column
df.loc[:, 'col1':'col4']                      # all rows, some columns
df.loc['row3':'row12', 'col1':'col4']         # a block of rows and columns
df.iloc[:3, 6:]                               # rows 0, 1, 2, columns 6 until the last one
df.iloc[[0,4,7], 0:2]                         # particular rows, columns 0 and 1
df['a':'b']                                   # rows a to b
df['b':'a':-1]                                # rows b to a (reversed oreder)

df[df.sales >= 100]                           # filter one column sales to be >= 100
df.loc[:, df.all()]                           # keeps only columns with all non-zero values
df.loc[:, df.any()]                           # keeps only columns with any non-zero entries
df.loc[:, df.isnull().any()]                  # keeps only columns which have at least one NaN
df.loc[:, df.notnull().all()]                 # keeps only columns without NaNs
df.dropna(how='any')                          # drop rows with any missings
df.dropna(how='all')                          # drop rows with all missings
df.dropna(thresh=1000, axis='columns')        # drop columns that have more than 1000 missings
df.col_a[df.col_b > 10] += 5                  # increase value of entries in col_a for which col_b > 10 by 5

df.apply(np.mean)                             # get column means

election = pd.read_csv("data/pennsylvania2012_turnout.csv")
red_vs_blue = {'Obama':'blue', 'Romney':'red'}                   # dictionary matching candidates to colours
election['color'] = election.winner.map(red_vs_blue)             # mapping winning candidate to colour in a new variable

# apply and map perform for-loops, so they should be avoided! use vectorized functions instead!
# these are compiled func written in C or Fortran. they are called Universal Functions or UFuncs in NumPy

from scipy.stats import zscore
turnout_zscore = zscore(election['turnout'])
election['turnout_zscore'] = turnout_zscore
print(election.head())


# Hierarchical indexes for multi-dimensional data ----------------------------------------------------------------------
sales = pd.DataFrame({'state': ['CA', 'CA', 'NY', 'NY', 'TX', 'TX'],
                      'month': [1, 2, 1, 2, 1, 2],
                      'eggs': [57, 100, 232, 431, 22, 24],
                      'salt': [34, 33, 56, 22, 52, 61]})
print(sales)

# Set and sort hierarchical index
sales = sales.set_index(['state', 'month']
sales = sales.sort_index()
print(sales)

# Lookup based on the outermost level of MultiIndex
print(sales.loc['NY'])

# Lookup based on multiple index levels
sales.loc['NY', 1]                              # NY, month 1
sales.loc[(['CA', 'TX'], 2), :]                 # CA and TX, month 2
sales.loc[(slice(None), 2), :]                  # all states, month 2


# Pivoting data frames -------------------------------------------------------------------------------------------------
users = pd.DataFrame({'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
                      'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
                      'visitors': [139, 257, 326, 457],
                      'signups': [7, 12, 3, 5]})

visitors = users.pivot(index = 'weekday', columns = 'city', values = 'visitors')
print(visitors)

signups = users.pivot(index = 'weekday', columns = 'city', values = 'signups')
print(signups)

allvars = users.pivot(index = 'weekday', columns = 'city')      # If 'values' not specified, uses all other columns
print(allvars)


# Stacking & unstacking DataFrames -------------------------------------------------------------------------------------
# Stack -> make df longer (move columns into levels of multiIndex)
# Unstack -> make df wider (move some levels of multiIndex into columns)
users2 = users.set_index(['city', 'weekday']).sort_index()

# Unstack users by 'city'
bycity = users2.unstack('city')
print(bycity)

# Stack back to the initial structure
newusers = bycity.stack(level='city')
print(newusers)
print(users2)

# Restoring the index order
print(newusers.swaplevel(0,1).sort_index())
print(users2)


# Melting DataFrames ---------------------------------------------------------------------------------------------------
# Restoring pivoted data to its original form; from wide shape to long shape
# id_vars: columns that should remain in the melted dataframe
# value_vars: columns to convert into values

print(visitors)
# Move the city names from the column labels to values in a single column called 'city'
visitors = users.pivot(index = 'weekday', columns = 'city', values = 'visitors')
visitors_noind = visitors.reset_index()
visitors_melt = pd.melt(visitors_noind, id_vars='weekday', value_name='visitors')

# Melting multiple columns
long_df = pd.melt(users, id_vars=['weekday', 'city'])
print(long_df)

# Obtaining key-value pairs with melt()
users_idx = users.set_index(['city', 'weekday'])
kv_pairs = pd.melt(users_idx, col_level=0)
print(kv_pairs)


# Pivot tables ---------------------------------------------------------------------------------------------------------
# pivot requiores unique index-column pairs to identify values in the new table.
# pivot_table uses reduction to deal with values for the same index-column pair (mean by default)

# Calclate count - two ways, same result
count_by_weekday1 = users.pivot_table(index = 'weekday', aggfunc='count')
count_by_weekday2 = users.pivot_table(index = 'weekday', aggfunc=len)

# Add totals in the margins
signups_and_visitors = users.pivot_table(index='weekday', aggfunc=sum)
print(signups_and_visitors)

signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum, margins=True)
print(signups_and_visitors_total)


# Group by -------------------------------------------------------------------------------------------------------------
titanic = pd.read_csv("data/titanic.csv")

# Number of passengers per ticket class (using survived column) and per class per embarking port
titanic.groupby('Pclass')['Survived'].count()
titanic.groupby(['Embarked', 'Pclass'])['Survived'].count()

# Groupby and aggregations
aggregated = titanic.groupby('Pclass')[['Age','Fare']].agg(['max', 'median'])

# Maximum age in each class
print(aggregated.loc[:, ('Age','max')])
# Median fare in each class
print(aggregated.loc[:, ('Fare', 'median')])

# Different aggregation for each variable, also user-defined functions
gapminder = pd.read_csv('data/gapminder_tidy.csv', index_col=['Year','region','Country']).sort_index()
by_year_region = gapminder.groupby(['Year', 'region'])
# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()
# Create the dictionary
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}
# Compute the total population, spread of per capita GDP values and average child mortality rate
aggregated = by_year_region.agg(aggregator)

# Aggregation on transformed dateime index
sales = pd.read_csv('data/sales-feb-2015.csv', index_col='Date', parse_dates=True)
by_day = sales.groupby(sales.index.strftime('%a'))                        # Transforms index to weekdays
units_sum = by_day['Units'].sum()

# Groupby and transformation
from scipy.stats import zscore
zscores_by_region = gapminder.groupby('region')['life', 'fertility'].transform(zscore)
outliers = (zscores_by_region['life'] < -3) | (zscores_by_region['fertility'] > 3)
gm_outliers = gapminder.loc[outliers]

# Imputation by group
# Fill in passenger's age with median age od their gender and pclass
by_sex_class = titanic.groupby(['Sex', 'Pclass'])
def impute_median(series):
    return series.fillna(series.median())
titanic.Age = by_sex_class.Age.transform(impute_median)

# Groupby and filtering
# Remove entries from companies that purchased less than 35 Units in the whole month
sales = pd.read_csv('data/sales-feb-2015.csv', index_col='Date', parse_dates=True)
by_company = sales.groupby('Company')
by_com_sum = by_company['Units'].sum()
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)

# Filtering and grouping with map()
# Compute survival rates for passengers by age (above 10 or not) and pclass
under10 = (titanic['Age'] < 10).map({True:'under 10', False:'over 10'})
survived_mean = titanic.groupby([under10, 'Pclass']).Survived.mean()       # groupby by both a series and a column!
print(survived_mean)

# idxmax, idxmin - row or col label where max/min value is located
maxlab = sales['Units'].idxmax()
sales.loc[maxlab]

minlab = sales['Units'].idxmin()
sales.loc[minlab]

# for col indexes: idmax(axis='columns')


# Importing multiple files into one dataframe --------------------------------------------------------------------------
filenames = ['file1.csv', 'file2.csv', 'file3.csv']
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))


# Appending & concatenating Series and DataFrames ----------------------------------------------------------------------
jan = pd.read_csv("data/sales-jan-2015.csv", parse_dates=True, index_col='Date')
feb = pd.read_csv("data/sales-feb-2015.csv", parse_dates=True, index_col='Date')
mar = pd.read_csv("data/sales-mar-2015.csv", parse_dates=True, index_col='Date')

# Concatenating pandas Series along row axis
units = []
# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month.Units)
# Concatenate the list
quarter1 = pd.concat(units, axis='rows')
# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Appending DataFrames with ignore_index
names_1881 = pd.read_csv('data/names1881.csv')
names_1981 = pd.read_csv('data/names1981.csv')
combined_names = names_1881.append(names_1981, ignore_index=True)
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)

# Concatenating pandas DataFrames along column axis
# The function pd.concat() can concatenate DataFrames horizontally as well as vertically (vertical is the default).
# To make the DataFrames stack horizontally, you have to specify the keyword argument axis=1 or axis='columns'.
combined = pd.concat([df1, df2], axis='columns')

# Concatenating vertically to get MultiIndexed rows & slicing
medal_types = ['bronze', 'silver', 'gold']
medals = []
for medal in medal_types:
    file_name = "data/Summer Olympic medals/%s_top5.csv" % medal
    medal_df = pd.read_csv(file_name, index_col="Country")
    medals.append(medal_df)

print(medals[1])
print(medals[2])

medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'], axis=0)
print(medals)

medals_sorted = medals.sort_index(level=0)
print(medals_sorted.loc[('bronze','Germany')])    # slicing on both multiindex levels
print(medals_sorted.loc['silver'])                # slicing on the outer multiindex level
idx = pd.IndexSlice                               # A slicer is required when slicing on the inner level of a MultiIndex
print(medals_sorted.loc[idx[:,'United Kingdom'], :])  #slicing on the inner level

# Concatenating horizontally to get MultiIndexed columns & slicing
dataframes = [pd.read_csv("data/feb-sales-Hardware.csv", index_col="Date", parse_dates=True),
              pd.read_csv("data/feb-sales-Software.csv", index_col="Date", parse_dates=True),
              pd.read_csv("data/feb-sales-Service.csv", index_col="Date", parse_dates=True)]
february = pd.concat(dataframes, keys=['Hardware', 'Software', 'Service'], axis=1)

# Slice rows between Feb. 2, 2015 to Feb. 8, 2015 from columns under 'Company'
february.loc['Feb 2, 2015':'Feb 8, 2015', idx[:, 'Company']]

# Concatenating DataFrames from a dict & slicing
jan = pd.read_csv("data/sales-jan-2015.csv", parse_dates=True, index_col='Date')
feb = pd.read_csv("data/sales-feb-2015.csv", parse_dates=True, index_col='Date')
mar = pd.read_csv("data/sales-mar-2015.csv", parse_dates=True, index_col='Date')

month_list = [('january', jan), ('february', feb), ('march', mar)]    # list of tuples
month_dict = dict()                                                   # empty dictionary

for month_name, month_data in month_list:
    month_dict[month_name] = month_data.groupby('Company').sum()

sales = pd.concat(month_dict)
print(sales)

# Slice: all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])

# Inner & outer joins
bronze = pd.read_csv("data/Summer Olympic medals/bronze_top5.csv", index_col='Country')
silver = pd.read_csv("data/Summer Olympic medals/silver_top5.csv", index_col='Country')
gold = pd.read_csv("data/Summer Olympic medals/gold_top5.csv", index_col='Country')
medal_list = [bronze, silver, gold]
# inner join - only countries in all dfs (intersection)
medals_inner = pd.concat(medal_list, keys=['bronze', 'silver', 'gold'], join='inner', axis=1)
print(medals_inner)
# outer join - all countries (union)
medals_outer = pd.concat(medal_list, keys=['bronze', 'silver', 'gold'], join='outer', axis=1)
print(medals_outer)


# Merging Data Frames --------------------------------------------------------------------------------------------------
bronze = pd.read_csv("data/Summer Olympic medals/bronze_top5.csv")
silver = pd.read_csv("data/Summer Olympic medals/silver_top5.csv")
gold = pd.read_csv("data/Summer Olympic medals/gold_top5.csv")
silver.columns = ["ctry", "Total"]

# inner is the default merge
inner_merged_1 = pd.merge(bronze, gold, on='Country', suffixes=['_bronze', '_gold'], how='inner')
# merging on columns with different names - both columns retained
inner_merged_2 = pd.merge(bronze, silver, left_on='Country', right_on="ctry", suffixes=['_bronze', '_silver'])

# Ordered merges
austin = pd.DataFrame({"date": ["2016-01-01", "2016-02-08", "2016-01-17"],
                       "rating": ["Cloudy", "Cloudy", "Sunny"]})
houston = pd.DataFrame({"date": ["2016-01-04", "2016-01-01", "2016-03-01"],
                        "rating": ["Rainy", "Cloudy", "Sunny"]})
# Merge with date sorted
tx_weather_suff = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus', '_hus'])
print(tx_weather_suff)
# Fill in missings with a forward fill
tx_weather_ffill = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus', '_hus'], fill_method='ffill')
print(tx_weather_ffill)

# Merge_alsof
# pd.merge_asof() function will also merge values in order using the on column, but for each row in the left DataFrame,
# only rows from the right DataFrame whose 'on' column values are less than the left value will be kept. This function
# can be used to align disparate datetime frequencies without having to first resample.
oil = pd.read_csv("data/oil_price.csv", parse_dates=['Date'])
auto = pd.read_csv("data/automobiles.csv", parse_dates=['yr'])

# Merge monthly oil prices (US dollars) into a full automobile fuel efficiency dataset.
# These datasets will align such that the first price of the year will be broadcast into the rows of the automobiles
# DataFrame. This is considered correct since by the start of any given year, most automobiles for that year will have
# already been manufactured.
merged = pd.merge_asof(auto, oil, left_on="yr", right_on="Date")