# Exploratory data analysis --------------------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv('data/dob_job_application_filings_subset.csv')
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()                              # Summary statistics for numeric variables
df['Borough'].value_counts(dropna=False)   # Frequency counts for categorical data

# Visual exploratory data analysis
import matplotlib.pyplot as plt
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)   # Histogram
plt.show()
df.boxplot(column='initial_cost', by='Borough', rot=90)                      # Boxplot
plt.show()
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)         # Scatterplot
plt.show()


# Tidying data for analysis --------------------------------------------------------------------------------------------
# Reshaping data using melt
# Melting data is the process of turning columns of your data into rows of data
#  - id_vars represent the columns of the data you do not want to melt (i.e., keep it in its current shape)
#  - value_vars represent the columns you do wish to melt into rows
airquality = pd.read_csv('data/airquality.csv')
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')


# Pivoting data (the opposite of melting: turn each unique value of a variable and turn it into a column)
#  - index parameter is used to specify the columns that you don't want pivoted
#  - columns (the name of the column you want to pivot)
#  - values (the values to be used when the column is pivoted)
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')
# Resetting the index, to get back the initial data frame
airquality_pivot = airquality_pivot.reset_index()


# Pivoting duplicated values
import numpy as np
airquality_dup = airquality_melt.append(airquality_melt)
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading',
                                              aggfunc=np.mean)
airquality_pivot = airquality_pivot.reset_index()


# Multiple information in one column
# Gender (m/f) and age group are stored together in the column name
tb = pd.read_csv('data/tb.csv')

tb_melt = pd.melt(tb, id_vars=['country', 'year'])
tb_melt['gender'] = tb_melt.variable.str[0]
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Variable stored in column with a delimiter
ebola = pd.read_csv('data/ebola.csv')

ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
ebola_melt['str_split'] = ebola_melt['type_country'].str.split('_')
ebola_melt['type'] = ebola_melt['str_split'].str.get(0)
ebola_melt['country'] = ebola_melt['str_split'].str.get(1)


# Combining data for analysis ------------------------------------------------------------------------------------------
# Concatinating
uber = pd.read_csv('data/nyc_uber_2014.csv')
uber1 = uber.loc[0:148, ]
uber2 = uber.loc[148:296, ]

row_concat = pd.concat([uber1, uber2])
col_concat = pd.concat([uber.Lat, uber.Lon], axis=1)

# Globbing (pattern matching for files names)
import glob
all_csv_files = glob.glob('data/*.csv')

frames = []
for csv in all_csv_files:
    df = pd.read_csv(csv)
    frames.append(df)
uber = pd.concat(frames)

# Merging
merged_df = pd.merge(left=df1, right=df2, left_on='key1', right_on='key2')

# Merging: many-to-one / one-to-many:
# rows from one of the data frames will be replicated

# Merging many-to-many (both data frames have keys with duplicates):
# for each duplicated key, every pairwise combination will be created


# Cleaning data --------------------------------------------------------------------------------------------------------
tips = pd.read_csv('data/tips.csv')
tips.info()

# Converting data types
tips.sex = tips.sex.astype('category')
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')  # Coerces strings to NaN

# Regular expressions
#  17     - \d             - any digit
#  17     - \d*            - any digit, match zero or more times
# $17     - \$\d*          - escape dollar sign + any digit (dollar = end of string)
# $17.00  - \$\d*\.\d*     - escape dollar, any digits, escape period, any digits (period - any character)
# $17.00  - \$\d*\.\d{2}   - escape dollar, any digits, escape period, 2 digits
# $17.000 - ^\$\d*\.\d{2}$ - start with $, escape dollar, any digits, escape period, 2 digits
# A       - \A             - capital letter A
# Austra  - \A\w*          - A + an arbitrary number of alphanumeric characters

import re
prog = re.compile('\d{3}-\d{3}-\d{4}')

result = prog.match('123-456-7890')
print(bool(result))

result = prog.match('1123-456-7890')
print(bool(result))

# \d is the pattern required to find digits. This should be followed with a + so that the previous element is matched
# one or more times. This ensures that 10 is viewed as one number and not as 1 and 0.
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
print(matches)
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)
pattern3 = bool(re.match(pattern='\A\w*', string='Australia'))
print(pattern3)


# Apply function
def recode_sex(sex_value):
    if sex_value == 'Male':
        return 1
    elif sex_value == 'Female':
        return 0
    else:
        return np.nan


tips['sex_recode'] = tips.sex.apply(recode_sex)
print(tips.head())

# Lambda functions
# Remove dollar sign using replace and using regex:
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])
print(tips.head())

# Duplicate and missing values
# Drop all duplicate rows:
df = df.drop_duplicates()

# Fill missing values with the mean:
airquality = pd.read_csv('data/airquality.csv')
oz_mean = airquality.Ozone.mean()
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Testing with asserts
# Assert statement returns nothing if it evaluates to True, and an error otherwise

# Assert that there are no missing values
# 1st all returns True/False for all coulmns, the 2nd gathers them into one True/False value
ebola = pd.read_csv('data/ebola.csv')
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
assert (ebola >= 0).all().all()

