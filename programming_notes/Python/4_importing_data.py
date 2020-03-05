# See which files are in the directory ---------------------------------------------------------------------------------
! ls

import os
wd = os.getcwd()
os.listdir(wd)


# Importing entire text files ------------------------------------------------------------------------------------------
file = open("data/sometextfile.txt", mode="r")  # r for read only (no writing)
file.read()
file.closed     # Check whether file is closed
file.close()    # Close file

# Read line by line, use context manager
with open("data/sometextfile.txt") as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())


# Import numeric flat files with NumPy ---------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

file = 'data/mnist.csv'
digits = np.loadtxt(file, delimiter=",")
print(type(digits))
# Select and reshape a row
im = digits[22, 1:]
im_sq = np.reshape(im, (28, 28))
# Plot reshaped data
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()

# Select rows or cols, or remove headers
np.loadtxt(file, delimiter="\t", skiprows=1, usecols=[0, 2])


# Working with mixed data types ----------------------------------------------------------------------------------------
# Import numeric data with string headers
file = 'data/seaslug.txt'
data = np.loadtxt(file, delimiter='\t', dtype=str)                      # import all as strings
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)  # remove headers

# genfromtxt() can figure out the type per columns; it creates a structured array
data = np.genfromtxt('data/titanic.csv', delimiter=',', names=True, dtype=None)
data[2]             # row access
data['Survived']    # column access
# np.recfromcsv() that behaves similarly to np.genfromtxt(), except that its default dtype is None
data = np.recfromcsv('data/titanic.csv', delimiter=',', names=True)


# Importing flat files using pandas ------------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv('data/titanic.csv')
df.head()

# Read the first 5 rows of the file into a DataFrame
data = pd.read_csv('data/mnist.csv', nrows = 5, header=None)
# Build a numpy array from the DataFrame
data_array = data.values
type(data_array)

# Import corrupted data: set separator, comment character in file and values to be set to NA
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')


# Importing data from other file types ---------------------------------------------------------------------------------
# Pickle files (native python files)
import pickle
with open('data.pkl', 'rb') as file:  # rb stands for read only for binary files
    d = pickle.load(file)

# Excel file
import pandas as pd
file = 'data/battledeath.xlsx'
xl = pd.ExcelFile(file)
print(xl.sheet_names)

df1 = xl.parse('2004')   # by sheet name
df2 = xl.parse(0)        # by sheet position
print(df1.head())
print(df2.head())

df1 = xl.parse(0, skiprows=1, names=['Country', 'AAM due to War (2002)'])
df2 = xl.parse(1, parse_cols=0, skiprows=1, names=['Country'])
print(df1.head())
print(df2.head())

# SAS file
# Import sas7bdat package
from sas7bdat import SAS7BDAT
with SAS7BDAT('data/sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
print(df_sas.head())

# Plot histogram of DataFrame features
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()

# Stata file
df = pd.read_stata('data/disarea.dta')
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of coutries')
plt.show()

# HDF5 file (Hierarchical Data Format for storing large data)
import numpy as np
import h5py
file = 'data/LIGO_data.hdf5'
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)

# Extracting data from HDF5 file
group = data['strain']
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
strain = data['strain']['Strain'].value
# Set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()

# MATLAB file
import scipy.io
mat = scipy.io.loadmat('data/albeck_gene_expression.mat')
print(type(mat))
# Imported MATLAB files are dictionaries:
# keys = MATLAB variable names
# values = objects assigned to variables
print(mat.keys())
print(type(mat['CYratioCyt']))
print(np.shape(mat['CYratioCyt']))

data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()


# Working with relational databases in Python --------------------------------------------------------------------------
# Querying relational databases in Python
# SELECT * FROM Orders -> select all cols from Orders table
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Chinook.sqlite')  # create a database engine
engine.table_names()                         #  heck table names
con = engine.connect()                       # connection object
rs = con.execute("SELECT * FROM Album")      # result object
df = pd.DataFrame(rs.fetchall())             # fetch all rows; fetchmany(size=5) fetches 5 rows
df.columns() = rs.keys()                     # get correct colnames
con.close()                                  # close connection

# Alternatively, use context manager
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Filtering with WHERE
# SELECT * FROM Customer WHERE Country = 'Canada'
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee WHERE EmployeeId >= 6')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Ordering SQL records
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())

# Querying relational databases directly with pandas
# The same as above, but in one line
df = pd.read_sql_query("SELECT * FROM Album", engine)

# Advanced Querying: exploiting table relationships
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

df = pd.read_sql_query(
    "SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000",
    engine)


