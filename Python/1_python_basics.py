# Basics ---------------------------------------------------------------------
## String methods
mystring = "poolhouse"
mystring.upper()
mystring.count("o")

## List methods
mylist = [11.25, 18.0, 20.0, 10.75, 9.50]
mylist.index(20)
mylist.count(14.5)
mylist.append(24.5)
mylist.reverse()

## Importing packages & functions
pip3 install numpy
import numpy
import numpy as np
from numpy import array
from numpy import array as my_fun_name


# NumPy ----------------------------------------------------------------------
## 2D arrays
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],           
            [188, 75.2]]
np_baseball = np.array(baseball)
np_baseball.shape
np_baseball[49,:]   # 50th row
np_baseball[:,1]    # 2nd col
np_baseball[123,0]  # 124th row, 1st col

## Statistics
np.mean(x)
np.median(x)
np.std(x)
np.corrcoef(x,y)

## Logical operators with NumPy
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
np.logical_or(my_house > 18.5,
			  my_house < 10)
np.logical_and(my_house < 11,
			   your_house < 11)


# Matplotlib -----------------------------------------------------------------
import matplotlib.pyplot as plt

plt.plot(x, y)                  # line plot
plt.scatter(x, y)               # scatter plot
			s = size, 
			c = color,
			alpha = transp)           
plt.hist(x, bins = 10)          # histogram
                                
plt.ylabel('ylab')              # ylab
plt.xscale('log')               # show x axis on the logarithmic scale
plt.title('title')              # add title
plt.text(xval, yval, 'text')    # add text
plt.grid(True)                  # add grids
 
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']
plt.xticks(tick_val, tick_lab)  # add ticks to the x axis

plt.show()                      # display the plot
plt.clf()                       # clean up the plot (can start afresh)


# Dictionaries ---------------------------------------------------------------
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

ind_ger = countries.index("germany")
capitals[ind_ger]

europe = {
    'spain':'madrid',
    'france':'paris',
    'germany':'berlin',
    'norway':'oslo'
}
europe.keys()               # show keys
europe['norway']            # print value using key
europe['italy'] = 'rome'    # add value to the dictionary
del(europe['australia'])    # remove element from the dictionary

## Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
		  
data = {'capital':'rome', 'population':59.83}   # Create sub-dictionary data
europe['italy'] = data                          # Add data to europe under key 'italy'


# Pandas ---------------------------------------------------------------------
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr    = [True, False, False, False, True, True, True]
cpc   = [809, 731, 588, 18, 200, 70, 45]
dict  = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars  = pd.DataFrame(dict)

## Specify row labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels

## Import from CSV
cars = pd.read_csv('cars.csv')
cars = pd.read_csv('cars.csv', index_col = 0)   # 1st col used as row labels

## Indexing with square brackets
cars['country']                      # country column as Pandas Series
cars[['country']]                    # country column as Pandas DataFrame
cars[['country', 'drives_right']]    # DataFrame with country and drives_right columns
cars[0:3]                            # first 3 observations
cars[3:6]                            # fourth, fifth and sixth observation

## Indexing with loc and iloc
cars.loc[['JAP']]                    # observation for Japan
cars.loc[['AUS', 'EG']]              # observations for Australia and Egypt
cars.loc[['MOR'], ['drives_right']]  # drives_right value of Morocco
cars.iloc[[4,5], [1,2]]              # same as above, with iloc
cars.loc[:, "drives_right"]          # column as Series
cars.loc[:, ["drives_right"]])       # column as DataFrame

## Filtering pd DataFrames
cars[cars['drives_right']]
cars[np.logical_and(cars['cars_per_cap'] > 100,
                    cars['cars_per_cap'] < 500)]


# Conditional Statements -----------------------------------------------------
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")


# Loops ----------------------------------------------------------------------
## While loop
error = 50.0
while error > 1 :
    error = error / 4
    print(error)

## For loop
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
for a in areas :
    print(a)
for index, a in enumerate(areas) :
    print("room " + str(index) + ": " + str(a))

## Looping over dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
for key, value in europe.items():
    print("the capital of " + key + " is " + value)
	
## Looping over NumPy array
for i in np.nditer(np_baseball):
    print(i)
	
## Looping over Pandas DataFrames
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))
	
## Loop adding new variable "COUNTRY" equal to uppercase of "country"
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = (row["country"]).upper()

## Add new variable using apply
cars["COUNTRY"] = cars["country"].apply(str.upper)


# Generating random numbers --------------------------------------------------
np.random.seed(123)      # Set seed
np.random.rand()         # Generate random float in (0, 1)
np.random.randint(1,7)   # Generate random integer between in (0, 6)


# Writing functions ----------------------------------------------------------
