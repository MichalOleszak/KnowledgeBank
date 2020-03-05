# Customizing plots ----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
year = np.array([1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980,
       1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
       1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
       2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011])
physical_sciences = np.array([13.8, 14.9, 14.8, 16.5, 18.2, 19.1, 20. , 21.3, 22.5, 23.7, 24.6,
       25.7, 27.3, 27.6, 28. , 27.5, 28.4, 30.4, 29.7, 31.3, 31.6, 32.6,
       32.6, 33.6, 34.8, 35.9, 37.3, 38.3, 39.7, 40.2, 41. , 42.2, 41.1,
       41.7, 42.1, 41.6, 40.8, 40.7, 40.7, 40.7, 40.2, 40.1])
computer_science = np.array([13.6, 13.6, 14.9, 16.4, 18.9, 19.8, 23.9, 25.7, 28.1, 30.2, 32.5,
       34.8, 36.3, 37.1, 36.8, 35.7, 34.7, 32.4, 30.8, 29.9, 29.4, 28.7,
       28.2, 28.5, 28.5, 27.5, 27.1, 26.8, 27. , 28.1, 27.7, 27.6, 27. ,
       25.1, 22.2, 20.6, 18.6, 17.6, 17.8, 18.1, 17.6, 18.2])

# Multiple plots on single axis
plt.plot(year, physical_sciences, color='blue')
plt.plot(year, computer_science, color='red')
plt.show()

# Plot different line plots on distinct axes
# plt.axes([xlo, ylo, width, height]) creates a set of axes with lower corner at coordinates (xlo, ylo)
# The coordinates and lengths are values between 0 and 1 representing lengths relative to the dimensions of the figure.
plt.axes([0.05, 0.05, 0.425, 0.9])
plt.plot(year, physical_sciences, color='blue')
plt.axes([0.525, 0.05, 0.425, 0.9])
plt.plot(year, computer_science, color='red')
plt.show()

# Subplots - same as above, but automatic
# plt.subplot(m, n, k) builds a grid of dimensions m by n with the kth subplot active
# (subplots are numbered starting from 1 row-wise from the top left corner of the subplot grid)
plt.subplot(1, 2, 1)
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

plt.subplot(1, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

plt.tight_layout()   # improve the spacing between subplots
plt.show()

# Customizing axes
# Zooming in by limiting the axes
plt.plot(year,computer_science, color='red')
plt.plot(year, physical_sciences, color='blue')
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')
plt.xlim(1990, 2010)
plt.ylim(0, 50)
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Instead of using xlim and ylim, one can call them in one go:
plt.axis([1990, 2010, 0, 50])

# Legends
plt.plot(year, computer_science, color='red', label='Computer Science')
plt.plot(year, physical_sciences, color='blue', label="Physical Sciences")
plt.legend(loc="lower center")
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()

# Annotations
plt.plot(year, computer_science, color='red', label='Computer Science')
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]

plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()

# Styles
import matplotlib.pyplot as plt
print(plt.style.available)

plt.style.use('ggplot')
plt.plot(year, computer_science, color='red', label='Computer Science')
plt.show()


# Working with 2D arrays -----------------------------------------------------------------------------------------------
# Generating meshes
import numpy as np
import matplotlib.pyplot as plt
# Generate two 1-D arrays
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)
# Generate 2-D arrays from u and v
X,Y = np.meshgrid(u, v)
# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2))
# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.colorbar()
plt.show()

# Plots overview
plt.subplot(2,2,1)
plt.contour(X, Y, Z)
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)
plt.tight_layout()  # Improve the spacing between subplots
plt.show()

# Modifying colormaps
plt.subplot(3,4,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

plt.subplot(3,4,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

plt.subplot(3,4,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')

plt.subplot(3,4,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')

plt.subplot(3,4,5)
plt.contourf(X,Y,Z,20, cmap='spring')
plt.colorbar()
plt.title('Spring')

plt.subplot(3,4,6)
plt.contourf(X,Y,Z,20, cmap='summer')
plt.colorbar()
plt.title('Summer')

plt.subplot(3,4,7)
plt.contourf(X,Y,Z,20, cmap='jet')
plt.colorbar()
plt.title('Jet')

plt.subplot(3,4,8)
plt.contourf(X,Y,Z,20, cmap='coolwarm')
plt.colorbar()
plt.title('Coolwarm')

plt.subplot(3,4,9)
plt.contourf(X,Y,Z,20, cmap='magma')
plt.colorbar()
plt.title('Magma')

plt.subplot(3,4,10)
plt.contourf(X,Y,Z,20, cmap='Reds')
plt.colorbar()
plt.title('Reds')

plt.subplot(3,4,11)
plt.contourf(X,Y,Z,20, cmap='Greens')
plt.colorbar()
plt.title('Greens')

plt.subplot(3,4,12)
plt.contourf(X,Y,Z,20, cmap='Blues')
plt.colorbar()
plt.title('Blues')

plt.tight_layout()
plt.show()

# Visualizing bivariate distributions ----------------------------------------------------------------------------------
mpg = np.array([18. ,  9. , 36.1, 18.5, 34.3, 32.9, 32.2, 22. , 15. , 17. , 44. ,
       24.5, 32. , 14. , 15. , 13. , 36. , 31. , 32. , 21.5, 19. , 17. ,
       16. , 15. , 23. , 26. , 32. , 24. , 21. , 31.3, 32.7, 15. , 23. ,
       17.6, 28. , 24. , 14. , 18.1, 36. , 29. , 35.1, 36. , 16.5, 16. ,
       29.9, 31. , 27.2, 14. , 32.1, 15. , 12. , 17.6, 25. , 28.4, 29. ,
       30.9, 20. , 20.8, 22. , 38. , 31. , 19. , 16. , 25. , 22. , 26. ,
       13. , 19.9, 11. , 28. , 15.5, 26. , 14. , 12. , 24.2, 25. , 22.5,
       26.8, 23. , 26. , 30.7, 31. , 27.2, 21.5, 29. , 20. , 13. , 14. ,
       38. , 13. , 24.5, 13. , 25. , 24. , 34.1, 13. , 44.6, 20.5, 18. ,
       23.2, 20. , 24. , 25.5, 36.1, 23. , 24. , 18. , 26.6, 32. , 20.3,
       27. , 17. , 21. , 13. , 24. , 17. , 39.1, 14.5, 13. , 20.2, 27. ,
       35. , 15. , 36.4, 30. , 31.9, 26. , 16. , 20. , 18.6, 14. , 25. ,
       33. , 14. , 18.5, 37.2, 18. , 44.3, 18. , 28. , 43.4, 20.6, 19.2,
       26.4, 18. , 28. , 26. , 13. , 25.8, 28.1, 13. , 16.5, 31.5, 24. ,
       15. , 18. , 33.5, 32.4, 27. , 13. , 31. , 28. , 27.2, 21. , 19. ,
       25. , 23. , 19. , 15.5, 23.9, 22. , 29. , 14. , 15. , 27. , 15. ,
       30.5, 25. , 17.5, 34. , 38. , 30. , 19.8, 25. , 21. , 26. , 16.5,
       18.1, 46.6, 21.5, 14. , 21.6, 15.5, 20.5, 23.9, 12. , 20.2, 34.4,
       23. , 24.3, 19. , 29. , 23.5, 34. , 37. , 33. , 18. , 15. , 34.7,
       19.4, 32. , 34.1, 33.7, 20. , 15. , 38.1, 26. , 27. , 16. , 17. ,
       13. , 28. , 14. , 31.5, 34.5, 11. , 16. , 31.6, 19.1, 18.5, 15. ,
       18. , 35. , 20.2, 13. , 31. , 22. , 11. , 33.5, 43.1, 25.4, 40.8,
       14. , 29.8, 16. , 20.6, 18. , 33. , 31.8, 13. , 20. , 32. , 13. ,
       23.7, 19.2, 37. , 18. , 19. , 32.3, 18. , 13. , 12. , 36. , 18.2,
       19. , 30. , 15. , 11. , 10. , 16. , 14. , 16.9, 13. , 25. , 21. ,
       21.1, 26. , 28. , 29. , 16. , 26.6, 19. , 32.8, 22. , 19. , 31. ,
       23. , 29.5, 17.5, 19. , 24. , 14. , 28. , 21. , 22.4, 36. , 18. ,
       16.2, 39.4, 30. , 18. , 17.5, 28.8, 22. , 34.2, 30.5, 16. , 38. ,
       41.5, 27.9, 22. , 29.8, 17.7, 15. , 14. , 15.5, 17.5, 12. , 29. ,
       15.5, 35.7, 26. , 30. , 33.8, 18. , 13. , 20. , 32.4, 16. , 27.5,
       23. , 14. , 17. , 16. , 23. , 24. , 27. , 15. , 27. , 28. , 14. ,
       33.5, 39. , 24. , 26.5, 19.4, 15. , 25.5, 14. , 27.4, 13. , 19. ,
       17. , 28. , 22. , 30. , 18. , 14. , 22. , 23.8, 24. , 26. , 26. ,
       30. , 29. , 14. , 25.4, 19. , 12. , 20. , 27. , 22.3, 10. , 19.2,
       26. , 16. , 37.3, 26. , 20.2, 13. , 21. , 25. , 20.5, 37.7, 36. ,
       20. , 37. , 18. , 27. , 29.5, 17.5, 25.1])
hp = np.array([ 88, 193,  60,  98,  78, 100,  75,  76, 130, 140,  52,  88,  84,
       148, 150, 130,  58,  82,  65, 110,  95, 110, 140, 170,  78,  90,
        96,  95, 110,  75, 132, 150,  83,  85,  86,  75, 140, 139,  70,
        52,  60,  84, 138, 180,  65,  67,  97, 150,  70, 100, 180, 129,
        95,  90,  83,  75, 100,  85, 112,  67,  65,  88, 100,  75, 100,
        70, 145, 110, 210,  80, 145,  69, 150, 198, 120,  92,  90, 115,
        95,  75,  76,  67,  71, 115,  84,  91, 150, 215,  67, 175,  60,
       175, 110,  95,  68, 150,  67,  95, 110, 105, 102, 110,  89,  66,
        88,  75,  78, 105,  70, 103,  60, 150,  72, 170,  90, 110,  58,
       152, 145, 139,  83,  69, 150,  67,  80,  71,  46, 105,  90, 110,
       175,  80,  74, 150, 150,  65, 100,  48, 105,  90,  48, 105, 105,
        88, 100,  75, 113, 190,  92,  80, 165, 180,  71,  97,  72, 105,
        90,  75,  88, 155,  68,  90,  84,  87, 112,  87, 125, 108, 142,
        97, 105,  75, 137, 150,  88, 145,  63,  95, 140,  88,  85,  70,
        85, 115,  86,  79, 120, 120,  65, 110, 220, 115, 170, 100,  90,
       225,  85,  65,  97,  90,  90,  49, 110,  70,  92,  53, 100, 190,
        63,  90,  67,  65,  75, 100, 110,  60,  93,  88, 150, 100, 150,
        88, 225,  68,  70, 208, 105,  74,  90, 110,  72,  97,  88,  88,
       129,  85,  86, 150,  70,  48,  77,  65, 175,  90, 150, 110, 130,
        53,  65, 158,  95,  61, 215, 100, 145,  68, 150,  88,  67, 105,
       175, 160,  74, 135, 100,  67, 198, 180, 215, 100, 225, 155, 170,
        81,  85,  95,  80,  92,  70, 149,  84,  97,  52,  72,  85,  52,
        95,  71, 140, 100,  96, 150,  75, 107, 110,  75,  97, 133,  70,
        67, 112, 145, 115,  98,  70,  78, 230,  63,  76, 105,  95,  62,
       165, 165, 160, 190,  95, 180,  78, 120,  80,  75,  68,  67,  95,
       140, 110,  72, 150,  95,  54, 153, 130, 170,  86,  97,  90, 145,
        86,  79, 165,  83,  64,  92,  72, 140, 150,  96, 150,  80, 130,
       100, 125,  90,  94,  76,  90, 150,  97,  85,  81,  78,  46,  84,
        70, 153, 116, 100, 167,  88,  88,  88, 200, 125,  92, 110,  69,
        67,  90, 150,  90,  71, 105,  62,  88, 122,  65,  88,  90,  68,
       110,  88])

# 2D histogram
plt.hist2d(hp, mpg, bins=(20, 20), range=((40, 235), (8, 48)))
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()

# Hexagonal histogram
plt.hexbin(hp, mpg, gridsize=(15, 12), extent=(40, 235, 8, 48))
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()

# Working with images --------------------------------------------------------------------------------------------------
# Load and show image
img = plt.imread('data/astronaut.png')
print(img.shape)
# Display the image
plt.imshow(img)
# Hide the axes
plt.axis('off')
plt.show()

# Pseudocolor plot from image data
# Compute the sum of the red, green and blue channels
intensity = img.sum(axis=2)
print(intensity.shape)
plt.imshow(intensity, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()

# Extent and aspect
# When using plt.imshow() to display an array, the default behavior is to keep pixels square so that the height to width
# ratio of the output matches the ratio determined by the shape of the array. In addition, by default, the x- and
# y-axes are labeled by the number of samples in each direction.
#
# The ratio of the displayed width to height is known as the image aspect and the range used to label the x- and y-axes
# is known as the image extent. The default aspect value of 'auto' keeps the pixels square and the extents are
# automatically computed from the shape of the array if not specified otherwise.

# Load the image into an array: img
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)

plt.tight_layout()
plt.show()

# Rescaling pixel intensities
image = plt.imread('data/Unequalized_Hawkes_Bay_NZ.png')
# Sometimes, low contrast images can be improved by rescaling their intensities.
# This image has no pixel values near 0 or near 255 (the limits of valid intensities).
# Translate and stretch the pixel intensities so that the intensities of the new image fill the range from 0 to 255.

# Extract minimum and maximum values from the image
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))
# Rescale the pixels
rescaled_image = 256*(image-pmin)/(pmax-pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." %
      (rescaled_image.min(), rescaled_image.max()))
# Display both images
plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)
plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image)
plt.show()


# Statistical plots with Seaborn ---------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
auto = pd.read_csv('data/auto-mpg.csv')

# Regression
sns.lmplot(x='weight', y='hp', data=auto)
plt.show()

# Regression residuals
sns.residplot(x='hp', y='mpg', data=auto, color='green')
plt.show()

# Higher-order regressions
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='order 1')
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='green', order=2, label='order 2')
plt.legend(loc='upper right')
plt.show()

# Grouping linear regressions by hue (one regression per level of a factor variable)
sns.lmplot(x='weight', y='hp', data=auto, hue='origin', palette='Set1')
plt.show()

# Grouping linear regressions by row or column (as above, in separate plots)
sns.lmplot(x='weight', y='hp', data=auto, row='origin', palette='Set1')
plt.show()

# Visualizing univariate distributions

# Strip plots
plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto, jitter=False)
plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, jitter=True, size=3)
plt.show()

# Swarm plots
plt.subplot(2,1,1)
sns.swarmplot(x='cyl', y='hp', data=auto)
plt.subplot(2,1,2)
sns.swarmplot(x='hp', y='cyl', data=auto, hue='origin', orient='h')
plt.show()

# Violin plots
plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)
plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, color='lightgray', inner=None)
sns.stripplot(x='cyl', y='hp', data=auto, jitter=True, size=1.5)
plt.show()

# Visualizing multivariate distributions

# Joint plots
sns.jointplot('hp', 'mpg', data=auto)
plt.show()

sns.jointplot('hp', 'mpg', data=auto, kind='hex')
plt.show()

sns.jointplot('hp', 'mpg', data=auto, kind='reg')
plt.show()

sns.jointplot('hp', 'mpg', data=auto, kind='resid')
plt.show()

sns.jointplot('hp', 'mpg', data=auto, kind='kde')
plt.show()

# Pairwise plots
# Only numeric columns are included
auto2 = auto[['hp', 'mpg', 'origin']]
sns.pairplot(auto2)
plt.show()

sns.pairplot(auto2, hue='origin', kind='reg')
plt.show()

# Heatmaps
auto3 = auto[['mpg', 'hp', 'weight', 'accel', 'displ']]
corr_matrix = auto3.corr()
sns.heatmap(corr_matrix)


# Visualizing time series ----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
stocks = pd.read_csv('data/stocks.csv', parse_dates=True, index_col='Date')
aapl = stocks[['AAPL']]
ibm = stocks[['IBM']]
csco = stocks[['CSCO']]
msft = stocks[['MSFT']]

# Multiple time series on common axes
plt.plot(aapl, color='blue', label='AAPL')
plt.plot(ibm, color='green', label='IBM')
plt.plot(csco, color='red', label='CSCO')
plt.plot(msft, color='magenta', label='MSFT')
plt.legend(loc='upper left')
plt.xticks(rotation=60)
plt.show()

# Multiple time series slices
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')
view = aapl['2007':'2008']    # Slice aapl from '2007' to '2008' inclusive
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()
plt.show()

# Plotting an inset view
plt.plot(aapl, color='blue')
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

view = aapl['2007-11':'2008-04']
plt.axes([0.25, 0.5, 0.35, 0.35])
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()

# Plotting moving averages
mean_30 = aapl.rolling(window=30).mean()
mean_75 = aapl.rolling(window=75).mean()
mean_125 = aapl.rolling(window=125).mean()
mean_250 = aapl.rolling(window=250).mean()

plt.subplot(2,2,1)
plt.plot(mean_30, color='green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

plt.subplot(2,2,2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

plt.subplot(2, 2, 3)
plt.plot(mean_125, color='magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

plt.subplot(2,2,4)
plt.plot(mean_250, color='cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')

plt.show()

# Plotting moving standard deviations

std_30 = aapl.rolling(window=30).std()
std_75 = aapl.rolling(window=75).std()
std_125 = aapl.rolling(window=125).std()
std_250 = aapl.rolling(window=250).std()

plt.plot(std_30, color='red', label='30d')
plt.plot(std_75, color='cyan', label='75d')
plt.plot(std_125, color='green', label='125d')
plt.plot(std_250, color='magenta', label='250d')
plt.legend(loc='upper left')
plt.title('Moving standard deviations')
plt.show()


# Histogram equalization in images -------------------------------------------------------------------------------------
# (sharpening the image by spreading out pixel intensities so that subtle contrasts are enhanced)
image = plt.imread('data/Unequalized_Hawkes_Bay_NZ.jpg')

# Extracting a histogram from a grayscale image
# For grayscale images, various image processing algorithms use an image histogram. Recall that an image is
# a two-dimensional array of numerical intensities. An image histogram, then, is computed by counting the occurences
# of distinct pixel intensities over all the pixels in the image.
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')
# Flatten the image into 1 dimension: pixels
pixels = image.flatten()
# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,255))
plt.title('Normalized histogram')
plt.hist(pixels, bins=64, range=(0,256), normed=True, color='red', alpha=0.4)
# Display the plot
plt.show()


# Cumulative Distribution Function from an image histogram
# Display image in top subplot using color map 'gray'
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')
# Flatten the image into 1 dimension: pixels
pixels = image.flatten()
# Display a histogram of the pixels in the bottom subplot
plt.subplot(2, 1, 2)
pdf = plt.hist(pixels, bins=64, range=(0, 256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')
# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0, 256),
               normed=True, cumulative=True,
               color='blue', alpha=0.4)
# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0, 256))
plt.grid('off')
plt.title('PDF & CDF (original image)')
plt.show()

# Histogram equalization is an image processing procedure that reassigns image pixel intensities. The basic idea is to
# use interpolation to map the original CDF of pixel intensities to a CDF that is almost a straight line. In essence,
# the pixel intensities are spread out and this has the practical effect of making a sharper, contrast-enhanced image.
# This is particularly useful in astronomy and medical imaging to help us see more features.

# Equalizing an image histogram
image = plt.imread('data/Unequalized_Hawkes_Bay_NZ.jpg')
# Flatten the image into 1 dimension: pixels
pixels = image.flatten()
# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)
# Reshape new_pixels as a 2-D array: new_image
new_image = np.reshape(new_pixels, newshape=image.shape)
# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image, cmap='gray')
# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')
# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid('off')
# Add title
plt.title('PDF & CDF (equalized image)')
# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
plt.show()



# Extracting histograms from a color image -----------------------------------------------------------------------------
image = plt.imread('data/helix_nebula_jpg.jpg')
# Display image in top subplot
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image)
# Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()
# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2,1,2)
plt.title('Histograms from color image')
plt.xlim((0,256))
plt.hist(red_pixels, bins=64, normed=True, color='red', alpha=0.2)
plt.hist(blue_pixels, bins=64, normed=True, color='blue', alpha=0.2)
plt.hist(green_pixels, bins=64, normed=True, color='green', alpha=0.2)
# Display the plot
plt.show()


# Extracting bivariate histograms from a color image -------------------------------------------------------------------
image = plt.imread('data/helix_nebula_jpg.jpg')

# Extract RGB channels and flatten into 1-D array
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(x=red_pixels, y=green_pixels, bins=(32,32))

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(x=green_pixels, y=blue_pixels, bins=(32,32))

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(x=blue_pixels, y=red_pixels, bins=(32,32))

# Display the plot
plt.show()