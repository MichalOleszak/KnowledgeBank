# Time series classification -------------------------------------------------------------------------------------------
# Find abnormal heartbeats in patients from audio files

# Visualize raw audio data ---------------------------------------------------------------------------------------------
import librosa as lr
from glob import glob
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from librosa.core import stft
from librosa.core import amplitude_to_db
from librosa.display import specshow

# List all the wav files in the folder
audio_files = glob('data/heartbeat_audio' + '/*.wav')
# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq
# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# Prep labels for normal and abnormal heatbeats
labels = pd.read_csv('data/heartbeat_audio/set_a.csv')
labels_train = labels[['fname', 'label']]
labels_train = labels_train[labels_train['label'].notnull()]
labels_train['label'] = ['normal' if label == 'normal' else 'abnormal' for label in labels_train['label']]
labels_train['fname'] = sum([re.findall(r'\d+', i) for i in labels_train['fname']], [])

# Load files (takes long)
audios = list()
times = list()
sfreqs = list()
for file in audio_files:
    a, s = lr.load(file)
    a = a[:3000]
    t = np.arange(0, len(a)) / s
    audios.append(a)
    sfreqs.append(s)
    times.append(t)

file_nums = pd.DataFrame([re.findall(r'\d+', i) for i in audio_files])
file_nums.columns = ["fname"]
dat = pd.merge(file_nums, labels_train, how='left', on="fname")
normal_ind = [i for i, x in enumerate(dat.label) if x == 'normal']
abnormal_ind = [i for i, x in enumerate(dat.label) if x == 'abnormal']

# Example normal and abnormal data frames for visual inspection
normal = pd.DataFrame(
    {'time': times[normal_ind[0]],
     '0': audios[normal_ind[1]],
     '1': audios[normal_ind[2]],
     '2': audios[normal_ind[3]]
    })
normal = normal.set_index(['time'])
abnormal = pd.DataFrame(
    {'time': times[abnormal_ind[0]],
     '0': audios[abnormal_ind[20]],
     '1': audios[abnormal_ind[15]],
     '2': audios[abnormal_ind[3]]
    })
abnormal = abnormal.set_index(['time'])

fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)
# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq
# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T
# Loop through each audio file / ax object and plot
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
plt.show()

# Average across the time dimension of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()

# Creating the features ------------------------------------------------------------------------------------------------
# In audio data, a common way to remove noise is to smooth the data and then half-wave-rectify it so that
# the total amount of sound energy over time is more distinguishable. Summary statistics of thus smoother data
# can be used as features.
all_audios = pd.DataFrame(audios).transpose()
all_audios['time'] = np.arange(0, len(all_audios)) / sfreq
all_audios.set_index('time', inplace=True)
# Plot the raw data first
all_audios.loc[:, 0].plot(figsize=(10, 5))
plt.show()
# Rectify the audio signal
audio_half_wave = all_audios.apply(np.abs)
audio_half_wave.loc[:, 0].plot(figsize=(10, 5))
plt.show()
# Smooth by applying a rolling mean
audio_half_wave_smooth = audio_half_wave.rolling(50).mean()
audio_half_wave_smooth.loc[:, 0].plot(figsize=(10, 5))
plt.show()
# Calculate features from the envelope
means = np.mean(audio_half_wave_smooth, axis=0)
stds = np.std(audio_half_wave_smooth, axis=0)
maxs = np.max(audio_half_wave_smooth, axis=0)
# Create the X and y arrays
labels = pd.read_csv('data/heartbeat_audio/set_a.csv')
X = np.column_stack([means, stds, maxs])
X = X[labels['label'].notnull(), :]
labels_train = labels[['label']]
labels_train = labels_train[labels_train['label'].notnull()]
y = ['normal' if label == 'normal' else 'abnormal' for label in labels_train['label']]
# Fit the model and score on testing data
model = LinearSVC()
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

# Use librosa to compute some tempo and rhythm features for heartbeat data
# Note that librosa functions tend to only operate on numpy arrays instead of DataFrames
# Calculate the tempo of the sounds
tempos = []
for col, i_audio in all_audios.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))
# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)
# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)
# Run the model with these new features
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
X = X[labels['label'].notnull(), :]
# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

# Spectrograms of heartbeat audio
# Spectral engineering is one of the most common techniques in machine learning for time series data. The first step
# in this process is to calculate a spectrogram of sound. This describes what spectral content (e.g., low and high
# pitches) are present in the sound over time.
one_audio = np.array(all_audios.loc[:, 7])
# Calculate short-term Fourier transform
spec = stft(one_audio, hop_length=2**4, n_fft=2**7)
# Convert into decibels
spec_db = amplitude_to_db(spec)
# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, one_audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=2**4)
plt.show()

# There is a lot more information in a spectrogram compared to a raw audio file.
# By computing the spectral features, you have a much better idea of what's going on.
# Calculate the spectral centroid and bandwidth for the spectrogram
spec = abs(spec)
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]
time_spec = np.arange(0, len(centroids))
# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow(spec_db, x_axis='time', y_axis='hz', hop_length=2**4)
ax.plot(time_spec, centroids)
ax.fill_between(time_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()

# Having many spectrograms (one per time series), spectral features can be used in the model as well
# Loop through each spectrogram
bandwidths = []
centroids = []
for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)
    centroids.append(this_mean_centroid)

# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape([-1, 1])
# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


# Predicting data over time --------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from functools import partial
prices = pd.read_csv('data/stocks.csv', date_parser='Date')
prices.set_index(pd.DatetimeIndex(prices['Date']), inplace=True)
prices.drop('Date', axis=1, inplace=True)

# Plot raw data over time
prices.plot()
plt.show()
# Scatter plot of Apple vs Microsoft
prices.plot.scatter('AAPL', 'MSFT')
plt.show()
# Encode time as the color of each datapoint to visualize how the relationship between these two variables changes
prices.plot.scatter('AAPL', 'MSFT', c=prices.index, cmap=plt.cm.viridis, colorbar=False)
plt.show()

# Fitting a simple regression model: predict Apple with other companies; not really good
X = np.array(prices[['IBM', 'CSCO', 'MSFT']])
y = np.array(prices[['AAPL']])
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)
# Visualizing predicted vs observed values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, shuffle=False, random_state=1)
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()

# Real life data is often messy
prices_messy = prices
prices_messy.loc[2000:2500, 'IBM'] = np.nan
prices_messy.loc[300:500, 'AAPL'] = np.nan
prices_messy.loc[3000:3250, 'MSFT'] = np.nan
# Visualize messy data
prices_messy.plot(legend=False)
plt.tight_layout()
plt.show()
# Count the missing values of each time series
missing_values = prices_messy.isna().sum()
print(missing_values)
# Imputing missing values with interpolation
# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):
    # Create a boolean mask for missing values
    missing_values = prices.isna()
    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)
    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()


# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices_messy, interpolation_type)
# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)
# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)

# A function that calculates the percent change of the latest data point from the mean of a window of previous points
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]
    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change


# Calculate the percent change over a rolling window.
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.plot()
plt.show()

# Handling outliers
# A function that replaces outlier data points with the median value from the entire time series.
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series


# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.plot()
plt.show()

# Engineering multiple rolling features at once
prices_perc = prices_perc[['AAPL']]
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]
# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)
# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()

# Percentiles and partial functions
percentiles = [1, 10, 25, 50, 75, 90, 99]
# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]
# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)
# Plot a subset of the result
ax = features_percentiles.plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

# Using "date" information
# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.weekday
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month
print(prices_perc)


# Creating features from the past --------------------------------------------------------------------------------------
prices = pd.read_csv('data/stocks.csv', date_parser='Date')
prices.set_index(pd.DatetimeIndex(prices['Date']), inplace=True)
prices.drop('Date', axis=1, inplace=True)
prices_perc = prices.rolling(20).apply(percent_change).AAPL

# Creating lags
shifts = np.arange(1, 11).astype(int)
# Use a dictionary comprehension to create name-value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}
# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)
# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()

# Visualize regression coefficients
# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))
# Fit the model
model = Ridge()
model.fit(X, y)
# Viz func
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax


# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()


# Cross-validation -----------------------------------------------------------------------------------------------------
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

# Cross-validation with shuffling (does not make sense foe time series!)
cv = ShuffleSplit(10, random_state=1)
# Iterate through CV splits
results = []
model = Ridge()
X = np.array(X)
y = np.array(y)
fig, axs = plt.subplots(2, 1)
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))
    # Plot the indices chosen for validation on each loop
    axs[0].scatter(tt, [0] * len(tt), marker='_', s=2, lw=40)
    axs[0].set(ylim=[-.1, .1], title='Test set indices (color=CV loop)',
               xlabel='Index of raw data')
    # Plot the model predictions on each iteration
    axs[1].plot(model.predict(X[tt]))
    axs[1].set(title='Test set predictions on each CV loop',
               xlabel='Prediction index')

# Cross-validation without shuffling: neighbor: CV predictions look smoother, like time series data
cv = KFold(n_splits=10, shuffle=False, random_state=1)
results = []
model = Ridge()
fig, axs = plt.subplots(2, 1)
for tr, tt in cv.split(X, y):
    model.fit(X[tr], y[tr])
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    axs[0].scatter(tt, [0] * len(tt), marker='_', s=2, lw=40)
    axs[0].set(ylim=[-.1, .1], title='Test set indices (color=CV loop)',
               xlabel='Index of raw data')
    axs[1].plot(model.predict(X[tt]))
    axs[1].set(title='Test set predictions on each CV loop',
               xlabel='Prediction index')

# Time-based cross-validation
cv = TimeSeriesSplit(10)
# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()


# Stationarity and stability -------------------------------------------------------------------------------------------
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

# Bootstrapping a confidence interval
def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles


# Calculating variability in model coefficients across CV splits
# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)
# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)
# Plot it
fig, ax = plt.subplots()
ax.scatter(prices_perc_shifted.columns, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(prices_perc_shifted.columns, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# Visualizing model score variability over time
def my_pearsonr (est, X, y):
    # Generate predictions and convert to a vector
    y_pred = est.predict(X).squeeze()
    # Use the numpy "corrcoef" function to calculate a correlation matrix
    my_corrcoef_matrix = np.corrcoef(y_pred, y.squeeze())
    # Return a single correlation value from the matrix
    my_corrcoef = my_corrcoef_matrix[1, 0]
    return my_corrcoef


# TimeSeriesSplit object will use successively-later indices for each test set.
# This means that you can treat the scores of your validation as a time series.
model = Ridge()
cv = TimeSeriesSplit(n_splits=100)
# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')
# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))
# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()

# Accounting for non-stationarity
# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]
# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)
# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()