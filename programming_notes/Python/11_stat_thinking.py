# Graphical EDA --------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# get data
from sklearn import datasets
iris = datasets.load_iris().data
setosa_petal_length = iris[0:50, 2]
virginica_petal_length = iris[100:150, 2]
versicolor_petal_length = iris[50:100, 2]
versicolor_petal_width = iris[50:100, 3]
# Set default Seaborn plot style
sns.set()


# Empirical cumulative distribution functions (ECDF)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    # ECDF goes from 1/n to 1 in equally spaced increments. The end value in np.arange() is not inclusive, so use n+1
    y = np.arange(1, n+1) / n
    return x, y


x_vers, y_vers = ecdf(versicolor_petal_length)
plt.plot(x_vers, y_vers, marker='.', linestyle='none')
plt.ylabel('ECDF')
plt.xlabel('Versicolor petal length')
plt.show()

# Multiple ECDFs in one plot
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

plt.show()


# Quatitative exploratory analysis -------------------------------------------------------------------------------------
np.mean(setosa_petal_length)
np.median(setosa_petal_length)
np.percentile(setosa_petal_length, [25, 50, 75])
np.var(setosa_petal_length)
np.std(setosa_petal_length)

# Compare percentiles to ECDF
percentiles = np.array([2.5, 25, 50, 75, 97.5])
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

plt.show()


# Covariance
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Pearson correlation
pearson_corr = np.corrcoef(versicolor_petal_length, versicolor_petal_width)


# Descrete distributions -----------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# random numbers between 0 and 1
np.random.random(size=3)

random_numbers = np.empty(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()
plt.hist(random_numbers)
plt.show()


# Bernoulli trials
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    n_success = 0
    for i in range(n):
        random_number = np.random.random()
        if random_number < p:
            n_success += 1
    return n_success


# Bank made 100 loans. Each loan's default prob is 0.05. What is the prob of a given number of defaults?
n_defaults = np.empty(100000)
for i in range(100000):
    n_defaults[i] = perform_bernoulli_trials(n=100, p=0.05)

x, y = ecdf(n_defaults)
plt.plot(x, y, marker='.', linestyle='none')
plt.show()

# What's the prob of 10 or more defaults?
ten_or_more_def = np.sum(n_defaults >= 10) / len(n_defaults)

# Number of r succeesses in n Bernoulli trial with success prob of p is binomial distributed
n_defaults_binomial = np.random.binomial(100, 0.05, 100000)
ten_or_more_def_binomial = np.sum(n_defaults_binomial >= 10) / len(n_defaults_binomial)

print(ten_or_more_def, ten_or_more_def_binomial)

# Plot the PMF of the Binomial distribution as a histogram, with bins centered on integers
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5
plt.hist(n_defaults, normed=True, bins=bins)
plt.xlabel('Number of defaults out of 100 loans')
plt.ylabel('Probability Mass Function (PMF)')
plt.show()


# Poisson processes and the Poisson distribution
#
# The timing of the next event is independent on when the previous event happened.
#
# The number r of arrivals of a Poisson process in a given time interval with average rate of lambda arrivals
# per interval is Poisson distributed.
#
# Poisson distr is the limit of Bernoulli distr for low success prob and large number of trials (so, for rare events).
# So, the Poisson distribution with arrival rate equal to np approximates a Binomial distribution for n Bernoulli
# trials with probability p of success (with n large and p small). Importantly, the Poisson distribution is often
# simpler to work with because it has only one parameter instead of two for the Binomial distribution.

samples_poisson = np.random.poisson(10, size=10000)
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]
for i in range(3):
    samples_binomial = np.random.binomial(n=n[i], p=p[i], size=10000)
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))

# he means are all about the same, which can be shown to be true by doing some pen-and-paper work. The standard
# deviation of the Binomial distribution gets closer and closer to that of the Poisson distribution as the
# probability p gets lower and lower.

# In baseball, a no-hitter is a game in which a pitcher does not allow the other team to get a hit.
# This is a rare event, happened only 251 times in the last 115 seasons (years). In 1990 and 2015 there were 7!
# What is the probability of having seven or more in a season?

n_nohitters = np.random.poisson(251/115, size=10000)
n_large = np.sum(n_nohitters >= 7)
p_large = n_large / 10000
print('Probability of seven or more no-hitters:', p_large)


# Continuous distributions ---------------------------------------------------------------------------------------------

# Normal distribution
# The Normal PDF
samples_std1 = np.random.normal(20, 1, 100000)
samples_std3 = np.random.normal(20, 3, 100000)
samples_std10 = np.random.normal(20, 10, 100000)

plt.hist(samples_std1, bins=100, normed=True, histtype='step')
plt.hist(samples_std3, bins=100, normed=True, histtype='step')
plt.hist(samples_std10, bins=100, normed=True, histtype='step')

plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

# The Normal CDF
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

plt.plot(x_std1, y_std1, marker='.', linestyle='none')
plt.plot(x_std3, y_std3, marker='.', linestyle='none')
plt.plot(x_std10, y_std10, marker='.', linestyle='none')

plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


# Exponential distribution
# The waiting time between arrivals of a Poisson process is exponentially distributed.

# Compute waiting time to observe two rare sport events one after another, with mean waiting times 764 and 715 games.
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    t1 = np.random.exponential(tau1, size)
    t2 = np.random.exponential(tau2, size)
    return t1 + t2


waiting_times = successive_poisson(764, 715, 100000)
plt.hist(waiting_times, bins=100, normed=True, histtype='step')
plt.xlabel("Waiting time")
plt.ylabel("PDF")
plt.show()


# Linear regression ----------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Get data
dat = pd.read_csv('data/literacy_birth_rate.csv')
dat = dat.loc[0:161, :]
literacy = dat['female literacy'].astype(float)
fertility = dat['fertility'].astype(float)

# Produce scatterplot of data
plt.scatter(literacy, fertility)
plt.margins(0.02)
plt.xlabel('literacy')
plt.ylabel('fertility')

# Fit regression
a, b = np.polyfit(literacy, fertility, deg=1)

# Draw regression line
x = np.array([0, 100])
y = y = a * x + b
plt.plot(x, y)
plt.show()


# Bootstrapping --------------------------------------------------------------------------------------------------------

# Get data
dat = pd.read_csv('data/sheffield_weather_station.csv')
rainfall = dat.groupby('yyyy')[['rain']].agg('sum').reset_index(drop=True)
rainfall = rainfall.values.tolist()
rainfall = [item for sublist in rainfall for item in sublist]

# Visualizing bootstrap samples
for _ in range(50):
    # Generate bootstrap sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))
    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none', color='gray', alpha=0.1)
# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')
# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
# Show the plot
plt.show()

# Functions for drawing bootstrap replciates
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    bs_replicates = np.empty(shape=size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates


# Bootstrap replicates of the mean and the SEM (standard error od the mean)

# It can be shown theoretically that under not-too-restrictive conditions, the value of the mean will always be
# Normally distributed. The standard deviation of this distribution, called the standard error of the mean, or SEM,
# is given by the standard deviation of the data divided by the square root of the number of data points.
bs_replicates = draw_bs_reps(rainfall, np.mean, 10000)
# Compute SEM according to Central Limit Theorem
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)
# Compute standard deviation of the mean with bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Bootstrapped 95% confidence interval
np.percentile(bs_replicates, [2.5, 97.5])


# Pairs bootstrap for linear regression (to get confidence intervals for intercept and slope) --------------------------
#  - resample data in pais (x and y)
#  - compute regression on resampled pairs: each parameter is a boostrap replicate
#  - get CI from percentiles of these replicates

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # Set up array of indices to sample from
    inds = np.arange(len(x))
    # Initialize replicates
    bs_slope_reps = np.empty(shape=size)
    bs_intercept_reps = np.empty(shape=size)
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps


# Get data
dat = pd.read_csv('data/literacy_birth_rate.csv')
dat = dat.loc[0:161, :]
literacy = dat['female literacy'].astype(float)
fertility = dat['fertility'].astype(float)

# Get CI for the slope
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(literacy, fertility, 1000)
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plotting bootstrap regressions
# Generate array of x-values for bootstrap lines
x = np.array([0, 100])
# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')
# Plot the true data
_ = plt.plot(literacy, fertility, marker='.', linestyle='none')
# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()


# Hypothesis testing ---------------------------------------------------------------------------------------------------

# Hypothesis testing = assessing how reasonable observed data are, assuming the hypothesis is true.

# If a null hypothesis assumes two variables to be identically distributed, we assume it is true, treat the two
# as one joint variable and permute it randomly, and then assign the permuted numbers back to the two variables.
# This way, we simulate the null hypothesis.

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # Concatenate the data sets
    data = np.concatenate((data1, data2))
    # Permute the concatenated array
    permuted_data = np.random.permutation(data)
    # Split the permuted array into two
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


# Visualizing permutation sampling
dat = pd.read_csv('data/sheffield_weather_station.csv')
rain_june = dat.loc[dat['mm'] == 6].rain.reset_index(drop=True).values.tolist()
rain_november = dat.loc[dat['mm'] == 11].rain.reset_index(drop=True).values.tolist()
# Is rainfall in november and june identically   distributed? See how their ECDFs would look if they were.
for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)
    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)
    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)
# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')
# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()
# None of the ECDFs from the permutation samples overlap with the observed data, suggesting that the hypothesis is not
# commensurate with the data. July and November rainfall are not identically distributed.


# Test statistics and p-values -----------------------------------------------------------------------------------------
# Test statistic = a single number that can be computed from observed data and from data simulated under the null
# p-value = prob of getting test statistic at least as extreme as what was observed, assuming the null is true

# Generating permutation replicates (single values of a statistic computed from a permutation sample)
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(shape=size)
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates


# Kleinteich and Gorb (Sci. Rep., 4, 5225, 2014) performed an interesting experiment with South American horned frogs.
# They held a plate connected to a force transducer, along with a bait fly, in front of them. They then measured the
# impact force and adhesive force of the frog's tongue when it struck the target. Frog A is an adult and Frog B is
# a juvenile. The researchers measured the impact force of 20 strikes for each frog.
# We will test the hypothesis that the two frogs have the same distribution of impact forces.

# Get data
df = pd.read_csv('data/frog_tongue.csv')[['ID', 'impact force (mN)']]
df = df.loc[df.ID.isin(['II', 'IV'])]
df = df.rename(columns={"impact force (mN)": "impact_force"})
df = df.assign(ID = ['A' if id == 'II' else 'B' for id in df['ID']])

# EDA
_ = sns.swarmplot('ID', 'impact_force', data=df)
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')
plt.show()
# Eyeballing it, it does not look like they come from the same distribution. Frog A, the adult, has three or four very
# hard strikes, and Frog B, the juvenile, has a couple weak ones. However, it is possible that with only 20 samples it
# might be too difficult to tell if they have difference distributions, so we should proceed with the hypothesis test.

# Permutation test for difference in means -----------------------------------------------------------------------------
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    diff = np.mean(data_1) - np.mean(data_2)
    return diff


# Compute difference of mean impact force from experiment
force_a = df.loc[df.ID == 'A'].impact_force.tolist()
force_b = df.loc[df.ID == 'B'].impact_force.tolist()
empirical_diff_means = diff_of_means(force_a, force_b)
# The difference in average strike foreces is 288 mN. We will compute the probability of getting at least a 288 mN
# difference in mean strike force under the hypothesis that the distributions of strike forces for the two frogs are
# identical. We use a permutation test with a test statistic of the difference of means to test this hypothesis.
perm_replicates = draw_perm_reps(force_a, force_b, diff_of_means, size=10000)
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)
print('p-value =', p)
# The p-value tells you that there is about a 0.6% chance that you would get the difference of means observed in the
# experiment if frogs were exactly the same.


# Pipeline for hypothesis testing --------------------------------------------------------------------------------------
# - Clearly state the null hypothesis
# - Define your test statistic
# - Generate many sets of simulated data assuming the null hypothesis is true
# - Compute the test statistic for each simulated data set
# - The p-value is the fraction of your simulated data sets for which the test statistic is at least as extreme as for
#   the real data


# A one-sample bootstrap hypothesis test (compare data set to a single number) -----------------------------------------
#
# Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces.
# Unfortunately, you do not have Frog C's impact forces available, but you know they have a mean of 550 mN.
# Because you don't have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that
# the forces from Frog B and Frog C come from the same distribution. You will therefore test another, less restrictive
# hypothesis: The mean strike force of Frog B is equal to that of Frog C.
# To set up the bootstrap hypothesis test, you will take the mean as our test statistic. The goal is to calculate the
# probability of getting a mean impact force less than or equal to what was observed for Frog B if the hypothesis that
# the true mean of Frog B's impact forces is equal to that of Frog C is true.

# Translate the impact forces of Frog B such that its mean is 550 mN.
translated_force_b = np.array(force_b) + 550 - np.mean(force_b)
# Take bootstrap replicates of Frog B's translated impact forces
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)
# Compute fraction of replicates that are less than the observed Frog B force (< because np.mean(force_b) < 550,
# so less is more extreme)
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
# Print the p-value
print('p = ', p)
# The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.


# A two-sample bootstrap test for identical distributions --------------------------------------------------------------
#
# We can test the same hypothesis that we tested with a permutation test: that the Frog A and Frog B have identically
# distributed impact forces. To do this test on two arrays with n1 and n2 entries, we do a very similar procedure as a
# permutation test. We concatenate the arrays, generate a bootstrap sample (instead of a permutation) from it, and take
# the first n1 entries of the bootstrap sample as belonging to the first data set and the last n2 as belonging to the
# second. We then compute the test statistic, e.g., the difference of means, to get a bootstrap replicate. The p-value
# is the number of bootstrap replicates for which the test statistic is greater than what was observed.

# Compute difference of mean impact force from experiment
empirical_diff_means = diff_of_means(force_a, force_b)
# Concatenate forces
forces_concat = np.concatenate((force_a, force_b))
# Initialize bootstrap replicates
bs_replicates = np.empty(shape=10000)
for i in range(10000):
    # Generate bootstrap sample
    bs_sample = np.random.choice(forces_concat, size=len(forces_concat))
    # Compute replicate
    bs_replicates[i] = diff_of_means(bs_sample[:len(force_a)], bs_sample[len(force_b):])
# Compute and print p-value
p = np.sum(bs_replicates > empirical_diff_means) / len(bs_replicates)
print('p-value =', p)
# p-value is close to the one from the permutation test. These two indeed test the same thing.
# However, the permutation test exactly simulates the null hypothesis that the data come from the same distribution,
# whereas the bootstrap test approximately simulates it. So, permutation test is preferred, as it is more exact.


# A two-sample bootstrap hypothesis test for difference of means -------------------------------------------------------
#
# We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, but not necessarily the
# same distribution. This is not possible with the permutation test.
# To do the two-sample bootstrap test, we shift both arrays to have the same mean, since we are simulating the
# hypothesis that their means are, in fact, equal. We then draw bootstrap samples out of the shifted arrays and compute
# the difference in means. This constitutes a bootstrap replicate, and we generate many of them. The p-value is the
# fraction of replicates with a difference in means greater than or equal to what was observed.

# Compute mean of all forces
mean_force = np.mean(forces_concat)
# Generate shifted data sets for both a and b such that the mean of each is the mean of the concatenated impact forces.
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force
# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)
# Get replicates of difference of means
bs_replicates = bs_replicates_a - bs_replicates_b
# Compute and print p-value
p = np.sum(bs_replicates >= np.mean(np.array(force_a) - np.array(force_b))) / len(bs_replicates)
print('p-value =', p)
# Not surprisingly, the more forgiving hypothesis, only that the means are equal as opposed to having identical
# distributions, gives a higher p-value.


# Some hypothesis testing examples -------------------------------------------------------------------------------------

# Example 1: permutation test for indentical fraction voting for
#
# The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. Excluding
# "present" and "abstain" votes, 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35
# Republicans voted nay. Did party affiliation make a difference in the vote?
# To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or
# her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the probability
# of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244.

# Construct arrays of data:
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)


def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    # Two input needed for using it in draw_perm_reps(), but the second not used
    frac = np.sum(dems) / len(dems)
    return frac


# Acquire permutation samples
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, 10000)

# Compute and print p-value
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# This small p-value suggests that party identity had a lot to do with the voting.


# Example 2: permutation test for equal means
#
# In 1920, Major League Baseball implemented important rule changes that ended the so-called dead ball era. Importantly,
# the pitcher was no longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. In this
# problem you will perform an A/B test to determine if these rule changes resulted in a slower rate of no-hitters (i.e.,
# longer average time between no-hitters) using the difference in mean inter-no-hitter time as your test statistic.

# The inter-no-hitter times for the respective eras:
nht_dead = np.array([-1, 894, 10, 130, 1, 934, 29, 6, 485, 254, 372, 81, 191, 355, 180, 286, 47, 269, 361, 173, 246,
                     492, 462, 1319, 58, 297, 31, 2970, 640, 237, 434, 570, 77, 271, 563, 3365, 89, 0, 379, 221, 479,
                     367, 628, 843, 1613, 1101, 215, 684, 814, 278, 324, 161, 219, 545, 715, 966, 624, 29, 450, 107, 20,
                     91, 1325, 124, 1468, 104, 1309, 429, 62, 1878, 1104, 123, 251, 93, 188, 983, 166, 96, 702, 23, 524,
                     26, 299, 59, 39, 12, 2, 308, 1114, 813, 887])
nht_live = np.array([645, 2088, 42, 2090, 11, 886, 1665, 1084, 2900, 2432, 750, 4021, 1070, 1765, 1322, 26, 548, 1525,
                     77, 2181, 2752, 127, 2147, 211, 41, 1575, 151, 479, 697, 557, 2267, 542, 392, 73, 603, 233, 255,
                     528, 397, 1529, 1023, 1194, 462, 583, 37, 943, 996, 480, 1497, 717, 224, 219, 1531, 498, 44, 288,
                     267, 600, 52, 269, 1086, 386, 176, 2199, 216, 54, 675, 1243, 463, 650, 171, 327, 110, 774, 509, 8,
                     197, 136, 12, 1124, 64, 380, 811, 232, 192, 731, 715, 226, 605, 539, 1491, 323, 240, 179, 702, 156,
                     82, 1397, 354, 778, 603, 1001, 385, 986, 203, 149, 576, 445, 180, 1403, 252, 675, 1351, 2983, 1568,
                     45, 899, 3260, 1025, 31, 100, 2055, 4043, 79, 238, 3931, 2351, 595, 110, 215, 0, 563, 206, 660,
                     242, 577, 179, 157, 192, 192, 1848, 792, 1693, 55, 388, 225, 1134, 1172, 1555, 31, 1582, 1044, 378,
                     1687, 2915, 280, 765, 2819, 511, 1521, 745, 2491, 580, 2072, 6450, 578, 745, 1075, 1103, 1549,
                     1520, 138, 1202, 296, 277, 351, 391, 950, 459, 62, 1056, 1128, 139, 420, 87, 71, 814, 603, 1349,
                     162, 1027, 783, 326, 101, 876, 381, 905, 156, 419, 239, 119, 129, 467])

# Compute the observed difference in mean inter-no-hitter times
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time
# Permutation test, H0: equal inter-no-hitter times in both epochs
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
print('p-val =', p)

# The p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual
# difference between the dead ball and live ball eras. This suggests strong statistical significance.


# Example 3: Hypothesis test for Pearson correlation
#
# Is literacy correlated with fertility? To test it, one needs to simulate the data assuming the null hypothesis of
# no correlation is true.
# Do a permutation test: Permute the literacy values but leave the fertility values fixed (no need to permute them
# too, correlations are broken anyway by permuting literacy) to generate a new set of (literacy, fertility) data.

dat = pd.read_csv('data/literacy_birth_rate.csv')
dat = dat.loc[0:161, :]
literacy = dat['female literacy'].astype(float)
fertility = dat['fertility'].astype(float)

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0,1]


# Compute observed correlation
r_obs = pearson_r(literacy, fertility)
# Initialize permutation replicates
perm_replicates = np.empty(10000)
# Draw replicates
for i in range(10000):
    # Permute literacy measurments
    literacy_permuted = np.random.permutation(literacy)
    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(literacy_permuted, fertility)
# Compute p-value: p
p = np.sum(perm_replicates <= r_obs) / len(perm_replicates)
print('p-val =', p)

# You got a p-value of zero. In hacker statistics, this means that your p-value is very low, since you never got
# a single replicate in the 10,000 you took that had a Pearson correlation as extreme as the observed one. You could try
# increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.

