import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from scipy.stats import uniform, norm, gamma, expon, poisson, binom, bernoulli


# Uniform distribution -------------------------------------------------------------------------------------------------
n = 10000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc=start, scale=width)

ax = sns.distplot(data_uniform,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15, 'alpha': 1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
plt.show()


# Normal distribution --------------------------------------------------------------------------------------------------
data_normal = norm.rvs(size=10000,loc=0,scale=1)
ax = sns.distplot(data_normal,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
plt.show()


# Gamma distribution ---------------------------------------------------------------------------------------------------
# The gamma distribution is a two-parameter family of continuous probability distributions. While it is used rarely in
# its raw form but other popularly used distributions like exponential, chi-squared, erlang distributions are special
#  cases of the gamma distribution. The gamma distribution can be parameterized in terms of a shape parameter α=k and
# an inverse scale parameter β=1/θ, called a rate parameter., the symbol Γ(n) is the gamma function and is defined
# as (n−1)!
data_gamma = gamma.rvs(a=5, size=10000)
ax = sns.distplot(data_gamma,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Gamma Distribution', ylabel='Frequency')
plt.show()


# Exponential distribution ---------------------------------------------------------------------------------------------
# The exponential distribution describes the time between events in a Poisson point process, i.e., a process in which
# events occur continuously and independently at a constant average rate. It has a parameter λ called rate parameter
data_expon = expon.rvs(scale=1,loc=0,size=1000)
ax = sns.distplot(data_expon,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Exponential Distribution', ylabel='Frequency')
plt.show()


# Poisson distribution -------------------------------------------------------------------------------------------------
# Poisson random variable is typically used to model the number of times an event happened in a time interval.
# For example, the number of users visited on a website in an interval can be thought of a Poisson process.
# Poisson distribution is described in terms of the rate (μ) at which the events happen. An event can occur 0, 1, 2, …
# times in an interval. The average number of events in an interval is designated λ (lambda).
# Lambda is the event rate, also called the rate parameter.
# Note that the normal distribution is a limiting case of Poisson distribution with the parameter λ→∞. Also, if the
# times between random events follow an exponential distribution with rate λ, then the total number of events in a time
# period of length t follows the Poisson distribution with parameter λt.
data_poisson = poisson.rvs(mu=3, size=10000)
ax = sns.distplot(data_poisson,
                  bins=30,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Poisson Distribution', ylabel='Frequency')
plt.show()


# Binomial distribution ------------------------------------------------------------------------------------------------
# A distribution where only two outcomes are possible, such as success or failure, gain or loss, win or lose and where
# the probability of success and failure is same for all the trials is called a Binomial Distribution. However, the
# outcomes need not be equally likely, and each trial is independent of each other. The parameters of a binomial
# distribution are n and p where n is the total number of trials, and p is the probability of success in each trial.
# Note that since the probability of success was greater than 0.5 the distribution is skewed towards the right side.
# Also, poisson distribution is a limiting case of a binomial distribution under the following conditions:
# 1. The number of trials is indefinitely large or n→∞
# 2. The probability of success for each trial is same and indefinitely small or p→0
# 3. np=λ is finite.
# Normal distribution is another limiting form of binomial distribution under the following conditions:
# 1.The  number of trials is indefinitely large, n→∞
# 2. Both p and q are not indefinitely small.
data_binom = binom.rvs(n=10,p=0.8,size=10000)
ax = sns.distplot(data_binom,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
plt.show()


# Bernoulli distribution -----------------------------------------------------------------------------------------------
# A Bernoulli distribution has only two possible outcomes, namely 1 (success) and 0 (failure), and a single trial,
# for example, a coin toss. So the random variable X which has a Bernoulli distribution can take value 1 with the
# probability of success, p, and the value 0 with the probability of failure, q or 1−p. The probabilities of success
# and failure need not be equally likely. The Bernoulli distribution is a special case of the binomial distribution
# where a single trial is conducted (n=1).
data_bern = bernoulli.rvs(size=10000,p=0.6)
ax= sns.distplot(data_bern,
                 kde=False,
                 color="skyblue",
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Bernoulli Distribution', ylabel='Frequency')
plt.show()