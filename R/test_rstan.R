# Intro -----------------------------------------------------------------------
library(rstan)

# Want to impose priors? Ultimately care about prediction? -> Bayesian
# Don't have conjugate priors? Code up a Metropolis-Hastings, optimizing
# Acceptance rates and proposals. Or: use RStan.

# HMC is a method to produce proposals for a MH algorithm that are accepted
# with high probability. Rather than have a proposal distribution we appeal 
# to Hamiltonian dynamics. Consider the target distribution as an inverted ice rink:
# - Give the particle some momentum.
# - It slides around the ice rink spending most time where the density is high.
# - Taking snap shots of this trajectory gives a proposal sample for the posterior.
# - We then correct using Metropolis-Hastings.

# HMC, like RWMH, requires some tuning, the number and size of the leapfrog steps.
# The "No-U-Turn Sampler" or NUTs (Hoffman and Gelman (2014)), optimises these adaptively.
# Too small number and size of steps leads to RW type behaviour while too big
# the trajectory starts to come back on itself. NUTS builds up a set of likely 
# candidate points and stops immediately when a trajectory starts to come back on itself.


# Example 1: 8schools ---------------------------------------------------------
# A hierarchical model is used to model the effect of coaching programs on college admissions tests.
# J - number of schools in the experiment
# y - effect of coaching
# sigma - estimated standard error of the effect

# The stan function accepts data as a named list or a character vector of object names
schools_data <- list(J = 8, 
                     y = c(28,  8, -3,  7, -1,  1, 18, 12),
                     sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

# The .stan file:
# - The data block specifies the data that is conditioned upon in Bayes Rule
# - The parameters block declares the parameters whose posterior distribution is sought:
#   (mean mu and std tau of the school effects + standardized school-level effect eta)
# - The model block -> the second argument to Stan's normal(???,???) distribution is the 
#   standard deviation, not the variance!

# Lookup Stan's version of R's statistical funcs
lookup("dnorm")

# The stan function wraps the following three steps:
# - Translate a model in Stan code to C++ code
# - Compile the C++ code to a dynamic shared object (DSO) and load the DSO
# - Sample given some user-specified data and other settings
fit1 <- stan(
  file = "C:/Users/Michal/Dropbox/R/test_packages/8schools.stan",
  data = schools_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 1,              # number of cores (using 2 just for the vignette)
  refresh = 1000          # show progress every 'refresh' iterations
)

# The last line of this output, lp__, is the logarithm of the (unnormalized) posterior density
print(fit1, pars=c("theta", "mu", "tau", "lp__"), probs = c(.1,.5,.9))

# The default plot shows posterior uncertainty intervals (by default 80% (inner) and 95% (outer)) 
# and the posterior median for all the parameters as well as lp__ (the log of posterior density 
# function up to an additive constant):
plot(fit1)

# The traceplot method is used to plot the time series of the posterior draws. If we include 
# the warmup draws by setting inc_warmup=TRUE, the background color of the warmup area is different 
# from the post-warmup phase:
traceplot(fit1, pars = c("mu", "tau"), inc_warmup = TRUE, nrow = 2)

# To assess the convergence of the Markov chains, in addition to visually inspecting traceplots we can 
# calculate the split Rhat statistic. Split Rhat is an updated version of the Rhat statistic proposed 
# in Gelman and Rubin (1992) that is based on splitting each chain into two halves.
print(fit1, pars = c("mu", "tau"))

# Get information on parameters related the performance of the sampler:
sampler_params <- get_sampler_params(fit1, inc_warmup = TRUE)
# all chains combined
summary(do.call(rbind, sampler_params), digits = 2)
# each chain separately
lapply(sampler_params, summary, digits = 2)

# The pairs plot can be used to get a sense of whether any sampling difficulties are occurring 
# in the tails or near the mode:
pairs(fit1, pars = c("mu", "tau", "lp__"), las = 1)
# In this plot, the marginal distribution of each selected parameter is included as a histogram along the diagonal. 
# By default, draws with below-median accept_stat__ (MCMC proposal acceptance rate) are plotted below the diagonal 
# and those with above-median accept_stat__ are plotted above the diagonal (this can be changed using the condition
# argument). Each off-diagonal square represents a bivariate distribution of the draws for the intersection of the 
# row-variable and the column-variable. Ideally, the below-diagonal intersection and the above-diagonal intersection 
# of the same two variables should have distributions that are mirror images of each other. Any yellow points would 
# indicate transitions where the maximum treedepth__ was hit, and red points indicate a divergent transition.

# The extract function (with its default arguments) returns a list with named components 
# corresponding to the model parameters.
list_of_draws <- extract(fit1)
print(names(list_of_draws))

# In this model the parameters mu and tau are scalars and theta is a vector with eight elements. 
# This means that the draws for mu and tau will be vectors (with length equal to the number of
# post-warmup iterations times the number of chains) and the draws for theta will be a matrix, 
# with each column corresponding to one of the eight components:
head(list_of_draws$mu)
head(list_of_draws$tau)
head(list_of_draws$theta)

# Converting draws to R-friednly objects
df_of_draws <- as.data.frame(fit1)
matrix_of_draws <- as.matrix(fit1)
array_of_draws <- as.array(fit1)

print(dim(matrix_of_draws))
print(dim(df_of_draws))
print(dim(array_of_draws))

# Posterior summary statistics and convergence diagnostics
fit_summary <- summary(fit1)
# In fit_summary$summary all chains are merged whereas fit_summary$c_summary contains 
# summaries for each chain individually. Typically we want the summary for all chains merged.
print(names(fit_summary))
# The summary is a matrix with rows corresponding to parameters and columns to the various summary 
# quantities. These include the posterior mean, the posterior standard deviation, and various 
# quantiles computed from the draws.
# For models fit using MCMC, also included in the summary are the Monte Carlo standard error (se_mean),
# the effective sample size (n_eff), and the R-hat statistic (Rhat).
print(fit_summary$summary)
summary(fit1, pars = c("mu", "tau"), probs = c(0.1, 0.9))$summary

# Sampler diagnostics
# The object returned by get_sampler_params is a list with one component (a matrix) per chain. 
# Each of the matrices has number of columns corresponding to the number of sampler parameters 
# and the column names provide the parameter names.
sampler_params <- get_sampler_params(fit1, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]
colnames(sampler_params_chain1)
# To do things like calculate the average value of accept_stat__ for each chain (or the maximum 
# value of treedepth__ for each chain if using the NUTS algorithm, etc.) the sapply function is 
# useful as it will apply the same function to each component of sampler_params.
sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))

# Model code
code <- get_stancode(fit1)
cat(code)

# Initia values
inits <- get_inits(fit1)
inits_chain1 <- inits[[1]]
print(inits_chain1)

# (P)RNG seed
print(get_seed(fit1))

# Warmup and sampling times
print(get_elapsed_time(fit1))


# Example 2: Bayesian LASSO ---------------------------------------------------
# - Data: n, p, Y, X, prior parameters, hyper-parameters
# - Parameters: beta, sigma^2
# - Model: Gaussian likelihood, Laplacian and Gamma priors
# - Output: Posterior samples, posterior predictive samples

library(car)  # for data
data(Prestige)

# Construct data
dat <- list(n = 102, p = 3, X = cbind(Prestige$education, Prestige$women, Prestige$prestige, 
                                      rep(1, 102)), y = Prestige$income, a = 10, b = 10, w = 100)

# Run 25000 warm-ups and 25000 samples
chain1 <- stan(file = 'C:/Users/Michal/Dropbox/R/test_packages/bayes_LASSO.stan',
               data = dat, iter = 50000, chains = 1, cores = 1)
params <- extract(chain1)

# Plotting posterior distributions
par(mfrow = c(1, 2))
plot(density(params$beta[, 1]), xlab = "beta1", ylab = "Density", main = "")
plot(density(params$beta[, 2]), xlab = "beta2", ylab = "Density", main = "")

# Plotting predictive distributions
par(mfrow = c(1, 2))
plot(density(params$y_predict[, 1]), xlab = "Income", ylab = "Density", main = "")
plot(density(params$y_predict[, 100]), xlab = "Income", ylab = "Density", main = "")

# Chain diagnostics
sampler_params <- get_sampler_params(chain1, inc_warmup = FALSE)
sampler_params[[1]][1:5,]

traceplot(chain1, pars = "beta", inc_warmup = FALSE)
traceplot(chain1, pars = "beta", inc_warmup = TRUE)   # see how sampler converges after burn-in
pairs(chain1, pars = "beta")

stan_rhat(chain1, pars = "beta")  # the distribution of the Rhat statistic
stan_ess(chain1, pars = "beta")   # the ratio of effective sample size to total sample size
stan_mcse(chain1, pars = "beta")  # the ratio of Monte Carlo standard error to posterior standard deviation for the estimated parameters
