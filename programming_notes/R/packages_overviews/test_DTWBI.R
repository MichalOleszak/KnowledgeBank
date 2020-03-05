# Info ----------------------------------------------------------------------------------------------------------------
# Functions to impute large gaps within time series based on Dynamic Time Warping methods. It
# contains all required functions to create large missing consecutive values within time series and to
# fill them, according to the paper Phan et al. (2017), <DOI:10.1016/j.patrec.2017.08.019>. Performance
# criteria are added to compare similarity between two signals (query and reference).

# Install & load 
devtools::install_url("http://mawenzi.univ-littoral.fr/DTWBI/package/DTWBI_1.0.tar.gz", dependencies = T)
library(DTWBI)

# Load package containing example dataset
library(TSA) 
data("google") 


# Create data with missings and impute --------------------------------------------------------------------------------
# Create a query and a reference signal
query <- ref <- as.numeric(google)

# Create a gap within query (10% of signal size)
query <- gapCreation(query, rate = 0.1)
plot(query$output_vector, type = "l")

# Fill gap using DTWBI algorithm
imputed <- DTWBI_univariate(query$output_vector, t = query$begin_gap, T = query$gap_size)
lines(imputed$output_vector, col = "red")

# Plot both time series
imp <- imputed$output_vector
imp[!is.na(imputed$input_vector)] <- NA

plot(query$output_vector, type = "l")
lines(imp, col = "red")


# Try different methods -----------------------------------------------------------------------------------------------
imputed_dtw <- DTWBI_univariate(query$output_vector, t = query$begin_gap, T = query$gap_size, DTW_method = "DTW")
imputed_ddtw <- DTWBI_univariate(query$output_vector, t = query$begin_gap, T = query$gap_size, DTW_method = "DDTW")
imputed_afbdtw <- DTWBI_univariate(query$output_vector, t = query$begin_gap, T = query$gap_size, DTW_method = "AFBDTW")

# Compare accuracy by RMSE
compute.rmse(ref, imputed_dtw$output_vector)
compute.rmse(ref, imputed_ddtw$output_vector)
compute.rmse(ref, imputed_afbdtw$output_vector)


# Compute accuracy measures of imputed vector against the true values -------------------------------------------------
# Similarity
# A higher similarity (Similarity in [0, 1]) highlights a more accurate method. 
compute.sim(ref, imputed$output_vector)

# FA2
# This FA2 corresponds to the percentage of pairs of values (xi, yi) satisfying the condition 
# 0.5 <= (Yi/Xi) <= 2. The closer FA2 is to 1, the more accurate is the imputation model. 
compute.fa2(ref, imputed$output_vector)

# Fractional bias
# A perfect imputation model gets FB = 0. An acceptable imputation model gives FB <= 0.3.
compute.fb(ref, imputed$output_vector)

# Fraction of Standard Deviation
# Values of FSD closer to zero indicate a better performance method for the imputation task.
compute.fsd(ref, imputed$output_vector)

# Normalized Mean Absolute Error
# A lower NMAE (NMAE in [0, inf]) value indicates a better performance method for the imputation task. 
compute.nmae(ref, imputed$output_vector)

# RMSE
compute.rmse(ref, imputed$output_vector)

