# Calculate VTI returns
re_turns <- zoo::coredata(na.omit(NPE::etf_env$re_turns[, "VTI", drop=FALSE]))
n_rows <- NROW(re_turns)

# Calculate sampled VTI returns
set.seed(1121)
boot_sample <- lapply(1:1000, function(x) {
  re_turns[sample.int(n_rows, replace = TRUE), , drop=FALSE]
})  # end sapply

# Mean and standard errors of location estimators from bootstrap
boot_data <- sapply(boot_sample, function(sampl_e) {
  c(median_est=NPE::med_ian(sampl_e),
    hl_est = NPE::hle(sampl_e),
    mean_est=mean(sampl_e))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))


# Mean and standard error of MAD estimator from bootstrap
boot_data <- sapply(boot_sample, function(sampl_e) {
  c(mad_est=NPE::calc_mad(sampl_e), std_dev=sd(sampl_e))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))

# Mean and standard error of different types of skewness estimators from bootstrap
boot_data <- sapply(boot_sample, function(sampl_e) {
  c(pearson_skew=NPE::calc_skew(sampl_e, typ_e="Pearson"),
    quantile_skew=NPE::calc_skew(sampl_e, typ_e="Quantile"), 
    nonparametric_skew=NPE::calc_skew(sampl_e, typ_e="Nonparametric"))
})  # end sapply
std_errors <- apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))
# The ratio of std_error to mean shows that the Nonparametric skewness 
# has the smallest standard error of all types of skewness.
std_errors[2, ]/std_errors[1, ]

# Mean and standard error of medcouple estimator from bootstrap
boot_data <- sapply(boot_sample, function(sampl_e) {
  c(med_couple=NPE::med_couple(sampl_e), 
    pearson_skew=NPE::calc_skew(sampl_e, typ_e="Pearson"))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))


## Standard error of Theil-Sen estimator

# Calculate VTI and XLF returns for a single month
re_turns <- zoo::coredata(na.omit(NPE::etf_env$re_turns["2019-11", c("VTI", "XLF")]))
n_rows <- NROW(re_turns)

# Calculate sampled VTI and XLF returns
set.seed(1121)
boot_sample <- lapply(1:1000, function(x) {
  re_turns[sample.int(n_rows, replace = TRUE), ]
})  # end sapply

# Mean and standard error of Theil-Sen estimator from bootstrap
boot_data <- sapply(boot_sample, function(sampl_e) {
  c(theilSen=NPE::theilSenEstimator(sampl_e[, "VTI"], sampl_e[, "XLF"])[2], 
    least_squares=unname(coef(lm(sampl_e[, "XLF"] ~ sampl_e[, "VTI"]))[2]))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))
