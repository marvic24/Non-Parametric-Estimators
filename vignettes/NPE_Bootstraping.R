library(HighFreq)

re_turns <- rutils::etf_env$re_turns[,"XLF"]
library(NPE)

re_turns <- na.omit(re_turns)
n_rows <- NROW(re_turns)

#bootstraping median
set.seed(1121)
boot_medians <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  NPE::med_ian(re_turns[boot_sample])
})

#bootstraping Mean
set.seed(1121)
boot_means <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  mean(re_turns[boot_sample])
})

#bootstapping Hodges-Lehmann Estimator
set.seed(1121)
boot_hle <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  NPE::hle(re_turns[boot_sample])
})

#Mean and standard error for median
c(mean=mean(boot_medians), std_error=sd(boot_medians))

#Mean and standard error for Hodges-Lehamnn Estimators
c(mean=mean(boot_hle), std_error=sd(boot_hle))

#Mean and standard error for mean
c(mean=mean(boot_means), std_error=sd(boot_means))



#bootstraping Median Absolute Deviations
set.seed(1121)
boot_mad <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  NPE::medianAbsoluteDeviation(re_turns[boot_sample])
})

#bootstapping Hodges-Lehmann Estimator
set.seed(1121)
boot_sd <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  sd(re_turns[boot_sample])
})

#Mean and standard error for Median Absolute Deviation
c(mean=mean(boot_medians), std_error=sd(boot_mad))

#Mean and standard error for Standard Deviations
c(mean=mean(boot_hle), std_error=sd(boot_sd))


library(e1071)

#bootstraping Medcouple
set.seed(1121)
boot_mc <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  NPE::med_couple(re_turns[boot_sample])
})

#bootstapping Skewness
set.seed(1121)
boot_skew <- sapply(1:100, function(x) {
  boot_sample <- sample.int(n_rows, replace = TRUE)
  skewness(re_turns[boot_sample])
})

#Mean and standard error for Medcouple
c(mean=mean(boot_medians), std_error=sd(boot_mc))

#Mean and standard error for Skewness
c(mean=mean(boot_hle), std_error=sd(boot_skew))



