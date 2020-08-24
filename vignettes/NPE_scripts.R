######################################################################
### R scripts for testing the package NPE


############################
# Installation and Loading of Package NPE

# Install external packages
if (!require("devtools")) install.packages("devtools")
if (!require("rutils")) install.packages("rutils")

# library(Rcpp)
# library(RcppArmadillo)
# library(RcppParallel)

# Install NPE package from local drive
setwd("C:/Develop/capstone/Sumit_Sethi/NPE")
devtools::document()
install.packages(pkgs="C:/Develop/capstone/Sumit_Sethi/NPE", repos=NULL, type="source")
# Install NPE package from Github
devtools::install_github(repo="marvic24/Non-Parametric-Estimators")
library(NPE)
# detach("package:NPE")

## Create package documentation
system("R CMD Rd2pdf C:/Develop/capstone/Sumit_Sethi/NPE")


## Tests for function NPE::med_ian()

# Calculate XLF returns
re_turns <- na.omit(rutils::etf_env$re_turns[ ,"XLF", drop=FALSE])
n_rows <- NROW(re_turns)


# The function NPE::med_ian() calculates the median
# da_ta <- rnorm(1e4)
all.equal(median(re_turns), NPE::med_ian(re_turns), check.attributes=FALSE)

# NPE::med_ian() is a little faster than stats::median()
library(microbenchmark)
summary(microbenchmark(
  Rcpp=NPE::med_ian(re_turns),
  Rcode=median(re_turns),
  times=10))[, c(1, 4, 5)]


# Benchmark the Median Absolute Deviation - Rcpp code vs R code.
summary(microbenchmark(
  Rcpp=NPE::calc_mad(re_turns),
  Rcode=mad(re_turns),
  times=10))[, c(1, 4, 5)]


## Tests for function NPE::calc_skew()

# The function NPE::calc_skew() calculates the skewness

calc_skewr <- function(x) {
  x <- (x-mean(x)); nr <- NROW(x);
  nr*sum(x^3)/(var(x))^1.5/(nr-1)/(nr-2)
}  # end calc_skewr

all.equal(calc_skewr(re_turns), calc_skew(re_turns), check.attributes=FALSE)

skew_ness <- sapply(rutils::etf_env$sym_bols, function(sym_bol) {
  calc_skew(na.omit(get(sym_bol, rutils::etf_env$re_turns)))
})  # end sapply

foo <- sapply(rutils::etf_env$sym_bols, function(sym_bol) {
  calc_skewnp(na.omit(get(sym_bol, rutils::etf_env$re_turns)))
})  # end sapply
bar <- sapply(rutils::etf_env$sym_bols, function(sym_bol) {
  calc_mad(na.omit(get(sym_bol, rutils::etf_env$re_turns)))
})  # end sapply
foo <- foo/bar
plot(skew_ness[c(-12, -16)], foo[c(-12, -16)])


summary(microbenchmark(
  Rcpp=calc_skew(re_turns),
  Rcode=calc_skewr(re_turns),
  times=10))[, c(1, 4, 5)]


## Benchmark for rolling median
# The function NPE::rolling_median() calculates the rolling median,
# the same as roll::roll_median().
all.equal(drop(NPE::rolling_median(re_turns, look_back=11))[-(1:10)],
          roll::roll_median(re_turns, width=11)[-(1:10)], check.attributes=FALSE)

# NPE::rolling_median() is about as fast as roll::roll_median().
summary(microbenchmark(
  parallel_Rcpp=NPE::rolling_median(re_turns, look_back=11),
  Rcpp=roll::roll_median(re_turns, width=11),
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary



# Benchmark for calculating rolling median. We are benchmarking this function against function "roll_median"
# (Also written in Rcpp) offered by library "roll". We are bit faster than this function due to 
# multithreaded approach.

summary(microbenchmark(
  Rcpp=rolling_median(re_turns, 30),
  roll_library=roll_median(re_turns, 30),
  times=10))[, c(1, 4, 5)]  



# Multi-Threaded function for calculating median absolute function over rolling window
rolling_mad(re_turns, 7)


x <- runif(n = 10, min = -100000, max = 100000)


# Benchmark the medcouple
summary(microbenchmark(
  Rcpp=NPE::med_couple(re_turns),
  Rcode=calc_skewr(re_turns),
  times=10))[, c(1, 4, 5)] 



# Hodges-Lehmann Estimator 
# Above 50 wilcox.test function will approximate the results.
da_ta <- coredata(re_turns)
all.equal(wilcox.test(da_ta, conf.int = TRUE)$estimate, NPE::hle(da_ta), check.attributes=FALSE)
summary(microbenchmark(
  Rcpp=NPE::hle(da_ta),
  Rcode=wilcox.test(da_ta, conf.int = TRUE)$estimate,
  times=10))[, c(1, 4, 5)] 

x <- round(runif(10), 2)
y <- round(runif(10), 2)
all.equal(wilcox.test(x, y)$p.value, drop(NPE::WilcoxanMannWhitneyTest(x, y)), check.attributes=FALSE)
summary(microbenchmark(
  Rcpp=WilcoxanMannWhitneyTest(x, y),
  Rcode=wilcox.test(x, y),
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary



## Theil-Sen Estimator 
# I've copied this code from - https://github.com/mrxiaohe/WRScpp
# R code to benchmark against this is available in package "WRS: - (https://www.r-bloggers.com/installation-of-wrs-package-wilcox-robust-statistics/)
x <- runif(10)
y <- runif(10)

library("WRS")
tsreg(x, y, FALSE)$coef    # there is very small difference in intercept because WRS package adjusts it for residuals and I don't.
TheilSenEstimator(x, y)


summary(microbenchmark(
  Rcpp=TheilSenEstimator(x, y),
  Rcode=tsreg(x, y, FALSE)$coef,
  times=10))[, c(1, 4, 5)] 


# PCA Using AcppArmadillo
x <- matrix(1:9, 3, 3)
calc_pca(x)
prcomp(x)

summary(microbenchmark(
  Rcpp=calc_pca(x),
  Rcode=prcomp(x),
  times=10))[, c(1, 4, 5)] 




############################
# Standard Errors of Nonparametric Estimators

# Calculate sampled XLF returns
set.seed(1121)
boot_sample <- lapply(1:1000, function(x) {
  re_turns[sample.int(n_rows, replace = TRUE)]
})  # end sapply


# Mean and standard error of median estimator from bootstrap
boot_data <- sapply(boot_sample, function(sample) {
  c(median_est=NPE::med_ian(sample),
    mean_est=mean(sample))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))


# Mean and standard error of MAD estimator from bootstrap
boot_data <- sapply(boot_sample, function(sample) {
  c(mad_est=NPE::calc_mad(sample), std_dev=sd(sample))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))

# Mean and standard error of medcouple estimator from bootstrap
boot_data <- sapply(boot_sample, function(sample) {
  c(med_couple=NPE::med_couple(sample), std_skew=calc_skewr(sample))
})  # end sapply
apply(boot_data, MARGIN=1, function(x) 
  c(mean=mean(x), std_error=sd(x)))




######################################################################
# Need to install BH to use boost libraries in this code.
# install.packages("BH")

# Non parametric tests
Rcpp::sourceCpp(file = "E:\\Summer term\\project\\NonParametricTests.cpp")

# Wilcoxan signed rank test.
x <- c(1.83,  0.50,  1.62,  2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
y <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.29)

wilcox.test(x, alternative = "greater")$p.value
WilcoxanSignedRankTest(x, alternative = "greater")

summary(microbenchmark(
  Rcpp=WilcoxanSignedRankTest(x, alternative = "greater"),
  Rcode=wilcox.test(x, alternative = "greater")$p.value,
  times=10))[, c(1, 4, 5)] 


# Wilcoxan-Mann_whitney rank sum test
x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
y <- c(1.15, 0.88, 0.90, 0.74, 1.21)

wilcox.test(x, y, alternative = "two.sided")$p.value
WilcoxanMannWhitneyTest(x, y, alternative = "two.sided")

summary(microbenchmark(
  Rcpp=WilcoxanMannWhitneyTest(x, y, alternative = "two.sided"),
  Rcode=wilcox.test(x, y, alternative = "greater")$p.value,
  times=10))[, c(1, 4, 5)] 


# Kruskal-Wallice test.
x <- c(2.9, 3.0, 2.5, 2.6, 3.2) 
y <- c(3.8, 2.7, 4.0, 2.4)      
z <- c(2.8, 3.4, 3.7, 2.2, 2.0)

kruskal.test(list(x, y, z))$p.value
KruskalWalliceTest(list(x, y, z))

summary(microbenchmark(
  Rcpp=KruskalWalliceTest(list(x, y, z)),
  Rcode=kruskal.test(list(x, y, z))$p.value,
  times=10))[, c(1, 4, 5)] 


# files with the bootstrap simulation 


