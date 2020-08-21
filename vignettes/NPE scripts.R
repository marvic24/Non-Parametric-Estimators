######################################################################
### R scripts for testing the package NPE


############################
# Installation and Loading

library(Rcpp)
library(RcppArmadillo)
library(RcppParallel)
library(microbenchmark)

install.packages("devtools")
devtools::install_github(repo = "marvic24/Non-Parametric-Estimators")
library(NPE)
# detach("package:NPE")

## Create package documentation
setwd("C:/Develop/capstone/Sumit_Sethi/NPE")
devtools::document()
system("R CMD Rd2pdf C:/Develop/capstone/Sumit_Sethi/NPE")


## Tests for function NPE::med_ian()

# The function NPE::med_ian() calculates the median
da_ta <- rnorm(1e3)
all.equal(median(da_ta), NPE::med_ian(da_ta))

# NPE::med_ian() is much faster than stats::median()
library(microbenchmark)
summary(microbenchmark(
  rcpp=NPE::med_ian(da_ta),
  rcode=median(da_ta),
  times=10))[, c(1, 4, 5)]


## Tests for function NPE::rolling_median()

# The function NPE::rolling_median() calculates the rolling median,
# the same as roll::roll_median().
da_ta <- rnorm(1e6)
all.equal(drop(NPE::rolling_median(da_ta, look_back=11))[-(1:10)],
          roll::roll_median(da_ta, width=11)[-(1:10)])

# NPE::rolling_median() is about as fast as roll::roll_median().
summary(microbenchmark(
  parallel_rcpp=NPE::rolling_median(da_ta, look_back=11),
  rcpp=roll::roll_median(da_ta, width=11),
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary






# Benchmark for calculating median - RCpp code vs R code.
summary(microbenchmark(
  rcpp=NPE::med_ian(x),
  rcode=median(x),
  times=10))[, c(1, 4, 5)]  

library(roll)
x <- rnorm(1e3)

# Benchmark for calculating rolling median. We are benchmarking this function against function "roll_median"
# (Also written in RCpp) offered by library "roll". We are bit faster than this function due to 
# multithreaded approach.

summary(microbenchmark(
  mycode=rolling_median(x, 30),
  roll_library=roll_median(x, 30),
  times=10))[, c(1, 4, 5)]  


x <- runif(n = 10, min = -100000, max = 100000)

# Benchmark for calculating Median absolute deviation - RCpp code vs R code.
summary(microbenchmark(
  RCpp=medianAbsoluteDeviation(x),
  R=mad(x),
  times=10))[, c(1, 4, 5)] 


# Multi-Threaded function for calculating median absolute function over rolling window
x <- c(1:1000)
rolling_mad(x, 7)


# Hodges-Lehmann Estimator 
x <- runif(100)  # above 50 wilcox.test function will approximate the results.
wilcox.test(x, conf.int = TRUE)$estimate
hle(x)
summary(microbenchmark(
  RCpp=hle(x),
  R=wilcox.test(x, conf.int = TRUE)$estimate,
  times=10))[, c(1, 4, 5)] 

x <- round(runif(10), 2)
y <- round(runif(10), 2)
all.equal(wilcox.test(x, y)$p.value, drop(NPE::WilcoxanMannWhitneyTest(x, y)))
summary(microbenchmark(
  rcpp=WilcoxanMannWhitneyTest(x, y),
  rcode=wilcox.test(x, y),
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary


## Theil-Sen Estimator 
# I've copied this code from - https://github.com/mrxiaohe/WRScpp
# R code to benchmark against this is available in package "WRS: - (https://www.r-bloggers.com/installation-of-wrs-package-wilcox-robust-statistics/)
x <- runif(10)
y <- runif(10)

library("WRS")
tsreg(x, y)$coef    # there is very small difference in intercept because WRS package adjusts it for residuals and I don't.
TheilSenEstimator(x, y)


summary(microbenchmark(
  RCpp=TheilSenEstimator(x, y),
  R=tsreg(x, y)$coef,
  times=10))[, c(1, 4, 5)] 


# PCA Using AcppArmadillo
x <- matrix(1:9, 3, 3)
calc_pca(x)
prcomp(x)

summary(microbenchmark(
  RCpp=calc_pca(x),
  R=prcomp(x),
  times=10))[, c(1, 4, 5)] 


#############################################################################################
# need to install BH to use boost libraries in this code.
# install.packages("BH")

# Non parametric tests
Rcpp::sourceCpp(file = "E:\\Summer term\\project\\NonParametricTests.cpp")

# Wilcoxan signed rank test.
x <- c(1.83,  0.50,  1.62,  2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
y <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.29)

wilcox.test(x, alternative = "greater")$p.value
WilcoxanSignedRankTest(x, alternative = "greater")

summary(microbenchmark(
  RCpp=WilcoxanSignedRankTest(x, alternative = "greater"),
  R=wilcox.test(x, alternative = "greater")$p.value,
  times=10))[, c(1, 4, 5)] 


# Wilcoxan-Mann_whitney rank sum test
x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
y <- c(1.15, 0.88, 0.90, 0.74, 1.21)

wilcox.test(x, y, alternative = "two.sided")$p.value
WilcoxanMannWhitneyTest(x, y, alternative = "two.sided")

summary(microbenchmark(
  RCpp=WilcoxanMannWhitneyTest(x, y, alternative = "two.sided"),
  R=wilcox.test(x, y, alternative = "greater")$p.value,
  times=10))[, c(1, 4, 5)] 


# Kruskal-Wallice test.
x <- c(2.9, 3.0, 2.5, 2.6, 3.2) 
y <- c(3.8, 2.7, 4.0, 2.4)      
z <- c(2.8, 3.4, 3.7, 2.2, 2.0)

kruskal.test(list(x, y, z))$p.value
KruskalWalliceTest(list(x, y, z))

summary(microbenchmark(
  RCpp=KruskalWalliceTest(list(x, y, z)),
  R=kruskal.test(list(x, y, z))$p.value,
  times=10))[, c(1, 4, 5)] 


# files with the bootstrap simulation 


