---
title: "Capstone Project: Nonparametric Feature Engineering for Machine Learning"
author: by Sumit Mahaveer Sethi
supervisor: Jerzy Pawlowski
output:
  pdf_document: default
  html_document: default
---

### Project supervisor: Jerzy Pawlowski

### Abstract

The objective of the capstone project will be to implement a library of *C++* functions for calculating nonparametric estimators of time series data.  The estimators can be used for feature engineering for machine learning applications.  The library will also implement fast rolling functions over time series data.


### Overview

The standard statistical estimators of the moments (mean, variance, skewness) are often used as features in machine learning models, but they are not always well suited as features.  First, because financial time series data is often far from normally distributed, which violates the assumptions of many models, leading to the underestimation of the standard errors of predictions.  Secondly, standard estimators are not the most efficient for skewed distributions in the presence of noise.  On the other hand, nonparametric estimators are more robust to noise and can offer a better bias-variance tradeoff.  But nonparametric estimators often require calculating the sorts, ranks, and the quantiles of data, which are time consuming.  It's therefore better to implement them using fast *C++* functions, rather than using *Python* or *R*.  In addition, the nonparametric estimators need to be applied over a rolling time window.  This can be achieved by applying the nonparametric estimators in a *C++* loop over the time series data.

The capstone project will implement a variety of nonparametric estimators, including estimators of location (median, Hodges-Lehmann), of dispersion (Median Absolute Deviation), of skewness (medcouple), and of dependency-covariance (Theil-Sen).  It will also implement the nonparametric statistics of the Wilcoxon Signed Rank test, the Mann-Whitney-Wilcoxon Rank Sum test, and the Kruskal-Wallis test.  It will also implement nonparametric regression and PCA.  Many of these statistics are already implemented, but they are not easily applied over a rolling time window in an efficient way.  

The emphasis will be on achieving very fast computation speeds.  Parallel processing will be employed on multi-core CPUs, to further accelerate the calculations.  


### Deliverables

The *C++* library will be part of an *R* package, allowing users to easily call the *C++* functions from *R*.  The *R* environment will serve as the user interface for the *C++* library.  

The nonparametric estimators will be applied to empirical time series data.  Their standard errors will be estimated using bootstrap simulation, and they will be compared to those of standard estimators, to demonstrate that nonparametric estimators offer a better bias-variance tradeoff.


### Tools 

The capstone project will use a number of open source *C++* libraries designed for high performance computing and machine learning, like for example: *Armadillo* and *RcppParallel* (for parallel computing).  The capstone project will leverage existing open source *R* packages with similar functionality, including the packages *Rcpp*, *roll*, *RcppRoll*, *MarkowitzR*, and *SharpeR*.
