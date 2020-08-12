# Non-Parametric-Estimators
The library implements a variety of nonparametric estimators, including estimators of location
(median, Hodges-Lehmann), of dispersion (Median Absolute Deviation), of skewness (medcouple), and of
dependency-covariance (Theil-Sen). It also implements the nonparametric statistics of the Wilcoxon
Signed Rank test, the Mann-Whitney-Wilcoxon Rank Sum test, and the Kruskal-Wallis test.
The emphasis is on achieving very fast computation speeds. Parallel processing is employed on
multi-core CPUs, to further accelerate the calculations.
 
# Installation and Loading
```r
install.packages("devtools")
devtools::install_github(repo = "marvic24/Non-Parametric-Estimators")
library(NPE)
```
