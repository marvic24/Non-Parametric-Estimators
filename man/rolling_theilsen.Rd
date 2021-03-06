% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{rolling_theilsen}
\alias{rolling_theilsen}
\title{Calculate the nonparametric Theil-Sen estimator of dependency-covariance for
two \emph{vectors} over rolling window using \code{RcppArmadillo} and 
\code{RcppParallel}}
\usage{
rolling_theilsen(vector_x, vector_y, look_back)
}
\arguments{
\item{\code{vector_x}}{A \emph{vector} independent (explanatory) data.}

\item{\code{vector_y}}{A \emph{vector} dependent data.}

\item{\code{look_back}}{The length of look back interval.}
}
\value{
A matrix \emph{matrix} containing two columns values i.e intercept and
  slope.
}
\description{
Calculate the nonparametric Theil-Sen estimator of dependency-covariance for
two \emph{vectors} over rolling window using \code{RcppArmadillo} and 
\code{RcppParallel}
}
\details{
The function \code{rolling_tse()} calculates the Theil-Sen
  estimator over rolling window using \code{RcppArmadillo} and \code{RcppParallel}.
  The function \code{rolling_tse()} is significantly faster than function
  it's  \code{R} implementation.
}
\examples{
\dontrun{
# Create vectors of random returns
vector_x <- rnorm(30)
vector_y <- rnorm(30)
# Define R function rolling_theilsenr
rolling_theilsenr <- function(x, y, look_back) {
  sapply(2:NROW(x), function(i) {
     NPE::theilSenEstimator(x[max(1, i-look_back+1):i], y[max(1, i-look_back+1):i])
  })  # end sapply
} 

# Compare rolling_theilsen() with rolling_theilsenr
all.equal((NPE::rolling_theilsen(vector_x, vector_y, 5 ))[-(1),], 
         t(rolling_theilsenr(vector_x, vector_y, 5)), check.attributes=FALSE)
# Compare the speed of RcppParallel with R code
library(microbenchmark)
summary(microbenchmark(
  Rcpp=NPE::rolling_theilsen(vector_x, vector_y, 10),
  Rcode=rolling_theilsenr(vector_x, vector_y, 10),
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary
}

}
