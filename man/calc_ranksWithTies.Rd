% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{calc_ranksWithTies}
\alias{calc_ranksWithTies}
\title{Calculate the ranks of the elements of a \emph{vector} or a single-column
\emph{time series} using \code{RcppArmadillo} and \code{boost}.}
\usage{
calc_ranksWithTies(vec_tor)
}
\arguments{
\item{\code{vec_tor}}{A \emph{vector} or a single-column \emph{time series}.}
}
\value{
A \emph{double vector} with the ranks of the elements of the
  \emph{vector}.
}
\description{
Calculate the ranks of the elements of a \emph{vector} or a single-column
\emph{time series} using \code{RcppArmadillo} and \code{boost}.
}
\details{
The function \code{calc_ranks()} calculates the ranks of the
  elements of a \emph{vector} or a single-column \emph{time series}.
  It \emph{averages} the ranks in case fo ties.
  It uses the \code{boost} function \code{boost::sort::parallel_stable_sort}
  for sorting array in parallel fashion.
}
\examples{
\dontrun{
# Create a vector of random data
da_ta <- round(runif (7), 2)
# Calculate the ranks of the elements in two ways
all.equal(rank(da_ta), drop(NPE::calc_ranksWithTies(da_ta)))
# Create a time series of random data
da_ta <- xts::xts(runif (7), seq.Date(Sys.Date(), by=1, length.out=7))
# Calculate the ranks of the elements in two ways
all.equal(rank(coredata(da_ta)), drop(NPE::calc_ranksWithTies(da_ta)))
# Compare the speed of this function with RcppArmadillo and R code
da_ta <- runif (7)
library(microbenchmark)
summary(microbenchmark(
  Rcpp=calc_ranks(da_ta),
  Rcode=rank(da_ta),
  boost=calc_ranksWithTies(da_ta) 
  times=10))[, c(1, 4, 5)]  # end microbenchmark summary
}

}
