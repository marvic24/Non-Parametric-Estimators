#define STRICT_R_HEADERS
#include <algorithm>

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;


//////////////////////////////////////////
// Functions for Non Parametric Estimators
//////////////////////////////////////////


////////////////////////////////////////////////////////////
//' Calculate the median of a  \emph{vector} or a single-column \emph{time series}
//' using \code{RcppArmadillo}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' 
//' @return A single \emph{double} value representing median of the vector.
//'
//' @details The function \code{med_ian()} calculates the median of the \emph{vector},
//'   using \code{RcppArmadillo}. The function \code{med_ian()} is several times faster
//'   than \code{median()} in \code{R}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' # Compare med_ian() with median()
//' all.equal(drop(HighFreq::med_ian(re_turns)), 
//'   median(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=HighFreq::med_ian(re_turns),
//'   rcode=median(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double med_ian(const arma::vec& vec_tor)
{
  return arma::median(vec_tor);
}





////////////////////////////////////////////////////////////
//' Worker function for calculating median over rolling window by using parallel processing.

struct parallel_rolling_median : public Worker
{
  //input vector 
  const RVector<double> vec_tor;
  const int look_back;
  
  //output (pass by reference)
  arma::vec& med_ians;
  
  //constructor
  parallel_rolling_median(const NumericVector vec_tor,
                          const int look_back,
                          arma::vec& med_ians) : vec_tor(vec_tor), look_back(look_back), med_ians(med_ians){}
  
  void operator()(std::size_t begin, std::size_t end){
    
    for(std::size_t i = begin; i < end; i++)
    {
      int start_index = std::max((std::size_t)(0), (i-look_back + 1));
      arma::vec temp(i-start_index+1);
      
      for(std::size_t j = start_index ; j<= i; j++)
        temp[j-start_index] = vec_tor[j];
      
      med_ians[i] = arma::median(temp);
    }
  }
};





////////////////////////////////////////////////////////////
//' Calculate the rolling median over a \emph{vector} or a single-column \emph{time series}
//' using \code{RcppArmadillo} and \code{RcppParallel}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{look_back} The length of look back interval, equal to the
//'   number of elements of data used for calculating the median.
//'   
//' @return A column \emph{vector} of the same length as the argument
//'   \code{vect_tor}.
//'
//' @details The function \code{rolling_median()} calculates a vector of
//'   rolling medians, over a \emph{vector} of data, using \emph{RcppArmadillo}
//'   and \emph{RcppParallel}. The function \code{rolling_median()} is faster
//'   than \code{roll::roll_median()} which uses \code{Rcpp}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' # Compare rolling_median() with roll::roll_median()
//' all.equal(drop(HighFreq::rolling_median(re_turns)), 
//'   roll::roll_median(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   parallel_rcpp=HighFreq::rolling_median(re_turns),
//'   rcpp=roll::roll_median(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::vec rolling_median(NumericVector vec_tor, int look_back)
{
  int n = vec_tor.size();
  arma::vec results(n);
  
  parallel_rolling_median media_n(vec_tor, look_back, results);
  
  results[0] = vec_tor[0];
  parallelFor(1, vec_tor.length(), media_n);
  
  return results;
}





////////////////////////////////////////////////////////////
//' Calculate the Median absolute deviation of a  \emph{vector} or a single-column
//'  \emph{time series} using \code{RcppArmadillo}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' 
//' @return A single \emph{double} value representing median absolue deviation of 
//'   the vector.
//'
//' @details The function \code{medianAbsoluteDeviation()} calculates the median of 
//'   the \emph{vector}, using \code{RcppArmadillo}. The function \code{medianAbsoluteDeviation()}
//'   is several times faster than \code{mad()} in \code{R}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' # Compare medianAbsoluteDeviation() with mad()
//' all.equal(drop(HighFreq::medianAbsoluteDeviation(re_turns)), 
//'   mad(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=HighFreq::medianAbsoluteDeviation(re_turns),
//'   rcode=mad(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double medianAbsoluteDeviation(arma::vec& vec_tor)
{
  return med_ian(arma::abs(vec_tor - med_ian(vec_tor)));
}





////////////////////////////////////////////////////////////
//' Worker function for calculating median absolute deviation over rolling window
//' by using parallel processing.


struct parallel_rolling_mad : public Worker
{
  //input vector 
  const RVector<double> vec_tor;
  int look_back;
  
  //output (pass by reference)
  arma::vec& m_ad;
  
  //constructor
  parallel_rolling_mad(const NumericVector vec_tor,
                          const int look_back,
                          arma::vec& m_ad) : vec_tor(vec_tor), look_back(look_back), m_ad(m_ad){}
  
  void operator()(std::size_t begin, std::size_t end){
    
    for(std::size_t i = begin; i < end; i++)
    {
      int start_index = std::max((std::size_t)(0), (i-look_back+1));
      
      arma::vec temp(i-start_index+1);
      for(std::size_t j = start_index ; j <= i; j++)
      {
        temp[j-start_index] = vec_tor[j];
      }
      
      m_ad[i] = med_ian(arma::abs(temp - med_ian(temp)));
    }
  }
};





////////////////////////////////////////////////////////////
//' Calculate the rolling median absolute deviation over a \emph{vector} or
//' a single-column \emph{time series} using \code{RcppArmadillo} and \code{RcppParallel}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{look_back} The length of look back interval, equal to the
//'   number of elements of data used for calculating the median.
//'   
//' @return A column \emph{vector} of the same length as the argument
//'   \code{vect_tor}.
//'
//' @details The function \code{rolling_mad()} calculates a vector of
//'   rolling medians, over a \emph{vector} of data, using \emph{RcppArmadillo}
//'   and \emph{RcppParallel}. 
//'   
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' rolling_mad(re_turns)
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::vec rolling_mad(NumericVector vec_tor, int look_back)
{
  int n = vec_tor.size();
  arma::vec results(n);
  
  parallel_rolling_mad ma_d(vec_tor, look_back, results);
  
  results[0] = 0;
  parallelFor(1, vec_tor.length(), ma_d);
  
  return results;
}





////////////////////////////////////////////////////////////
//' Worker function for calculating pair averages needed for Hodges-Lehmann estimator
//' by using parallel processing.

struct pair_averages : public Worker
{
  //input vector 
  const RVector<double> vec_tor;
  int n;
  
  //output (pass by reference)
  arma::vec& ave_rages;
  
  //constructor
  pair_averages(const NumericVector vec_tor, arma::vec& ave_rages) : vec_tor(vec_tor), ave_rages(ave_rages){ n = vec_tor.size();}
  
  void operator()(std::size_t begin_index, std::size_t end_index){

    for(std::size_t i = begin_index; i < (end_index); i++)
    {
      for(std::size_t j = (i+1) ; j< (size_t)(n); j++)
      {
        int idx = (n*(n-1)/2) - (n-i)*(n-i-1)/2 - (i+1);
        
        ave_rages[idx + j] = (vec_tor[i] + vec_tor[j])/2;
      }
    }
  }
  
};






////////////////////////////////////////////////////////////
//' Calculate the non parametric Hodges-Lehmann estimator of location for a
//' \emph{vector} or a single-column \emph{time series} using \code{RcppArmadillo}
//' and \code{RcppParallel}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' 
//' @return A single \emph{double} value representing Hodges-Lehmann estimator of 
//'   the vector.
//'
//' @details The function \code{hle()} calculates the Hodges-Lehmann estimator of 
//'   the \emph{vector}, using \code{RcppArmadillo} and \code{RcppParallel}. The 
//'   function \code{hle()} is very much faster than function \code{wilcox.test()}
//'   in \code{R}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' # Compare hle() with wilcox.test()
//' all.equal(drop(HighFreq::hle(re_turns)), 
//'   wilcox.test(re_turns, conf.int = TRUE))
//' # Compare the speed of RcppParallel with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=HighFreq::hle(re_turns),
//'   rcode=wilcox.test(re_turns, conf.int = TRUE),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double hle(NumericVector vec_tor)
{
  int n = vec_tor.size();
  arma::vec pairs(n*(n-1)/2);
  
  pair_averages avera_ges(vec_tor, pairs);
  
  parallelFor(0, vec_tor.length()-1, avera_ges);

  return med_ian(pairs);
}





//Function to calculate slopesof the all pairs for Theil-Sen Estimator.

NumericVector outer_pos( arma::vec vector_x, arma::vec vector_y ){
  NumericVector output;
  int n = vector_x.size();
  double temp;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      temp = vector_x[j] - vector_x[i];
      if(temp > 0){
        output.push_back(( vector_y[j] - vector_y[i]) / temp);
      }
    }
  } 
  return output;
}

//Ranks vector_y according to sorted vector_X.
NumericVector ts_proc( arma::vec vector_x, arma::vec vector_y ){
  arma::uvec ind = arma::sort_index( vector_x );
  
  return outer_pos( vector_x.elem( ind ), vector_y.elem( ind ) );
}





////////////////////////////////////////////////////////////
//' Calculate the non parametric Theil-Sen estimator of dependency-covariance for two
//' \emph{vectors}  using \code{RcppArmadillo}
//' 
//' @param \code{vector_x} A \emph{vector} independent (explanatory) data.
//' @param \code{vector_y} A \emph{vector} dependent data.
//' 
//' @return A column \emph{vector} containing two values i.e intercept and slope
//'
//' @details The function \code{TheilSenEstimator()} calculates the Theil-Sen estimator of 
//'   the \emph{vector}, using \code{RcppArmadillo} . The function \code{TheilSenEstimator()}
//'   is significantly faster than function \code{WRS::tsreg()} in \code{R}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' vector_x <- rnorm(10)
//' vactor_y <- rnorm(10)
//' # Compare TheilSenEstimator() with tsreg()
//' # Compare the speed of RcppParallel with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=HighFreq::TheilSenEstimator(vector_x, vector_y),
//'   rcode=WRS(vector_x, vector_y),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]  
NumericVector TheilSenEstimator(arma::vec x, arma::vec y)
{
  NumericVector coef(2);
  NumericVector v1v2 = ts_proc( x, y );
  //int n_s = v1v2.size();	
  coef[1] = med_ian( v1v2 );
  
  coef[0] = med_ian(y - coef[1] * x);
  return coef;
}





////////////////////////////////////////////////////////////
//' Performs a principle component analysis on given \emph{matrix} or \emph{time
//' series} using \code{RcppArmadillo}.
//' 
//' @param \code{mat_rix} A \emph{matrix} or a \emph{time series}.
//'
//' @return A \emph{matrix} of variable loadings (i.e. a matrix whose columns contain
//'   the eigenvectors).
//'
//' @details The function \code{calc_pca()} performs a principle component analysis
//'    on a \emph{matrix} using \code{RcppArmadillo}. 
//'   
//' @examples
//' \dontrun{
//' # Create a matrix of random returns
//' re_turns <- matrix(rnorm(5e6), nc=5)
//' # Compare calc_pca() with standard prcomp()
//' all.equal(drop(HighFreq::calc_pca(re_turns)), 
//'   prcomp(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=HighFreq::calc_pca(re_turns),
//'   rcode=prcomp(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::mat calc_pca(arma::mat& mat_rix) {
  return arma::princomp(mat_rix);
}
