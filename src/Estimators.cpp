#define STRICT_R_HEADERS
#include <algorithm>
#include <vector>

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace RcppParallel;
using namespace Rcpp;
using namespace arma;
using namespace std;

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
//' all.equal(drop(NPE::med_ian(re_turns)), 
//'   median(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::med_ian(re_turns),
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
  // Input vector 
  const RVector<double> vec_tor;
  const int look_back;
  
  // Output (pass by reference)
  arma::vec& med_ians;
  
  // Constructor
  parallel_rolling_median(const NumericVector vec_tor,
                          const int look_back,
                          arma::vec& med_ians) : vec_tor(vec_tor), look_back(look_back), med_ians(med_ians){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t i = begin; i < end; i++) {
      int start_index = std::max((std::size_t)(0), (i-look_back + 1));
      arma::vec temp(i-start_index+1);
      
      for (std::size_t j = start_index; j <= i; j++)
        temp[j-start_index] = vec_tor[j];
      
      med_ians[i] = arma::median(temp);
    }  // end for
  }  // end operator
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
//' all.equal(drop(NPE::rolling_median(re_turns, look_back=11)), 
//'   roll::roll_median(re_turns, width=11))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   parallel_rcpp=NPE::rolling_median(re_turns),
//'   rcpp=roll::roll_median(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::vec rolling_median(NumericVector vec_tor, int look_back) {
  int n = vec_tor.size();
  arma::vec results(n);
  
  parallel_rolling_median media_n(vec_tor, look_back, results);
  
  results[0] = vec_tor[0];
  parallelFor(1, vec_tor.length(), media_n);
  
  return results;
}  // end rolling_median





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
//' all.equal(drop(NPE::medianAbsoluteDeviation(re_turns)), 
//'   mad(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::medianAbsoluteDeviation(re_turns),
//'   rcode=mad(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double medianAbsoluteDeviation(arma::vec& vec_tor) {
  
  return med_ian(arma::abs(vec_tor - med_ian(vec_tor)));
  
}  // end medianAbsoluteDeviation





////////////////////////////////////////////////////////////
//' Worker function for calculating median absolute deviation over rolling window
//' by using parallel processing.

// Define structure 
struct parallel_rolling_mad : public Worker
{
  // input vector 
  const RVector<double> vec_tor;
  int look_back;
  
  // Output (pass by reference)
  arma::vec& m_ad;
  
  // Define constructor
  parallel_rolling_mad(const NumericVector vec_tor,
                          const int look_back,
                          arma::vec& m_ad) : vec_tor(vec_tor), look_back(look_back), m_ad(m_ad){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t i = begin; i < end; i++) {
      int start_index = std::max((std::size_t)(0), (i-look_back+1));
      
      arma::vec temp(i-start_index+1);
      
      for (std::size_t j = start_index; j <= i; j++) {
        temp[j-start_index] = vec_tor[j];
      }  // end if
      
      m_ad[i] = med_ian(arma::abs(temp - med_ian(temp)));
    }  // end for
  }  // end operator
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
}  // end rolling_mad





////////////////////////////////////////////////////////////
//' Worker function for calculating pair averages needed for Hodges-Lehmann estimator
//' by using parallel processing.

struct pair_averages : public Worker
{
  // Input vector 
  const RVector<double> vec_tor;
  int n;
  
  // Output (pass by reference)
  arma::vec& ave_rages;
  
  // Constructor
  pair_averages(const NumericVector vec_tor, arma::vec& ave_rages) : vec_tor(vec_tor), ave_rages(ave_rages){ n = vec_tor.size();}
  
  void operator()(std::size_t begin_index, std::size_t end_index){

    for (std::size_t i = begin_index; i < (end_index); i++) {
      for (std::size_t j = (i+1); j< (size_t)(n); j++) {
        int idx = (n*(n-1)/2) - (n-i)*(n-i-1)/2 - (i+1);
        
        ave_rages[idx + j] = (vec_tor[i] + vec_tor[j])/2;
      }  // end for
    }  // end for
  }  // end operator
  
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
//' all.equal(drop(NPE::hle(re_turns)), 
//'   wilcox.test(re_turns, conf.int = TRUE))
//' # Compare the speed of RcppParallel with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::hle(re_turns),
//'   rcode=wilcox.test(re_turns, conf.int = TRUE),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double hle(NumericVector vec_tor) {
  
  int n = vec_tor.size();
  arma::vec pairs(n*(n-1)/2);
  
  pair_averages avera_ges(vec_tor, pairs);
  
  parallelFor(0, vec_tor.length()-1, avera_ges);

  return med_ian(pairs);
}  // end hle








// Function to calculate slopes of the all pairs for Theil-Sen Estimator.
NumericVector ts_proc(arma::vec vector_x, arma::vec vector_y) {

  NumericVector output;
  int n = vector_x.size();
  double temp;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      temp = vector_x[j] - vector_x[i];
      if (temp > 0) {
        output.push_back((vector_y[j] - vector_y[i]) / temp);
      }  // end if
    }  // end for
  }  // end for
  return output;
  
}  // end ts_proc





////////////////////////////////////////////////////////////
//' Calculate the non parametric Theil-Sen estimator of dependency-covariance for two
//' \emph{vectors}  using \code{RcppArmadillo}
//' 
//' @param \code{vector_x} A \emph{vector} independent (explanatory) data.
//' @param \code{vector_y} A \emph{vector} dependent data.
//' 
//' @return A column \emph{vector} containing two values i.e intercept and slope
//'
//' @details The function \code{theilSenEstimator()} calculates the Theil-Sen estimator of 
//'   the \emph{vector}, using \code{RcppArmadillo} . The function \code{theilSenEstimator()}
//'   is significantly faster than function \code{WRS::tsreg()} in \code{R}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' vector_x <- rnorm(10)
//' vactor_y <- rnorm(10)
//' # Compare theilSenEstimator() with tsreg()
//' # Compare the speed of RcppParallel with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::theilSenEstimator(vector_x, vector_y),
//'   rcode=WRS(vector_x, vector_y),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]  
NumericVector theilSenEstimator(arma::vec x, arma::vec y) {
  NumericVector coef(2);
  NumericVector v1v2 = ts_proc(x, y);
  //int n_s = v1v2.size();	
  coef[1] = med_ian(v1v2);
  
  coef[0] = med_ian(y - coef[1] * x);
  return coef;
}  // end theilSenEstimator





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
//' all.equal(drop(NPE::calc_pca(re_turns)), 
//'   prcomp(re_turns))
//' # Compare the speed of RcppArmadillo with R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::calc_pca(re_turns),
//'   rcode=prcomp(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::mat calc_pca(arma::mat& mat_rix) {
  
  return arma::princomp(mat_rix);
  
}  // end calc_pca





// Template Function to return sign of the datatype.
template <typename T>
int signum(T val) {
  return (T(0) < val) - (val < T(0));
}

// Template Function fo calculate the sum of the elements
template <typename Container, typename Out = typename Container::value_type>
Out sum(const Container& C) {
  return std::accumulate(C.begin(), C.end(), Out());
}



// This computes the weighted median of array A with corresponding weights W.

double wmedian(const std::vector<double>& A, const std::vector<long>& W){
  
  typedef pair<double, long> aw_t;
  
  long n = A.size();
  std::vector<aw_t> AW(n);
  
  for (long i = 0; i < n; i++)
    AW[i] = make_pair(A[i], W[i]);
  
  long wtot = sum(W);
  
  long beg = 0;
  long end = n - 1;
  
  while (true) {
    long mid = (beg + end) / 2;
    
    std::nth_element(AW.begin(), AW.begin() + mid, AW.end(),
                [](const aw_t& l, const aw_t& r) {return l.first > r.first; });
    
    double trial = AW[mid].first;
    
    long wleft = 0, wright = 0;
    for (aw_t& aw : AW) {
      double a;
      long w;
      
      std::tie(a, w) = std::move(aw);
      
      if (a > trial)
        wleft += w;
      else
        // This also includes a == trial, i.e. the "middle" weight.
        wright += w;
      
    } // end for
    
    
    if (2 * wleft > wtot)
      end = mid;
    else if (2 * wright < wtot)
      beg = mid;
    else
      return trial;
    
  } // end while
  
  
  return 0;
  
}

// Dunction to calculate medcouple.
// This code is refered from http://inversethought.com/hg/medcouple/file/default/jmedcouple.c%2B%2B.
double medcouple(const NumericVector X, double eps1, double eps2)
  
{
  
  long n = X.size(), n2 = (n - 1) / 2;
  
  if (n < 3)
    return 0;
  
  
  NumericVector Z = clone(X);
  std::sort(Z.begin(), Z.end(), std::greater<double>());
  
  
  double Zmed;
  if (n % 2 == 1)
    Zmed = Z[n2];
  else
    Zmed = (Z[n2] + Z[n2 + 1]) / 2;
  
  
  // Check if the median is at the edges up to relative epsilon
  if (abs(Z[0] - Zmed) < eps1 * (eps1 + abs(Zmed)))
    return -1.0;
  
  if (abs(Z[n - 1] - Zmed) < eps1 * (eps1 + abs(Zmed)))
    return 1.0;
  
  
  // Center Z wrt median, so that median(Z) = 0.
  std::for_each(Z.begin(), Z.end(), [&](double& z) { z -= Zmed; });
  
  
  // Scale inside [-0.5, 0.5], for greater numerical stability.
  double Zden = 2 * std::max(Z[0], -Z[n - 1]);
  
  std::for_each(Z.begin(), Z.end(), [&](double& z) {z /= Zden; });
  
  Zmed /= Zden;
  
  double Zeps = eps1 * (eps1 + abs(Zmed));
  
  // These overlap on the entries that are tied with the median
  std::vector<double> Zplus, Zminus;
  
  std::copy_if(Z.begin(), Z.end(), std::back_inserter(Zplus),
          
          [=](double z) {return z >= -Zeps; });
  
  std::copy_if(Z.begin(), Z.end(), std::back_inserter(Zminus),
          
          [=](double z) {return Zeps >= z; });
  
  
  long n_plus = Zplus.size();
  long n_minus = Zminus.size();
  
  
  /*
   
   Kernel function h for the medcouple, closing over the values of
   
   Zplus and Zminus just defined above.
   
   
   In case a and be are within epsilon of the median, the kernel
   
   is the signum of their position.
   
   */
  
  auto h_kern = [&](long i, long j) {
    
    double a = Zplus[i];
    double b = Zminus[j];
    
    double h;
    
    if (abs(a - b) <= 2 * eps2)
      h = signum(n_plus - 1 - i - j);
    else
      h = (a + b) / (a - b);
    
    return h;
    
  }; // end h_kern
  
  
  // Init left and right borders
  
  std::vector<long> L(n_plus, 0);
  std::vector<long> R(n_plus, n_minus - 1);
  
  long Ltot = 0;
  long Rtot = n_minus * n_plus;
  long medc_idx = Rtot / 2;
  
  // kth pair algorithm (Johnson & Mizoguchi)
  while (Rtot - Ltot > n_plus) {
    
    // First, compute the median inside the given bounds
    std::vector<double> A;
    std::vector<long> W;
    
    for (long i = 0; i < n_plus; i++) {
      if (L[i] <= R[i]) {
        
        A.push_back(h_kern(i, (L[i] + R[i]) / 2));
        W.push_back(R[i] - L[i] + 1);
        
      } // end if
      
    } // end for
    
    double Am = wmedian(A, W);
    double Am_eps = eps1 * (eps1 + abs(Am));
    
    // Compute new left and right boundaries, based on the weighted median
    
    std::vector<long> P(n_plus), Q(n_plus);
    
    {
      long j = 0;
      
      for (long i = n_plus - 1; i >= 0; i--) {
        
        while (j < n_minus and h_kern(i, j) - Am > Am_eps)
          j++;
        
        P[i] = j - 1;
        
      } // end for
    } // end scope
    
    
    {
      long j = n_minus - 1;
      for (long i = 0; i < n_plus; i++) {
        
        while (j >= 0 and h_kern(i, j) - Am < -Am_eps)
          j--;
        
        Q[i] = j + 1;
        
      } // end for
      
    } // end scope
    
    
    long sumP = sum(P) + n_plus;
    long sumQ = sum(Q);
    
    if (medc_idx <= sumP - 1) {
      R = P;
      Rtot = sumP;
      
    } // end if
    
    else {
      if (medc_idx > sumQ - 1) {
        L = Q;
        Ltot = sumQ;
      } // end if
      else
        return Am;
      
    } // end else
    
  } // end while
  
  
  // Didn't find the median, but now we have a very small search space
  // to find it in, just between the left and right boundaries. This
  // space is of size Rtot - Ltot which is <= n_plus
  
  std::vector<double> A;
  
  for (long i = 0; i < n_plus; i++) {
    
    for (long j = L[i]; j <= R[i]; j++)
      
      A.push_back(h_kern(i, j));
    
  } // end for
  
  std::nth_element(A.begin(), A.begin() + (medc_idx - Ltot), A.end(),
              
              [](double x, double y) {return x > y; });
  
  
  
  double Am = A[medc_idx - Ltot];
  return Am;
  
} // end medcouple






////////////////////////////////////////////////////////////
//' Calculate the medcouple of a  \emph{vector} or a single-column \emph{time series}
//' using \code{Rcpp}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{eps1} A \emph{double} Tolerance of the algorithm.
//' @param \code{eps2} A \emph{couble} Tolerance of the algorithm..
//' 
//' 
//' @return A single \emph{double} value representing medcouple of the vector.
//'
//' @details The function \code{med_couple()} calculates the medcouple of the \emph{vector},
//'   using \code{Rcpp}. The function \code{med_couple()} is several times faster
//'   than \code{mc()} in package \code{robustbase}.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random returns
//' re_turns <- rnorm(1e6)
//' # Compare med_couple() with mc()
//' all.equal(drop(NPE::med_couple(re_turns)), 
//'   robustbase::mc(re_turns))
//' # Compare the speed of NPE with Robustbase code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=NPE::med_couple(re_turns),
//'   robustbase=robustbase::mc(re_turns),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double med_couple(NumericVector x, double eps1 = 1e-14, double eps2 = 1e-15){
  
  return medcouple(x, eps1, eps2);
  
}

