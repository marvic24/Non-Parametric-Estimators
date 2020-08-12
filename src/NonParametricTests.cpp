#include <boost/sort/sort.hpp>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(BH)]]


//Enum representing alternatives for the hypothesis test
enum alternatives{two_sided, greater, less};

//Function returning enum values for corresponding string names of alternatives.
alternatives hash_it(const std::string& alternative)
{
  if(alternative == "greater" || alternative == "g" ) return  alternatives::greater;
  else if(alternative == "less" || alternative == "l" ) return alternatives::less;
  else return alternatives::two_sided;
}


//'This is an overload of a function \code{calc_ranksWithTies()}, which returns ranks of
//'the \emph{vector} or a single column \emph{time-series}. It also returns a \code{boolean}
//'variable indicating if there are ties in the data or not.
//'There is a function for calculating ranks in rcpp::armadillo, but it doesn't handle ties!

arma::vec calc_ranksWithTies(const arma::vec& vec_tor, bool &ties)
{
  typedef std::pair<double, int> ai_t;
  
  int n = vec_tor.n_elem;
  std::vector<ai_t> AI(n);
  
  for (int i = 0; i < n; i++)
    AI[i] = std::make_pair(vec_tor[i], i);
  
  boost::sort::parallel_stable_sort(AI.begin(), AI.end(), [](const ai_t& l, const ai_t& r) {return l.first < r.first; });
  
  
  arma::vec rnk(n);
  double rank = 1;
  int i = 0;
  int temp = 1;
  
  while (i < n)
  {
    int j = i;
    
    while ((j < n-1) && (AI[j].first == AI[j + 1].first))
    {
      ties = true;
      j++;
    }
    
    
    temp = j - i + 1;
    
    for (j = 0; j < temp; j++)
    {
      int id = AI[i + j].second;
      rnk[id] = rank + (temp - 1) * 0.5;
    }
    
    rank += temp;
    i += temp;
  }
  
  return rnk;
}





////////////////////////////////////////////////////////////
//' Calculate the ranks of the elements of a \emph{vector} or a single-column
//' \emph{time series} using \code{RcppArmadillo} and \code{boost}.
//' 
//' @param \code{vec_tor} A \emph{vector} or a single-column \emph{time series}.
//'
//' @return A \emph{double vector} with the ranks of the elements of the
//'   \emph{vector}.
//'
//' @details The function \code{calc_ranks()} calculates the ranks of the
//'   elements of a \emph{vector} or a single-column \emph{time series}.
//'   It \emph{averages} the ranks in case fo ties.
//'   It uses the \code{boost} function \code{boost::sort::parallel_stable_sort}
//'   for sorting array in parallel fashion.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random data
//' da_ta <- round(runif(7), 2)
//' # Calculate the ranks of the elements in two ways
//' all.equal(rank(da_ta), drop(HighFreq::calc_ranksWithTies(da_ta)))
//' # Create a time series of random data
//' da_ta <- xts::xts(runif(7), seq.Date(Sys.Date(), by=1, length.out=7))
//' # Calculate the ranks of the elements in two ways
//' all.equal(rank(coredata(da_ta)), drop(HighFreq::calc_ranksWithTies(da_ta)))
//' # Compare the speed of this function with RcppArmadillo and R code
//' da_ta <- runif(7)
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=calc_ranks(da_ta),
//'   rcode=rank(da_ta),
//'   boost=calc_ranksWithTies(da_ta) 
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
arma::vec calc_ranksWithTies(NumericVector vec_tor)
{
  bool ties = false;
  return calc_ranksWithTies(vec_tor, ties);
}





//'This function calculates the p values using normal approximation in case 
//'1. exact calculation is not requested and sample size is greater than 50.
//'2. There are ties in the data.
double pnorm_approximation(double z, const double& sigma, const bool& correct, const std::string& alternative = "two.sided")
{
  double correction = 0;
  if(correct)
  {
    switch(hash_it(alternative))
    {
    case alternatives::greater:{
      correction = 0.5;
      break;
    }
    case alternatives::less:{
      correction = -0.5;
      break;
    } 
    default:{
      correction = sign(z)*0.5;
      break;
    } 
    }
  }
  
  z = (z-correction)/sigma;
  double p =0;
  
  switch(hash_it(alternative))
  {
  case alternatives::greater:{
    p = R::pnorm(z, 0, 1, FALSE, FALSE);
    break;
  }
  case alternatives::less:{
    p = R::pnorm(z , 0, 1, TRUE, FALSE);
    break;
  } 
  default:{
    p = 2*(R::pnorm(z , 0, 1, TRUE, FALSE) <  R::pnorm(z, 0, 1, FALSE, FALSE) ?  R::pnorm(z, 0, 1, TRUE, FALSE) :  R::pnorm(z, 0, 1, FALSE, FALSE));
    break;
  }
  }
  
  return p;
}





////////////////////////////////////////////////////////////
//' Performs one sample Wilcoxan ranked sum test on \emph{vector} or a single-column
//' \emph{time series} using \code{RcppArmadillo} and \code{boost}.
//' 
//' @param \code{x} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{mu} A \emph{double} specifing an optional parameter used 
//'   to form null hypothesis. Default value is \emph{zero}.
//' @param \code{alternative} a \emph{character} string specifying the alternative
//'   hypothesis. It must be one of :
//'   \itemize{
//'     \item "two.sided" two tailed test.
//'     \item "greater" greater(right) tailed test.
//'     \item "less" smaller(left) tailed test.
//'   }
//'   (The default is \emph{two.sided} test.)
//' @param \code{exact} A {boolean} indicating whether an exact p-value should be computed.
//' @param \code{correct} A {boolean} indicating whether to apply continuity correction
//'   in normal approximation for the p-value.  
//'
//' @return A \emph{double} indicating p-value of the test.
//'
//' @details The function \code{WilcoxanSignedRankTest()} carries out the wilcoxan signed 
//'   rank test on \emph{vec_tor} and returns the \emph{p-value} of the test.
//'   By default (if \code{exact} is not specified), an exact p-value is computed if sample 
//'   contains less than 50 finite values and there are no ties. Otherwise, a normal approximation
//'   is used.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random data
//' da_ta <- round(runif(7), 2)
//' # Carry out wilcoxan signed rank test on the elements in two ways
//' all.equal(wilcox.test(da_ta)$p.value, drop(HighFreq::WilcoxanSignedRankTest(da_ta)))
//' # Create a time series of random data
//' da_ta <- xts::xts(runif(7), seq.Date(Sys.Date(), by=1, length.out=7))
//' # Calculate the ranks of the elements in two ways
//' all.equal(wilcox.test(coredata(da_ta))$p.value, drop(HighFreq::WilcoxanSignedRankTest(da_ta)))
//' # Compare the speed of Rcpp and R code
//' da_ta <- runif(10)
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=WilcoxanSignedRankTest(da_ta),
//'   rcode=wilcox.test(da_ta),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double WilcoxanSignedRankTest(arma::vec& x, const double& mu =0, const std::string& alternative = "two.sided", bool exact = false, const bool correct = true)
{
  x = x- mu;
  bool ties = false;
  arma::vec ranks = calc_ranksWithTies(abs(x), ties);
  
  int n = x.size();
  
  if(n<50)
    exact = true;
  
  //double statistic = 0;
  arma::uvec idx = find(ranks >0);
  double statistic = arma::accu(ranks.elem(idx));
  
  
  double p = 0.0;
  
  if(exact && !ties)
  {
    switch(hash_it(alternative))
    {
    case alternatives::greater:{
      p = R::psignrank(statistic-1, n, FALSE, FALSE);
      break;
    }
    case alternatives::less:{
      p = R::psignrank(statistic, n, TRUE, FALSE);
      break;
    } 
    default:{
      if(statistic > (n*(n+1)/4))
        p = R::psignrank(statistic-1, n, FALSE, FALSE);
      else
        p = R::psignrank(statistic, n, TRUE, FALSE);
      p = 2*p < 1.0 ? 2*p:1.0;
      break;
    }
    }
  }
  else
  {
    std::map<double, int> ties_table;
    for(double i:ranks)
      ties_table[i]++;
    
    NumericVector nties(ties_table.size());
    for(auto const i:ties_table)
      nties.push_back(i.second);
    
    double z = statistic -  n*(n+1)/4;
    double var_iance = n*(n+1)*(2*n+1)/24 - sum(nties*nties*nties - nties)/48;
    double sigma = std::sqrt(var_iance);
    
    p = pnorm_approximation(z, sigma, correct, alternative);
    
  }
  return p;
}





////////////////////////////////////////////////////////////
//' Performs two sample Wilcoxan-Mann-Whitney rank sum test also known as 
//' Mann-Whitney U Test on \emph{vector} or a single-column \emph{time series}
//' using \code{RcppArmadillo} and \code{boost}.
//' 
//' @param \code{x} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{y} A \emph{vector} or a single-column \emph{time series}.
//' @param \code{mu} A \emph{double} specifing an optional parameter used 
//'   to form null hypothesis. Default value is \emph{zero}.
//' @param \code{alternative} a \emph{character} string specifying the alternative
//'   hypothesis. It must be one of :
//'   \itemize{
//'     \item "two.sided" two tailed test.
//'     \item "greater" greater(right) tailed test.
//'     \item "less" smaller(left) tailed test.
//'   }
//'   (The default is \emph{two.sided} test.)
//' @param \code{exact} A {boolean} indicating whether an exact p-value should be computed.
//' @param \code{correct} A {boolean} indicating whether to apply continuity correction
//'   in normal approximation for the p-value.  
//'
//' @return A \emph{double} indicating p-value of the test.
//'
//' @details The function \code{WilcoxanMannWhitneyTest()} carries out the wilcoxan-Mann-Whitney
//'   signed rank test on \emph{x} & \emph{y} and returns the \emph{p-value} of the test.
//'   By default (if \code{exact} is not specified), an exact p-value is computed if sample 
//'   contains less than 50 finite values and there are no ties. Otherwise, a normal approximation
//'   is used.
//'
//' @examples
//' \dontrun{
//' # Create a vector of random data
//' x <- round(runif(10), 2)
//' y <- round(runif(10), 2)
//' # Carry out WMW signed rank test on the elements in two ways
//' all.equal(wilcox.test(x, y)$p.value, drop(HighFreq::WilcoxanMannWhitneyTest(x, y)))
//' # Compare the speed of Rcpp and R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=WilcoxanMannWhitneyTest(x, y),
//'   rcode=wilcox.test(x, y),
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double WilcoxanMannWhitneyTest(arma::vec& x, const arma::vec& y, const double& mu =0, const std::string& alternative = "two.sided", bool exact = false, const bool correct = true)
{
  x = x-mu;
  
  arma::vec xy = join_cols(x, y);
  
  bool ties = false;
  arma::vec ranks = calc_ranksWithTies(xy, ties);
  
  int n_x = x.n_elem;
  int n_y = y.n_elem;
  int n_xy = n_x+n_y;
  
  exact = (n_x <50) && (n_y <50);
  
  std::map<double, double> table;
  for(int i=0; i<n_xy; i++)
  {
    table[xy[i]] = ranks[i];
  }
  
  arma::vec x_ranks(n_x);
  for(int i =0; i<n_x; i++)
    x_ranks[i] = table[x[i]];
  
  double statistic = arma::accu(x_ranks) - n_x*(n_x+1)/2;
  double p = 0;
  
  if(exact && !ties)
  {
    switch(hash_it(alternative))
    {
    case alternatives::greater:{
      p = R::pwilcox(statistic-1, n_x, n_y, FALSE, FALSE);
      break;
    }
    case alternatives::less:{
      p = R::pwilcox(statistic, n_x, n_y, TRUE, FALSE);
      break;
    } 
    default:{
      if(statistic > (n_x*n_y/2))
        p = R::pwilcox(statistic-1, n_x, n_y, FALSE, FALSE);
      else
        p = R::pwilcox(statistic, n_x, n_y, TRUE, FALSE);
      p = 2*p < 1.0 ? 2*p:1.0;
      break;
    }
    }
  }
  
  else
  {
    std::map<double, int> ties_table;
    for(double i:ranks)
      ties_table[i]++;
    
    NumericVector nties(ties_table.size());
    for(auto const i:ties_table)
      nties.push_back(i.second);
    
    double z = statistic -  n_x*n_y/2;
    double var_iance = (n_x*n_y/12)*(n_x + n_y + 1) - sum(nties*nties*nties - nties)/((n_x + n_y)*(n_x + n_y - 1));
    double sigma = std::sqrt(var_iance);
    
    p = pnorm_approximation(z, sigma, correct, alternative);
    
  }
  
  return p;
}





////////////////////////////////////////////////////////////
//' Performs a Kruskal-Wallis rank sum test. using \code{Rcpp} and \code{boost}.
//' 
//' @param \code{x} A \emph{List} of numeric data vectors
//' 
//' @return A \emph{double} indicating p-value of the test.
//'
//' @details The function \code{KruskalWalliceTest()} performs a Kruskal-Wallis rank 
//'   sum test of the null hypothesis that the location parameters of the distribution
//'   of x are the same in each group. The alternative is that they differ in
//'   at least in one.
//'
//' @examples
//' \dontrun{
//' x <- c(2.9, 3.0, 2.5, 2.6, 3.2) # normal subjects
//' y <- c(3.8, 2.7, 4.0, 2.4)      # with obstructive airway disease
//' z <- c(2.8, 3.4, 3.7, 2.2, 2.0) # with asbestosis
//'
//' # Carry out Kruskal wallice rank sum test on the elements in two ways
//' all.equal(kruskal.test(list(x, y, z))$p.value, drop(HighFreq::KruskalWalliceTest(list(x, y, z))))
//' # Compare the speed of Rcpp and R code
//' library(microbenchmark)
//' summary(microbenchmark(
//'   rcpp=KruskalWalliceTest(list(x, y, z)),
//'   rcode=kruskal.test(list(x, y, z))$p.value,
//'   times=10))[, c(1, 4, 5)]  # end microbenchmark summary
//' }
//' 
//' @export
// [[Rcpp::export]]
double KruskalWalliceTest(List x)
{
  int n = x.size();
  arma::vec elements;
  arma::vec sizes(n);
  
  for(int i=0; i<n; ++i)
  {
    arma::vec temp = as<arma::vec>(x[i]);
    elements = join_cols(elements, temp);
    sizes[i] = temp.n_elem;
  }
  
  int n_e = elements.n_elem;
  
  bool ties = false;
  arma::vec ranks = calc_ranksWithTies(elements, ties);
  
  int k=0;
  double statistic = 0;
  
  for(int i=0; i<n; i++)
  {
    int m = sizes[i];
    double rank_sum = 0;
    for(int j=0; j<m; j++)
    {
      rank_sum += ranks[j+k];
      k++;
    }
    
    statistic += (rank_sum*rank_sum)/m;
    
  }
  
  std::map<double, int> ties_table;
  for(double i:ranks)
    ties_table[i]++;
  
  NumericVector nties(ties_table.size());
  for(auto const i:ties_table)
    nties.push_back(i.second);
  
  statistic = ((12*statistic/(n_e*(n_e+1))-3*(n_e+1))/
    (1-sum(nties*nties*nties - nties)/(n_e*n_e*n_e - n_e)));
  
  int paramter = n-1;
  
  double p = R::pchisq(statistic, paramter, false, false); 
  
  return p; 
}


