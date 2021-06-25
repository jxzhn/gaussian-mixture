# include "gmm.hpp"
# include <stdio.h>

extern "C" {

__attribute__((visibility ("default")))
void gmmFit(const double* data, double* weights, double* means, double* covariances, int numData, int dim, int nComponent, double tol, int maxIter) {
    
    GaussianMixture gmm(dim, nComponent, weights, means, covariances, tol, maxIter);
    gmm.fit(data, numData);
    
}

} // extern "C"