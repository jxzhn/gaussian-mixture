# include "gmm.hpp"
# include <stdio.h>
# include <sys/time.h>

extern "C" {

__attribute__((visibility ("default")))
void gmmFit(const float* data, float* weights, float* means, float* covariances, int numData, int dim, int nComponent, float tol, int maxIter) {
    GaussianMixture gmm(dim, nComponent, weights, means, covariances, tol, maxIter);

    double t1 = wall_time();

    gmm.fit(data, numData);

    double t2 = wall_time();
    printf("fitting finished in %lf seconds\n", t2 - t1);
}

}