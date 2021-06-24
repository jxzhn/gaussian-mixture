# include "gmm.hpp"
# include <stdio.h>
# include <sys/time.h>

inline double wall_time() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

extern "C" {

__attribute__((visibility ("default")))
void gmmFit(const double* data, double* weights, double* means, double* covariances, int numData, int dim, int nComponent, double tol, int maxIter) {
    GaussianMixture gmm(dim, nComponent, weights, means, covariances, tol, maxIter);

    double t1 = wall_time();

    gmm.fit(data, numData);

    double t2 = wall_time();
    printf("fitting finished in %lf seconds\n", t2 - t1);
}

}