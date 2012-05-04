#include "mex.h"

#include "kmeans.c"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    if (nrhs < 2)
        mexErrMsgTxt("kmeans: not enough arguments.");
    if (nrhs > 3)
        mexErrMsgTxt("kmeans: too many arguments.");
    if (nlhs > 2)
        mexErrMsgTxt("kmeans: at most two outputs.");

    if (!mxIsNumeric(prhs[0]) || !mxIsNumeric(prhs[1]) || !mxIsNumeric(prhs[1]))
        mexErrMsgTxt("kmeans: arguments must be numeric.");
    /* TODO: check that X is dense */
    if (!mxIsScalar(prhs[1]))
        mexErrMsgTxt("kmeans: cluster count must be scalar.");
    if (nrhs > 2 && !mxIsScalar(prhs[2]))
        mexErrMsgTxt("kmeans: population size must be scalar.");

    Problem p;

    N = mxGetM(prhs[0]);
    D = mxGetN(prhs[0]);
    C = (int)mxGetPr(prhs[1])[0];
    P = nrhs <= 2 ? 1 : (int)mxGetPr(prhs[2])[0];
    p.data = mxGetPr(prhs[0]);

    mwSize dims[2] = {N,1};
    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, false);
    int *clusters = (int*)mxGetPr(plhs[0]);

    double t = kmeans(p, clusters);

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        mxGetPr(plhs[1])[0] = t;
    }

    return;
}
