#include "mex.h"
#include "cuda_match.h"
#include <cstring>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nlhs > 1 || nrhs > 3)
        mexErrMsgTxt("Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("Inputs must be single float matrix.");
    if (mxGetM(prhs[0]) != mxGetM(prhs[1]))
        mexErrMsgTxt("Features dimensions are not equal.");

    int vdim = mxGetM(prhs[0]);
    int K1 = mxGetN(prhs[0]);
    int K2 = mxGetN(prhs[1]);
    float *A = (float *)mxGetData(prhs[0]);
    float *B = (float *)mxGetData(prhs[1]);
    float thresh = 1.5;
    if (nrhs != 2)
        thresh = *(float *)mxGetData(prhs[2]);

    //printf("%d %d %d %d\n", mxGetN(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[1]), mxGetM(prhs[1]));

    int sz;
    float *C = (float *)malloc(sizeof(float) * K1);
    sz = cuda_match_two_set(A, K1, B, K2, vdim, C, thresh);

    mwSize dims[2] = {2, sz};

    plhs[0] = mxCreateNumericMatrix (2, sz, mxSINGLE_CLASS, mxREAL);

    float *result = (float *)mxGetData(plhs[0]);
    memcpy(result, C, sz * 2 * sizeof(float));

}