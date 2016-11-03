#include "cuda_match.h"
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <math_constants.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>



#define BATCH_SIZE 128

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        assert(0);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "unknown CUBLAS error";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d \n error  %s\nterminating!\n", file, line, 
			_cublasGetErrorEnum(err)); 
			cudaDeviceReset(); assert(0); 
	}
}


inline int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
#define cublasSafeCall( err)  __cublasSafeCall(err, __FILE__, __LINE__)


int cuda_match_two_set(const float *A, const int K1, const float *B, const int K2, const int Ndims, float *C, const float Thresh)
{    
	
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

   

    thrust::device_vector<float> x(BATCH_SIZE * iDivUp(K1, BATCH_SIZE) * Ndims);
    thrust::copy(A, A + K1 * Ndims, x.begin());
    thrust::device_vector<float> y(B, B + K2 * Ndims);    
   

    thrust::host_vector<float> h_dots(K2 * BATCH_SIZE);
    thrust::device_vector<float> d_dots = h_dots;

    thrust::host_vector<float> h_dist(K2 * BATCH_SIZE);
    thrust::device_vector<float> d_dist(K2 * BATCH_SIZE);

    

	thrust::host_vector<int> cpu_min_index(BATCH_SIZE * iDivUp(K1, BATCH_SIZE));
	thrust::host_vector<int> cpu_min_index2(BATCH_SIZE * iDivUp(K1, BATCH_SIZE));
	thrust::host_vector<float> cpu_min_value2(BATCH_SIZE * iDivUp(K1, BATCH_SIZE));
	thrust::host_vector<float> cpu_min_value(BATCH_SIZE * iDivUp(K1, BATCH_SIZE));

	
    float alpha = 1.f;
    float beta  = 0.f;
   
    for (int b = 0; b < iDivUp(K1, BATCH_SIZE); b++)
    {
    	alpha = 1.f;
    	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K2, BATCH_SIZE, Ndims, &alpha,
                               thrust::raw_pointer_cast(y.data()), Ndims, 
                               (float*)(thrust::raw_pointer_cast(x.data())) + b * BATCH_SIZE * Ndims , Ndims, &beta,
                               thrust::raw_pointer_cast(d_dots.data()), K2));
    	
#ifdef DEBUG
    	printf("(A*B)\n");
    	h_dots = d_dots;
    	for(int i = 0; i < BATCH_SIZE; i++)
    	{
		    for(int j = 0; j < K2; j++)
		    	printf("%f ", h_dots[i*K2+j]);
		    printf("\n");
    	}
    	fflush( stdout );
#endif

    	alpha = -2.f;
    	thrust::fill(d_dist.begin(), d_dist.end(), 2.f);
    	cublasSafeCall(cublasSaxpy(handle, K2 * BATCH_SIZE, &alpha, 
    								thrust::raw_pointer_cast(d_dots.data()), 1,
    								thrust::raw_pointer_cast(d_dist.data()), 1));
#ifdef DEBUG
    	printf("2 - 2(A*B)\n");
    	h_dist = d_dist;
    	for(int i = 0; i < BATCH_SIZE; i++)
    	{
		    for(int j = 0; j < K2; j++)
		    	printf("%f ", h_dist[i*K2+j]);
		    printf("\n");
    	}
    	fflush( stdout );
#endif
    	//find best and second best
    	for (int i = 0; i < BATCH_SIZE; i++)
		{
			cublasIsamin(handle, K2, ((float*)thrust::raw_pointer_cast(d_dist.data())) + K2 * i, 1, 
						((int*)thrust::raw_pointer_cast(cpu_min_index.data())) + i + b * BATCH_SIZE);
		}		
		h_dist = d_dist;
		for(int i  = 0; i < BATCH_SIZE; i++)
		{
			cpu_min_value[i + b * BATCH_SIZE] = h_dist[i*K2+cpu_min_index[i + b * BATCH_SIZE] - 1];
			h_dist[i*K2+cpu_min_index[i + b * BATCH_SIZE] -1] = 100000.f;
		}
		d_dist = h_dist;	    
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			cublasIsamin(handle, K2, ((float*)thrust::raw_pointer_cast(d_dist.data())) + K2 * i, 1, 
						((int*)thrust::raw_pointer_cast(cpu_min_index2.data())) + i + b * BATCH_SIZE);
		}
		for(int i  = 0; i < BATCH_SIZE; i++)
		{
			cpu_min_value2[i + b * BATCH_SIZE] = h_dist[i*K2+cpu_min_index2[i + b * BATCH_SIZE] - 1];
		}
    }

#ifdef DEBUG
	for (int i = 0; i < K1; i++)
	{
		printf("%d ", cpu_min_index[i]);
	}
	for (int i = 0; i < K1; i++)
	{
		printf("%d ", cpu_min_index2[i]);
	}
#endif

	int num_match = 0;
	for (int i = 0; i < K1; i++)
	{
		if (cpu_min_value[i] * Thresh < cpu_min_value2[i])
		{
			C[num_match*2] = i + 1; // for MATLAB 1-based index
			C[num_match*2 + 1] = cpu_min_index[i];
			num_match++;
		}
	}

	cublasSafeCall(cublasDestroy(handle));
	return num_match;
}




