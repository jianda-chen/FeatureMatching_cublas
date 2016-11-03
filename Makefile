CUDA_ARCH :=    -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50

all:
	nvcc -m64 -dc $(CUDA_ARCH) -ccbin="g++" -Xcompiler -fPIC  -L"/usr/local/cuda/lib64" -lcublas_device -lcudadevrt -o cuda_match.o -c cuda_match_batch.cu 
	nvcc $(CUDA_ARCH) -ccbin="g++" -Xcompiler -fPIC -L"/usr/local/cuda/lib64" -lcublas_device -lcudadevrt -lcublas cuda_match.o -shared -o libcuda_match.so
	mex -L"/usr/local/cuda/lib64" -I"/usr/local/cuda/include" mex_cuda_match.cpp libcuda_match.so
test:
	nvcc -g -G -DDEBUG -m64 -dc $(CUDA_ARCH) -ccbin="g++" -Xcompiler -fPIC  -L"/usr/local/cuda/lib64" -lcublas_device -lcudadevrt -o cuda_match.o -c cuda_match_batch.cu 
	nvcc -g -G -DDEBUG $(CUDA_ARCH) -ccbin="g++" -Xcompiler -fPIC -L"/usr/local/cuda/lib64" -lcublas_device -lcudadevrt -lcublas cuda_match.o -shared -o libcuda_match.so
	g++ -g -ggdb -DDEBUG -L"/usr/local/cuda/lib64" -I"/usr/local/cuda/include" test.cpp libcuda_match.so -o test 

clean:
	rm *.a *.o *.mexa64 *.so test



