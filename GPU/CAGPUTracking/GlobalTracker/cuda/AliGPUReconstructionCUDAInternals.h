#ifndef ALIGPURECONSTRUCTIONCUDAINTERNALS_H
#define ALIGPURECONSTRUCTIONCUDAINTERNALS_H

#include <cuda.h>

struct AliGPUReconstructionCUDAInternals
{
	CUcontext CudaContext; //Pointer to CUDA context
	cudaStream_t* CudaStreams; //Pointer to array of CUDA Streams
};

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)

static int GPUFailedMsgA(const long long int error, const char* file, int line)
{
	//Check for CUDA Error and in the case of an error display the corresponding error string
	if (error == cudaSuccess) return(0);
	printf("CUDA Error: %lld / %s (%s:%d)\n", error, cudaGetErrorString((cudaError_t) error), file, line);
	return(1);
}

static_assert(std::is_convertible<cudaEvent_t, void*>::value, "CUDA event type incompatible to deviceEvent");

#endif
