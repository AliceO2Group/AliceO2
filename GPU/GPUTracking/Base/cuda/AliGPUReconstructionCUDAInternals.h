#ifndef ALIGPURECONSTRUCTIONCUDAINTERNALS_H
#define ALIGPURECONSTRUCTIONCUDAINTERNALS_H

#include <cuda.h>

struct AliGPUReconstructionCUDAInternals
{
	CUcontext CudaContext; //Pointer to CUDA context
	cudaStream_t CudaStreams[GPUCA_GPUCA_MAX_STREAMS]; //Pointer to array of CUDA Streams
};

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

static int GPUFailedMsgAI(const long long int error, const char* file, int line)
{
	//Check for CUDA Error and in the case of an error display the corresponding error string
	if (error == cudaSuccess) return(0);
	printf("CUDA Error: %lld / %s (%s:%d)\n", error, cudaGetErrorString((cudaError_t) error), file, line);
	return 1;
}

static void GPUFailedMsgA(const long long int error, const char* file, int line)
{
	if (GPUFailedMsgAI(error, file, line)) throw std::runtime_error("CUDA Failure");
}

static_assert(std::is_convertible<cudaEvent_t, void*>::value, "CUDA event type incompatible to deviceEvent");

#endif
