#ifndef ALIGPURECONSTRUCTIONCUDAINTERNALS_H
#define ALIGPURECONSTRUCTIONCUDAINTERNALS_H

#include <cuda.h>

struct AliGPUReconstructionCUDAInternals
{
	CUcontext CudaContext; //Pointer to CUDA context
	cudaStream_t* CudaStreams; //Pointer to array of CUDA Streams
};

#endif
