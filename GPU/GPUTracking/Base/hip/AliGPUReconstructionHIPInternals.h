#ifndef ALIGPURECONSTRUCTIONHIPINTERNALS_H
#define ALIGPURECONSTRUCTIONHIPINTERNALS_H

struct AliGPUReconstructionHIPInternals
{
	hipStream_t HIPStreams[GPUCA_MAX_STREAMS]; //Pointer to array of HIP Streams
};

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

static int GPUFailedMsgAI(const long long int error, const char* file, int line)
{
	//Check for HIP Error and in the case of an error display the corresponding error string
	if (error == hipSuccess) return(0);
	printf("HIP Error: %lld / %s (%s:%d)\n", error, hipGetErrorString((hipError_t) error), file, line);
	return 1;
}

static void GPUFailedMsgA(const long long int error, const char* file, int line)
{
	if (GPUFailedMsgAI(error, file, line)) throw std::runtime_error("HIP Failure");
}

static_assert(std::is_convertible<hipEvent_t, void*>::value, "HIP event type incompatible to deviceEvent");

#endif
