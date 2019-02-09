#ifndef ALIGPUMEMORYRESOURCE_H
#define ALIGPUMEMORYRESOURCE_H

#include "AliTPCCommonDef.h"
#include "AliGPUProcessor.h"

class AliGPUMemoryResource
{
	friend class AliGPUReconstruction;
	friend class AliGPUReconstructionCPU;
	
public:
	enum MemoryType {MEMORY_HOST = 1, MEMORY_GPU = 2, MEMORY_INPUT = 7, MEMORY_OUTPUT = 11, MEMORY_INOUT = 15, MEMORY_SCRATCH = 16, MEMORY_SCRATCH_HOST = 17, MEMORY_PERMANENT = 32, MEMORY_CUSTOM = 64, MEMORY_CUSTOM_TRANSFER = 128};
	enum AllocationType {ALLOCATION_AUTO = 0, ALLOCATION_INDIVIDUAL = 1, ALLOCATION_GLOBAL = 2};
	
#ifndef GPUCA_GPUCODE
	AliGPUMemoryResource(AliGPUProcessor* proc, void* (AliGPUProcessor::*setPtr)(void*), MemoryType type, const char* name = "") :
		mProcessor(proc), mPtr(nullptr), mPtrDevice(nullptr), mSetPointers(setPtr), mType(type), mSize(0), mName(name)
	{}
	AliGPUMemoryResource(const AliGPUMemoryResource&) CON_DEFAULT;
#endif
	
#ifndef __OPENCL__
	void* SetPointers(void* ptr) {return (mProcessor->*mSetPointers)(ptr);}
	void* SetDevicePointers(void* ptr) {return (mProcessor->mDeviceProcessor->*mSetPointers)(ptr);}
	void* Ptr() {return mPtr;}
	void* PtrDevice() {return mPtrDevice;}
	size_t Size() const {return mSize;}
	const char* Name() const {return mName;}
#endif

private:
	AliGPUProcessor* mProcessor;
	void* mPtr;
	void* mPtrDevice;
	void* (AliGPUProcessor::* mSetPointers)(void*);
	MemoryType mType;
	size_t mSize;
	const char* mName;
};

#endif
