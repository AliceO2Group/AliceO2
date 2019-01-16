#ifndef ALIGPUMEMORYRESOURCE_H
#define ALIGPUMEMORYRESOURCE_H

#include "AliTPCCommonDef.h"
#include "AliGPUProcessor.h"

class AliGPUMemoryResource
{
	friend class AliGPUReconstruction;
	friend class AliGPUReconstructionDeviceBase;
	
public:
	enum MemoryType {MEMORY_INPUT = 0, MEMORY_SCRATCH = 1, MEMORY_OUTPUT = 2, MEMORY_SCRATCH_HOST = 3, MEMORY_PERMANENT = 4, MEMORY_CUSTOM = 5};
	enum AllocationType {ALLOCATION_AUTO = 0, ALLOCATION_INDIVIDUAL = 1, ALLOCATION_GLOBAL = 2};
	
#ifndef GPUCA_GPUCODE
	AliGPUMemoryResource(AliGPUProcessor* proc, void* (AliGPUProcessor::*setPtr)(void*), MemoryType type) :
		mProcessor(proc), mPtr(nullptr), mPtrDevice(nullptr), mSetPointers(setPtr), mType(type), mSize(0)
	{
	}
	AliGPUMemoryResource(const AliGPUMemoryResource&) CON_DEFAULT;
#endif
	
#ifndef __OPENCL__
	void* SetPointers(void* ptr) {return (mProcessor->*mSetPointers)(ptr);}
	void* SetDevicePointers(void* ptr) {return (mProcessor->mDeviceProcessor->*mSetPointers)(ptr);}
	void* Ptr() {return mPtr;}
	void* PtrDevice() {return mPtrDevice;}
	size_t Size() const {return mSize;}
#endif

private:
	AliGPUProcessor* mProcessor;
	void* mPtr;
	void* mPtrDevice;
	void* (AliGPUProcessor::* mSetPointers)(void*);
	MemoryType mType;
	size_t mSize;
};

#endif
