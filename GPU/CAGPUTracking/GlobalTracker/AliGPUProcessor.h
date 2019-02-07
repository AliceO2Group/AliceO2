#ifndef ALIGPUPROCESSOR_H
#define ALIGPUPROCESSOR_H

#include "AliTPCCommonDef.h"
#include "AliGPUTPCDef.h"

#ifndef GPUCA_GPUCODE
#include <cstddef>
#include <algorithm>
#endif

class AliGPUReconstruction;
MEM_CLASS_PRE() class AliGPUCAParam;

class AliGPUProcessor
{
	friend class AliGPUReconstruction;
	friend class AliGPUReconstructionDeviceBase;
	friend class AliGPUMemoryResource;
	
public:
	enum ProcessorType {PROCESSOR_TYPE_CPU = 0, PROCESSOR_TYPE_DEVICE = 1, PROCESSOR_TYPE_SLAVE = 2};

#ifndef GPUCA_GPUCODE
	AliGPUProcessor();
	~AliGPUProcessor();
	AliGPUProcessor(const AliGPUProcessor&) CON_DELETE;
	AliGPUProcessor& operator= (const AliGPUProcessor&) CON_DELETE;
#endif

	GPUconstantref() const MEM_CONSTANT(AliGPUCAParam) &GetParam() const {return *mCAParam;}
	const AliGPUReconstruction& GetRec() const {return *mRec;}

#ifndef __OPENCL__
	void InitGPUProcessor(AliGPUReconstruction* rec, ProcessorType type = PROCESSOR_TYPE_CPU, AliGPUProcessor* slaveProcessor = NULL);
	void Clear();

	//Helpers for memory allocation
	constexpr static size_t MIN_ALIGNMENT = 64;

	template <size_t alignment = MIN_ALIGNMENT> static inline size_t getAlignment(size_t addr)
	{
		static_assert((alignment & (alignment - 1)) == 0, "Invalid alignment, not power of 2");
		if (alignment <= 1) return 0;
		size_t mod = addr & (alignment - 1);
		if (mod == 0) return 0;
		return (alignment - mod);
	}
	template <size_t alignment = MIN_ALIGNMENT> static inline size_t nextMultipleOf(size_t size)
	{
		return size + getAlignment<alignment>(size);
	}
	template <size_t alignment = MIN_ALIGNMENT> static inline void* alignPointer(void* ptr)
	{
		return(reinterpret_cast<void*>(nextMultipleOf<alignment>(reinterpret_cast<size_t>(ptr))));
	}
	template <size_t alignment = MIN_ALIGNMENT> static inline size_t getAlignment(void* addr)
	{
		return(getAlignment<alignment>(reinterpret_cast<size_t>(addr)));
	}
	template <size_t alignment = MIN_ALIGNMENT, class S> static inline S* getPointerWithAlignment(size_t& basePtr, size_t nEntries = 1)
	{
		if (basePtr == 0) basePtr = 1;
		constexpr size_t maxAlign = (alignof(S) > alignment) ? alignof(S) : alignment;
		basePtr += getAlignment<maxAlign>(basePtr);
		S* retVal = (S*) (basePtr);
		basePtr += nEntries * sizeof(S);
		return retVal;
	}
	template <size_t alignment = MIN_ALIGNMENT, class S> static inline S* getPointerWithAlignment(void*& basePtr, size_t nEntries = 1)
	{
		return getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
	}

	template <size_t alignment = MIN_ALIGNMENT, class T, class S> static inline void computePointerWithAlignment(T*& basePtr, S*& objPtr, size_t nEntries = 1, bool runConstructor = false)
	{
		objPtr = getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
		if (runConstructor)
		{
			for (size_t i = 0;i < nEntries;i++)
			{
				new (objPtr + i) S;
			}
		}
	}
#endif

protected:
	void AllocateAndInitializeLate() {mAllocateAndInitializeLate = true;}
	
	AliGPUReconstruction* mRec;
	ProcessorType mGPUProcessorType;
	AliGPUProcessor* mDeviceProcessor;
	GPUconstantref() const MEM_CONSTANT(AliGPUCAParam) *mCAParam;

private:
	bool mAllocateAndInitializeLate;
};

#endif
