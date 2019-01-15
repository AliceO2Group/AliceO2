#ifndef ALIGPUPROCESSOR_H
#define ALIGPUPROCESSOR_H

#include "AliTPCCommonDef.h"

#ifndef GPUCA_GPUCODE
#include <cstddef>
#endif

class AliGPUReconstruction;

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
	void InitGPUProcessor(AliGPUReconstruction* rec, ProcessorType type = PROCESSOR_TYPE_CPU, AliGPUProcessor* slaveProcessor = NULL);
#endif

	void* InputMemory() const;
	void* ScratchMemory() const;
	size_t InputMemorySize() const;
	size_t ScratchMemorySize() const;

protected:
	AliGPUReconstruction* mRec;
	ProcessorType mGPUProcessorType;
	AliGPUProcessor* mDeviceProcessor;
	
	int mMemoryResInput;
	int mMemoryResOutput;
    int mMemoryResScratch;
    int mMemoryResScratchHost;
};

#endif
