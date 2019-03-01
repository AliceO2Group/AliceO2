#include "GPUProcessor.h"
#include "GPUReconstruction.h"
#include "GPUReconstructionDeviceBase.h"

GPUProcessor::GPUProcessor() : mRec(NULL), mGPUProcessorType(PROCESSOR_TYPE_CPU), mDeviceProcessor(NULL), mCAParam(NULL), mAllocateAndInitializeLate(false)
{
}

GPUProcessor::~GPUProcessor()
{
	if (mRec && mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		Clear();
	}
}

void GPUProcessor::InitGPUProcessor(GPUReconstruction* rec, GPUProcessor::ProcessorType type, GPUProcessor* slaveProcessor)
{
	mRec = rec;
	mGPUProcessorType = type;
	if (slaveProcessor) slaveProcessor->mDeviceProcessor = this;
	mCAParam = type == PROCESSOR_TYPE_DEVICE ? ((GPUReconstructionDeviceBase*) rec)->DeviceParam() : &rec->GetParam();
}

void GPUProcessor::Clear()
{
	mRec->FreeRegisteredMemory(this, true);
}
