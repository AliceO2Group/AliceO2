#include "AliGPUProcessor.h"
#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionDeviceBase.h"

AliGPUProcessor::AliGPUProcessor() :
	mRec(NULL), mGPUProcessorType(PROCESSOR_TYPE_CPU), mDeviceProcessor(NULL), mCAParam(NULL)
{
}

AliGPUProcessor::~AliGPUProcessor()
{
	if (mRec && mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		mRec->FreeRegisteredMemory(this);
	}
}

void AliGPUProcessor::InitGPUProcessor(AliGPUReconstruction* rec, AliGPUProcessor::ProcessorType type, AliGPUProcessor* slaveProcessor)
{
	mRec = rec;
	mGPUProcessorType = type;
	if (slaveProcessor) slaveProcessor->mDeviceProcessor = this;
	mCAParam = type == PROCESSOR_TYPE_DEVICE ? ((AliGPUReconstructionDeviceBase*) rec)->DeviceParam() : &rec->GetParam();
}
