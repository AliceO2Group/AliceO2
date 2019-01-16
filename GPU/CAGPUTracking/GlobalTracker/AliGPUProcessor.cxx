#include "AliGPUProcessor.h"
#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionDeviceBase.h"

AliGPUProcessor::AliGPUProcessor() :
	mRec(NULL), mGPUProcessorType(PROCESSOR_TYPE_CPU), mDeviceProcessor(NULL), mParam(NULL),
	mMemoryResInput(-1), mMemoryResOutput(-1), mMemoryResScratch(-1), mMemoryResScratchHost(-1), mMemoryResPermanent(-1)
{
}

AliGPUProcessor::~AliGPUProcessor()
{
	if (mMemoryResInput != -1) mRec->FreeRegisteredMemory(mMemoryResInput);
	if (mMemoryResOutput != -1) mRec->FreeRegisteredMemory(mMemoryResOutput);
	if (mMemoryResScratch != -1) mRec->FreeRegisteredMemory(mMemoryResScratch);
	if (mMemoryResScratchHost != -1) mRec->FreeRegisteredMemory(mMemoryResScratchHost);
	if (mMemoryResPermanent != -1) mRec->FreeRegisteredMemory(mMemoryResPermanent);
}

void AliGPUProcessor::InitGPUProcessor(AliGPUReconstruction* rec, AliGPUProcessor::ProcessorType type, AliGPUProcessor* slaveProcessor)
{
	mRec = rec;
	mGPUProcessorType = type;
	if (slaveProcessor) slaveProcessor->mDeviceProcessor = this;
	mParam = type == PROCESSOR_TYPE_DEVICE ? ((AliGPUReconstructionDeviceBase*) rec)->DeviceParam() : &rec->GetParam();
}

void* AliGPUProcessor::InputMemory() const {return(mRec->Res(mMemoryResInput).Ptr()); }
void* AliGPUProcessor::ScratchMemory() const {return(mRec->Res(mMemoryResScratch).Ptr()); }
size_t AliGPUProcessor::InputMemorySize() const {return(mRec->Res(mMemoryResInput).Size()); }
size_t AliGPUProcessor::ScratchMemorySize() const {return(mRec->Res(mMemoryResScratch).Size()); }
