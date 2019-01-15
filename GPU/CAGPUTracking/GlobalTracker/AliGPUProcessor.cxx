#include "AliGPUProcessor.h"
#include <cstddef>

AliGPUProcessor::AliGPUProcessor() :
	mRec(NULL),
	mGPUProcessorType(PROCESSOR_TYPE_CPU)
{
	
}

void AliGPUProcessor::InitGPUProcessor(AliGPUReconstruction* rec, AliGPUProcessor::ProcessorType type)
{
	mRec = rec;
	mGPUProcessorType = type;
}
