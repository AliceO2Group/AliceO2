#include "AliGPUReconstructionCUDA.h"
#include "AliHLTTPCCAGPUTrackerNVCC.h"

AliGPUReconstructionCUDA::AliGPUReconstructionCUDA() : AliGPUReconstruction()
{
    mTPCTracker.reset(new AliHLTTPCCAGPUTrackerNVCC);
}

AliGPUReconstructionCUDA::~AliGPUReconstructionCUDA()
{
    
}

AliGPUReconstruction* AliGPUReconstruction_Create_CUDA()
{
	return new AliGPUReconstructionCUDA;
}
