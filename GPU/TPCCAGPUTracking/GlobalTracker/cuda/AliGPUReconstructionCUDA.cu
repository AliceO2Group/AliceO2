#include "AliGPUReconstructionCUDA.h"
#include "AliHLTTPCCAGPUTrackerNVCC.h"

AliGPUReconstructionCUDA::AliGPUReconstructionCUDA() : AliGPUReconstruction(CUDA)
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
