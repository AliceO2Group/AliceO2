#include "AliGPUReconstructionOCL.h"
#include "AliHLTTPCCAGPUTrackerOpenCL.h"

AliGPUReconstructionOCL::AliGPUReconstructionOCL() : AliGPUReconstruction(OCL)
{
    mTPCTracker.reset(new AliHLTTPCCAGPUTrackerOpenCL);
}

AliGPUReconstructionOCL::~AliGPUReconstructionOCL()
{
    
}

AliGPUReconstruction* AliGPUReconstruction_Create_OCL()
{
	return new AliGPUReconstructionOCL;
}
