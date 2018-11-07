#include "AliGPUReconstructionOCL.h"
#include "AliHLTTPCCAGPUTrackerOpenCL.h"

#ifdef HAVE_O2HEADERS
#include "ITStracking/TrackerTraitsCPU.h"
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsCPU : public TrackerTraits {}; }}
#endif

AliGPUReconstructionOCL::AliGPUReconstructionOCL() : AliGPUReconstruction(OCL)
{
    mTPCTracker.reset(new AliHLTTPCCAGPUTrackerOpenCL);
    mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsCPU);
}

AliGPUReconstructionOCL::~AliGPUReconstructionOCL()
{
    
}

AliGPUReconstruction* AliGPUReconstruction_Create_OCL()
{
	return new AliGPUReconstructionOCL;
}
