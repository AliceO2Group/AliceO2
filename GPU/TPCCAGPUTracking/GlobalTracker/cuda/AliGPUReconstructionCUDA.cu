#include "AliGPUReconstructionCUDA.h"
#include "AliHLTTPCCAGPUTrackerNVCC.h"

#ifdef HAVE_O2HEADERS
#include "ITStrackingCUDA/TrackerTraitsNV.h"

#ifndef HLTCA_BUILD_O2_LIB
#include "TrackerTraitsNV.cu"
#include "Context.cu"
#include "Stream.cu"
#include "DeviceStoreNV.cu"
#include "Utils.cu"
#endif

#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsNV : public TrackerTraits {}; }}
#endif

AliGPUReconstructionCUDA::AliGPUReconstructionCUDA() : AliGPUReconstructionDeviceBase(CUDA)
{
    mTPCTracker.reset(new AliHLTTPCCAGPUTrackerNVCC);
    mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsNV);
}

AliGPUReconstructionCUDA::~AliGPUReconstructionCUDA()
{
#if defined(HLTCA_STANDALONE) && !defined(HLTCA_BUILD_O2_LIB)
    mITSTrackerTraits.release();
#endif
}

AliGPUReconstruction* AliGPUReconstruction_Create_CUDA()
{
	return new AliGPUReconstructionCUDA;
}
