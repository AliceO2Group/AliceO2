#ifndef GPUCONSTANTMEM_H
#define GPUCONSTANTMEM_H

#include "GPUTPCTracker.h"
#include "GPUParam.h"

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUTPCGMMerger.h"
#include "GPUITSFitter.h"
#else
class GPUTPCGMMerger {};
class GPUITSFitter {};
#endif

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB))
#include "GPUTRDTracker.h"
#else
class GPUTRDTracker {void SetMaxData(){}};
#endif

MEM_CLASS_PRE()
struct GPUConstantMem
{
	MEM_LG(GPUParam) param;
	MEM_LG(GPUTPCTracker) tpcTrackers[GPUCA_NSLICES];
	GPUTPCGMMerger tpcMerger;
	GPUTRDTracker trdTracker;
	GPUITSFitter itsFitter;
};

#endif
