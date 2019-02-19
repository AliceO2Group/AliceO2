#ifndef ALIGPUCONSTANTMEM_H
#define ALIGPUCONSTANTMEM_H

#include "AliGPUTPCTracker.h"
#include "AliGPUParam.h"

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "AliGPUTPCGMMerger.h"
#else
class AliGPUTPCGMMerger {};
#endif

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB))
#include "AliGPUTRDTracker.h"
#else
class AliGPUTRDTracker {void SetMaxData(){}};
#endif

MEM_CLASS_PRE()
struct AliGPUConstantMem
{
	MEM_LG(AliGPUParam) param;
	MEM_LG(AliGPUTPCTracker) tpcTrackers[GPUCA_NSLICES];
	AliGPUTPCGMMerger tpcMerger;
	AliGPUTRDTracker trdTracker;
};

#endif
