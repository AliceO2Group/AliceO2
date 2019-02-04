#ifndef ALIGPUCADATATYPES_H
#define ALIGPUCADATATYPES_H

#include "AliGPUTPCTracker.h"
#include "AliGPUCAParam.h"

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
struct AliGPUCAWorkers
{
	MEM_LG(AliGPUTPCTracker) tpcTrackers[GPUCA_NSLICES];
	AliGPUTPCGMMerger tpcMerger;
	AliGPUTRDTracker trdTracker;
};

MEM_CLASS_PRE()
struct AliGPUCAConstants
{
	MEM_LG(AliGPUCAParam) param;
};

MEM_CLASS_PRE()
struct AliGPUCAConstantMem : public MEM_LG(AliGPUCAConstants), public MEM_LG(AliGPUCAWorkers)
{};

#endif
