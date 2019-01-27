#ifndef ALIGPUCADATATYPES_H
#define ALIGPUCADATATYPES_H

#include "AliGPUTPCTracker.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCGMMerger.h"

#if !defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB)
#include "AliGPUTRDTracker.h"
#else
class AliGPUTRDTracker {void SetMaxData(){}};
#endif

struct AliGPUCAConstantMem
{
	AliGPUCAParam param;
	AliGPUTPCTracker tpcTrackers[36];
	AliGPUTPCGMMerger tpcMerger;
	AliGPUTRDTracker trdTracker;
};

#endif
