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

struct AliGPUCAWorkers
{
	AliGPUTPCTracker tpcTrackers[36];
	AliGPUTPCGMMerger tpcMerger;
	AliGPUTRDTracker trdTracker;
};

struct AliGPUCAConstants
{
	AliGPUCAParam param;
};

struct AliGPUCAConstantMem : public AliGPUCAConstants, public AliGPUCAWorkers
{};

#endif
