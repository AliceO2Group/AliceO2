#ifndef ALIGPUCADATATYPES_H
#define ALIGPUCADATATYPES_H

#include "AliGPUTPCTracker.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCGMMerger.h"
#include "AliGPUTRDTracker.h"

struct AliGPUCAConstantMem
{
	AliGPUCAParam param;
	AliGPUTPCTracker tpcTrackers[36];
	AliGPUTPCGMMerger tpcMerger;
	AliGPUTRDTracker trdTracker;
};

#endif
