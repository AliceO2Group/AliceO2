#ifndef ALIGPUCADATATYPES_H
#define ALIGPUCADATATYPES_H

#include "AliGPUTPCTracker.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCGMMerger.h"

struct AliGPUCAConstantMem
{
	AliGPUCAParam param;
	AliGPUTPCTracker tpcTrackers[36];
	AliGPUTPCGMMerger tpcMerger;
};

#endif
