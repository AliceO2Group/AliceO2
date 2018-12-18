#ifndef ALIGPUCADATATYPES_H
#define ALIGPUCADATATYPES_H

#include "AliHLTTPCCATracker.h"
#include "AliGPUCAParam.h"
#include "AliHLTTPCGMMerger.h"

struct AliGPUCAConstantMem
{
	AliGPUCAParam param;
	AliHLTTPCCATracker tpcTrackers[36];
	AliHLTTPCGMMerger tpcMerger;
};

#endif
