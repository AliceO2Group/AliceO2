#ifndef GPUITSTRACK_H
#define GPUITSTRACK_H

#include "AliGPUTPCGMTrackParam.h"

namespace o2 { namespace ITS {

class GPUITSTrack : public AliGPUTPCGMTrackParam
{
public:
	AliGPUTPCGMTrackParam::AliGPUTPCOuterParam mOuterParam;
	float mAlpha;
	int mClusters[7];
};

}}

#endif
