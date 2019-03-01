#ifndef GPUITSTRACK_H
#define GPUITSTRACK_H

#include "GPUTPCGMTrackParam.h"

namespace o2 { namespace ITS {

class GPUITSTrack : public GPUTPCGMTrackParam
{
public:
	GPUTPCGMTrackParam::GPUTPCOuterParam mOuterParam;
	float mAlpha;
	int mClusters[7];
};

}}

#endif
