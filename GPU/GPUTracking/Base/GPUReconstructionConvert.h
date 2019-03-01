#ifndef GPURECONSTRUCTIONCONVERT_H
#define GPURECONSTRUCTIONCONVERT_H

#include <memory>
#include "GPUTPCSettings.h"
class GPUTPCClusterData;

struct ClusterNativeAccessExt;
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class GPUReconstructionConvert
{
public:
	constexpr static unsigned int NSLICES = GPUCA_NSLICES;
	static void ConvertNativeToClusterData(ClusterNativeAccessExt* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin = 0);
};

#endif
