#include "GPUO2Interface.h"
#include "GPUTPCGMMergedTrack.h"

namespace o2::gpu
{
struct GPUConstantMem;
};

using namespace o2::gpu;

void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
{
  for (unsigned int i = 0; i < filter->size(); i++) {
    auto& trk = ioPtrs->mergedTracks[i];
    (*filter)[i] = fabsf(trk.GetParam().GetQPt()) < 1.0f;
  }
}
