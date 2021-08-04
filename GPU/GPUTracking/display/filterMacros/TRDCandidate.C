#include "GPUO2Interface.h"
#include "GPUConstantMem.h"
using namespace o2::gpu;

void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
{
  for (unsigned int i = 0; i < filter->size(); i++) {
    (*filter)[i] = processors->trdTrackerGPU.PreCheckTrackTRDCandidate(ioPtrs->mergedTracks[i]) && processors->trdTrackerGPU.CheckTrackTRDCandidate((GPUTRDTrackGPU)ioPtrs->mergedTracks[i]);
  }
}
