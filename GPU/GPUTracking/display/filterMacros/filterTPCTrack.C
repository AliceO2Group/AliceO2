#include "GPUO2Interface.h"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsTPC/TrackTPC.h"
#endif

namespace o2::gpu
{
struct GPUConstantMem;
};

using namespace o2::gpu;

void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
{
  for (uint32_t i = 0; i < filter->size(); i++) {
    auto& trk = ioPtrs->outputTracksTPCO2[i];
    (*filter)[i] = fabsf(trk.getQ2Pt()) < 1.0f;
  }
}
