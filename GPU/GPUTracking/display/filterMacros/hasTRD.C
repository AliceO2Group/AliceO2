#include "GPUO2Interface.h"
using namespace o2::gpu;

void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
{
  if (!ioPtrs->tpcLinkTRD) {
    return;
  }
  for (unsigned int i = 0; i < filter->size(); i++) {
    (*filter)[i] = ioPtrs->tpcLinkTRD[i] != -1;
  }
}
