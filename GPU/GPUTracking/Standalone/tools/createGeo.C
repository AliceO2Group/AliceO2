#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TSystem.h>
#include "TRDBase/TRDGeometryFlat.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"

using namespace GPUCA_NAMESPACE::gpu;

void createGeo()
{
  o2::base::GeometryManager::loadGeometry();
  o2::trd::TRDGeometry gm;
  gm.createPadPlaneArray();
  gm.createClusterMatrixArray();
  o2::trd::TRDGeometryFlat gf(gm);
  gSystem->Load("libO2GPUTracking.so");
  GPUReconstruction* rec = GPUReconstruction::CreateInstance(GPUReconstruction::DeviceType::CPU);
  GPUChainTracking* chain = rec->AddChain<GPUChainTracking>();
  chain->SetTRDGeometry(&gf);
  rec->DumpSettings();
}

#endif
