#include <TSystem.h>
#include "TRDBase/TRDGeometryFlat.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"

void createGeo()
{
  o2::Base::GeometryManager::loadGeometry();
  o2::trd::TRDGeometry gm;
  gm.createPadPlaneArray();
  gm.createClusterMatrixArray();
  o2::trd::TRDGeometryFlat gf(gm);
  gSystem->Load("libO2GPUTracking.so");
  GPUReconstruction* rec = GPUReconstruction::CreateInstance(GPUReconstruction::DeviceType::CPU);
  rec->SetTRDGeometry(gf);
  rec->DumpSettings();
}
