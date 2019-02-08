#include <TSystem.h>
#include "TRDBase/TRDGeometryFlat.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "AliCAGPUO2Interface.h"
#include "AliGPUReconstruction.h"

void createGeo()
{
  o2::Base::GeometryManager::loadGeometry();
  o2::trd::TRDGeometry gm;
  gm.createPadPlaneArray();
  gm.createClusterMatrixArray();
  o2::trd::TRDGeometryFlat gf(gm);
  gSystem->Load("libO2CAGPUTracking.so");
  AliGPUReconstruction* rec = AliGPUReconstruction::CreateInstance(AliGPUReconstruction::DeviceType::CPU);
  rec->SetTRDGeometry(gf);
  rec->DumpSettings();
}
