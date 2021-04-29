#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <FairLogger.h>
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include "DataFormatsParameters/GRPObject.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#endif

int initSimGeom(std::string geom = "");

int initSimGeomAndField(std::string geom = "", std::string grpFileName = "o2sim_grp.root", std::string grpName = "GRP")
{
  int res = 0;
  res = initSimGeom(geom);
  if (res) {
    return res;
  }
  res = o2::base::Propagator::initFieldFromGRP(grpFileName, grpName);
  return res;
}

int initSimGeom(std::string geom)
{
  /// load geometry from the file
  o2::base::GeometryManager::loadGeometry(geom);
  return 0;
}
