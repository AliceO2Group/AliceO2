#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <FairLogger.h>
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include "DataFormatsParameters/GRPObject.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#endif

int initFieldFromGRP(const o2::parameters::GRPObject* grp);
int initFieldFromGRP(const std::string grpFileName = "o2sim_grp.root", std::string grpName = "GRP");
int initSimGeom(std::string geom = "");

int initSimGeomAndField(std::string geom = "", std::string grpFileName = "o2sim_grp.root", std::string grpName = "GRP")
{
  int res = 0;
  res = initSimGeom(geom);
  if (res) {
    return res;
  }
  res = initFieldFromGRP(grpFileName, grpName);
  return res;
}

int initSimGeom(std::string geom)
{
  /// load geometry from the file
  o2::base::GeometryManager::loadGeometry(geom);
  return 0;
}

//____________________________________________________________
int initFieldFromGRP(const std::string grpFileName, std::string grpName)
{
  /// load grp and init magnetic field
  std::cout << "Loading field from GRP of " << grpFileName << std::endl;
  TFile flGRP(grpFileName.data());
  if (flGRP.IsZombie()) {
    std::cout << "Failed to open " << grpFileName << std::endl;
    return -10;
  }
  auto grp =
    static_cast<o2::parameters::GRPObject*>(flGRP.GetObjectChecked(grpName.data(), o2::parameters::GRPObject::Class()));
  if (!grp) {
    std::cout << "Did not find GRP object named " << grpName << std::endl;
    return -12;
  }
  grp->print();

  return initFieldFromGRP(grp);
}

//____________________________________________________________
int initFieldFromGRP(const o2::parameters::GRPObject* grp)
{
  /// init mag field from GRP data and attach it to TGeoGlobalMagField

  if (TGeoGlobalMagField::Instance()->IsLocked()) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2::field::MagneticField::kOverrideGRP)) {
      std::cout << "ExpertMode!!! GRP information will be ignored" << std::endl;
      std::cout << "ExpertMode!!! Running with the externally locked B field" << std::endl;
      return 0;
    } else {
      std::cout << "Destroying existing B field instance" << std::endl;
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2::field::MagneticField::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent(), o2::field::MagneticField::kConvLHC, grp->getFieldUniformity());
  TGeoGlobalMagField::Instance()->SetField(fld);
  TGeoGlobalMagField::Instance()->Lock();
  std::cout << "Running with the B field constructed out of GRP" << std::endl;
  std::cout << "Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via" << std::endl;
  std::cout << "auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )"
            << std::endl;

  return 0;
}
