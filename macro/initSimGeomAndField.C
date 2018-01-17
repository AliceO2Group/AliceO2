#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include <string>
#include <FairLogger.h>
#endif

int initFieldFromGRP(const o2::parameters::GRPObject* grp);
int initFieldFromGRP(const std::string grpFileName="o2sim_grp.root",std::string grpName="GRP");
int initSimGeom(std::string geomFileName="O2geometry.root", std::string geomName="FAIRGeom");


int initSimGeomAndField(std::string geomFileName="O2geometry.root",
			std::string grpFileName="o2sim_grp.root",
			std::string geomName="FAIRGeom",
			std::string grpName="GRP")
{
  
  int res = 0;
  res = initSimGeom(geomFileName,geomName);
  if (!res) {
    return res;
  }
  res = initFieldFromGRP(grpFileName,grpName);
  return res;
}

int initSimGeom(std::string geomFileName, std::string geomName)
{
  /// load geometry from the file
  LOG(INFO)<<"Loading geometry from "<<geomFileName<<FairLogger::endl;
  TFile flGeom(geomFileName.data());
  if ( flGeom.IsZombie() ) {
    LOG(ERROR)<<"Failed to open "<<geomFileName<<FairLogger::endl;    
    return -1;
  }
  if ( !flGeom.Get(geomName.data()) ) {
    LOG(ERROR)<<"Did not find geometry named "<<geomName<<FairLogger::endl;        
    return -2;
  }
  return 0;
}


//____________________________________________________________
int initFieldFromGRP(const std::string grpFileName,std::string grpName)
{
  /// load grp and init magnetic field
  LOG(INFO)<<"Loading field from GRP of "<<grpFileName<<FairLogger::endl;
  TFile flGRP(grpFileName.data());
  if ( flGRP.IsZombie() ) {
    LOG(ERROR)<<"Failed to open "<<grpFileName<<FairLogger::endl;    
    return -10;
  }
  auto grp = static_cast<o2::parameters::GRPObject*>
    (flGRP.GetObjectChecked(grpName.data(),o2::parameters::GRPObject::Class()));
  if (!grp) {
    LOG(ERROR)<<"Did not find GRP object named "<<grpName<<FairLogger::endl;        
    return -12;
  }
  grp->print();

  return initFieldFromGRP(grp);

}

//____________________________________________________________
int initFieldFromGRP(const o2::parameters::GRPObject* grp)
{
  /// init mag field from GRP data and attach it to TGeoGlobalMagField
  
  if ( TGeoGlobalMagField::Instance()->IsLocked() ) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2::field::MagneticField::kOverrideGRP)) {
      LOG(WARNING)<<"ExpertMode!!! GRP information will be ignored"<<FairLogger::endl;
      LOG(WARNING)<<"ExpertMode!!! Running with the externally locked B field"<<FairLogger::endl;
      return 0;
    }
    else {
      LOG(INFO)<<"Destroying existing B field instance"<<FairLogger::endl;
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2::field::MagneticField::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent());
  TGeoGlobalMagField::Instance()->SetField( fld );
  TGeoGlobalMagField::Instance()->Lock();
  LOG(INFO)<<"Running with the B field constructed out of GRP"<<FairLogger::endl;
  LOG(INFO)<<"Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via"<<FairLogger::endl;
  LOG(INFO)<<"auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )"
	   <<FairLogger::endl;
  
  return 0;
}
