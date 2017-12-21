#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include <string>
#endif


using o2field = o2::field::MagneticField;
using GRP = o2::parameters::GRPObject;
void initFieldFromGRP(const o2::parameters::GRPObject* grp);

int initSimGeomAndField(std::string geomFileName="O2geometry.root",
			std::string grpFileName="o2sim_grp.root",
			std::string geomName="FAIRGeom",
			std::string grpName="GRP")
{
  printf("Loading geometry from %s\n",geomFileName.data());
  TFile flGeom(geomFileName.data());
  if ( flGeom.IsZombie() ) {
    return -1;
  }
  if ( !flGeom.Get(geomName.data()) ) {
    return -2;
  }

  //
  printf("Loading field from GRP of %s\n",grpFileName.data());
  TFile flGRP(grpFileName.data());
  if ( flGRP.IsZombie() ) {
    return -10;
  }
  auto grp = static_cast<GRP*>(flGRP.GetObjectChecked(grpName.data(),
						      GRP::Class()));
  if (!grp) {
    return -12;
  }
  grp->print();
  initFieldFromGRP(grp);
  return 0;
}

void initFieldFromGRP(const GRP* grp)
{
  if ( TGeoGlobalMagField::Instance()->IsLocked() ) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2field::kOverrideGRP)) {
      printf("ExpertMode!!! GRP information will be ignored\n");
      printf("ExpertMode!!! Running with the externally locked B field\n");
      return;
    }
    else {
      printf("Destroying existing B field instance\n");
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2field::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent());
  TGeoGlobalMagField::Instance()->SetField( fld );
  TGeoGlobalMagField::Instance()->Lock();
  printf("Running with the B field constructed out of GRP\n");
  printf("Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via\n");
  printf("auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )\n");
  
}
