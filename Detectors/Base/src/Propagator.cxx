// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsBase/Propagator.h"
#include <FairLogger.h>
#include <FairRunAna.h> // eventually will get rid of it
#include <TGeoGlobalMagField.h>
#include "DetectorsBase/GeometryManager.h"
#include "Field/MagFieldFast.h"
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"

using namespace o2::Base;

Propagator::Propagator()
{
  ///< construct checking if needed components were initialized

  // we need the geoemtry loaded
  if (!gGeoManager) {
    LOG(FATAL) << "No active geometry!" << FairLogger::endl;
  }

  o2::field::MagneticField* slowField = nullptr;
  slowField = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!slowField) {
    LOG(WARNING) << "No Magnetic Field in TGeoGlobalMagField, checking legacy FairRunAna" << FairLogger::endl;
    slowField = dynamic_cast<o2::field::MagneticField*>(FairRunAna::Instance()->GetField());
  }
  if (!slowField) {
    LOG(FATAL) << "Magnetic field is not initialized!" << FairLogger::endl;
  }
  if (!slowField->getFastField()) {
    slowField->AllowFastField(true);
  }
  mField = slowField->getFastField();
}

//_______________________________________________________________________
bool Propagator::PropagateToXBxByBz(o2::track::TrackParCov& track, float xToGo, float mass,
				    float maxSnp, float maxStep, int matCorr, int signCorr)
{
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction (if <0 then q=2)
  // maxStep  - maximal step for propagation
  //
  //----------------------------------------------------------------
  const float Epsilon = 0.00001;
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  std::array<float, 3> b;
  while (std::abs(dx) > Epsilon) {
    auto step = std::min(std::abs(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();
    mField->Field(xyz0, b.data());

    if (!track.propagateTo(x, b))
      return false;
    if (maxSnp > 0 && std::abs(track.getSnp()) >= maxSnp)
      return false;

    if (matCorr) {
      auto xyz1 = track.getXYZGlo();
      auto mb = GeometryManager::MeanMaterialBudget(xyz0, xyz1);
      if (signCorr < 0) {
        mb.length = -mb.length;
      }
      //
      if (!track.correctForMaterial(mb.meanX2X0, mb.meanRho * mb.length, mass))
        return false;
    }
    dx = xToGo - track.getX();
  }
  return true;
}

//____________________________________________________________
int Propagator::initFieldFromGRP(const std::string grpFileName,std::string grpName)
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
int Propagator::initFieldFromGRP(const o2::parameters::GRPObject* grp)
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
