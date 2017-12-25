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
#include <TGeoGlobalMagField.h>
#include "DetectorsBase/GeometryManager.h"
#include "Field/MagFieldFast.h"
#include "Field/MagneticField.h"

using namespace o2::Base::Track;

Propagator::Propagator()
{
  ///< construct checking if needed components were initialized

  // we need the geoemtry loaded
  if (!gGeoManager) {
    LOG(FATAL) << "No active geometry!" << FairLogger::endl;
  }

  o2::field::MagneticField* slow = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!slow) {
    LOG(FATAL) << "Magnetic field is not initialized!" << FairLogger::endl;
  }
  slow->AllowFastField(true);
  mField = slow->getFastField();
}

//_______________________________________________________________________
bool Propagator::PropagateToXBxByBz(TrackParCov& track, float xToGo, float mass, float maxSnp, float maxStep,
                                    int matCorr, int signCorr)
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
