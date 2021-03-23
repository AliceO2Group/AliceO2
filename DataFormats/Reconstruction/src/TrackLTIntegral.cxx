// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "CommonConstants/PhysicsConstants.h"
#include "MathUtils/Utils.h"

namespace o2
{
namespace track
{

//_____________________________________________________
GPUd() void TrackLTIntegral::print() const
{
#ifndef GPUCA_GPUCODE_DEVICE
  printf("L(cm): %6.2f, X2X0: %e XRho: %e TOF(ps): ", getL(), getX2X0(), getXRho());
  if (isTimeNotNeeded()) {
    printf(" Times not filled");
  } else {
    for (int i = 0; i < getNTOFs(); i++) {
      printf(" %7.1f |", getTOF(i));
    }
  }
  printf("\n");
#endif
}

//_____________________________________________________
GPUd() void TrackLTIntegral::addStep(float dL, float p2Inv)
{
  ///< add step in cm to integrals
  mL += dL;
  if (isTimeNotNeeded()) {
    return;
  }
  const float dTns = dL * 1000.f / o2::constants::physics::LightSpeedCm2NS; // time change in ps for beta = 1 particle
  for (int id = 0; id < getNTOFs(); id++) {
    const float m2z = track::PID::getMass2Z(id);
    const float betaInv = math_utils::sqrt(1.f + m2z * m2z * p2Inv);
    mT[id] += dTns * betaInv;
  }
}

} // namespace track
} // namespace o2
