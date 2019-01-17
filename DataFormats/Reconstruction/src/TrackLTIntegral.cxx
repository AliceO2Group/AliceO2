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
#include <cmath>

using namespace o2::track;

//_____________________________________________________
void TrackLTIntegral::print() const
{
  printf("L(cm): %6.2f, X2X0: %5.3f TOF(ps): ", getL(), getX2X0());
  for (int i = 0; i < getNTOFs(); i++) {
    printf(" %7.1f |", getTOF(i));
  }
  printf("\n");
}

//_____________________________________________________
void TrackLTIntegral::addStep(float dL, const TrackPar& track)
{
  ///< add step in cm to integrals
  mL += dL;
  float p2 = track.getP2Inv();
  float dTns = dL * 1000.f / o2::constants::physics::LightSpeedCm2NS; // time change in ps for beta = 1 particle
  for (int id = 0; id < getNTOFs(); id++) {
    float m2z = o2::track::PID::getMass2Z(id);
    float betaInv = std::sqrt(1.f + m2z * m2z * p2);
    mT[id] += dTns * betaInv;
  }
}
