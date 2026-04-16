// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/Decay3Body.h"

using namespace o2::dataformats;

Decay3Body::Decay3Body(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz, const Track& tr0, const Track& tr1, const Track& tr2, o2::track::PID pid)
  : mProngs{tr0, tr1, tr2}
{
  std::array<float, 21> cov{}, cov0{}, cov1{}, cov2{};
  tr0.getCovXYZPxPyPzGlo(cov0);
  tr1.getCovXYZPxPyPzGlo(cov1);
  tr2.getCovXYZPxPyPzGlo(cov2);
  constexpr int MomInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
  for (int i = 0; i < 6; i++) {
    cov[i] = covxyz[i];
    cov[MomInd[i]] = cov0[MomInd[i]] + cov1[MomInd[i]] + cov2[MomInd[i]];
  }
  this->set(xyz, pxyz, cov, tr0.getCharge() + tr1.getCharge() + tr2.getCharge(), true, pid);
}

float Decay3Body::calcMass2(float mass0, float mass1, float mass2) const
{
  auto p2 = getP2();
  auto energy = std::sqrt(mass0 + mProngs[0].getP2()) + std::sqrt(mass1 + mProngs[1].getP2()) + std::sqrt(mass1 + mProngs[2].getP2());
  return energy * energy - p2;
}
