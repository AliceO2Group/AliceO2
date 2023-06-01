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

#include "ReconstructionDataFormats/V0.h"

using namespace o2::dataformats;

V0::V0(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz,
       const o2::track::TrackParCov& trPos, const o2::track::TrackParCov& trNeg,
       GIndex trPosID, GIndex trNegID, o2::track::PID pid)
  : mProngIDs{trPosID, trNegID}, mProngs{trPos, trNeg}
{
  std::array<float, 21> covV{0.}, covP, covN;
  trPos.getCovXYZPxPyPzGlo(covP);
  trNeg.getCovXYZPxPyPzGlo(covN);
  constexpr int MomInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
  for (int i = 0; i < 6; i++) {
    covV[i] = covxyz[i];
    covV[MomInd[i]] = covP[MomInd[i]] + covN[MomInd[i]];
  }
  this->set(xyz, pxyz, covV, trPos.getCharge() + trNeg.getCharge(), true, pid);
  this->checkCorrelations();
}

float V0::calcMass2(float massPos2, float massNeg2) const
{
  auto p2 = getP2();
  auto p2pos = mProngs[0].getP2(), p2neg = mProngs[1].getP2();
  auto energy = std::sqrt(massPos2 + p2pos) + std::sqrt(massNeg2 + p2neg);
  return energy * energy - p2;
}
