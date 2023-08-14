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

#include "ReconstructionDataFormats/Cascade.h"

using namespace o2::dataformats;
/*
Cascade::Cascade(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz, const std::array<float, 6>& covxyz,
                 const o2::track::TrackParCov& v0, const o2::track::TrackParCov& bachelor, o2::track::PID pid) : mProngs{v0, bachelor}
{
  std::array<float, 21> covC{0.}, covV{}, covB{};
  v0.getCovXYZPxPyPzGlo(covV);
  bachelor.getCovXYZPxPyPzGlo(covB);
  constexpr int MomInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
  for (int i = 0; i < 6; i++) {
    covC[i] = covxyz[i];
    covC[MomInd[i]] = covV[MomInd[i]] + covB[MomInd[i]];
  }
  this->set(xyz, pxyz, covC, v0.getCharge() + bachelor.getCharge(), true, pid);
  this->checkCorrelations();
  setV0Track(v0);
  setBachelorTrack(bachelor);
}
*/
