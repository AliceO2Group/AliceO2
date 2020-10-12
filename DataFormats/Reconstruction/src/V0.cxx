// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/V0.h"

using namespace o2::dataformats;

V0::V0(const std::array<float, 3>& xyz, const std::array<float, 3>& pxyz,
       const o2::track::TrackParCov& trPos, const o2::track::TrackParCov& trNeg,
       GIndex trPosID, GIndex trNegID)
  : o2::track::TrackParCov{xyz, pxyz, 0, false}, mProngIDs{trPosID, trNegID}, mProngs{trPos, trNeg}
{
}

float V0::calcMass2(float massPos2, float massNeg2) const
{
  auto p2 = getP2();
  auto p2pos = mProngs[0].getP2(), p2neg = mProngs[1].getP2();
  auto energy = std::sqrt(massPos2 + p2pos) + std::sqrt(massNeg2 + p2neg);
  return energy * energy - p2;
}
