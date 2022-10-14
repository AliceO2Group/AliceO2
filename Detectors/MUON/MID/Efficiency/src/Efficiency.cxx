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

/// \file   MID/Efficiency/src/Efficiency.cxx
/// \brief  Computes the MID chamber efficiency
/// \author Livia Terlizzi <Livia.Terlizzi at cern.ch>
/// \date   20 September 2022

#include "MIDEfficiency/Efficiency.h"

#include "MIDBase/DetectorParameters.h"

namespace o2
{
namespace mid
{

void Efficiency::process(gsl::span<const mid::Track> midTracks)
{

  for (auto& track : midTracks) {
    auto deIdMT11 = track.getFiredDEId();
    auto isRight = detparams::isRightSide(deIdMT11);
    auto rpcLine = detparams::getRPCLine(deIdMT11);
    auto effFlag = track.getEfficiencyFlag();
    if (effFlag < 0) {
      continue;
    }

    for (int ich = 0; ich < 4; ++ich) {
      bool isFiredBP = track.isFiredChamber(ich, 0);
      bool isFiredNBP = track.isFiredChamber(ich, 1);
      mEff[static_cast<int>(ElementType::Plane)].addEntry(isFiredBP, isFiredNBP, ich, 0, 0);
      if (effFlag < 2) {
        continue;
      }
      auto deId = detparams::getDEId(isRight, ich, rpcLine);
      mEff[static_cast<int>(ElementType::RPC)].addEntry(isFiredBP, isFiredNBP, deId, 0, 0);
      if (effFlag < 3) {
        continue;
      }
      mEff[static_cast<int>(ElementType::Board)].addEntry(isFiredBP, isFiredNBP, deId, track.getFiredColumnId(), track.getFiredLineId());
    }
  }
}

} // namespace mid
} // namespace o2