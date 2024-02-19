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

/// \file   MID/Efficiency/src/ChamberEfficiency.cxx
/// \brief  Implementation of the chamber efficiency for the MID RPC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2019

#include "MIDEfficiency/ChamberEfficiency.h"

#include <iostream>
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/Mapping.h"

namespace o2
{
namespace mid
{

void ChamberEfficiency::setFromCounters(const std::vector<ChEffCounter>& counters)
{
  mCounters.clear();
  for (auto& count : counters) {
    mCounters[detparams::makeUniqueFEEId(count.deId, count.columnId, count.lineId)] = count;
  }
}

std::vector<ChEffCounter> ChamberEfficiency::getCountersAsVector() const
{
  std::vector<ChEffCounter> counters;
  for (auto& count : mCounters) {
    counters.emplace_back(count.second);
  }
  return counters;
}

EffCountType ChamberEfficiency::convert(EffType type) const
{
  switch (type) {
    case EffType::BendPlane:
      return EffCountType::BendPlane;
    case EffType::NonBendPlane:
      return EffCountType::BendPlane;
    case EffType::BothPlanes:
      return EffCountType::BothPlanes;
  }
  return EffCountType::BothPlanes; // Never used
}

double ChamberEfficiency::getEfficiency(int deId, int columnId, int lineId, EffType type) const
{
  auto idx = detparams::makeUniqueFEEId(deId, columnId, lineId);
  auto entryTot = mCounters.find(idx);
  if (entryTot == mCounters.end()) {
    std::cerr << "Warning: no efficiency found for deId: " << deId << "  column: " << columnId << "  line: " << lineId << "  type: " << static_cast<int>(type) << "\n";
    return -1.;
  }
  return static_cast<double>(entryTot->second.getCounts(convert(type))) / static_cast<double>(entryTot->second.getCounts(EffCountType::AllTracks));
}

//______________________________________________________________________________
void ChamberEfficiency::addEntry(bool isEfficientBP, bool isEfficientNBP, int deId, int columnId, int lineId)
{
  auto& entry = mCounters[detparams::makeUniqueFEEId(deId, columnId, lineId)];
  entry.deId = deId;
  entry.columnId = columnId;
  entry.lineId = lineId;
  ++entry.counts[3];
  if (isEfficientBP) {
    ++entry.counts[0];
  }
  if (isEfficientNBP) {
    ++entry.counts[1];
  }
  if (isEfficientBP && isEfficientNBP) {
    ++entry.counts[2];
  }
}

ChamberEfficiency createDefaultChamberEfficiency()
{
  /// Creates the default parameters
  ChamberEfficiency effMap;
  Mapping mapping;
  uint32_t nevents = 10000;
  for (int ide = 0; ide < detparams::NDetectionElements; ++ide) {
    for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      for (int iline = mapping.getFirstBoardBP(icol, ide); iline <= mapping.getLastBoardBP(icol, ide); ++iline) {
        for (uint32_t ievent = 0; ievent < nevents; ++ievent) {
          effMap.addEntry(true, true, ide, icol, iline);
        }
      }
    }
  }

  return effMap;
}

} // namespace mid
} // namespace o2
