// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/ChamberEfficiency.cxx
/// \brief  Implementation of the chamber efficiency for the MID RPC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2019

#include "MIDBase/ChamberEfficiency.h"

#include <iostream>
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/Mapping.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
int ChamberEfficiency::typeToIdx(EffType type) const
{
  /// Converts efficiency types to indexes
  switch (type) {
    case EffType::BendPlane:
      return 0;
    case EffType::NonBendPlane:
      return 1;
    case EffType::BothPlanes:
      return 2;
  }

  // This will never be reached, but it is needed to avoid compiler warnings
  return -1;
}

//______________________________________________________________________________
int ChamberEfficiency::indexToInt(int deId, int columnId, int line) const
{
  /// Converts indexes to integer for easier map access
  return line + 10 * columnId + 100 * deId;
}

//______________________________________________________________________________
double ChamberEfficiency::getEfficiency(int deId, int columnId, int line, EffType type) const
{
  /// Gets the efficiency
  /// \par deId Detection element ID
  /// \par columnId Column ID
  /// \par line line of the local board in the RPC
  int idx = indexToInt(deId, columnId, line);
  auto entryTot = mCounters.find(idx);
  if (entryTot == mCounters.end()) {
    std::cerr << "Warning: no efficiency found for deId: " << deId << "  column: " << columnId << "  line: " << line << "  type: " << typeToIdx(type) << "\n";
    return -1.;
  }
  return static_cast<double>(entryTot->second[typeToIdx(type)]) / static_cast<double>(entryTot->second[3]);
}

//______________________________________________________________________________
void ChamberEfficiency::addEntry(bool isEfficientBP, bool isEfficientNBP, int deId, int columnId, int line)
{
  /// Adds an entry
  /// \par passed Number of efficient events
  /// \par total Total number of events
  /// \par deId Detection element ID
  /// \par columnId Column ID
  /// \par line line of the local board in the RPC
  auto& entry = mCounters[indexToInt(deId, columnId, line)];
  ++entry[3];
  if (isEfficientBP) {
    ++entry[0];
  }
  if (isEfficientNBP) {
    ++entry[1];
  }
  if (isEfficientBP && isEfficientNBP) {
    ++entry[2];
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
          effMap.addEntry(true, true, ide, icol, iline);
          effMap.addEntry(true, true, ide, icol, iline);
        }
      }
    }
  }

  return std::move(effMap);
}

} // namespace mid
} // namespace o2
