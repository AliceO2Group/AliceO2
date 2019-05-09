// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/ChamberEfficiencyResponse.cxx
/// \brief  Implementation MID RPC efficiency response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2019

#include "MIDSimulation/ChamberEfficiencyResponse.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
ChamberEfficiencyResponse::ChamberEfficiencyResponse(const ChamberEfficiency& efficiencyMap) : mEfficiency(efficiencyMap)
{
  /// Constructor
}

//______________________________________________________________________________
bool ChamberEfficiencyResponse::isEfficient(int deId, int columnId, int line, bool& isEfficientBP, bool& isEfficientNBP)
{
  /// \par deId Detection element ID
  /// \par columnId Column ID
  /// \par line line of the local board in the RPC
  /// \par isEfficientBP The BP was efficient
  /// \par isEfficientNBP The NBP was efficient
  /// Returns true if the chamber is efficient

  // P(B) : probability to fire bending plane
  double effBP = mEfficiency.getEfficiency(deId, columnId, line, ChamberEfficiency::EffType::BendPlane);

  // P(BN) : probability to fire bending and non-bending plane
  double effBoth = mEfficiency.getEfficiency(deId, columnId, line, ChamberEfficiency::EffType::BothPlanes);

  isEfficientBP = (mRandom(mGenerator) <= effBP);

  // P(N) : probability to fire non-bending plane
  double effNBP = 0.;
  if (isEfficientBP) {
    // P(N|B) = P(BN) / P(B)
    effNBP = effBoth / effBP;
  } else {
    // P(N|!B) = ( P(N) - P(BN) ) / ( 1 - P(B) )
    effNBP = (mEfficiency.getEfficiency(deId, columnId, line, ChamberEfficiency::EffType::NonBendPlane) - effBoth) / (1. - effBP);
  }

  isEfficientNBP = (mRandom(mGenerator) <= effNBP);

  return (isEfficientBP || isEfficientNBP);
}

ChamberEfficiencyResponse createDefaultChamberEfficiencyResponse()
{
  return ChamberEfficiencyResponse(createDefaultChamberEfficiency());
}

} // namespace mid
} // namespace o2
