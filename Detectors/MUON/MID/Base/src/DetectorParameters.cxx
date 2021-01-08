// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/DetectorParameters.cxx
/// \brief  Implementation of detector params for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#include "MIDBase/DetectorParameters.h"

#include <stdexcept>
#include <sstream>
#include <string>

namespace o2
{
namespace mid
{
namespace detparams
{
void assertDEId(int deId)
{
  /// Checks if the detection element ID is valid
  if (deId < 0 || deId > NDetectionElements) {
    throw std::out_of_range("Detection element ID must be between 0 and 72");
  }
}

int getDEId(bool isRight, int chamber, int rpc)
{
  /// Gets detection element Id
  /// \param isRight RPC is in right side
  /// \param chamber The chamber ID (0-3)
  /// \param rpc RPC ID (0-8)
  int deOffset = (isRight) ? 0 : NDetectionElementsPerSide;
  return deOffset + NRPCLines * chamber + rpc;
}

std::string getDEName(int deId)
{
  /// Gets the detection element name from its ID
  /// \param deId The detection element ID
  int chId = getChamber(deId);
  int stId = 1 + chId / 2;
  int planeId = 1 + chId % 2;
  std::stringstream deName;
  deName << "MT" << stId << planeId << ((deId / NDetectionElementsPerSide == 0) ? "In" : "Out") << (deId % NRPCLines) + 1;
  return deName.str();
}

} // namespace detparams
} // namespace mid
} // namespace o2
