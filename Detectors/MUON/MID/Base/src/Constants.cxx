// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/Constants.cxx
/// \brief  Implementation of constants for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 March 2018
#include "MIDBase/Constants.h"

#include <stdexcept>
#include <sstream>
#include <string>

namespace o2
{
namespace mid
{
constexpr std::array<const double, 4> Constants::sScaleFactors;
constexpr std::array<const double, 4> Constants::sDefaultChamberZ;

void Constants::assertDEId(int deId)
{
  /// Checks if the detection element ID is valid
  if (deId < 0 || deId > sNDetectionElements) {
    throw std::out_of_range("Detection element ID must be between 0 and 72");
  }
}

//_____________________________________________________________________________
std::string Constants::getDEName(int deId)
{
  /// Gets the detection element name from its ID
  /// @param deId The detection element ID
  int chId = getChamber(deId);
  int stId = 1 + chId / 2;
  int planeId = 1 + chId % 2;
  std::stringstream deName;
  deName << "MT" << stId << planeId << ((deId / 36 == 0) ? "In" : "Out") << (deId % 9) + 1;
  return deName.str();
}

} // namespace mid
} // namespace o2
