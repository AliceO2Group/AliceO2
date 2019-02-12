// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file MathUtils.h
/// \brief Transformation functions for the coordinates
///

#ifndef O2_MFT_MATHUTILS_H_
#define O2_MFT_MATHUTILS_H_

#include <array>
#include <cmath>

#include "MFTTracking/Constants.h"
#include "MFTTracking/Definitions.h"

namespace o2
{
namespace MFT
{

namespace MathUtils
{
Float_t calculatePhiCoordinate(const Float_t, const Float_t);
Float_t calculateRCoordinate(const Float_t, const Float_t);
Float_t getNormalizedPhiCoordinate(const Float_t);
} // namespace MathUtils

inline Float_t MathUtils::calculatePhiCoordinate(const Float_t xCoordinate, const Float_t yCoordinate)
{
  return std::atan2(-yCoordinate, -xCoordinate) + Constants::Math::Pi;
}

inline Float_t MathUtils::calculateRCoordinate(const Float_t xCoordinate, const Float_t yCoordinate)
{
  return std::sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
}

inline Float_t MathUtils::getNormalizedPhiCoordinate(const Float_t phiCoordinate)
{
  return (phiCoordinate < 0)
           ? phiCoordinate + Constants::Math::TwoPi
           : (phiCoordinate > Constants::Math::TwoPi) ? phiCoordinate - Constants::Math::TwoPi : phiCoordinate;
}

} // namespace MFT
} // namespace o2

#endif /* O2_MFT_MATHUTILS_H_ */
