// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/MpArea.cxx
/// \brief  Implementation of mapping area for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 December 2017
#include "MIDBase/MpArea.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
MpArea::MpArea(float x1, float y1, float x2, float y2) : mPositions()
{
  /// Constructor
  mPositions[0] = (x1 < x2) ? x1 : x2;
  mPositions[2] = (x1 < x2) ? x2 : x1;
  mPositions[1] = (y1 < y2) ? y1 : y2;
  mPositions[3] = (y1 < y2) ? y2 : y1;
}

//______________________________________________________________________________
float MpArea::getCenterX() const
{
  /// Return x position of the area center
  return 0.5 * (getXmax() + getXmin());
}

//______________________________________________________________________________
float MpArea::getCenterY() const
{
  /// Return y position of the area center
  return 0.5 * (getYmax() + getYmin());
}

//______________________________________________________________________________
float MpArea::getHalfSizeX() const
{
  /// Return half size in x
  return 0.5 * (getXmax() - getXmin());
}

//______________________________________________________________________________
float MpArea::getHalfSizeY() const
{
  /// Return half size in y
  return 0.5 * (getYmax() - getYmin());
}

//______________________________________________________________________________
bool MpArea::isValid() const
{
  /// Check if area is valid
  for (int icoor = 0; icoor < 2; ++icoor) {
    if (mPositions[icoor] == mPositions[icoor + 2])
      return false;
  }
  return true;
}

} // namespace mid
} // namespace o2
