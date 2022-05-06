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

/// \file   MID/src/MCLabel.cxx
/// \brief  Implementation of MC label for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 February 2019

#include "DataFormatsMID/MCLabel.h"

ClassImp(o2::mid::MCLabel);

namespace o2
{
namespace mid
{

MCLabel::MCLabel(int trackID, int eventID, int srcID, int deId, int columnId, int cathode, int firstStrip, int lastStrip) : o2::MCCompLabel(trackID, eventID, srcID, false)
{
  /// Constructor
  setDEId(deId);
  setColumnId(columnId);
  setCathode(cathode);
  setFirstStrip(firstStrip);
  setLastStrip(lastStrip);
}

void MCLabel::set(int value, unsigned int mask, unsigned int offset)
{
  /// Sets the value
  mStripsInfo &= ~(mask << offset);
  mStripsInfo |= ((value & mask) << offset);
}

bool MCLabel::operator==(const MCLabel& other) const
{
  if (compare(other) != 1) {
    return false;
  }
  return (mStripsInfo == other.mStripsInfo);
}

} // namespace mid
} // namespace o2
