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

/// \file   Constants.h
/// \brief  General constants in FV0
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland

#ifndef ALICEO2_FV0_CONSTANTS_
#define ALICEO2_FV0_CONSTANTS_

#include "FV0Base/Geometry.h"

namespace o2
{
namespace fv0
{

struct Constants {
  static constexpr int nChannelsPerPm = 12; // Fixed now together with the production of PMs - will remain constant
  static constexpr int nPms = 6;            // Number of processing modules (PMs); 1 PM per ring, 2 PMs needed for ring 5
  static constexpr int nTcms = 1;           // Number of trigger and clock modules (TCMs)
  static constexpr int nGbtLinks = nPms + nTcms;
  static constexpr int nFv0Channels = Geometry::getNumberOfReadoutChannels();
  static constexpr int nFv0ChannelsPlusRef = nFv0Channels + 1;
};

} // namespace fv0
} // namespace o2
#endif
