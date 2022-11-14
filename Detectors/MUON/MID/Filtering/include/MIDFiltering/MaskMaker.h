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

/// \file   MIDFiltering/MaskMaker.h
/// \brief  Function to produce the MID masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 January 2020
#ifndef O2_MID_MASKMAKER_H
#define O2_MID_MASKMAKER_H

#include <cstdint>
#include "MIDFiltering/ChannelScalers.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/FEEIdConfig.h"

namespace o2
{
namespace mid
{

std::vector<ColumnData> makeBadChannels(const ChannelScalers& scalers, double timeOrTriggers, double threshold);
std::vector<ColumnData> makeMasks(const ChannelScalers& scalers, double timeOrTriggers, double threshold, const std::vector<ColumnData>& refMasks = {});
std::vector<ColumnData> makeDefaultMasks();
std::vector<ColumnData> makeDefaultMasksFromCrateConfig(const FEEIdConfig& feeIdConfig = FEEIdConfig(), const CrateMasks& crateMasks = CrateMasks());

} // namespace mid
} // namespace o2

#endif /* O2_MID_MASKMAKER_H */
