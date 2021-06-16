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

/// @file   LinkZSToDigitsSpec.h
/// @author Jens Wiechula
/// @since  2020-02-20
/// @brief  Processor spec for running link based zero suppressed data to digit converter

#ifndef TPC_LinkZSToDigitsSpec_H_
#define TPC_LinkZSToDigitsSpec_H_

#include "Framework/DataProcessorSpec.h"
#include <string_view>

namespace o2
{
namespace tpc
{

/// Processor to convert link based zero suppressed data to simulation digits
o2::framework::DataProcessorSpec getLinkZSToDigitsSpec(int channel, const std::string_view inputDef, std::vector<int> const& tpcSectors);

} // end namespace tpc
} // end namespace o2

#endif // TPC_LinkZSToDigitsSpec_H_
