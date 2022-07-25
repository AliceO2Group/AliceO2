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

/// @file   RawToDigitsSpec.h
/// @author Jens Wiechula
/// @since  2020-01-17
/// @brief  Processor spec for running TPC GBT raw frame to digit converter

#ifndef TPC_RAWTODIGITSSPEC_H_
#define TPC_RAWTODIGITSSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include <string_view>

namespace o2
{
namespace tpc
{

/// create a processor spec
/// read simulated TPC clusters from file and publish
o2::framework::DataProcessorSpec getRawToDigitsSpec(int channel, const std::string inputSpec, bool ignoreDistStf, std::vector<int> const& tpcSectors, bool sendCEdigits = false);

} // end namespace tpc
} // end namespace o2

#endif // TPC_RAWTODIGITSSPEC_H_
