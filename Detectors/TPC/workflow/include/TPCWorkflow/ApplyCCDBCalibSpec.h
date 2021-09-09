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

#ifndef O2_TPC_ApplyCCDBCalibSpec_H
#define O2_TPC_ApplyCCDBCalibSpec_H

/// @file   ApplyCCDBCalibSpec.h
/// @brief  Device to apply calibrations loaded from CCDB

namespace o2::framework
{
struct DataProcessorSpec;
} // namespace o2::framework

namespace o2::tpc
{
o2::framework::DataProcessorSpec getApplyCCDBCalibSpec();
} // namespace o2::tpc
#endif
