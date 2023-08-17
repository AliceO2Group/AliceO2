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

/// @file   OccupancyFilterSpec.h
/// @author Jens Wiechula
/// @since  2021-11-19
/// @brief  Processor spec for filtering krypton raw data

#ifndef TPC_OccupancyFilterSpec_H_
#define TPC_OccupancyFilterSpec_H_

#include "Framework/DataProcessorSpec.h"

namespace o2::tpc
{

o2::framework::DataProcessorSpec getOccupancyFilterSpec();

} // namespace o2::tpc

#endif
