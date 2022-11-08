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

#ifndef O2_TPC_DCSConfigSpec_H_
#define O2_TPC_DCSConfigSpec_H_

/// @file   DCSConfigSpec.h
/// @author Jens Wiechula
/// @brief  DCS configuration processing

#include "Framework/DataProcessorSpec.h"
using namespace o2::framework;

namespace o2::tpc
{

/// create DCS processor
DataProcessorSpec getDCSConfigSpec();

} // end namespace o2::tpc

#endif
