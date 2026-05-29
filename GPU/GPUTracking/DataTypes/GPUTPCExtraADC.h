// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCExtraADC.h
/// \author Felix Weiglhofer

#include "GPUDefConstantsAndSettings.h"
#include "DataFormatsTPC/Digit.h"
#include <array>
#include <vector>

namespace o2::gpu
{
struct GPUTPCExtraADC {
  std::array<std::vector<tpc::Digit>, tpc::constants::MAXSECTOR> digitsBySector;
};
} // namespace o2::gpu
