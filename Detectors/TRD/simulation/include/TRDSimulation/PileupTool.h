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

#ifndef ALICEO2_TRD_PILEUPTOOL_H_
#define ALICEO2_TRD_PILEUPTOOL_H_

#include "DataFormatsTRD/SignalArray.h" // for SignalArray
#include "DataFormatsTRD/Constants.h"

#include <array>
#include <deque>
#include <unordered_map>

namespace o2
{
namespace trd
{

using SignalContainer = std::unordered_map<int, SignalArray>;

struct PileupTool {
  SignalContainer addSignals(std::deque<std::array<SignalContainer, constants::MAXCHAMBER>>&, const double&);
};

} // namespace trd
} // namespace o2
#endif
