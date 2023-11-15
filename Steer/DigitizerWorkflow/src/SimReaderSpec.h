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

#ifndef O2_STEER_SIMREADERSPEC_H
#define O2_STEER_SIMREADERSPEC_H

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace steer
{
struct SubspecRange {
  int min = 0;
  int max = 0;
};

o2::framework::DataProcessorSpec getSimReaderSpec(SubspecRange range, const std::vector<std::string>& simprefixes, const std::vector<int>& tpcsectors, bool withTrigger = false);
}
} // namespace o2

#endif // O2_STEER_SIMREADERSPEC_H
