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

#ifndef STEER_DIGITIZERWORKFLOW_SRC_TPCDIGITROOTWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_TPCDIGITROOTWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include <vector>
#include <numeric> // std::iota

namespace o2
{
namespace tpc
{
/// get the processor spec
/// the laneConfiguration is a vector of subspecs which the processor subscribes to
o2::framework::DataProcessorSpec getTPCDigitRootWriterSpec(std::vector<int> const& laneConfiguration, bool mctruth);

// numberofsourcedevices is the number of devices we receive digits from
inline o2::framework::DataProcessorSpec getTPCDigitRootWriterSpec(int numberofsourcedevices = 1)
{
  std::vector<int> defaultConfiguration(numberofsourcedevices);
  std::iota(defaultConfiguration.begin(), defaultConfiguration.end(), 0);
  return getTPCDigitRootWriterSpec(defaultConfiguration, true);
}

} // namespace tpc
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_TPCDIGITROOTWRITERSPEC_H_ */
