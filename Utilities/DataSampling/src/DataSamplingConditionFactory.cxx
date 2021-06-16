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

/// \file DataSamplingConditionFactory.cxx
/// \brief Implementation of DataSamplingConditionFactory
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <memory>
#include <stdexcept>

#include "DataSampling/DataSamplingConditionFactory.h"
using namespace o2::framework;

namespace o2::utilities
{

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::create(std::string name)
{
  if (name == "random" || name == "DataSamplingConditionRandom") {
    return createDataSamplingConditionRandom();
  } else if (name == "payloadSize" || name == "DataSamplingConditionPayloadSize") {
    return createDataSamplingConditionPayloadSize();
  } else if (name == "nConsecutive" || name == "DataSamplingConditionNConsecutive") {
    return createDataSamplingConditionNConsecutive();
  } else if (name == "custom" || name == "DataSamplingConditionCustom") {
    return createDataSamplingConditionCustom();
  }
  throw std::runtime_error("DataSamplingCondition '" + name + "' unknown.");
}

} // namespace o2::utilities
