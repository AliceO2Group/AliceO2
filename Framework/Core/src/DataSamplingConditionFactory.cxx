// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionFactory.cxx
/// \brief Implementation of DataSamplingConditionFactory
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <memory>

#include "Framework/DataSamplingConditionFactory.h"
#include "Framework/Logger.h"

namespace o2
{
namespace framework
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

} // namespace framework
} // namespace o2
