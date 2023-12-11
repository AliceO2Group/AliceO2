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

/// \file Utils.cxx
/// \brief Implementation of generic utils for DPL devices, v0.1
///
/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#include "DPLUtils/Utils.h"
#include "Framework/DataSpecUtils.h"

using namespace o2::framework;

namespace o2
{
namespace workflows
{

// Method to convert an OutputSpec in a Output.
Output getOutput(const o2f::OutputSpec outputSpec)
{
  auto concrete = DataSpecUtils::asConcreteDataMatcher(outputSpec);
  return Output{concrete.origin, concrete.description, concrete.subSpec};
}

// This method can convert a vector of OutputSpec into a vector of Output.
// This is useful for DPL devices, to avoid specifying both OutputSpec and Output in define
std::shared_ptr<std::vector<Output>> getOutputList(const Outputs outputSpecs)
{
  std::shared_ptr<std::vector<o2f::Output>> outputList = std::make_shared<std::vector<o2f::Output>>();

  for (const auto& itOutputSpec : outputSpecs) {
    (*outputList).emplace_back(getOutput(itOutputSpec));
  }

  return outputList;
};

} // namespace workflows
} // namespace o2
