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
#ifndef O2_FRAMEWORK_DANGLINGEDGESCONTEXT_H_
#define O2_FRAMEWORK_DANGLINGEDGESCONTEXT_H_

#include <vector>
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"

namespace o2::framework
{
class DataOutputDirector;

struct OutputTaskInfo {
  uint32_t id;
  std::string name;
};

struct OutputObjectInfo {
  uint32_t id;
  std::vector<std::string> bindings;
};

// This will keep track of the inputs which have
// been requested and for which we will need to inject
// some source device.
struct DanglingEdgesContext {
  std::vector<InputSpec> requestedAODs;
  std::vector<OutputSpec> providedAODs;
  std::vector<InputSpec> requestedDYNs;
  std::vector<OutputSpec> providedDYNs;
  std::vector<InputSpec> requestedIDXs;
  std::vector<OutputSpec> providedTIMs;
  std::vector<InputSpec> requestedTIMs;
  std::vector<OutputSpec> providedOutputObjHist;
  std::vector<InputSpec> spawnerInputs;

  // These are the timestamped tables which are required to
  // inject the the CCDB objecs.
  std::vector<InputSpec> analysisCCDBInputs;

  // Needed to created the hist writer
  std::vector<OutputTaskInfo> outTskMap;
  std::vector<OutputObjectInfo> outObjHistMap;

  // Needed to create the output director
  std::vector<InputSpec> outputsInputs;
  std::vector<bool> isDangling;

  // Needed to create the aod writer
  std::vector<InputSpec> outputsInputsAOD;
};
} // namespace o2::framework

extern template class std::vector<o2::framework::OutputObjectInfo>;
extern template class std::vector<o2::framework::OutputTaskInfo>;

#endif // O2_FRAMEWORK_DANGLINGEDGESCONTEXT_H_
