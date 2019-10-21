// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICESPEC_H
#define FRAMEWORK_DEVICESPEC_H

#include "Framework/WorkflowSpec.h"
#include "Framework/ComputingResource.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelInfo.h"
#include "Framework/DeviceControl.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ForwardRoute.h"
#include "Framework/InputRoute.h"
#include "Framework/OutputRoute.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"

#include <vector>
#include <string>
#include <map>
#include <utility>

namespace o2
{
namespace framework
{

/// Concrete description of the device which will actually run
/// a DataProcessor.
struct DeviceSpec {
  std::string name;
  std::string id;
  std::vector<InputChannelSpec> inputChannels;
  std::vector<OutputChannelSpec> outputChannels;
  std::vector<std::string> arguments;
  std::vector<ConfigParamSpec> options;

  AlgorithmSpec algorithm;

  std::vector<InputRoute> inputs;
  std::vector<OutputRoute> outputs;
  std::vector<ForwardRoute> forwards;
  size_t rank;   // Id of a parallel processing I am part of
  size_t nSlots; // Total number of parallel units I am part of
  size_t inputTimesliceId;
  /// The completion policy to use for this device.
  CompletionPolicy completionPolicy;
  DispatchPolicy dispatchPolicy;
  ComputingResource resource;
};

} // namespace framework
} // namespace o2
#endif
