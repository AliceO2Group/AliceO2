// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSORSPEC_H
#define FRAMEWORK_DATAPROCESSORSPEC_H

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataRef.h"
#include "Framework/DataAllocator.h"
#include "Framework/AlgorithmSpec.h"

#include <vector>
#include <string>

namespace o2 {
namespace framework {

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

class ConfigParamRegistry;
class ServiceRegistry;

struct DataProcessorSpec {
  std::string name;
  Inputs inputs;
  Outputs outputs;
  AlgorithmSpec algorithm;

  Options options;
  // FIXME: not used for now...
  std::vector<std::string> requiredServices;
  // FIXME: for the moment I put them here, but it's a hack
  //        since we do not want to expose this to users...
  //        Maybe we should have a ParallelGroup kind of node
  //        which embdes them and hides them from users.
  size_t rank = 0;
  size_t nSlots = 1;
  /// Which timeslice of the input is being processed by the associated
  /// DataProcessor. We do not need to keep track of how many are there,
  /// because time slices are completely independent from one another.
  /// This should not be set directly, but really be managed by the
  /// topology builder. Notice also that we need a number for the out
  /// put, but this is actually to be handled in the actual DeviceSpec.
  size_t inputTimeSliceId = 0;
  size_t maxInputTimeslices = 1;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATAPROCESSORSPEC_H
