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
#ifndef O2_FRAMEWORK_DATAPROCESSORSPEC_H_
#define O2_FRAMEWORK_DATAPROCESSORSPEC_H_

#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataProcessorLabel.h"
#include "Framework/DataRef.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/CommonServices.h"

#include <string>
#include <vector>

namespace o2::framework
{

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

class ConfigParamRegistry;
class ServiceRegistry;

struct DataProcessorMetadata {
  std::string key;
  std::string value;
};

struct DataProcessorSpec {
  std::string name;
  Inputs inputs;
  Outputs outputs;
  AlgorithmSpec algorithm;

  Options options = {};
  /// A set of services which are required to run
  /// this data processor spec. Defaults to the old
  /// list of hardcoded services. If you want
  /// to override them, make sure you request at least
  /// CommonServices::requiredServices() otherwise things
  /// will go horribly wrong.
  std::vector<ServiceSpec> requiredServices = CommonServices::defaultServices();
  /// Labels associated to the DataProcessor. These can
  /// be used to group different DataProcessor together
  /// and they can allow to filter different groups, e.g.
  /// use push/pull rather than pub/sub for all the edges
  /// which involve a DataProcessorSpec with a given label.
  /// Examples labels could be "reco", "qc".
  std::vector<DataProcessorLabel> labels = {};

  /// Extra key, value pairs which can be used to describe extra information
  /// about a given data processor.
  std::vector<DataProcessorMetadata> metadata = {};

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

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAPROCESSORSPEC_H_
