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
#ifndef O2_FRAMEWORK_COMMONDATAPROCESSORS_H_
#define O2_FRAMEWORK_COMMONDATAPROCESSORS_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/InputSpec.h"

#include <vector>
#include <string>

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
} // namespace o2::framework
extern template class std::vector<o2::framework::OutputObjectInfo>;
extern template class std::vector<o2::framework::OutputTaskInfo>;
namespace o2::framework
{
/// Helpers to create a few general data processors
struct CommonDataProcessors {
  /// Match all inputs of kind ATSK and write them to a ROOT file,
  /// one root file per originating task.
  static DataProcessorSpec getOutputObjHistSink(std::vector<OutputObjectInfo> const& objmap,
                                                std::vector<OutputTaskInfo> const& tskmap);
  /// Given the list of @a danglingInputs @return a DataProcessor which does
  /// a binary dump for all the dangling inputs matching the Timeframe
  /// lifetime. @a unmatched will be filled with all the InputSpecs which are
  /// not going to be used by the returned DataProcessorSpec.
  static DataProcessorSpec getGlobalFileSink(std::vector<InputSpec> const& danglingInputs,
                                             std::vector<InputSpec>& unmatched);
  /// Given the list of @a danglingInputs @return a DataProcessor which
  /// exposes them through a FairMQ channel.
  /// @fixme: for now we only support shmem and ipc
  /// @fixme: for now only the dangling inputs are forwarded.
  static DataProcessorSpec getGlobalFairMQSink(std::vector<InputSpec> const& danglingInputs);

  /// writes inputs of kind AOD to file
  static DataProcessorSpec getGlobalAODSink(std::shared_ptr<DataOutputDirector> dod,
                                            std::vector<InputSpec> const& outputInputs);

  /// @return a dummy DataProcessorSpec which requires all the passed @a InputSpec
  /// and simply discards them. @a rateLimitingChannelConfig is the configuration
  /// for the rate limiting channel, if any required.
  static DataProcessorSpec getDummySink(std::vector<InputSpec> const& danglingInputs, std::string rateLimitingChannelConfig);
  static AlgorithmSpec wrapWithRateLimiting(AlgorithmSpec spec);
};

} // namespace o2::framework

#endif // o2_framework_CommonDataProcessors_H_INCLUDED
