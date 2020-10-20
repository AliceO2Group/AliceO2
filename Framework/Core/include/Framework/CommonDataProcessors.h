// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_CommonDataProcessors_H_INCLUDED
#define o2_framework_CommonDataProcessors_H_INCLUDED

#include "Framework/DataProcessorSpec.h"
#include "Framework/InputSpec.h"
#include "Framework/DataOutputDirector.h"
#include "TTree.h"

#include <vector>

namespace o2::framework
{
using outputTasks = std::vector<std::pair<uint32_t, std::string>>;
using outputObjects = std::vector<std::pair<uint32_t, std::vector<std::string>>>;

/// Helpers to create a few general data processors
struct CommonDataProcessors {
  /// Match all inputs of kind HIST and write them to a ROOT file,
  /// one root file per originating task.
  static DataProcessorSpec getHistogramRegistrySink(outputObjects const& objmap, const outputTasks& tskmap);
  /// Match all inputs of kind ATSK and write them to a ROOT file,
  /// one root file per originating task.
  static DataProcessorSpec getOutputObjSink(outputObjects const& objmap, const outputTasks& tskmap);
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
                                            std::vector<InputSpec> const& OutputInputs);

  /// @return a dummy DataProcessorSpec which requires all the passed @a InputSpec
  /// and simply discards them.
  static DataProcessorSpec getDummySink(std::vector<InputSpec> const& danglingInputs);
};

} // namespace o2::framework

#endif // o2_framework_CommonDataProcessors_H_INCLUDED
