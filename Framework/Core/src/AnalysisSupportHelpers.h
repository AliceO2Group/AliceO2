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
#ifndef O2_FRAMEWORK_ANALYSISSUPPORTHELPERS_H_
#define O2_FRAMEWORK_ANALYSISSUPPORTHELPERS_H_

#include "Framework/OutputSpec.h"
#include "Framework/InputSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include <array>

namespace o2::framework
{
static constexpr std::array<header::DataOrigin, 3> AODOrigins{header::DataOrigin{"AOD"}, header::DataOrigin{"AOD1"}, header::DataOrigin{"AOD2"}};
static constexpr std::array<header::DataOrigin, 5> extendedAODOrigins{header::DataOrigin{"AOD"}, header::DataOrigin{"AOD1"}, header::DataOrigin{"AOD2"}, header::DataOrigin{"DYN"}, header::DataOrigin{"AMD"}};
static constexpr std::array<header::DataOrigin, 4> writableAODOrigins{header::DataOrigin{"AOD"}, header::DataOrigin{"AOD1"}, header::DataOrigin{"AOD2"}, header::DataOrigin{"DYN"}};

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
//
struct AnalysisContext {
  std::vector<InputSpec> requestedAODs;
  std::vector<OutputSpec> providedAODs;
  std::vector<InputSpec> requestedDYNs;
  std::vector<OutputSpec> providedDYNs;
  std::vector<InputSpec> requestedIDXs;
  std::vector<OutputSpec> providedOutputObjHist;
  std::vector<InputSpec> spawnerInputs;

  std::vector<OutputTaskInfo> outTskMap;
  std::vector<OutputObjectInfo> outObjHistMap;
};

// Helper class to be moved in the AnalysisSupport plugin at some point
struct AnalysisSupportHelpers {
  /// Helper functions to add AOD related internal devices.
  /// FIXME: moved here until we have proper plugin based amendment
  ///        of device injection
  static void addMissingOutputsToReader(std::vector<OutputSpec> const& providedOutputs,
                                        std::vector<InputSpec> const& requestedInputs,
                                        DataProcessorSpec& publisher);
  static void addMissingOutputsToSpawner(std::vector<OutputSpec> const& providedSpecials,
                                         std::vector<InputSpec> const& requestedSpecials,
                                         std::vector<InputSpec>& requestedAODs,
                                         DataProcessorSpec& publisher);
  static void addMissingOutputsToBuilder(std::vector<InputSpec> const& requestedSpecials,
                                         std::vector<InputSpec>& requestedAODs,
                                         std::vector<InputSpec>& requestedDYNs,
                                         DataProcessorSpec& publisher);

  /// Match all inputs of kind ATSK and write them to a ROOT file,
  /// one root file per originating task.
  static DataProcessorSpec getOutputObjHistSink(std::vector<OutputObjectInfo> const& objmap,
                                                std::vector<OutputTaskInfo> const& tskmap);
  /// writes inputs of kind AOD to file
  static DataProcessorSpec getGlobalAODSink(std::shared_ptr<DataOutputDirector> dod,
                                            std::vector<InputSpec> const& outputInputs);
};

}; // namespace o2::framework

#endif // O2_FRAMEWORK_ANALYSISSUPPORTHELPERS_H_
