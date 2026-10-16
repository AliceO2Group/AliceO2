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
#include "Framework/DanglingEdgesContext.h"
#include "Headers/DataHeader.h"
#include <array>

namespace o2::framework
{
static constexpr std::array<header::DataOrigin, 5> AODOrigins{header::DataOrigin{"AOD"}, header::DataOrigin{"AOD1"}, header::DataOrigin{"AOD2"}, header::DataOrigin{"EMB"}, header::DataOrigin{"AMD"}};
static constexpr std::array<header::DataOrigin, 3> writableAODOrigins{header::DataOrigin{"AOD"}, header::DataOrigin{"AOD1"}, header::DataOrigin{"AOD2"}};

class DataOutputDirector;
struct ConfigContext;

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
  static void addMissingOutputsToSlicer(std::vector<InputSpec> const& requestedSLCs,
                                        DataProcessorSpec& publisher);
  /// Split the requested slice infos into groups by the device providing the sliced table
  /// (the AOD reader, if none of the providers has it) and create a slicer device for each
  /// group, so that each slicer depends on a single device and does not create loops.
  /// Each slicer is returned together with the name of its provider.
  static std::vector<std::pair<std::string, DataProcessorSpec>> makeSlicers(std::vector<InputSpec> const& requestedSLCs,
                                                                            std::vector<DataProcessorSpec const*> const& providers,
                                                                            std::vector<std::vector<InputSpec>>& slicerGroups);

  /// Match all inputs of kind ATSK and write them to a ROOT file,
  /// one root file per originating task.
  static DataProcessorSpec getOutputObjHistSink(ConfigContext const&);
  /// writes inputs of kind AOD to file
  static DataProcessorSpec getGlobalAODSink(ConfigContext const&);
  /// Get the data director
  static std::shared_ptr<DataOutputDirector> getDataOutputDirector(ConfigContext const& ctx);
};

}; // namespace o2::framework

#endif // O2_FRAMEWORK_ANALYSISSUPPORTHELPERS_H_
