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

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TPCWorkflow/LaserTrackFilterSpec.h"
#include "DataFormatsTPC/TrackTPC.h"

using namespace o2::framework;
using namespace o2::tpc;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"enable-writer", VariantType::Bool, false, {"selection string input specs"}},
  };

  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  using namespace o2::tpc;

  WorkflowSpec workflow;
  workflow.emplace_back(getLaserTrackFilter());

  if (config.options().get<bool>("enable-writer")) {
    const char* processName = "tpc-laser-track-writer";
    const char* defaultFileName = "tpc-laser-tracks.root";
    const char* defaultTreeName = "tpcrec";

    // branch definitions for RootTreeWriter spec
    using TrackOutputType = std::vector<o2::tpc::TrackTPC>;

    // a spectator callback which will be invoked by the tree writer with the extracted object
    // we are using it for printing a log message
    auto logger = BranchDefinition<TrackOutputType>::Spectator([](TrackOutputType const& tracks) {
      LOG(info) << "writing " << tracks.size() << " track(s)";
    });
    auto tracksdef = BranchDefinition<TrackOutputType>{InputSpec{"inputTracks", "TPC", "LASERTRACKS", 0}, //
                                                       "TPCTracks", "track-branch-name",                  //
                                                       1,                                                 //
                                                       logger};                                           //

    workflow.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                              std::move(tracksdef))());
  }

  return workflow;
}
