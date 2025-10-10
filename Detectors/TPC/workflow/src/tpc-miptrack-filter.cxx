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
#include "TPCWorkflow/MIPTrackFilterSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using GID = o2::dataformats::GlobalTrackID;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"enable-writer", VariantType::Bool, false, {"selection string input specs"}},
    {"use-global-tracks", VariantType::Bool, false, {"use global matched tracks instead of TPC only"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
  };

  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  const auto useGlobal = config.options().get<bool>("use-global-tracks");
  WorkflowSpec workflow;

  const auto useMC = false;
  auto srcTracks = GID::getSourcesMask("TPC");
  const auto srcCls = GID::getSourcesMask("");
  if (useGlobal) {
    srcTracks = GID::getSourcesMask("ITS,TPC,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  }

  workflow.emplace_back(getMIPTrackFilterSpec(srcTracks));

  if (config.options().get<bool>("enable-writer")) {
    const char* processName = "tpc-mips-writer";
    const char* defaultFileName = "tpc-mips.root";
    const char* defaultTreeName = "tpcrec";

    // branch definitions for RootTreeWriter spec
    using TrackOutputType = std::vector<o2::tpc::TrackTPC>;

    // a spectator callback which will be invoked by the tree writer with the extracted object
    // we are using it for printing a log message
    auto logger = BranchDefinition<TrackOutputType>::Spectator([](TrackOutputType const& tracks) {
      LOG(info) << "writing " << tracks.size() << " track(s)";
    });
    auto tracksdef = BranchDefinition<TrackOutputType>{InputSpec{"inputTracks", "TPC", "MIPS", 0, Lifetime::Sporadic}, //
                                                       "TPCTracks", "track-branch-name",                               //
                                                       1,                                                              //
                                                       logger};                                                        //

    workflow.push_back(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                              std::move(tracksdef))());
  }

  return workflow;
}
