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

#include <vector>
#include <string>
#include "fmt/format.h"

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "DetectorsCalibration/Utils.h"
#include "TPCCalibration/TrackDump.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::tpc
{

class TrackAndClusterFilterDevice : public o2::framework::Task
{
 public:
  TrackAndClusterFilterDevice() = default;

  void init(o2::framework::InitContext& ic) final
  {
    mTrackDump.outputFileName = ic.options().get<std::string>("output-file");
    mTrackDump.writeTracks = ic.options().get<bool>("write-tracks");
    mTrackDump.writeGlobal = ic.options().get<bool>("write-global-cluster-info");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // for (auto const& ref : pc.inputs()) {
    //   const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    //   LOGP(info, "Specifier: {}/{}/{} Part {} of {}", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dh->splitPayloadIndex, dh->splitPayloadParts);
    // }
    const auto tracks = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
    const auto clRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
    const auto& clustersInputs = getWorkflowTPCInput(pc);

    std::vector<TrackTPC> filteredTracks;
    std::copy_if(tracks.begin(), tracks.end(), std::back_inserter(filteredTracks),
                 [this](const auto& track) { return isGood(track); });

    LOGP(info, "Filtered {} good tracks out of {} total tpc tracks", filteredTracks.size(), tracks.size());

    mTrackDump.filter(tracks, clustersInputs->clusterIndex, clRefs);
  }

  void endOfStream(o2::framework::EndOfStreamContext& /*ec*/) final
  {
    stop();
  }

  void stop() final
  {
    mTrackDump.finalize();
  }

 private:
  TrackDump mTrackDump;

  bool isGood(const TrackTPC& track)
  {
    if (track.getP() < 0.02) {
      return false;
    }

    if (track.getNClusters() < 60) {
      return false;
    }

    if (track.getdEdx().dEdxTotTPC < 20) {
      return false;
    }

    return true;
  }
};

DataProcessorSpec getTrackAndClusterFilterSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-track-and-cluster-filter",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackAndClusterFilterDevice>()},
    Options{
      {"output-file", VariantType::String, "filtered-track-and-clusters.root", {"output file name"}},
      {"write-tracks", VariantType::Bool, true, {"dump filtered tracks and clusters"}},
      {"write-global-cluster-info", VariantType::Bool, false, {"write simple clusters tree"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc