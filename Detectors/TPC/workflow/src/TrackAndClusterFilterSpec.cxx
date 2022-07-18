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
#include "DataFormatsTPC/TrackCuts.h"
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
    const double mindEdx = ic.options().get<double>("min-dedx");
    const double maxdEdx = ic.options().get<double>("max-dedx");
    const double minP = ic.options().get<double>("min-momentum");
    const double maxP = ic.options().get<double>("max-momentum");
    const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));

    mCuts.setPMin(minP);
    mCuts.setPMax(maxP);
    mCuts.setNClusMin(minClusters);
    mCuts.setdEdxMin(mindEdx);
    mCuts.setdEdxMax(maxdEdx);
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
                 [this](const auto& track) { return mCuts.goodTrack(track); });

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
  TrackCuts mCuts{};
};

DataProcessorSpec getTrackAndClusterFilterSpec(const std::string dataDescriptionStr)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  o2::header::DataDescription dataDescription;
  if (dataDescriptionStr.size() > dataDescription.size + 1) {
    LOGP(fatal, "Size of {} is larger than {}", dataDescriptionStr, dataDescription.size);
  }
  for (size_t i = 0; i < dataDescriptionStr.size(); ++i) {
    dataDescription.str[i] = dataDescriptionStr[i];
  }
  inputs.emplace_back("trackTPC", "TPC", dataDescription, 0, Lifetime::Timeframe);
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
      {"min-dedx", VariantType::Double, 20., {"minimum dEdx cut"}},
      {"max-dedx", VariantType::Double, 1e10, {"maximum dEdx cut"}},
      {"min-momentum", VariantType::Double, 0.2, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Double, 1e10, {"maximum momentum cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
