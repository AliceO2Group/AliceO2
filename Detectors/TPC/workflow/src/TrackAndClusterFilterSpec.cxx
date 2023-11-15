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
  TrackAndClusterFilterDevice(bool writeMC) { mTrackDump.writeMC = writeMC; };

  void init(o2::framework::InitContext& ic) final
  {
    mTrackDump.outputFileName = ic.options().get<std::string>("output-file");
    mTrackDump.writeTracks = !ic.options().get<bool>("dont-write-tracks");
    mTrackDump.writeGlobal = ic.options().get<bool>("write-global-cluster-info");

    // cluster write types
    mTrackDump.clusterStorageType = TrackDump::ClStorageType::InsideTrack;
    const auto clWriteType = ic.options().get<int>("clusters-write-type");
    if (clWriteType >= 0 && clWriteType <= 3) {
      mTrackDump.clusterStorageType = (TrackDump::ClStorageType)clWriteType;
    } else {
      LOGP(error, "clWriteType {} unknown, using default", clWriteType);
    }

    mTrackDump.noTrackClusterType = TrackDump::ClUnStorageType::DontStore;
    const auto clNoTrackWriteType = ic.options().get<int>("notrack-clusters-write-type");
    if (clNoTrackWriteType >= 0 && clNoTrackWriteType <= 2) {
      mTrackDump.noTrackClusterType = (TrackDump::ClUnStorageType)clNoTrackWriteType;
    } else {
      LOGP(error, "clNoTrackWriteType {} unknown, using default", clNoTrackWriteType);
    }

    // track cuts
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
    const auto mcLabel = (mTrackDump.writeMC) ? pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTruth") : gsl::span<o2::MCCompLabel>();
    const auto& clustersInputs = getWorkflowTPCInput(pc);

    std::vector<TrackTPC> filteredTracks;
    std::vector<o2::MCCompLabel> filteredMCLabels;
    for (size_t iTrack = 0; iTrack < tracks.size(); iTrack++) {
      if (mCuts.goodTrack(tracks[iTrack])) {
        filteredTracks.emplace_back(tracks[iTrack]);
        if (mTrackDump.writeMC) {
          filteredMCLabels.emplace_back(mcLabel[iTrack]);
        }
      }
    }

    LOGP(info, "Filtered {} good tracks with {} MC labels out of {} total tpc tracks", filteredTracks.size(), filteredMCLabels.size(), tracks.size());

    mTrackDump.filter(filteredTracks, clustersInputs->clusterIndex, clRefs, filteredMCLabels);
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

DataProcessorSpec getTrackAndClusterFilterSpec(const std::string dataDescriptionStr, const bool writeMC)
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
  if (writeMC) {
    inputs.emplace_back("trackTPCMCTruth", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-and-cluster-filter",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackAndClusterFilterDevice>(writeMC)},
    Options{
      {"output-file", VariantType::String, "filtered-tracks-and-clusters.root", {"output file name"}},
      {"dont-write-tracks", VariantType::Bool, false, {"don't dump filtered tracks and clusters"}},
      {"write-global-cluster-info", VariantType::Bool, false, {"write simple clusters tree"}},
      {"clusters-write-type", VariantType::Int, 0, {"how to store clusters associated to tracks: 0 - vector in track, 1 - separate branch, 2 - separate tree, 3 - spearate file"}},
      {"notrack-clusters-write-type", VariantType::Int, 0, {"clusters not associated to tracks: 0 - don't write, 1 - separate tree, 2 - spearate file"}},
      {"min-dedx", VariantType::Double, 20., {"minimum dEdx cut"}},
      {"max-dedx", VariantType::Double, 1e10, {"maximum dEdx cut"}},
      {"min-momentum", VariantType::Double, 0.2, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Double, 1e10, {"maximum momentum cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
