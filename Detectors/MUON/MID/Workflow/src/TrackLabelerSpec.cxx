// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/TrackLabelerSpec.cxx
/// \brief  Data processor spec for MID TrackLabeler device
/// \author Diego Stocco <diego.stocco at cern.ch>
/// \date   11 June 2019

#include "MIDWorkflow/TrackLabelerSpec.h"

#include "Framework/DataRefUtils.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/Cluster3D.h"
#include "DataFormatsMID/Track.h"
#include "MIDSimulation/MCClusterLabel.h"
#include "MIDSimulation/TrackLabeler.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{
class TrackLabelerDeviceDPL
{
 public:
  TrackLabelerDeviceDPL(const char* inputClustersBinding, const char* inputTracksBinding, const char* inputLabelsBinding) : mInputClustersBinding(inputClustersBinding), mInputTracksBinding(inputTracksBinding), mInputLabelsBinding(inputLabelsBinding), mTrackLabeler(){};
  ~TrackLabelerDeviceDPL() = default;

  void init(of::InitContext& ic)
  {
  }

  void run(of::ProcessingContext& pc)
  {
    auto msgClusters = pc.inputs().get(mInputClustersBinding.c_str());
    gsl::span<const Cluster3D> clusters = of::DataRefUtils::as<const Cluster3D>(msgClusters);

    auto msgTracks = pc.inputs().get(mInputTracksBinding.c_str());
    gsl::span<const Track> tracks = of::DataRefUtils::as<const Track>(msgTracks);

    std::unique_ptr<const o2::dataformats::MCTruthContainer<MCClusterLabel>> labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<MCClusterLabel>*>(mInputLabelsBinding.c_str());

    mTrackLabeler.process(clusters, tracks, *labels);
    pc.outputs().snapshot(of::Output{"MID", "TRACKSLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTracksLabels());
    LOG(INFO) << "Sent " << mTrackLabeler.getTracksLabels().getIndexedSize() << " indexed tracks";
    pc.outputs().snapshot(of::Output{"MID", "TRCLUSLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTrackClustersLabels());
    LOG(INFO) << "Sent " << mTrackLabeler.getTrackClustersLabels().getIndexedSize() << " indexed track clusters";
  }

 private:
  std::string mInputClustersBinding;
  std::string mInputTracksBinding;
  std::string mInputLabelsBinding;
  TrackLabeler mTrackLabeler;
};

framework::DataProcessorSpec getTrackLabelerSpec()
{
  std::string inputClustersBinding = "mid_trackclusters_mc";
  std::string inputTracksBinding = "mid_tracks_mc";
  std::string inputLabelsBinding = "mid_clusterlabels";

  std::vector<of::InputSpec> inputSpecs{
    of::InputSpec{inputClustersBinding, "MID", "TRCLUS_MC"},
    of::InputSpec{inputTracksBinding, "MID", "TRACKS_MC"},
    of::InputSpec{inputLabelsBinding, "MID", "CLUSTERSLABELS"},
  };
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{"MID", "TRACKSLABELS"}, of::OutputSpec{"MID", "TRCLUSLABELS"}};

  return of::DataProcessorSpec{
    "TrackLabeler",
    {inputSpecs},
    {outputSpecs},
    of::adaptFromTask<o2::mid::TrackLabelerDeviceDPL>(inputClustersBinding.c_str(), inputTracksBinding.c_str(), inputLabelsBinding.c_str())};
}
} // namespace mid
} // namespace o2