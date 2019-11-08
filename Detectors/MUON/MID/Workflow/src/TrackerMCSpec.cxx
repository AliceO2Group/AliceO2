// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/TrackerMCSpec.cxx
/// \brief  Data processor spec for MID MC tracker device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 September 2019

#include "MIDWorkflow/TrackerMCSpec.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/Cluster3D.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "MIDSimulation/Geometry.h"
#include "MIDTracking/Tracker.h"
#include "DetectorsBase/GeometryManager.h"
#include "MIDSimulation/MCClusterLabel.h"
#include "MIDSimulation/TrackLabeler.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{
class TrackerMCDeviceDPL
{
 public:
  void init(o2::framework::InitContext& ic)
  {
    auto geoFilename = ic.options().get<std::string>("geometry-filename");
    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry(geoFilename);
    }

    mTracker = std::make_unique<Tracker>(createTransformationFromManager(gGeoManager));

    if (!mTracker->init()) {
      LOG(ERROR) << "Initialization of MID tracker device failed";
    }
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get("mid_clusters");
    gsl::span<const Cluster2D> clusters = of::DataRefUtils::as<const Cluster2D>(msg);

    auto msgROF = pc.inputs().get("mid_clusters_rof");
    gsl::span<const ROFRecord> inROFRecords = of::DataRefUtils::as<const ROFRecord>(msgROF);

    std::unique_ptr<const o2::dataformats::MCTruthContainer<MCClusterLabel>> labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<MCClusterLabel>*>("mid_clusterlabels");

    mTracker->process(clusters, inROFRecords);
    mTrackLabeler.process(mTracker->getClusters(), mTracker->getTracks(), *labels);

    pc.outputs().snapshot(of::Output{"MID", "TRACKS", 0, of::Lifetime::Timeframe}, mTracker->getTracks());
    LOG(DEBUG) << "Sent " << mTracker->getTracks().size() << " tracks.";
    pc.outputs().snapshot(of::Output{"MID", "TRACKCLUSTERS", 0, of::Lifetime::Timeframe}, mTracker->getClusters());
    LOG(DEBUG) << "Sent " << mTracker->getClusters().size() << " track clusters.";

    pc.outputs().snapshot(of::Output{"MID", "TRACKSROF", 0, of::Lifetime::Timeframe}, mTracker->getTrackROFRecords());
    LOG(DEBUG) << "Sent " << mTracker->getTrackROFRecords().size() << " ROFs.";
    pc.outputs().snapshot(of::Output{"MID", "TRCLUSROF", 0, of::Lifetime::Timeframe}, mTracker->getClusterROFRecords());
    LOG(DEBUG) << "Sent " << mTracker->getClusterROFRecords().size() << " ROFs.";

    pc.outputs().snapshot(of::Output{"MID", "TRACKSLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTracksLabels());
    LOG(DEBUG) << "Sent " << mTrackLabeler.getTracksLabels().getIndexedSize() << " indexed tracks.";
    pc.outputs().snapshot(of::Output{"MID", "TRCLUSLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTrackClustersLabels());
    LOG(DEBUG) << "Sent " << mTrackLabeler.getTrackClustersLabels().getIndexedSize() << " indexed track clusters.";
  }

 private:
  std::unique_ptr<Tracker> mTracker{nullptr};
  TrackLabeler mTrackLabeler{};
};

framework::DataProcessorSpec getTrackerMCSpec()
{

  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_clusters", "MID", "CLUSTERS"}, of::InputSpec{"mid_clusters_rof", "MID", "CLUSTERSROF"}, of::InputSpec{"mid_clusterlabels", "MID", "CLUSTERSLABELS"}};

  std::vector<of::OutputSpec> outputSpecs{
    of::OutputSpec{"MID", "TRACKS"},
    of::OutputSpec{"MID", "TRACKCLUSTERS"},
    of::OutputSpec{"MID", "TRACKSROF"},
    of::OutputSpec{"MID", "TRCLUSROF"},
    of::OutputSpec{"MID", "TRACKSLABELS"},
    of::OutputSpec{"MID", "TRCLUSLABELS"}};

  return of::DataProcessorSpec{
    "TrackerMC",
    {inputSpecs},
    {outputSpecs},
    of::adaptFromTask<o2::mid::TrackerMCDeviceDPL>(),
    of::Options{
      {"geometry-filename", of::VariantType::String, "O2geometry.root", {"Name of the geometry file"}}}};
}
} // namespace mid
} // namespace o2
