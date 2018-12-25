// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackerSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/Task.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"

#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

class TrackerDPL : public Task
{
 public:
  TrackerDPL() = default;
  ~TrackerDPL() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  o2::ITS::TrackerTraitsCPU mTraits;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<o2::ITS::Tracker> mTracker = nullptr;
};

void TrackerDPL::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::Base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    o2::Base::GeometryManager::loadGeometry();
    o2::ITS::GeometryTGeo* geom = o2::ITS::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                              o2::TransformType::T2G));

    mTracker = std::make_unique<o2::ITS::Tracker>(&mTraits);
    double origD[3] = { 0., 0., 0. };
    mTracker->setBz(field->getBz(origD));
  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filename.c_str() << " file !";
    mState = 0;
  }
  mState = 1;
}

void TrackerDPL::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  auto compClusters = pc.inputs().get<const std::vector<o2::ITSMFT::CompClusterExt>>("compClusters");
  auto clusters = pc.inputs().get<const std::vector<o2::ITSMFT::Cluster>>("clusters");
  auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
  auto rofs = pc.inputs().get<const std::vector<o2::ITSMFT::ROFRecord>>("ROframes");

  LOG(INFO) << "ITSTracker pulled " << clusters.size() << " clusters, "
            << labels->getIndexedSize() << " MC label objects , in "
            << rofs.size() << " RO frames";

  std::vector<o2::ITS::TrackITS> tracks;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  std::vector<o2::ITS::TrackITS> allTracks;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> allTrackLabels;

  std::uint32_t roFrame = 0;
  o2::ITS::ROframe event(0);

  bool continuous = mGRP->isDetContinuousReadOut("ITS");
  LOG(INFO) << "ITSTracker RO: continuous=" << continuous;

  if (continuous) {
    int nclLeft = clusters.size();
    while (nclLeft > 0) {
      int nclUsed = o2::ITS::IOUtils::loadROFrameData(roFrame, event, &clusters, labels.get());
      if (nclUsed) {
        LOG(INFO) << "ROframe: " << roFrame << ", clusters left: " << nclLeft;
        event.addPrimaryVertex(0.f, 0.f, 0.f); //FIXME :  run an actual vertex finder !
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracks.swap(mTracker->getTracks());
        LOG(INFO) << "Found tracks: " << tracks.size();
        trackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
        std::copy(tracks.begin(), tracks.end(), std::back_inserter(allTracks));
        allTrackLabels.mergeAtBack(trackLabels);
        nclLeft -= nclUsed;
      }
      roFrame++;
    }
  } else {
    o2::ITS::IOUtils::loadEventData(event, &clusters, labels.get());
    event.addPrimaryVertex(0.f, 0.f, 0.f); //FIXME :  run an actual vertex finder !
    mTracker->clustersToTracks(event);
    allTracks.swap(mTracker->getTracks());
    allTrackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
  }

  LOG(INFO) << "ITSTracker pushed " << allTracks.size() << " tracks";
  pc.outputs().snapshot(Output{ "ITS", "TRACKS", 0, Lifetime::Timeframe }, allTracks);
  pc.outputs().snapshot(Output{ "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe }, allTrackLabels);

  mState = 2;
  //pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getTrackerSpec()
{
  return DataProcessorSpec{
    "its-tracker",
    Inputs{
      InputSpec{ "compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "clusters", "ITS", "CLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe },
      InputSpec{ "ROframes", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe } },
    Outputs{
      OutputSpec{ "ITS", "TRACKS", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ adaptFromTask<TrackerDPL>() },
    Options{
      { "grp-file", VariantType::String, "o2sim_grp.root", { "Name of the output file" } },
    }
  };
}

} // namespace ITS
} // namespace o2
