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

#include "MFTWorkflow/TrackerSpec.h"

#include "MFTTracking/ROframe.h"
#include "MFTTracking/IOUtils.h"
#include "MFTTracking/Tracker.h"
#include "MFTTracking/TrackCA.h"
#include "MFTBase/GeometryTGeo.h"

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void TrackerDPL::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    o2::base::GeometryManager::loadGeometry();
    o2::mft::GeometryTGeo* geom = o2::mft::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                              o2::TransformType::T2G));

    mTracker = std::make_unique<o2::mft::Tracker>();
    double origD[3] = {0., 0., 0.};
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

  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto clusters = pc.inputs().get<const std::vector<o2::itsmft::Cluster>>("clusters");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "MFTTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  LOG(INFO) << "MFTTracker pulled " << clusters.size() << " clusters in "
            << rofs.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = mUseMC ? pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release() : nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    // get the array as read-only span, a snapshot of the object is sent forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(INFO) << labels->getIndexedSize() << " MC label objects , in "
              << mc2rofs.size() << " MC events";
  }

  std::vector<o2::mft::TrackMFTExt> tracks;
  std::vector<int> allClusIdx;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  auto& allTracks = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> allTrackLabels;
  std::vector<o2::mft::TrackLTF> tracksLTF;
  auto& allTracksLTF = pc.outputs().make<std::vector<o2::mft::TrackLTF>>(Output{"MFT", "TRACKSLTF", 0, Lifetime::Timeframe});
  std::vector<o2::mft::TrackCA> tracksCA;
  auto& allTracksCA = pc.outputs().make<std::vector<o2::mft::TrackCA>>(Output{"MFT", "TRACKSCA", 0, Lifetime::Timeframe});

  std::uint32_t roFrame = 0;
  o2::mft::ROframe event(0);

  Bool_t continuous = mGRP->isDetContinuousReadOut("MFT");
  LOG(INFO) << "MFTTracker RO: continuous=" << continuous;

  // snippet to convert found tracks to final output tracks with separate cluster indices
  auto copyTracks = [](auto& tracks, auto& allTracks, auto& allClusIdx, int offset = 0) {
    for (auto& trc : tracks) {
      trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters();
      for (int ic = 0; ic < ncl; ic++) {
        allClusIdx.push_back(trc.getClusterIndex(ic) + offset);
      }
      allTracks.emplace_back(trc);
    }
  };

  if (continuous) {
    for (const auto& rof : rofs) {
      Int_t nclUsed = o2::mft::ioutils::loadROFrameData(rof, event, gsl::span(clusters.data(), clusters.size()), labels);
      if (nclUsed) {
        event.setROFrameId(roFrame);
        event.initialise();
        LOG(INFO) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracks.swap(mTracker->getTracks());
        tracksLTF.swap(event.getTracksLTF());
        tracksCA.swap(event.getTracksCA());
        LOG(INFO) << "Found tracks: " << tracks.size();
        LOG(INFO) << "Found tracks LTF: " << tracksLTF.size();
        LOG(INFO) << "Found tracks CA: " << tracksCA.size();
        trackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
        int first = allTracks.size();
        int number = tracks.size();
        int shiftIdx = -rof.getFirstEntry();
        rofs[roFrame].setFirstEntry(first);
        rofs[roFrame].setNEntries(number);
        std::copy(tracksLTF.begin(), tracksLTF.end(), std::back_inserter(allTracksLTF));
        std::copy(tracksCA.begin(), tracksCA.end(), std::back_inserter(allTracksCA));
        copyTracks(tracks, allTracks, allClusIdx, shiftIdx);
        allTrackLabels.mergeAtBack(trackLabels);
      }
      roFrame++;
    }
  } else {
    /*
    o2::mft::IOUtils::loadEventData(event, &clusters, labels.get());
    mTracker->clustersToTracks(event);
    allTracks.swap(mTracker->getTracks());
    allTrackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
    */
  }

  LOG(INFO) << "MFTTracker pushed " << allTracks.size() << " tracks";
  LOG(INFO) << "MFTTracker pushed " << allTracksLTF.size() << " tracks LTF";
  LOG(INFO) << "MFTTracker pushed " << allTracksCA.size() << " tracks CA";
  if (mUseMC) {
    pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"MFT", "MFTTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusters", "MFT", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "MFTClusterROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSLTF", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSCA", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "MFTTrackROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "MFT", "MFTClusterMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "MFTTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}},
    }};
}

} // namespace mft
} // namespace o2
