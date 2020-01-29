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
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"

namespace o2
{
using namespace framework;
namespace its
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void TrackerDPL::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    base::GeometryManager::loadGeometry();
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(utils::bit2Mask(TransformType::T2L, TransformType::T2GRot,
                                          TransformType::T2G));

    mTracker = std::make_unique<Tracker>(&mTrackerTraits);
    mVertexer = std::make_unique<Vertexer>(&mVertexerTraits);
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

  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  auto clusters = pc.inputs().get<gsl::span<o2::itsmft::Cluster>>("clusters");
  auto rofs = pc.inputs().get<std::vector<o2::itsmft::ROFRecord>>("ROframes"); // use vector rather than span, since used for output

  LOG(INFO) << "ITSTracker pulled " << clusters.size() << " clusters, "
            << rofs.size() << " RO frames and ";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  std::vector<itsmft::MC2ROFRecord> mc2rofs;
  if (mIsMC) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release();
    // use vector rather than span since we are using it also for the output
    mc2rofs = pc.inputs().get<std::vector<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(INFO) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
  }

  std::vector<o2::its::TrackITSExt> tracks;
  std::vector<int> allClusIdx;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  std::vector<o2::its::TrackITS> allTracks;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> allTrackLabels;

  std::vector<o2::itsmft::ROFRecord> vertROFvec;
  std::vector<Vertex> vertices;

  std::uint32_t roFrame = 0;
  ROframe event(0);

  bool continuous = mGRP->isDetContinuousReadOut("ITS");
  LOG(INFO) << "ITSTracker RO: continuous=" << continuous;

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
      int nclUsed = ioutils::loadROFrameData(rof, event, clusters, labels);
      if (nclUsed) {
        LOG(INFO) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mVertexer->clustersToVertices(event);
        auto vtxVecLoc = mVertexer->exportVertices();
        event.addPrimaryVertices(vtxVecLoc);
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracks.swap(mTracker->getTracks());
        LOG(INFO) << "Found tracks: " << tracks.size();
        int first = allTracks.size();
        int number = tracks.size();
        trackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
        int shiftIdx = -rof.getFirstEntry();
        rofs[roFrame].setFirstEntry(first);
        rofs[roFrame].setNEntries(number);
        copyTracks(tracks, allTracks, allClusIdx, shiftIdx);
        allTrackLabels.mergeAtBack(trackLabels);

        // for vertices output
        auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
        vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
        vtxROF.setNEntries(vtxVecLoc.size());
        for (const auto& vtx : vtxVecLoc) {
          vertices.push_back(vtx);
        }
      }
      roFrame++;
    }
  } else {
    ioutils::loadEventData(event, clusters, labels);
    event.addPrimaryVertex(0.f, 0.f, 0.f); //FIXME :  run an actual vertex finder !
    mTracker->clustersToTracks(event);
    tracks.swap(mTracker->getTracks());
    copyTracks(tracks, allTracks, allClusIdx);
    allTrackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
  }

  LOG(INFO) << "ITSTracker pushed " << allTracks.size() << " tracks";
  pc.outputs().snapshot(Output{"ITS", "TRACKS", 0, Lifetime::Timeframe}, allTracks);
  pc.outputs().snapshot(Output{"ITS", "TRACKCLSID", 0, Lifetime::Timeframe}, allClusIdx);
  pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
  pc.outputs().snapshot(Output{"ITS", "ITSTrackROF", 0, Lifetime::Timeframe}, rofs);
  pc.outputs().snapshot(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe}, vertices);
  pc.outputs().snapshot(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe}, vertROFvec);
  if (mIsMC) {
    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }

  mState = 2;
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusters", "ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}}}};
}

} // namespace its
} // namespace o2
