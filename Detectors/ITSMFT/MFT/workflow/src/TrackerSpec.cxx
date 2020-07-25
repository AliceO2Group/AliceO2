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
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"

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

    mTracker = std::make_unique<o2::mft::Tracker>(mUseMC);
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    mTracker->setBz(field->getBz(centerMFT));
  } else {
    throw std::runtime_error(o2::utils::concat_string("Cannot retrieve GRP from the ", filename));
  }

  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mDict.readBinaryFile(dictFile);
    LOG(INFO) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
  }
}

void TrackerDPL::run(ProcessingContext& pc)
{
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "TRACKSROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  LOG(INFO) << "MFTTracker pulled " << compClusters.size() << " compressed clusters in "
            << rofsinput.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = mUseMC ? pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release() : nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    // get the array as read-only span, a snapshot of the object is sent forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(INFO) << labels->getIndexedSize() << " MC label objects , in "
              << mc2rofs.size() << " MC events";
  }

  //std::vector<o2::mft::TrackMFTExt> tracks;
  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> allTrackLabels;
  std::vector<o2::mft::TrackLTF> tracksLTF;
  auto& allTracksLTF = pc.outputs().make<std::vector<o2::mft::TrackLTF>>(Output{"MFT", "TRACKSLTF", 0, Lifetime::Timeframe});
  std::vector<o2::mft::TrackCA> tracksCA;
  auto& allTracksCA = pc.outputs().make<std::vector<o2::mft::TrackCA>>(Output{"MFT", "TRACKSCA", 0, Lifetime::Timeframe});
  auto& allTracksMFT = pc.outputs().make<std::vector<o2::mft::TrackLTF>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});

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

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
  if (continuous) {
    for (auto& rof : rofs) {
      int nclUsed = ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels);
      if (nclUsed) {
        event.setROFrameId(roFrame);
        event.initialise();
        LOG(INFO) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracksLTF.swap(event.getTracksLTF());
        tracksCA.swap(event.getTracksCA());
        LOG(INFO) << "Found tracks LTF: " << tracksLTF.size();
        LOG(INFO) << "Found tracks CA: " << tracksCA.size();
        trackLabels = mTracker->getTrackLabels(); /// FIXME: assignment ctor is not optimal.
        int first = allTracksMFT.size();
        int number = tracksLTF.size() + tracksCA.size();
        int shiftIdx = -rof.getFirstEntry();
        rof.setFirstEntry(first);
        rofs[roFrame].setFirstEntry(first);
        rof.setNEntries(number);
        copyTracks(tracksLTF, allTracksMFT, allClusIdx, shiftIdx);
        copyTracks(tracksCA, allTracksMFT, allClusIdx, shiftIdx);
        std::copy(tracksLTF.begin(), tracksLTF.end(), std::back_inserter(allTracksLTF));
        std::copy(tracksCA.begin(), tracksCA.end(), std::back_inserter(allTracksCA));
        allTrackLabels.mergeAtBack(trackLabels);
      }
      roFrame++;
    }
  }

  LOG(INFO) << "MFTTracker pushed " << allTracksMFT.size() << " tracks";
  LOG(INFO) << "MFTTracker pushed " << allTracksLTF.size() << " tracks LTF";
  LOG(INFO) << "MFTTracker pushed " << allTracksCA.size() << " tracks CA";
  if (mUseMC) {
    pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
}

DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSLTF", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSCA", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKSROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "TRACKCLSID", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "MFT", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}},
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}}}};
}

} // namespace mft
} // namespace o2
