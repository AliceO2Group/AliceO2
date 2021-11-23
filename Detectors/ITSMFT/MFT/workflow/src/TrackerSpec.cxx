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
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void TrackerDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    Bool_t continuous = mGRP->isDetContinuousReadOut("MFT");
    LOG(info) << "MFTTracker RO: continuous=" << continuous;

    o2::base::GeometryManager::loadGeometry();
    o2::mft::GeometryTGeo* geom = o2::mft::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));

    // tracking configuration parameters
    auto& trackingParam = MFTTrackingParam::Instance();
    // create the tracker: set the B-field, the configuration and initialize

    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    auto Bz = field->getBz(centerMFT);
    if (Bz == 0 || trackingParam.forceZeroField) {
      LOG(info) << "Starting MFT Linear tracker: Field is off!";
      mFieldOn = false;
      mTrackerL = std::make_unique<o2::mft::Tracker<TrackLTFL>>(mUseMC);
      mTrackerL->initConfig(trackingParam, true);
      mTrackerL->initialize(trackingParam.FullClusterScan);
    } else {
      LOG(info) << "Starting MFT tracker: Field is on!";
      mFieldOn = true;
      mTracker = std::make_unique<o2::mft::Tracker<TrackLTF>>(mUseMC);
      mTracker->setBz(Bz);
      mTracker->initConfig(trackingParam, true);
      mTracker->initialize(trackingParam.FullClusterScan);
    }
  } else {
    throw std::runtime_error(o2::utils::Str::concat_string("Cannot retrieve GRP from the ", filename));
  }

  std::string dictPath = o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance().dictFilePath;
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::MFT, dictPath);
  if (o2::utils::Str::pathExists(dictFile)) {
    mDict.readFromFile(dictFile);
    LOG(info) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(info) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
  }
}

void TrackerDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto ntracks = 0;

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "MFTTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  LOG(info) << "MFTTracker pulled " << compClusters.size() << " compressed clusters in "
            << rofsinput.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = mUseMC ? pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release() : nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    // get the array as read-only span, a snapshot of the object is sent forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(info) << labels->getIndexedSize() << " MC label objects , in "
              << mc2rofs.size() << " MC events";
  }

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  std::vector<o2::MCCompLabel> trackLabels;
  std::vector<o2::MCCompLabel> allTrackLabels;
  std::vector<o2::mft::TrackLTF> tracks;
  std::vector<o2::mft::TrackLTFL> tracksL;
  auto& allTracksMFT = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});

  std::uint32_t roFrame = 0;

  if (mFieldOn) {
    o2::mft::ROframe<TrackLTF> event(0);

    // tracking configuration parameters
    auto& trackingParam = MFTTrackingParam::Instance();

    // snippet to convert found tracks to final output tracks with separate cluster indices
    auto copyTracks = [&event](auto& tracks, auto& allTracks, auto& allClusIdx) {
      for (auto& trc : tracks) {
        trc.setExternalClusterIndexOffset(allClusIdx.size());
        int ncl = trc.getNumberOfPoints();
        for (int ic = 0; ic < ncl; ic++) {
          auto externalClusterID = trc.getExternalClusterIndex(ic);
          allClusIdx.push_back(externalClusterID);
        }
        allTracks.emplace_back(trc);
      }
    };

    gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
    for (auto& rof : rofs) {
      int nclUsed = ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels, mTracker.get());
      if (nclUsed) {
        event.setROFrameId(roFrame);
        event.initialize(trackingParam.FullClusterScan);
        LOG(debug) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracks.swap(event.getTracks());
        ntracks += tracks.size();

        if (mUseMC) {
          mTracker->computeTracksMClabels(tracks);
          trackLabels.swap(mTracker->getTrackLabels());
          std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
          trackLabels.clear();
        }

        LOG(debug) << "Found MFT tracks: " << tracks.size();
        int first = allTracksMFT.size();
        int number = tracks.size();
        rof.setFirstEntry(first);
        rof.setNEntries(number);
        copyTracks(tracks, allTracksMFT, allClusIdx);
      }
      roFrame++;
    }
  } else { // Use Linear Tracker for Field off
    o2::mft::ROframe<TrackLTFL> event(0);

    // tracking configuration parameters
    auto& trackingParam = MFTTrackingParam::Instance();

    // snippet to convert found tracks to final output tracks with separate cluster indices
    auto copyTracks = [&event](auto& tracks, auto& allTracks, auto& allClusIdx) {
      for (auto& trc : tracks) {
        trc.setExternalClusterIndexOffset(allClusIdx.size());
        int ncl = trc.getNumberOfPoints();
        for (int ic = 0; ic < ncl; ic++) {
          auto externalClusterID = trc.getExternalClusterIndex(ic);
          allClusIdx.push_back(externalClusterID);
        }
        allTracks.emplace_back(trc);
      }
    };

    gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
    for (auto& rof : rofs) {
      int nclUsed = ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels, mTrackerL.get());
      if (nclUsed) {
        event.setROFrameId(roFrame);
        event.initialize(trackingParam.FullClusterScan);
        LOG(debug) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;
        mTrackerL->setROFrame(roFrame);
        mTrackerL->clustersToTracks(event);
        tracksL.swap(event.getTracks());
        ntracks += tracksL.size();

        if (mUseMC) {
          mTrackerL->computeTracksMClabels(tracksL);
          trackLabels.swap(mTrackerL->getTrackLabels());
          std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
          trackLabels.clear();
        }

        LOG(debug) << "Found MFT tracks: " << tracks.size();
        int first = allTracksMFT.size();
        int number = tracksL.size();
        rof.setFirstEntry(first);
        rof.setNEntries(number);
        copyTracks(tracksL, allTracksMFT, allClusIdx);
      }
      roFrame++;
    }
  }
  LOG(info) << "MFTTracker pushed " << allTracksMFT.size() << " tracks";

  if (mUseMC) {
    pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
  mTimer.Stop();
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "MFT Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "MFTTrackROF", 0, Lifetime::Timeframe);
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
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}}}};
}

} // namespace mft
} // namespace o2
