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
#include <future>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{
//#define _TIMING_

void TrackerDPL::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  for (int sw = 0; sw < NStopWatches; sw++) {
    mTimer[sw].Stop();
    mTimer[sw].Reset();
  }

  // tracking configuration parameters
  auto& trackingParam = MFTTrackingParam::Instance(); // to avoid loading interpreter during the run
}

void TrackerDPL::run(ProcessingContext& pc)
{
  mTimer[SWTot].Start(false);

  updateTimeDependentParams(pc);
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto ntracks = 0;

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"MFT", "MFTTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  ROFFilter filter = [](const o2::itsmft::ROFRecord& r) { return true; };

  LOG(info) << "MFTTracker pulled " << compClusters.size() << " compressed clusters in " << rofsinput.size() << " RO frames";

  auto& trackingParam = MFTTrackingParam::Instance();
  if (trackingParam.irFramesOnly) {
    // selects only those ROFs that overlap ITS IRFrame
    LOG(info) << "MFTTracker IRFrame filter enabled: loading ITS IR Frames. ";
    auto irFrames = pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("IRFramesITS");
    filter = createIRFrameFilter(irFrames);

    if (fair::Logger::Logging(fair::Severity::debug)) {
      for (const auto& irf : irFrames) {
        LOG(debug) << "IRFrame.info = " << irf.info << " ; min = " << irf.getMin().bc << " ; max = " << irf.getMax().bc;
      }
    }
  }

  if (trackingParam.isMultCutRequested()) {
    LOG(info) << "MFTTracker multiplicity filter enabled. ROF selection: Min nClusters =  " << trackingParam.cutMultClusLow << " ; Max nClusters = " << trackingParam.cutMultClusHigh;
  }

  const dataformats::MCTruthContainer<MCCompLabel>* labels = mUseMC ? pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release() : nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    // get the array as read-only span, a snapshot of the object is sent forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(info) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
  }

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"MFT", "TRACKCLSID", 0, Lifetime::Timeframe});
  std::vector<o2::MCCompLabel> trackLabels;
  std::vector<o2::MCCompLabel> allTrackLabels;
  std::vector<o2::mft::TrackLTF> tracks;
  std::vector<o2::mft::TrackLTFL> tracksL;
  auto& allTracksMFT = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});

  std::uint32_t roFrameId = 0;
  int nROFs = rofs.size();
  auto rofsPerWorker = std::max(1, nROFs / mNThreads);
  LOG(debug) << "nROFs = " << nROFs << " rofsPerWorker = " << rofsPerWorker;

  auto loadData = [&, this](auto& trackerVec, auto& roFrameDataVec) {
    auto& tracker = trackerVec[0]; // Use first tracker to load the data: serial operation
    gsl::span<const unsigned char>::iterator pattIt = patterns.begin();

    auto iROF = 0;

    for (const auto& rof : rofs) {
      int worker = std::min(int(iROF / rofsPerWorker), mNThreads - 1);
      auto& roFrameData = roFrameDataVec[worker].emplace_back();
      int nclUsed = ioutils::loadROFrameData(rof, roFrameData, compClusters, pattIt, mDict, labels, tracker.get(), filter);
      LOG(debug) << "ROframeId: " << iROF << ", clusters loaded : " << nclUsed << " on worker " << worker;
      iROF++;
    }
  };

  auto launchTrackFinder = [](auto* tracker, auto* workerROFs) {
#ifdef _TIMING_
    long tStart = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count(), tStartROF = tStart, tEnd = tStart;
    size_t rofCNT = 0;
#endif
    for (auto& rofData : *workerROFs) {
      tracker->findTracks(rofData);
#ifdef _TIMING_
      long tEndROF = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      LOGP(info, "launchTrackFinder| tracker:{} did {}-th ROF in {} mus: {} clusters -> {} tracks", tracker->getTrackerID(), ++rofCNT, tEndROF - tStartROF, rofData.getTotalClusters(), rofData.getTracks().size());
      tStartROF = tEnd = tEndROF;
#endif
    }
#ifdef _TIMING_
    LOGP(info, "launchTrackFinder| done: tracker:{} processed {} ROFS in {} mus", tracker->getTrackerID(), workerROFs->size(), tEnd - tStart);
#endif
  };

  auto launchFitter = [](auto* tracker, auto* workerROFs) {
#ifdef _TIMING_
    long tStart = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
#endif
    for (auto& rofData : *workerROFs) {
      tracker->fitTracks(rofData);
    }
#ifdef _TIMING_
    long tEnd = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
    LOGP(info, "launchTrackFitter| done: tracker:{} fitted  {} ROFS in {} mus", tracker->getTrackerID(), workerROFs->size(), tEnd - tStart);
#endif
  };

  auto runMFTTrackFinder = [&, this](auto& trackerVec, auto& roFrameDataVec) {
    std::vector<std::future<void>> finder;
    for (int i = 0; i < mNThreads; i++) {
      auto& tracker = trackerVec[i];
      auto& workerData = roFrameDataVec[i];
      auto f = std::async(std::launch::async, launchTrackFinder, tracker.get(), &workerData);
      finder.push_back(std::move(f));
    }

    for (int i = 0; i < mNThreads; i++) {
      finder[i].wait();
    }
  };

  auto runTrackFitter = [&, this](auto& trackerVec, auto& roFrameDataVec) {
    std::vector<std::future<void>> fitter;
    for (int i = 0; i < mNThreads; i++) {
      auto& tracker = trackerVec[i];
      auto& workerData = roFrameDataVec[i];
      auto f = std::async(std::launch::async, launchFitter, tracker.get(), &workerData);
      fitter.push_back(std::move(f));
    }

    for (int i = 0; i < mNThreads; i++) {
      fitter[i].wait();
    }
  };

  // snippet to convert found tracks to final output tracks with separate cluster indices
  auto copyTracks = [](auto& new_tracks, auto& allTracks, auto& allClusIdx) {
    for (auto& trc : new_tracks) {
      trc.setExternalClusterIndexOffset(allClusIdx.size());
      int ncl = trc.getNumberOfPoints();
      for (int ic = 0; ic < ncl; ic++) {
        auto externalClusterID = trc.getExternalClusterIndex(ic);
        allClusIdx.push_back(externalClusterID);
      }
      allTracks.emplace_back(trc);
    }
  };

  if (mFieldOn) {

    std::vector<std::vector<o2::mft::ROframe<TrackLTF>>> roFrameVec(mNThreads); // One vector of ROFrames per thread
    LOG(debug) << "Reserving ROFs ";

    for (auto& rof : roFrameVec) {
      rof.reserve(rofsPerWorker);
    }
    LOG(debug) << "Loading data into ROFs.";

    mTimer[SWLoadData].Start(false);
    loadData(mTrackerVec, roFrameVec);
    mTimer[SWLoadData].Stop();

    LOG(debug) << "Running MFT Track finder.";

    mTimer[SWFindMFTTracks].Start(false);
    runMFTTrackFinder(mTrackerVec, roFrameVec);
    mTimer[SWFindMFTTracks].Stop();

    LOG(debug) << "Runnig track fitter.";

    mTimer[SWFitTracks].Start(false);
    runTrackFitter(mTrackerVec, roFrameVec);
    mTimer[SWFitTracks].Stop();

    if (mUseMC) {
      LOG(debug) << "Computing MC Labels.";

      mTimer[SWComputeLabels].Start(false);
      auto& tracker = mTrackerVec[0];

      for (int i = 0; i < mNThreads; i++) {
        for (auto& rofData : roFrameVec[i]) {
          tracker->computeTracksMClabels(rofData.getTracks());
          trackLabels.swap(tracker->getTrackLabels());
          std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
          trackLabels.clear();
        }
      }
      mTimer[SWComputeLabels].Stop();
    }

    auto rof = rofs.begin();

    for (int i = 0; i < mNThreads; i++) {
      for (auto& rofData : roFrameVec[i]) {
        int ntracksROF = 0, firstROFTrackEntry = allTracksMFT.size();
        tracks.swap(rofData.getTracks());
        ntracksROF = tracks.size();
        copyTracks(tracks, allTracksMFT, allClusIdx);

        rof->setFirstEntry(firstROFTrackEntry);
        rof->setNEntries(ntracksROF);
        *rof++;
        roFrameId++;
      }
    }

  } else {
    LOG(debug) << "Field is off! ";
    std::vector<std::vector<o2::mft::ROframe<TrackLTFL>>> roFrameVec(mNThreads); // One vector of ROFrames per thread
    LOG(debug) << "Reserving ROFs ";

    for (auto& rof : roFrameVec) {
      rof.reserve(rofsPerWorker);
    }
    LOG(debug) << "Loading data into ROFs.";

    mTimer[SWLoadData].Start(false);
    loadData(mTrackerLVec, roFrameVec);
    mTimer[SWLoadData].Stop();

    LOG(debug) << "Running MFT Track finder.";

    mTimer[SWFindMFTTracks].Start(false);
    runMFTTrackFinder(mTrackerLVec, roFrameVec);
    mTimer[SWFindMFTTracks].Stop();

    LOG(debug) << "Runnig track fitter.";

    mTimer[SWFitTracks].Start(false);
    runTrackFitter(mTrackerLVec, roFrameVec);
    mTimer[SWFitTracks].Stop();

    if (mUseMC) {
      LOG(debug) << "Computing MC Labels.";

      mTimer[SWComputeLabels].Start(false);
      auto& tracker = mTrackerLVec[0];

      for (int i = 0; i < mNThreads; i++) {
        for (auto& rofData : roFrameVec[i]) {
          tracker->computeTracksMClabels(rofData.getTracks());
          trackLabels.swap(tracker->getTrackLabels());
          std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
          trackLabels.clear();
        }
      }
      mTimer[SWComputeLabels].Stop();
    }

    auto rof = rofs.begin();

    for (int i = 0; i < mNThreads; i++) {
      for (auto& rofData : roFrameVec[i]) {
        int ntracksROF = 0, firstROFTrackEntry = allTracksMFT.size();
        tracksL.swap(rofData.getTracks());
        ntracksROF = tracksL.size();
        copyTracks(tracksL, allTracksMFT, allClusIdx);
        rof->setFirstEntry(firstROFTrackEntry);
        rof->setNEntries(ntracksROF);
        *rof++;
        roFrameId++;
      }
    }
  }

  LOG(info) << "MFTTracker pushed " << allTracksMFT.size() << " tracks";

  if (mUseMC) {
    pc.outputs().snapshot(Output{"MFT", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }

  mTimer[SWTot].Stop();
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  for (int i = 0; i < NStopWatches; i++) {
    LOGF(info, "Timing %18s: Cpu: %.3e s; Real: %.3e s in %d slots", TimerName[i], mTimer[i].CpuTime(), mTimer[i].RealTime(), mTimer[i].Counter() - 1);
  }
}
///_______________________________________
void TrackerDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    bool continuous = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::MFT);
    LOG(info) << "MFTTracker RO: continuous =" << continuous;
    mMFTTriggered = !continuous;
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    if (mMFTTriggered) {
      setMFTROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // MFT ROFrame duration in \mus
    } else {
      setMFTROFrameLengthInBC(alpParams.roFrameLengthInBC); // MFT ROFrame duration in BC
    }

    o2::mft::GeometryTGeo* geom = o2::mft::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));
    // tracking configuration parameters
    auto& trackingParam = MFTTrackingParam::Instance();
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    auto Bz = field->getBz(centerMFT);
    if (Bz == 0 || trackingParam.forceZeroField) {
      LOG(info) << "Starting MFT Linear tracker: Field is off!";
      LOG(info) << "  MFT tracker running with " << mNThreads << " threads";
      mFieldOn = false;
      for (auto i = 0; i < mNThreads; i++) {
        auto& tracker = mTrackerLVec.emplace_back(std::make_unique<o2::mft::Tracker<TrackLTFL>>(mUseMC));
        tracker->setBz(0);
        tracker->configure(trackingParam, i);
      }
    } else {
      LOG(info) << "Starting MFT tracker: Field is on! Bz = " << Bz;
      LOG(info) << "  MFT tracker running with " << mNThreads << " threads";
      mFieldOn = true;
      for (auto i = 0; i < mNThreads; i++) {
        auto& tracker = mTrackerVec.emplace_back(std::make_unique<o2::mft::Tracker<TrackLTF>>(mUseMC));
        tracker->setBz(Bz);
        tracker->configure(trackingParam, i);
      }
    }
  }
}

///_______________________________________
void TrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mDict = (const o2::itsmft::TopologyDictionary*)obj;
  }
}

///_______________________________________
void TrackerDPL::setMFTROFrameLengthMUS(float fums)
{
  mMFTROFrameLengthMUS = fums;
  mMFTROFrameLengthMUSInv = 1. / mMFTROFrameLengthMUS;
  mMFTROFrameLengthInBC = std::max(1, int(mMFTROFrameLengthMUS / (o2::constants::lhc::LHCBunchSpacingNS * 1e-3)));
}

///_______________________________________
void TrackerDPL::setMFTROFrameLengthInBC(int nbc)
{
  mMFTROFrameLengthInBC = nbc;
  mMFTROFrameLengthMUS = nbc * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
  mMFTROFrameLengthMUSInv = 1. / mMFTROFrameLengthMUS;
}

///_______________________________________
DataProcessorSpec getTrackerSpec(bool useMC, int nThreads)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "MFT", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("MFT/Calib/ClusterDictionary"));

  auto& trackingParam = MFTTrackingParam::Instance();
  if (trackingParam.irFramesOnly) {
    inputs.emplace_back("IRFramesITS", "ITS", "IRFRAMES", 0, Lifetime::Timeframe);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
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
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest, useMC, nThreads)},
    Options{}};
}

} // namespace mft
} // namespace o2
