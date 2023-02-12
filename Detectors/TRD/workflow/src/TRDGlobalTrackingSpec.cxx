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

/// @file   TRDGlobalTrackingSpec.cxx

#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "TRDBase/Geometry.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TPCBase/ParameterElectronics.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "GPUWorkflowHelper/GPUWorkflowHelper.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "CommonConstants/GeomConstants.h"
#include "ITStracking/IOUtils.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsFT0/RecPoints.h"
#include "ITSReconstruction/RecoGeomHelper.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"

// GPU header
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUSettings.h"
#include "GPUDataTypes.h"
#include "GPUTRDDef.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDInterfaces.h"
#include "GPUTRDGeometry.h"

#include <regex>
#include <algorithm>
#include <numeric>

using namespace o2::framework;
using namespace o2::gpu;
using namespace o2::globaltracking;
using namespace o2::trd::constants;

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

void TRDGlobalTracking::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mTimer.Stop();
  mTimer.Reset();
}

void TRDGlobalTracking::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  o2::tpc::CorrectionMapsLoader::extractCCDBInputs(pc);
  // pc.inputs().get<TopologyDictionary*>("cldict"); // called by the RecoContainer to trigger finaliseCCDB
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // init-once stuff

    auto geo = Geometry::instance();
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));
    geo->createPadPlaneArray();
    geo->createClusterMatrixArray();
    mFlatGeo = std::make_unique<GeometryFlat>(*geo);

    GPURecoStepConfiguration cfgRecoStep;
    cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
    cfgRecoStep.inputs.clear();
    cfgRecoStep.outputs.clear();
    mRec = GPUReconstruction::CreateInstance("CPU", true);
    mRec->SetSettings(o2::base::Propagator::Instance()->getNominalBz(), &cfgRecoStep);
    mRec->GetNonConstParam().rec.trd.useExternalO2DefaultPropagator = true;
    mChainTracking = mRec->AddChain<GPUChainTracking>();

    mTracker = new GPUTRDTracker();
    mTracker->SetNCandidates(mRec->GetProcessingSettings().trdNCandidates); // must be set before initialization
    if (mStrict && mRec->GetProcessingSettings().trdNCandidates == 1) {
      LOG(error) << "Strict matching mode requested, but tracks with another close hypothesis will not be rejected. Please set trdNCandidates to at least 3.";
    }
    mTracker->SetProcessPerTimeFrame(true);
    mTracker->SetGenerateSpacePoints(false); // set to true to force space point calculation by the TRD tracker itself

    mRec->RegisterGPUProcessor(mTracker, false);
    mChainTracking->SetTRDGeometry(std::move(mFlatGeo));
    if (mRec->Init()) {
      LOG(fatal) << "GPUReconstruction could not be initialized";
    }

    mRecoParam.setBfield(o2::base::Propagator::Instance()->getNominalBz());

    mTracker->PrintSettings();
    LOG(info) << "Strict matching mode is " << ((mStrict) ? "ON" : "OFF");
    LOGF(info, "The search road in time for ITS-TPC tracks is set to %.1f sigma and %.2f us are added to it on top",
         mRec->GetParam().rec.trd.nSigmaTerrITSTPC, mRec->GetParam().rec.trd.addTimeRoadITSTPC);

    /// Get the PID model if requested
    if (mWithPID) {
      mBase = getTRDPIDBase(mPolicy);
      mBase->init(pc);
    }
  }

  bool updateCalib = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateCalib = true;
  }

  if (mTPCVDriftHelper.isUpdated()) {
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    mTPCTBinMUS = elParam.ZbinWidth;
    mTPCTBinMUSInv = 1. / mTPCTBinMUS;
    auto& vd = mTPCVDriftHelper.getVDriftObject();
    mTPCVdrift = vd.getVDrift();
    mTPCTDriftOffset = vd.getTimeOffset();
    LOGP(info, "Updating TPC VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper.getSourceName());
    mTracker->SetTPCVdrift(mTPCVdrift);
    mTracker->SetTPCTDriftOffset(mTPCTDriftOffset);
    mTPCVDriftHelper.acknowledgeUpdate();
    updateCalib = true;
  }
  if (updateCalib) {
    auto& vd = mTPCVDriftHelper.getVDriftObject();
    mTPCCorrMapsLoader.updateVDrift(vd.corrFact, vd.refVDrift, vd.getTimeOffset());
  }
}

void TRDGlobalTracking::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

void TRDGlobalTracking::fillMCTruthInfo(const TrackTRD& trk, o2::MCCompLabel lblSeed, std::vector<o2::MCCompLabel>& lblContainerTrd, std::vector<o2::MCCompLabel>& lblContainerMatch, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkltLabels) const
{
  // Check MC labels of the TRD tracklets attached to the track seed.
  // Set TRD track label to the most frequent tracklet label.
  // Fake flag is set if either not all TRD tracklets have most frequent label
  // or if the seeding label is different from the most frequent TRD label.
  // In case multiple tracklet labels occur most often we choose the one which matches the label of the seed, or,
  // if that is not the case one of the most frequent labels is chosen arbitrarily
  LOG(debug) << "Checking seed with label: " << lblSeed;
  std::unordered_map<o2::MCCompLabel, unsigned int> labelCounter;
  int maxOccurences = 0;
  for (int iLy = 0; iLy < constants::NLAYER; ++iLy) {
    auto trkltIndex = trk.getTrackletIndex(iLy);
    if (trkltIndex == -1) {
      // no tracklet in this layer
      continue;
    }
    const auto& lblsTrklt = trkltLabels->getLabels(trkltIndex);
    for (const auto lblTrklt : lblsTrklt) {
      int nOcc = ++labelCounter[lblTrklt];
      if (nOcc > maxOccurences) {
        maxOccurences = nOcc;
      }
    }
  }
  o2::MCCompLabel mostFrequentLabel;
  for (const auto& [lbl, count] : labelCounter) {
    LOG(debug) << "Label " << lbl << " occured " << count << " times.";
    if (count == maxOccurences) {
      if (lblSeed == lbl) {
        // most frequent label matches seed label
        mostFrequentLabel = lbl;
        mostFrequentLabel.setFakeFlag(maxOccurences != trk.getNtracklets());
        lblContainerTrd.push_back(mostFrequentLabel);
        lblContainerMatch.push_back(lblSeed); // is not fake by definition, since the seed label matches the TRD track label
        return;
      } else {
        // maybe multiple labels occur with the same frequency and the next one might match the seed?
        mostFrequentLabel = lbl;
      }
    }
  }
  mostFrequentLabel.setFakeFlag(maxOccurences != trk.getNtracklets());
  lblContainerTrd.push_back(mostFrequentLabel);
  lblSeed.setFakeFlag(lblSeed != mostFrequentLabel);
  lblContainerMatch.push_back(lblSeed);
}

void TRDGlobalTracking::fillTrackTriggerRecord(const std::vector<TrackTRD>& tracks, std::vector<TrackTriggerRecord>& trigRec, const gsl::span<const o2::trd::TriggerRecord>& trackletTrigRec) const
{
  // after the tracking is done we assemble here a TrackTriggerRecord similar to the TriggerRecord
  // which for each TRD trigger stores the index of the first found track and the total number of tracks

  int nTracksCurr = 0;
  int iTrackFirst = 0;
  int prevCollisionID = -1;

  for (size_t iTrk = 0; iTrk < tracks.size(); ++iTrk) {
    const auto& trk = tracks[iTrk];
    auto collisionID = trk.getCollisionId();
    if (iTrk == 0) {
      prevCollisionID = collisionID;
    }
    if (collisionID != prevCollisionID) {
      // we have a track from a new trigger within the same TF
      trigRec.emplace_back(trackletTrigRec[prevCollisionID].getBCData(), iTrackFirst, nTracksCurr);
      iTrackFirst += nTracksCurr;
      prevCollisionID = collisionID;
      nTracksCurr = 0;
    }
    ++nTracksCurr;
  }
  if (nTracksCurr > 0) {
    // this is the last trigger record for this TF, we can take the collision ID from the last track
    trigRec.emplace_back(trackletTrigRec[tracks.back().getCollisionId()].getBCData(), iTrackFirst, nTracksCurr);
  }
}

void TRDGlobalTracking::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  o2::globaltracking::RecoContainer inputTracks;
  inputTracks.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc);
  mChainTracking->ClearIOPointers();

  mTPCClusterIdxStruct = &inputTracks.inputsTPCclusters->clusterIndex;
  mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, &mTPCCorrMapsLoader, o2::base::Propagator::Instance()->getNominalBz(), inputTracks.getTPCTracksClusterRefs().data(), inputTracks.clusterShMapTPC.data(), nullptr, o2::base::Propagator::Instance());
  auto tmpInputContainer = getRecoInputContainer(pc, &mChainTracking->mIOPtrs, &inputTracks, mUseMC);
  auto tmpContainer = GPUWorkflowHelper::fillIOPtr(mChainTracking->mIOPtrs, inputTracks, mUseMC, nullptr, GTrackID::getSourcesMask("TRD"), mTrkMask, GTrackID::mask_t{GTrackID::MASK_NONE});
  mTrackletsRaw = inputTracks.getTRDTracklets();
  mTrackletsCalib = inputTracks.getTRDCalibratedTracklets();
  mTPCTracksArray = inputTracks.getTPCTracks();
  if (GTrackID::includesDet(GTrackID::DetID::ITS, mTrkMask)) {
    // load ITS tracks and clusters needed for the refit
    mITSTracksArray = inputTracks.getITSTracks();
    mITSTrackClusIdx = inputTracks.getITSTracksClusterRefs();
    mITSABRefsArray = inputTracks.getITSABRefs();
    mITSABTrackClusIdx = inputTracks.getITSABClusterRefs();
    const auto clusITS = inputTracks.getITSClusters();
    const auto patterns = inputTracks.getITSClustersPatterns();
    auto pattIt = patterns.begin();
    mITSClustersArray.clear();
    mITSClustersArray.reserve(clusITS.size());
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, mITSDict);
  }

  LOGF(info, "There are %i tracklets in total from %i trigger records", mChainTracking->mIOPtrs.nTRDTracklets, mChainTracking->mIOPtrs.nTRDTriggerRecords);
  LOGF(info, "As input seeds are available: %i ITS-TPC matched tracks and %i TPC tracks", mChainTracking->mIOPtrs.nTracksTPCITSO2, mChainTracking->mIOPtrs.nOutputTracksTPCO2);

  std::vector<o2::MCCompLabel> matchLabelsITSTPC;
  std::vector<o2::MCCompLabel> trdLabelsITSTPC;
  std::vector<o2::MCCompLabel> matchLabelsTPC;
  std::vector<o2::MCCompLabel> trdLabelsTPC;
  gsl::span<const o2::MCCompLabel> tpcTrackLabels;
  gsl::span<const o2::MCCompLabel> itstpcTrackLabels;
  if (mUseMC) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, mTrkMask)) {
      itstpcTrackLabels = inputTracks.getTPCITSTracksMCLabels();
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, mTrkMask)) {
      tpcTrackLabels = inputTracks.getTPCTracksMCLabels();
    }
  }
  mTracker->Reset();
  mRec->PrepareEvent();
  mRec->SetupGPUProcessor(mTracker, true);

  // check trigger record filter setting
  bool foundFilteredTrigger = false;
  for (unsigned int iTrig = 0; iTrig < mChainTracking->mIOPtrs.nTRDTriggerRecords; ++iTrig) {
    if (mChainTracking->mIOPtrs.trdTrigRecMask[iTrig] == 0) {
      foundFilteredTrigger = true;
    }
    LOGF(debug, "TRD trigger %u added with time %f", iTrig, mChainTracking->mIOPtrs.trdTriggerTimes[iTrig]);
  }
  if (!foundFilteredTrigger && mTrigRecFilter) {
    static bool warningSent = false;
    if (!warningSent) {
      LOG(warning) << "Trigger filtering requested, but no TRD trigger is actually masked. Can be that none needed to be masked or that the setting was not active for the tracklet transformer";
      warningSent = true;
    }
  } else if (foundFilteredTrigger && !mTrigRecFilter) {
    LOG(error) << "Trigger filtering is not requested, but masked TRD triggers are found. Rerun tracklet transformer without trigger filtering";
  }

  // load input tracks
  const auto& trackTune = o2::globaltracking::TrackTuneParams::Instance();
  LOG(debug) << "Start loading input seeds into TRD tracker";
  int nTracksLoadedITSTPC = 0;
  int nTracksLoadedTPC = 0;
  // load ITS-TPC matched tracks
  for (unsigned int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nTracksTPCITSO2; ++iTrk) {
    const auto& trkITSTPC = mChainTracking->mIOPtrs.tracksTPCITSO2[iTrk];
    GPUTRDTracker::HelperTrackAttributes trkAttribs;
    trkAttribs.mTime = trkITSTPC.getTimeMUS().getTimeStamp();
    trkAttribs.mTimeAddMax = trkITSTPC.getTimeMUS().getTimeStampError() * mRec->GetParam().rec.trd.nSigmaTerrITSTPC + mRec->GetParam().rec.trd.addTimeRoadITSTPC;
    trkAttribs.mTimeSubMax = trkITSTPC.getTimeMUS().getTimeStampError() * mRec->GetParam().rec.trd.nSigmaTerrITSTPC + mRec->GetParam().rec.trd.addTimeRoadITSTPC;
    GPUTRDTrack trkLoad(trkITSTPC);
    if (trackTune.tpcCovOuterType != o2::globaltracking::TrackTuneParams::AddCovType::Disable) {
      trkLoad.updateCov(trackTune.tpcCovOuter, trackTune.tpcCovOuterType == o2::globaltracking::TrackTuneParams::AddCovType::WithCorrelations);
    }
    auto trackGID = GTrackID(iTrk, GTrackID::ITSTPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw(), true, &trkAttribs)) {
      continue;
    }
    ++nTracksLoadedITSTPC;
    LOGF(debug, "Loaded ITS-TPC track %i with time %f. Window from %f to %f", nTracksLoadedITSTPC, trkAttribs.mTime, trkAttribs.mTime - trkAttribs.mTimeSubMax, trkAttribs.mTime + trkAttribs.mTimeAddMax);
  }
  // load TPC-only tracks
  for (unsigned int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nOutputTracksTPCO2; ++iTrk) {
    if (mChainTracking->mIOPtrs.tpcLinkITS && mChainTracking->mIOPtrs.tpcLinkITS[iTrk] != -1) {
      // this TPC tracks has already been matched to ITS and the ITS-TPC track has already been loaded in the tracker
      continue;
    }
    const auto& trkTpc = mChainTracking->mIOPtrs.outputTracksTPCO2[iTrk];
    GPUTRDTracker::HelperTrackAttributes trkAttribs;
    trkAttribs.mTime = trkTpc.getTime0() * mTPCTBinMUS - mTPCTDriftOffset; // account for the eventual time bias for TPC tracks time
    trkAttribs.mTimeAddMax = trkTpc.getDeltaTFwd() * mTPCTBinMUS;
    trkAttribs.mTimeSubMax = trkTpc.getDeltaTBwd() * mTPCTBinMUS;
    if (trkTpc.hasASideClustersOnly()) {
      trkAttribs.mSide = -1;
    } else if (trkTpc.hasCSideClustersOnly()) {
      trkAttribs.mSide = 1;
    }
    GPUTRDTrack trkLoad(trkTpc);
    if (trackTune.tpcCovOuterType != o2::globaltracking::TrackTuneParams::AddCovType::Disable) {
      trkLoad.updateCov(trackTune.tpcCovOuter, trackTune.tpcCovOuterType == o2::globaltracking::TrackTuneParams::AddCovType::WithCorrelations);
    }
    auto trackGID = GTrackID(iTrk, GTrackID::TPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw(), true, &trkAttribs)) {
      continue;
    }
    ++nTracksLoadedTPC;
    LOGF(debug, "Loaded TPC track %i with time %f. Window from %f to %f", nTracksLoadedTPC, trkAttribs.mTime, trkAttribs.mTime - trkAttribs.mTimeSubMax, trkAttribs.mTime + trkAttribs.mTimeAddMax);
  }
  LOGF(info, "%i tracks are loaded into the TRD tracker. Out of those %i ITS-TPC tracks and %i TPC tracks", nTracksLoadedITSTPC + nTracksLoadedTPC, nTracksLoadedITSTPC, nTracksLoadedTPC);

  // start the tracking
  // mTracker->DumpTracks();
  mChainTracking->DoTRDGPUTracking<GPUTRDTrackerKernels::o2Version>(mTracker);
  // mTracker->DumpTracks();

  // finished tracking, now collect the output
  std::vector<TrackTRD> tracksOutITSTPC;
  std::vector<TrackTRD> tracksOutTPC;
  std::vector<TrackTriggerRecord> trackTrigRecITSTPC;
  std::vector<TrackTriggerRecord> trackTrigRecTPC;
  GPUTRDTrack* tracksOutRaw = mTracker->Tracks();
  std::vector<unsigned int> trackIdxArray(mTracker->NTracks()); // track indices sorted by trigger record index
  std::iota(trackIdxArray.begin(), trackIdxArray.end(), 0);
  std::sort(trackIdxArray.begin(), trackIdxArray.end(), [tracksOutRaw](int lhs, int rhs) { return tracksOutRaw[lhs].getCollisionId() < tracksOutRaw[rhs].getCollisionId(); });

  std::vector<std::pair<uint8_t, uint8_t>> pileUpDist;
  bool ft0Seen = false;
  if (mTrkMask[GTrackID::FT0]) { // pile-up tagging was requested
    long maxDiffFwd = mTracker->Param().rec.trd.pileupFwdNBC;
    long maxDiffBwd = mTracker->Param().rec.trd.pileupBwdNBC;
    auto ft0recPoints = inputTracks.getFT0RecPoints();
    auto trdTriggers = tmpInputContainer->mTriggerRecords;
    ft0Seen = ft0recPoints.size() > 0;
    pileUpDist.resize(trdTriggers.size(), {0, 0});
    size_t curFT0 = 0;
    for (size_t itrd = 0; itrd < trdTriggers.size(); itrd++) {
      const auto& trig = trdTriggers[itrd];
      uint8_t fwd = 0, bwd = 0;
      for (size_t ft0id = curFT0; ft0id < ft0recPoints.size(); ft0id++) {
        const auto& f0rec = ft0recPoints[ft0id];
        if (f0rec.getTrigger().getVertex()) {
          auto bcdiff = trig.getBCData().toLong() - f0rec.getInteractionRecord().toLong();
          if (bcdiff > maxDiffBwd) {
            curFT0 = ft0id + 1;
            continue;
          }
          if (bcdiff > 0) { // pre-trigger pileup, maxDiffBwd is guaranteed to be < max uint8_t
            if (bwd == 0) {
              bwd = uint8_t(bcdiff);
            }
          } else {
            if (bcdiff < -maxDiffFwd) {
              break;
            }
            fwd = uint8_t(-bcdiff); // post-trigger pileup, maxDiffFwd is guaranteed to be < max uint8_t
          }
        }
      }
      pileUpDist[itrd] = {bwd, fwd};
    }
  }

  int nTrackletsAttached = 0; // only used for debug information
  int nTracksFailedTPCTRDRefit = 0;
  int nTracksFailedITSTPCTRDRefit = 0;
  for (int iTrk = 0; iTrk < mTracker->NTracks(); ++iTrk) {
    const auto& trdTrack = mTracker->Tracks()[trackIdxArray[iTrk]];
    if (trdTrack.getCollisionId() < 0) {
      // skip tracks without TRD tracklets (the collision ID for the TRD tracks is initialized to -1 and only changed if a tracklet is attached to the track)
      continue;
    }
    if (mStrict && (trdTrack.getIsAmbiguous() || trdTrack.getReducedChi2() > mTracker->Param().rec.trd.chi2StrictCut)) {
      // skip tracks which have another hypothesis close to the best one or which do are above strict chi2 threshold
      continue;
    }
    if (trdTrack.getNtracklets() < mTracker->Param().rec.trd.nTrackletsMin) {
      continue;
    }
    if (trdTrack.getChi2() / trdTrack.getNtracklets() > mTracker->Param().rec.trd.maxChi2Red) {
      continue;
    }
    nTrackletsAttached += trdTrack.getNtracklets();
    auto trackGID = trdTrack.getRefGlobalTrackId();
    if (trackGID.includesDet(GTrackID::Source::ITS)) {
      // this track is from an ITS-TPC seed
      tracksOutITSTPC.push_back(trdTrack);
      if (ft0Seen) {
        tracksOutITSTPC.back().setPileUpDistance(pileUpDist[trdTrack.getCollisionId()].first, pileUpDist[trdTrack.getCollisionId()].second);
      } else {
        tracksOutITSTPC.back().setPileUpDistance(mTracker->Param().rec.trd.pileupBwdNBC, mTracker->Param().rec.trd.pileupFwdNBC);
      }
      if (!refitITSTPCTRDTrack(tracksOutITSTPC.back(), mChainTracking->mIOPtrs.trdTriggerTimes[trdTrack.getCollisionId()], &inputTracks) || std::isnan(tracksOutITSTPC.back().getSnp())) {
        tracksOutITSTPC.pop_back();
        ++nTracksFailedITSTPCTRDRefit;
        continue;
      }
      if (mUseMC) {
        fillMCTruthInfo(trdTrack, itstpcTrackLabels[trackGID], trdLabelsITSTPC, matchLabelsITSTPC, inputTracks.getTRDTrackletsMCLabels());
      }
      if (mWithPID) {
        tracksOutITSTPC.back().setSignal(mBase->process(trdTrack, inputTracks, false));
      }
    } else {
      // this track is from a TPC-only seed
      tracksOutTPC.push_back(trdTrack);
      if (ft0Seen) {
        tracksOutTPC.back().setPileUpDistance(pileUpDist[trdTrack.getCollisionId()].first, pileUpDist[trdTrack.getCollisionId()].second);
      } else {
        tracksOutTPC.back().setPileUpDistance(mTracker->Param().rec.trd.pileupBwdNBC, mTracker->Param().rec.trd.pileupFwdNBC);
      }
      if (!refitTPCTRDTrack(tracksOutTPC.back(), mChainTracking->mIOPtrs.trdTriggerTimes[trdTrack.getCollisionId()], &inputTracks) || std::isnan(tracksOutTPC.back().getSnp())) {
        tracksOutTPC.pop_back();
        ++nTracksFailedTPCTRDRefit;
        continue;
      }
      if (mUseMC) {
        fillMCTruthInfo(trdTrack, tpcTrackLabels[trackGID], trdLabelsTPC, matchLabelsTPC, inputTracks.getTRDTrackletsMCLabels());
      }
      if (mWithPID) {
        tracksOutTPC.back().setSignal(mBase->process(trdTrack, inputTracks, true));
      }
    }
  }

  fillTrackTriggerRecord(tracksOutITSTPC, trackTrigRecITSTPC, tmpInputContainer->mTriggerRecords);
  fillTrackTriggerRecord(tracksOutTPC, trackTrigRecTPC, tmpInputContainer->mTriggerRecords);

  LOGF(info, "The TRD tracker found %lu tracks from TPC seeds and %lu tracks from ITS-TPC seeds and attached in total %i tracklets out of %i",
       tracksOutTPC.size(), tracksOutITSTPC.size(), nTrackletsAttached, mChainTracking->mIOPtrs.nTRDTracklets);
  LOGF(info, "Number of tracks failed in the refit: TPC-TRD (%i), ITS-TPC-TRD (%i)", nTracksFailedTPCTRDRefit, nTracksFailedITSTPCTRDRefit);

  uint32_t ss = o2::globaltracking::getSubSpec(mStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, mTrkMask)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCH_ITSTPC", 0, Lifetime::Timeframe}, tracksOutITSTPC);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0, Lifetime::Timeframe}, trackTrigRecITSTPC);
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe}, matchLabelsITSTPC);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe}, trdLabelsITSTPC);
    }
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, mTrkMask)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCH_TPC", ss, Lifetime::Timeframe}, tracksOutTPC);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_TPC", ss, Lifetime::Timeframe}, trackTrigRecTPC);
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC", ss, Lifetime::Timeframe}, matchLabelsTPC);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC_TRD", ss, Lifetime::Timeframe}, trdLabelsTPC);
    }
  }

  mTimer.Stop();
}

bool TRDGlobalTracking::refitITSTPCTRDTrack(TrackTRD& trk, float timeTRD, o2::globaltracking::RecoContainer* recoCont)
{
  auto propagator = o2::base::Propagator::Instance();

  // refit ITS-TPC-TRD track outwards to outermost TRD space point (start with ITS outer track parameters)
  auto& outerParam = trk.getOuterParam();
  auto detRefs = recoCont->getSingleDetectorRefs(trk.getRefGlobalTrackId());
  int nCl = -1, clEntry = -1, nClRefit = 0, clRefs[14];
  float chi2Out = 0, timeZErr = 0.;
  bool pileUpOn = trk.hasPileUpInfo(); // distance to farthest collision within the pileup integration time is set
  if (pileUpOn) {
    timeTRD += trk.getPileUpTimeShiftMUS(); // shift to average pileup position
  }
  auto geom = o2::its::GeometryTGeo::Instance();
  auto matCorr = o2::base::Propagator::MatCorrType(mRec->GetParam().rec.trd.matCorrType);
  if (detRefs[GTrackID::ITS].isIndexSet()) { // this is ITS track
    const auto& trkITS = mITSTracksArray[detRefs[GTrackID::ITS]];
    outerParam = trkITS.getParamOut();
    nCl = trkITS.getNumberOfClusters();
    clEntry = trkITS.getFirstClusterEntry();
    chi2Out = trkITS.getChi2();
    for (int icl = 0; icl < nCl; icl++) {            // clusters are stored from outer to inner layers
      clRefs[icl] = mITSTrackClusIdx[clEntry + icl]; // from outer to inner layer
    }
  } else { // this is ITS-AB track, will need to refit including the ITS part
    const auto& trkITSABref = mITSABRefsArray[detRefs[GTrackID::ITSAB]];
    nCl = trkITSABref.getNClusters();
    clEntry = trkITSABref.getFirstEntry();
    outerParam = recoCont->getTPCITSTrack(trk.getRefGlobalTrackId()); // start from the inner kinematics of ITS-TPC
    outerParam.resetCovariance(100);                                  // reset covariance to something big
    // refit
    for (int icl = 0; icl < nCl; icl++) {                                                              // clusters are stored from inner to outer layers
      const auto& clus = mITSClustersArray[clRefs[nCl - icl - 1] = mITSABTrackClusIdx[clEntry + icl]]; // register in clRefs from outer to inner layer
      if (!outerParam.rotate(geom->getSensorRefAlpha(clus.getSensorID())) ||
          !propagator->propagateToX(outerParam, clus.getX(), propagator->getNominalBz(), o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr)) {
        break;
      }
      chi2Out += outerParam.getPredictedChi2(clus);
      if (!outerParam.update(clus)) {
        break;
      }
      nClRefit++;
    }
    if (nClRefit != nCl) {
      LOG(debug) << "ITS-AB refit outward failed";
      return false;
    }
  }

  int retVal = mTPCRefitter->RefitTrackAsTrackParCov(outerParam, mTPCTracksArray[detRefs[GTrackID::TPC]].getClusterRef(), (timeTRD + mTPCTDriftOffset) * mTPCTBinMUSInv, &chi2Out, true, false); // outward refit
  if (retVal < 0) {
    LOG(debug) << "TPC refit outwards failed";
    return false;
  }

  if (!refitTRDTrack(trk, chi2Out, false)) {
    LOG(debug) << "TRD refit outwards failed";
    return false;
  }

  // refit ITS-TPC-TRD track inwards to innermost ITS cluster
  // here we also calculate the LT integral for matching to TOF
  float chi2In = 0.f;
  if (!refitTRDTrack(trk, chi2In, true)) {
    LOG(debug) << "TRD refit inwards failed";
    return false;
  }
  auto posStart = trk.getXYZGlo();
  retVal = mTPCRefitter->RefitTrackAsTrackParCov(trk, mTPCTracksArray[detRefs[GTrackID::TPC]].getClusterRef(), (timeTRD + mTPCTDriftOffset) * mTPCTBinMUSInv, &chi2In, false, false); // inward refit
  if (retVal < 0) {
    LOG(debug) << "TPC refit inwards failed";
    return false;
  }
  auto posEnd = trk.getXYZGlo();
  // account path integrals
  float dX = posEnd.x() - posStart.x(), dY = posEnd.y() - posStart.y(), dZ = posEnd.z() - posStart.z(), d2XY = dX * dX + dY * dY;
  if (std::abs(o2::base::Propagator::Instance()->getNominalBz()) > 0.01) { // circular arc = 2*R*asin(dXY/2R)
    float b[3];
    o2::math_utils::Point3D<float> posAv(0.5 * (posEnd.x() + posStart.x()), 0.5 * (posEnd.y() + posStart.y()), 0.5 * (posEnd.z() + posStart.z()));
    propagator->getFieldXYZ(posAv, b);
    float curvH = std::abs(0.5f * trk.getCurvature(b[2])), arcXY = 1. / curvH * std::asin(curvH * std::sqrt(d2XY));
    d2XY = arcXY * arcXY;
  }
  auto lInt = std::sqrt(d2XY + dZ * dZ);
  trk.getLTIntegralOut().addStep(lInt, trk.getP2Inv());
  // trk.getLTIntegralOut().addX2X0(lInt * mTPCmeanX0Inv); // do we need to account for the material budget here? probably

  const auto& trackTune = o2::globaltracking::TrackTuneParams::Instance();
  if (trackTune.tpcCovInnerType != o2::globaltracking::TrackTuneParams::AddCovType::Disable) {
    if (!propagator->PropagateToXBxByBz(trk, o2::constants::geom::XTPCInnerRef, o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr, &trk.getLTIntegralOut())) {
      LOG(debug) << "Propagation to TPC inner reference X for ITS refit inwards failed";
      return false;
    }
    bool preserveCorrelation = trackTune.tpcCovInnerType == o2::globaltracking::TrackTuneParams::AddCovType::WithCorrelations;
    trk.updateCov(trackTune.tpcCovInner, preserveCorrelation);
  }

  nClRefit = 0;
  for (int icl = 0; icl < nCl; icl++) {
    const auto& clus = mITSClustersArray[clRefs[icl]];
    if (!trk.rotate(geom->getSensorRefAlpha(clus.getSensorID())) ||
        // note: here we also calculate the L,T integral (in the inward direction, but this is irrelevant)
        // note: we should eventually use TPC pid in the refit (TODO)
        // note: since we are at small R, we can use field BZ component at origin rather than 3D field
        !propagator->propagateToX(trk, clus.getX(), propagator->getNominalBz(), o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr, &trk.getLTIntegralOut())) {
      break;
    }
    chi2In += trk.getPredictedChi2(clus);
    if (!trk.update(clus)) {
      break;
    }
    nClRefit++;
  }
  if (nClRefit != nCl) {
    LOG(debug) << "ITS refit inwards failed";
    return false;
  }
  // We need to update the LTOF integral by the distance to the "primary vertex"
  // We want to leave the track at the the position of its last update, so we do a fast propagation on the TrackPar copy of trfit,
  // and since for the LTOF calculation the material effects are irrelevant, we skip material corrections
  const o2::dataformats::VertexBase vtxDummy; // at the moment using dummy vertex: TODO use MeanVertex constraint instead
  o2::track::TrackPar trkPar(trk);
  if (!propagator->propagateToDCA(vtxDummy.getXYZ(), trkPar, propagator->getNominalBz(), o2::base::Propagator::MAX_STEP, matCorr, nullptr, &trk.getLTIntegralOut())) {
    LOG(error) << "LTOF integral might be incorrect";
  }
  return true;
}

bool TRDGlobalTracking::refitTPCTRDTrack(TrackTRD& trk, float timeTRD, o2::globaltracking::RecoContainer* recoCont)
{
  auto propagator = o2::base::Propagator::Instance();
  auto matCorr = o2::base::Propagator::MatCorrType(mRec->GetParam().rec.trd.matCorrType);
  // refit TPC-TRD track outwards toward outermost TRD space point
  auto& outerParam = trk.getOuterParam();
  auto detRefs = recoCont->getSingleDetectorRefs(trk.getRefGlobalTrackId());
  outerParam = trk;
  float chi2Out = 0, timeZErr = 0.;
  bool pileUpOn = trk.hasPileUpInfo(); // distance to farthest collision within the pileup integration time is set
  if (pileUpOn) {
    timeTRD += trk.getPileUpTimeShiftMUS(); // shift to average pileup position
  }
  int retVal = mTPCRefitter->RefitTrackAsTrackParCov(outerParam, mTPCTracksArray[detRefs[GTrackID::TPC]].getClusterRef(), (timeTRD + mTPCTDriftOffset) * mTPCTBinMUSInv, &chi2Out, true, false); // outward refit
  if (retVal < 0) {
    LOG(debug) << "TPC refit outwards failed";
    return false;
  }
  if (pileUpOn) { // account pileup time uncertainty in Z errors
    timeZErr = mTPCVdrift * trk.getPileUpTimeErrorMUS();
    outerParam.updateCov(timeZErr, o2::track::CovLabels::kSigZ2);
  }
  if (!refitTRDTrack(trk, chi2Out, false)) {
    LOG(debug) << "TRD refit outwards failed";
    return false;
  }

  // refit TPC-TRD track inwards toward inner TPC radius
  float chi2In = 0.f;
  if (!refitTRDTrack(trk, chi2In, true)) {
    LOG(debug) << "TRD refit inwards failed";
    return false;
  }
  auto posStart = trk.getXYZGlo();
  retVal = mTPCRefitter->RefitTrackAsTrackParCov(trk, mTPCTracksArray[detRefs[GTrackID::TPC]].getClusterRef(), (timeTRD + mTPCTDriftOffset) * mTPCTBinMUSInv, &chi2In, false, false); // inward refit
  if (retVal < 0) {
    LOG(debug) << "TPC refit inwards failed";
    return false;
  }
  if (pileUpOn) { // account pileup time uncertainty in Z errors
    trk.updateCov(timeZErr, o2::track::CovLabels::kSigZ2);
  }

  auto posEnd = trk.getXYZGlo();
  // account path integrals
  float dX = posEnd.x() - posStart.x(), dY = posEnd.y() - posStart.y(), dZ = posEnd.z() - posStart.z(), d2XY = dX * dX + dY * dY;
  if (std::abs(o2::base::Propagator::Instance()->getNominalBz()) > 0.01) { // circular arc = 2*R*asin(dXY/2R)
    float b[3];
    o2::math_utils::Point3D<float> posAv(0.5 * (posEnd.x() + posStart.x()), 0.5 * (posEnd.y() + posStart.y()), 0.5 * (posEnd.z() + posStart.z()));
    propagator->getFieldXYZ(posAv, b);
    float curvH = std::abs(0.5f * trk.getCurvature(b[2])), arcXY = 1. / curvH * std::asin(curvH * std::sqrt(d2XY));
    d2XY = arcXY * arcXY;
  }
  auto lInt = std::sqrt(d2XY + dZ * dZ);
  trk.getLTIntegralOut().addStep(lInt, trk.getP2Inv());
  // trk.getLTIntegralOut().addX2X0(lInt * mTPCmeanX0Inv); // do we need to account for the material budget here? probably?

  if (!propagator->PropagateToXBxByBz(trk, o2::constants::geom::XTPCInnerRef, o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr, &trk.getLTIntegralOut())) {
    LOG(debug) << "Final propagation to inner TPC radius failed (not removing the track because of this)";
  }
  propagator->estimateLTFast(trk.getLTIntegralOut(), trk); // guess about initial value for the track integral from the origin
  return true;
}

bool TRDGlobalTracking::refitTRDTrack(TrackTRD& trk, float& chi2, bool inwards)
{
  auto propagator = o2::base::Propagator::Instance();

  int lyStart = inwards ? NLAYER - 1 : 0;
  int direction = inwards ? -1 : 1;
  int lyEnd = inwards ? -1 : NLAYER;
  o2::track::TrackParCov* trkParam = nullptr;
  o2::track::TrackLTIntegral* tofL = nullptr;
  auto matCorr = o2::base::Propagator::MatCorrType(mRec->GetParam().rec.trd.matCorrType);

  if (inwards) {
    trkParam = &trk;
    tofL = &trk.getLTIntegralOut();
  } else {
    trkParam = &trk.getOuterParam();
    trkParam->setUserField(trk.getUserField()); // pileup timing info

    const auto& trackTune = o2::globaltracking::TrackTuneParams::Instance();
    if (trackTune.tpcCovOuterType != o2::globaltracking::TrackTuneParams::AddCovType::Disable) {
      if (!propagator->PropagateToXBxByBz(*trkParam, o2::constants::geom::XTPCOuterRef, o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr)) {
        LOG(debug) << "Propagation to TPC outer reference X for TRD outward refit failed";
        return false;
      }
      bool preserveCorrelation = trackTune.tpcCovOuterType == o2::globaltracking::TrackTuneParams::AddCovType::WithCorrelations;
      trkParam->updateCov(trackTune.tpcCovOuter, preserveCorrelation);
    }
  }

  if (inwards) {
    // reset covariance to something big for inwards refit
    trkParam->resetCovariance(100);
  }
  for (int iLy = lyStart; iLy != lyEnd; iLy += direction) {
    int trkltId = trk.getTrackletIndex(iLy);
    if (trkltId < 0) {
      continue;
    }
    int trkltDet = mTrackletsRaw[trkltId].getDetector();
    int trkltSec = trkltDet / (NLAYER * NSTACK);
    if (trkltSec != o2::math_utils::angle2Sector(trkParam->getAlpha())) {
      if (!trkParam->rotate(o2::math_utils::sector2Angle(trkltSec))) {
        LOGF(debug, "Track at alpha=%.2f could not be rotated in tracklet coordinate system with alpha=%.2f", trkParam->getAlpha(), o2::math_utils::sector2Angle(trkltSec));
        return false;
      }
    }
    if (!propagator->PropagateToXBxByBz(*trkParam, mTrackletsCalib[trkltId].getX(), o2::base::Propagator::MAX_SIN_PHI, o2::base::Propagator::MAX_STEP, matCorr, tofL)) {
      LOGF(debug, "Track propagation failed in layer %i (pt=%f, xTrk=%f, xToGo=%f)", iLy, trkParam->getPt(), trkParam->getX(), mTrackletsCalib[trkltId].getX());
      return false;
    }
    const PadPlane* pad = Geometry::instance()->getPadPlane(trkltDet);
    float tilt = tan(TMath::DegToRad() * pad->getTiltingAngle()); // tilt is signed! and returned in degrees
    float tiltCorrUp = tilt * (mTrackletsCalib[trkltId].getZ() - trkParam->getZ());
    float zPosCorrUp = mTrackletsCalib[trkltId].getZ() + mRecoParam.getZCorrCoeffNRC() * trkParam->getTgl();
    float padLength = pad->getRowSize(mTrackletsRaw[trkltId].getPadRow());
    if (!((trkParam->getSigmaZ2() < (padLength * padLength / 12.f)) && (std::fabs(mTrackletsCalib[trkltId].getZ() - trkParam->getZ()) < padLength))) {
      tiltCorrUp = 0.f;
    }

    std::array<float, 2> trkltPosUp{mTrackletsCalib[trkltId].getY() - tiltCorrUp, zPosCorrUp};
    std::array<float, 3> trkltCovUp;
    mRecoParam.recalcTrkltCov(tilt, trkParam->getSnp(), pad->getRowSize(mTrackletsRaw[trkltId].getPadRow()), trkltCovUp);

    chi2 += trkParam->getPredictedChi2(trkltPosUp, trkltCovUp);
    if (!trkParam->update(trkltPosUp, trkltCovUp)) {
      LOGF(debug, "Failed to update track with space point in layer %i", iLy);
      return false;
    }
  }
  if (!inwards) { // to make sure that the inward fit will start from the trkParam
    ((o2::track::TrackParCov&)trk) = *trkParam;
  }
  return true;
}

void TRDGlobalTracking::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD global tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, GTrackID::mask_t src, bool trigRecFilterActive, bool strict, bool withPID, PIDPolicy policy)
{
  std::vector<OutputSpec> outputs;
  uint32_t ss = o2::globaltracking::getSubSpec(strict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  if (strict) {
    dataRequest->setMatchingInputStrict();
  }
  auto trkSrc = src;
  trkSrc |= GTrackID::getSourcesMask("TPC");
  dataRequest->requestClusters(GTrackID::getSourcesMask("TRD"), useMC);
  dataRequest->requestTPCClusters(false); // only needed for refit, don't care about labels
  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    // ITS clusters are only needed if we match to ITS-TPC tracks
    dataRequest->requestITSClusters(false); // only needed for refit, don't care about labels
    trkSrc |= GTrackID::getSourcesMask("ITS");
  }
  dataRequest->requestTracks(trkSrc, useMC);
  auto& inputs = dataRequest->inputs;
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(inputs);

  // Request PID policy data
  if (withPID) {
    switch (policy) {
      case PIDPolicy::LQ1D:
        // inputs.emplace_back("LQ1D", "TRD", "LQ1D", 0, Lifetime::Condition, ccdbParamSpec("TRD/ppPID/LQ1D"));
        break;
      case PIDPolicy::LQ3D:
        // inputs.emplace_back("LQ3D", "TRD", "LQ3D", 0, Lifetime::Condition, ccdbParamSpec("TRD/ppPID/LQ3D"));
        break;
      case PIDPolicy::Test:
        inputs.emplace_back("mlTest", "TRD", "MLTEST", 0, Lifetime::Condition, ccdbParamSpec("TRD_test/pid/xgb1"));
        break;
      case PIDPolicy::Dummy:
        break;
      default:
        throw std::runtime_error("Unable to load requested PID policy data!");
    }
  }

  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCH_ITSTPC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe);
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe);
    }
    if (withPID) {
      outputs.emplace_back(o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0, Lifetime::Timeframe);
    }
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCH_TPC", ss, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_TPC", ss, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC", ss, Lifetime::Timeframe);
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC_TRD", ss, Lifetime::Timeframe);
    }
    if (trigRecFilterActive) {
      LOG(info) << "Matching to TPC-only tracks requested, but IRs without ITS contribution are filtered out (used strict matching mode to constrain TPC tracks before matching to ITS)";
    }
  }

  std::string processorName = o2::utils::Str::concat_string("trd-globaltracking", GTrackID::getSourcesNames(src));
  std::regex reg("[,\\[\\]]+");
  processorName = regex_replace(processorName, reg, "_");

  return DataProcessorSpec{
    processorName,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDGlobalTracking>(useMC, withPID, policy, dataRequest, ggRequest, src, trigRecFilterActive, strict)},
    Options{}};
}

} // namespace trd
} // namespace o2
