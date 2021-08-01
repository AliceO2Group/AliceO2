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
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "GPUWorkflowHelper/GPUWorkflowHelper.h"
#include "Framework/ConfigParamRegistry.h"

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

#include <algorithm>

using namespace o2::framework;
using namespace o2::gpu;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

void TRDGlobalTracking::init(InitContext& ic)
{

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  auto geo = Geometry::instance();
  geo->createPadPlaneArray();
  geo->createClusterMatrixArray();
  mFlatGeo = std::make_unique<GeometryFlat>(*geo);

  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    LOG(INFO) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(INFO) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }

  //-------- init GPU reconstruction --------//
  GPURecoStepConfiguration cfgRecoStep;
  cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
  cfgRecoStep.inputs.clear();
  cfgRecoStep.outputs.clear();
  mRec = GPUReconstruction::CreateInstance("CPU", true);
  mRec->SetSettings(o2::base::Propagator::Instance()->getNominalBz(), &cfgRecoStep);

  mChainTracking = mRec->AddChain<GPUChainTracking>();

  mTracker = new GPUTRDTracker();
  mTracker->SetNCandidates(mRec->GetProcessingSettings().trdNCandidates); // must be set before initialization
  mTracker->SetProcessPerTimeFrame(true);
  mTracker->SetGenerateSpacePoints(false); // set to true to force space point calculation by the TRD tracker itself

  mRec->RegisterGPUProcessor(mTracker, false);
  mChainTracking->SetTRDGeometry(std::move(mFlatGeo));
  if (mRec->Init()) {
    LOG(FATAL) << "GPUReconstruction could not be initialized";
  }

  mTracker->PrintSettings();

  mTimer.Stop();
  mTimer.Reset();
}

void TRDGlobalTracking::updateTimeDependentParams()
{
  // strictly speaking, one should do this only in case of the CCDB objects update
  // TODO: add CCDB interface
  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  auto& gasParam = o2::tpc::ParameterGas::Instance();
  mTPCTBinMUS = elParam.ZbinWidth;
  mTPCVdrift = gasParam.DriftV;
  mTracker->SetTPCVdrift(mTPCVdrift);
}

void TRDGlobalTracking::fillTrackTriggerRecord(const std::vector<TrackTRD>& tracks, std::vector<TrackTriggerRecord>& trigRec, const gsl::span<const o2::trd::TriggerRecord>& trackletTrigRec) const
{
  int currTrigRec = 0;
  int nTracksCurr = 0;
  int iTrackFirst = 0;
  for (const auto& trk : tracks) {
    if (trk.getCollisionId() != currTrigRec) {
      // new collision ID, create new track trigger record
      trigRec.emplace_back(trackletTrigRec[currTrigRec].getBCData(), iTrackFirst, nTracksCurr);
      currTrigRec = trk.getCollisionId();
      iTrackFirst += nTracksCurr;
      nTracksCurr = 0;
    }
    ++nTracksCurr;
  }
  if (nTracksCurr > 0) {
    // create track trigger record for remaining track range
    trigRec.emplace_back(trackletTrigRec[currTrigRec].getBCData(), iTrackFirst, nTracksCurr);
  }
}

void TRDGlobalTracking::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  mChainTracking->ClearIOPointers();
  o2::globaltracking::RecoContainer inputTracks;
  inputTracks.collectData(pc, *mDataRequest);
  auto tmpInputContainer = getRecoInputContainer(pc, &mChainTracking->mIOPtrs, &inputTracks);
  auto tmpContainer = GPUWorkflowHelper::fillIOPtr(mChainTracking->mIOPtrs, inputTracks, mUseMC, nullptr, GTrackID::getSourcesMask("TRD"), mTrkMask, GTrackID::mask_t{GTrackID::MASK_NONE});
  LOGF(INFO, "There are %i tracklets in total from %i trigger records", mChainTracking->mIOPtrs.nTRDTracklets, mChainTracking->mIOPtrs.nTRDTriggerRecords);
  LOGF(INFO, "As input seeds are available: %i ITS-TPC matched tracks and %i TPC tracks", mChainTracking->mIOPtrs.nTracksTPCITSO2, mChainTracking->mIOPtrs.nOutputTracksTPCO2);

  mTracker->Reset();
  updateTimeDependentParams();
  mRec->PrepareEvent();
  mRec->SetupGPUProcessor(mTracker, true);

  // check trigger record filter setting
  bool foundFilteredTrigger = false;
  for (int iTrig = 0; iTrig < mChainTracking->mIOPtrs.nTRDTriggerRecords; ++iTrig) {
    if (mChainTracking->mIOPtrs.trdTrigRecMask[iTrig] == 0) {
      foundFilteredTrigger = true;
    }
  }
  if (!foundFilteredTrigger && mTrigRecFilter) {
    LOG(WARNING) << "Trigger filtering requested, but no TRD trigger is actually masked. Can be that none needed to be masked or that the setting was not active for the tracklet transformer";
  } else if (foundFilteredTrigger && !mTrigRecFilter) {
    LOG(ERROR) << "Trigger filtering is not requested, but masked TRD triggers are found. Rerun tracklet transformer without trigger filtering";
  }

  // load input tracks
  LOG(DEBUG) << "Start loading input seeds into TRD tracker";
  int nTracksLoadedITSTPC = 0;
  int nTracksLoadedTPC = 0;
  // load ITS-TPC matched tracks
  for (int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nTracksTPCITSO2; ++iTrk) {
    const auto& trkITSTPC = mChainTracking->mIOPtrs.tracksTPCITSO2[iTrk];
    GPUTRDTracker::HelperTrackAttributes trkAttribs;
    trkAttribs.mTime = trkITSTPC.getTimeMUS().getTimeStamp();
    trkAttribs.mTimeAddMax = trkITSTPC.getTimeMUS().getTimeStampError() * mRec->GetParam().rec.trd.nSigmaTerrITSTPC;
    trkAttribs.mTimeSubMax = trkITSTPC.getTimeMUS().getTimeStampError() * mRec->GetParam().rec.trd.nSigmaTerrITSTPC;
    GPUTRDTrack trkLoad(trkITSTPC);
    auto trackGID = GTrackID(iTrk, GTrackID::ITSTPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw(), true, &trkAttribs)) {
      continue;
    }
    ++nTracksLoadedITSTPC;
    LOGF(DEBUG, "Loaded ITS-TPC track %i with time %f", nTracksLoadedITSTPC, trkAttribs.mTime);
  }
  // load TPC-only tracks
  for (int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nOutputTracksTPCO2; ++iTrk) {
    if (mChainTracking->mIOPtrs.tpcLinkITS && mChainTracking->mIOPtrs.tpcLinkITS[iTrk] != -1) {
      // this TPC tracks has already been matched to ITS and the ITS-TPC track has already been loaded in the tracker
      continue;
    }
    const auto& trkTpc = mChainTracking->mIOPtrs.outputTracksTPCO2[iTrk];
    GPUTRDTracker::HelperTrackAttributes trkAttribs;
    trkAttribs.mTime = trkTpc.getTime0() * mTPCTBinMUS;
    trkAttribs.mTimeAddMax = trkTpc.getDeltaTFwd() * mTPCTBinMUS;
    trkAttribs.mTimeSubMax = trkTpc.getDeltaTBwd() * mTPCTBinMUS;
    if (trkTpc.hasASideClustersOnly()) {
      trkAttribs.mSide = -1;
    } else if (trkTpc.hasCSideClustersOnly()) {
      trkAttribs.mSide = 1;
    }
    GPUTRDTrack trkLoad(trkTpc);
    auto trackGID = GTrackID(iTrk, GTrackID::TPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw(), true, &trkAttribs)) {
      continue;
    }
    ++nTracksLoadedTPC;
    LOGF(DEBUG, "Loaded TPC track %i with time %f", nTracksLoadedTPC, trkAttribs.mTime);
  }
  LOGF(INFO, "%i tracks are loaded into the TRD tracker. Out of those %i ITS-TPC tracks and %i TPC tracks", nTracksLoadedITSTPC + nTracksLoadedTPC, nTracksLoadedITSTPC, nTracksLoadedTPC);

  // start the tracking
  //mTracker->DumpTracks();
  mTracker->DoTracking(mChainTracking);
  //mTracker->DumpTracks();

  // finished tracking, now collect the output
  std::vector<TrackTRD> tracksOutITSTPC;
  std::vector<TrackTRD> tracksOutTPC;
  std::vector<TrackTriggerRecord> trackTrigRecITSTPC;
  std::vector<TrackTriggerRecord> trackTrigRecTPC;
  GPUTRDTrack* tracksOutRaw = mTracker->Tracks();
  std::vector<unsigned int> trackIdxArray(mTracker->NTracks()); // track indices sorted by trigger record index
  std::iota(trackIdxArray.begin(), trackIdxArray.end(), 0);
  std::sort(trackIdxArray.begin(), trackIdxArray.end(), [tracksOutRaw](int lhs, int rhs) { return tracksOutRaw[lhs].getCollisionId() < tracksOutRaw[rhs].getCollisionId(); });

  int nTrackletsAttached = 0; // only used for debug information
  for (int iTrk = 0; iTrk < mTracker->NTracks(); ++iTrk) {
    const auto& trdTrack = mTracker->Tracks()[trackIdxArray[iTrk]];
    if (trdTrack.getCollisionId() < 0) {
      // skip tracks without TRD tracklets (the collision ID for the TRD tracks is initialized to -1 and only changed if a tracklet is attached to the track)
      continue;
    }
    nTrackletsAttached += trdTrack.getNtracklets();
    auto trackGID = trdTrack.getRefGlobalTrackId();
    if (trackGID.includesDet(GTrackID::Source::ITS)) {
      // this track is from an ITS-TPC seed
      tracksOutITSTPC.push_back(trdTrack);
    } else {
      // this track is from a TPC-only seed
      tracksOutTPC.push_back(trdTrack);
    }
  }

  fillTrackTriggerRecord(tracksOutITSTPC, trackTrigRecITSTPC, tmpInputContainer->mTriggerRecords);
  fillTrackTriggerRecord(tracksOutTPC, trackTrigRecTPC, tmpInputContainer->mTriggerRecords);

  LOGF(INFO, "The TRD tracker found %lu tracks from TPC seeds and %lu tracks from ITS-TPC seeds and attached in total %i tracklets out of %i",
       tracksOutTPC.size(), tracksOutITSTPC.size(), nTrackletsAttached, mChainTracking->mIOPtrs.nTRDTracklets);

  uint32_t ss = o2::globaltracking::getSubSpec(mStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  if (inputTracks.isTrackSourceLoaded(GTrackID::Source::ITSTPC)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCHTRD_GLO", 0, Lifetime::Timeframe}, tracksOutITSTPC);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRG_GLO", 0, Lifetime::Timeframe}, trackTrigRecITSTPC);
  }
  if (inputTracks.isTrackSourceLoaded(GTrackID::Source::TPC)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCHTRD_TPC", ss, Lifetime::Timeframe}, tracksOutTPC);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRG_TPC", ss, Lifetime::Timeframe}, trackTrigRecTPC);
  }

  mTimer.Stop();
}

void TRDGlobalTracking::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TRD global tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, GTrackID::mask_t src, bool trigRecFilterActive, bool strict)
{
  std::vector<OutputSpec> outputs;
  uint32_t ss = o2::globaltracking::getSubSpec(strict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  if (strict) {
    dataRequest->setMatchingInputStrict();
  }
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GTrackID::getSourcesMask("TRD"), useMC);
  auto& inputs = dataRequest->inputs;


  if (useMC) {
    LOG(FATAL) << "MC usage must be disabled for this workflow, since it is not yet implemented";
  }

  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCHTRD_GLO", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRKTRG_GLO", 0, Lifetime::Timeframe);
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCHTRD_TPC", ss, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRKTRG_TPC", ss, Lifetime::Timeframe);
    if (trigRecFilterActive) {
      LOG(ERROR) << "Matching to TPC-only tracks requested, but IR without ITS contribution are filtered out. This does not lead to a crash, but it deteriorates the matching efficiency.";
    }
  }

  std::string processorName = o2::utils::Str::concat_string("trd-globaltracking", GTrackID::getSourcesNames(src));
  std::replace(processorName.begin(), processorName.end(), ',', '_');

  return DataProcessorSpec{
    processorName,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDGlobalTracking>(useMC, dataRequest, src, trigRecFilterActive, strict)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace trd
} // namespace o2
