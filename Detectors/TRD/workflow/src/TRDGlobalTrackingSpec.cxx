// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsTRD/TrackTRD.h"

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
  //mTracker->SetDoImpactAngleHistograms(true);

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

void TRDGlobalTracking::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  o2::globaltracking::RecoContainer inputTracks;
  inputTracks.collectData(pc, *mDataRequest);
  auto tmpInputContainer = getRecoInputContainer(pc, &mChainTracking->mIOPtrs, &inputTracks);
  LOGF(INFO, "There are %i tracklets in total from %i trigger records", mChainTracking->mIOPtrs.nTRDTracklets, mChainTracking->mIOPtrs.nTRDTriggerRecords);
  LOGF(INFO, "As input seeds are available: %i ITS-TPC matched tracks and %i TPC tracks", mChainTracking->mIOPtrs.nTracksTPCITSO2, mChainTracking->mIOPtrs.nOutputTracksTPCO2);
  if (tmpInputContainer->mNTracklets != tmpInputContainer->mNSpacePoints) {
    LOGF(FATAL, "Number of calibrated tracklets (%i) differs from the number of uncalibrated tracklets (%i)", tmpInputContainer->mNSpacePoints, tmpInputContainer->mNTracklets);
  }

  mTracker->Reset();
  mTracker->ResetImpactAngleHistograms();
  updateTimeDependentParams();
  mRec->PrepareEvent();
  mRec->SetupGPUProcessor(mTracker, true);

  // load input tracks
  LOG(DEBUG) << "Start loading input seeds into TRD tracker";
  int nTracksLoadedITSTPC = 0;
  int nTracksLoadedTPC = 0;
  std::vector<int> loadedTPCtracks;
  // load ITS-TPC matched tracks
  for (int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nTracksTPCITSO2; ++iTrk) {
    const auto& trkITSTPC = mChainTracking->mIOPtrs.tracksTPCITSO2[iTrk];
    GPUTRDTrack trkLoad(trkITSTPC, mTPCVdrift);
    auto trackGID = GTrackID(iTrk, GTrackID::ITSTPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw())) {
      continue;
    }
    loadedTPCtracks.push_back(trkITSTPC.getRefTPC());
    ++nTracksLoadedITSTPC;
    LOGF(DEBUG, "Loaded ITS-TPC track %i with time %f", nTracksLoadedITSTPC, trkLoad.getTime());
  }
  // load TPC-only tracks
  for (int iTrk = 0; iTrk < mChainTracking->mIOPtrs.nOutputTracksTPCO2; ++iTrk) {
    if (std::find(loadedTPCtracks.begin(), loadedTPCtracks.end(), iTrk) != loadedTPCtracks.end()) {
      // this TPC tracks has already been matched to ITS and the ITS-TPC track has already been loaded in the tracker
      continue;
    }
    const auto& trkTpc = mChainTracking->mIOPtrs.outputTracksTPCO2[iTrk];
    GPUTRDTrack trkLoad(trkTpc, mTPCTBinMUS, mTPCVdrift, iTrk);
    auto trackGID = GTrackID(iTrk, GTrackID::TPC);
    if (mTracker->LoadTrack(trkLoad, trackGID.getRaw())) {
      continue;
    }
    ++nTracksLoadedTPC;
    LOGF(DEBUG, "Loaded TPC track %i with time %f", nTracksLoadedTPC, trkLoad.getTime());
  }
  LOGF(INFO, "%i tracks are loaded into the TRD tracker. Out of those %i ITS-TPC tracks and %i TPC tracks", nTracksLoadedITSTPC + nTracksLoadedTPC, nTracksLoadedITSTPC, nTracksLoadedTPC);

  // start the tracking
  //mTracker->DumpTracks();
  mTracker->DoTracking(mChainTracking);
  //mTracker->DumpTracks();

  // finished tracking, now collect the output
  std::vector<TrackTRD> tracksOutITSTPC;
  std::vector<TrackTRD> tracksOutTPC;
  int nTrackletsAttached = 0; // only used for debug information
  for (int iTrk = 0; iTrk < mTracker->NTracks(); ++iTrk) {
    const auto& trdTrack = mTracker->Tracks()[iTrk];
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
  LOGF(INFO, "The TRD tracker found %lu tracks from TPC seeds and %lu tracks from ITS-TPC seeds and attached in total %i tracklets out of %i",
       tracksOutTPC.size(), tracksOutITSTPC.size(), nTrackletsAttached, mChainTracking->mIOPtrs.nTRDTracklets);

  // Temporary until it is transferred to its own DPL device for calibrations
  mCalibVDrift.setAngleDiffSums(mTracker->AngleDiffSums());
  mCalibVDrift.setAngleDiffCounters(mTracker->AngleDiffCounters());
  mCalibVDrift.process();

  if (inputTracks.isTrackSourceLoaded(GTrackID::Source::ITSTPC)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCHTRD_GLO", 0, Lifetime::Timeframe}, tracksOutITSTPC);
  }
  if (inputTracks.isTrackSourceLoaded(GTrackID::Source::TPC)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCHTRD_TPC", 0, Lifetime::Timeframe}, tracksOutTPC);
  }

  mTimer.Stop();
}

void TRDGlobalTracking::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TRD global tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, GTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;

  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(src, false);
  auto& inputs = dataRequest->inputs;

  inputs.emplace_back("trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);

  if (useMC) {
    LOG(FATAL) << "MC usage must be disabled for this workflow, since it is not yet implemented";
  }

  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCHTRD_GLO", 0, Lifetime::Timeframe);
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MATCHTRD_TPC", 0, Lifetime::Timeframe);
  }

  std::string processorName = o2::utils::Str::concat_string("trd-globaltracking", GTrackID::getSourcesNames(src));
  std::replace(processorName.begin(), processorName.end(), ',', '_');

  return DataProcessorSpec{
    processorName,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDGlobalTracking>(useMC, dataRequest)},
    Options{}};
}

} // namespace trd
} // namespace o2
