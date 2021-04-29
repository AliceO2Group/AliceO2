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
#include "CommonConstants/LHCConstants.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

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

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

o2::globaltracking::DataRequest dataRequestTRD;

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
  mTracker->SetProcessPerTimeFrame();
  mTracker->SetNMaxCollisions(mRec->GetProcessingSettings().trdNMaxCollisions);
  mTracker->SetTrkltTransformNeeded(!mUseTrackletTransform);
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
  inputTracks.collectData(pc, dataRequestTRD);
  const auto tracksITSTPC = inputTracks.getTPCITSTracks<o2::dataformats::TrackTPCITS>();
  const auto tracksTPC = inputTracks.getTPCTracks<o2::tpc::TrackTPC>();
  const auto trackletsTRD = pc.inputs().get<gsl::span<o2::trd::Tracklet64>>("trdtracklets");
  const auto triggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("trdtriggerrec");

  int nTracksITSTPC = tracksITSTPC.size();
  int nTracksTPC = tracksTPC.size();
  int nCollisions = triggerRecords.size();
  int nTracklets = trackletsTRD.size();
  LOGF(INFO, "There are %i tracklets in total from %i trigger records", nTracklets, nCollisions);
  LOGF(INFO, "As input seeds are available: %i ITS-TPC matched tracks and %i TPC tracks", nTracksITSTPC, nTracksTPC);

  const gsl::span<const CalibratedTracklet>* cTrkltsPtr = nullptr;
  using cTrkltType = std::decay_t<decltype(pc.inputs().get<gsl::span<CalibratedTracklet>>(""))>;
  std::optional<cTrkltType> cTrklts;
  int nTrackletsCal = 0;

  if (mUseTrackletTransform) {
    cTrklts.emplace(pc.inputs().get<gsl::span<CalibratedTracklet>>("trdctracklets"));
    cTrkltsPtr = &cTrklts.value();
    nTrackletsCal = cTrkltsPtr->size();
    LOGF(INFO, "Got %i calibrated tracklets as input", nTrackletsCal);
    if (nTracklets != nTrackletsCal) {
      LOGF(FATAL, "Number of calibrated tracklets (%i) differs from the number of uncalibrated tracklets (%i)", nTrackletsCal, nTracklets);
    }
  }

  std::vector<float> trdTriggerTimes;
  std::vector<int> trdTriggerIndices;

  for (int iEv = 0; iEv < nCollisions; ++iEv) {
#ifdef MS_GSL_V3
    const auto& trg = triggerRecords[iEv];
#else
    const auto& trg = triggerRecords.at(iEv);
#endif
    int nTrackletsCurrent = trg.getNumberOfTracklets();
    int iFirstTracklet = trg.getFirstTracklet();
    int64_t evTime = trg.getBCData().toLong() * o2::constants::lhc::LHCBunchSpacingNS; // event time in ns
    trdTriggerTimes.push_back(evTime / 1000.);
    trdTriggerIndices.push_back(iFirstTracklet);
    LOGF(DEBUG, "Event %i: Occured at %li us after SOR, contains %i tracklets, index of first tracklet is %i", iEv, evTime / 1000, nTrackletsCurrent, iFirstTracklet);
  }

  mTracker->Reset();
  updateTimeDependentParams();

  // the number of tracks loaded into the TRD tracker depends on the defined input sources
  // TPC-only tracks which are already matched to the ITS will not be loaded as seeds for the tracking
  // => the maximum number of seeds it the number of TPC-only tracks. If only ITS-TPC matches are considered than that
  //    of course defines the number of input tracks
  mChainTracking->mIOPtrs.nMergedTracks = (nTracksTPC == 0) ? nTracksITSTPC : nTracksTPC;
  mChainTracking->mIOPtrs.nTRDTracklets = nTracklets;
  mChainTracking->AllocateIOMemory();
  mRec->PrepareEvent();
  mRec->SetupGPUProcessor(mTracker, true);

  LOG(DEBUG) << "Start loading input into TRD tracker";

  int nTracksLoadedITSTPC = 0;
  int nTracksLoadedTPC = 0;
  std::vector<int> loadedTPCtracks;

  // load ITS-TPC matched tracks
  for (const auto& match : tracksITSTPC) {
    GPUTRDTrack trkLoad(match, mTPCVdrift);
    if (mTracker->LoadTrack(trkLoad)) {
      continue;
    }
    loadedTPCtracks.push_back(match.getRefTPC());
    ++nTracksLoadedITSTPC;
    LOGF(DEBUG, "Loaded ITS-TPC track %i with time %f", nTracksLoadedITSTPC, trkLoad.getTime());
  }

  // load TPC-only tracks
  for (int iTrk = 0; iTrk < tracksTPC.size(); ++iTrk) {
    if (std::find(loadedTPCtracks.begin(), loadedTPCtracks.end(), iTrk) != loadedTPCtracks.end()) {
      // this TPC tracks has already been matched to ITS and the ITS-TPC track has already been loaded in the tracker
      continue;
    }
    const auto& trkTpc = tracksTPC[iTrk];
    GPUTRDTrack trkLoad(trkTpc, mTPCTBinMUS, mTPCVdrift, iTrk);
    if (mTracker->LoadTrack(trkLoad)) {
      continue;
    }
    ++nTracksLoadedTPC;
    LOGF(DEBUG, "Loaded TPC track %i with time %f", nTracksLoadedTPC, trkLoad.getTime());
  }
  LOGF(INFO, "%i tracks are loaded into the TRD tracker. Out of those %i ITS-TPC tracks and %i TPC tracks", nTracksLoadedITSTPC + nTracksLoadedTPC, nTracksLoadedITSTPC, nTracksLoadedTPC);

  // load the TRD tracklets
  for (int iTrklt = 0; iTrklt < nTracklets; ++iTrklt) {
    auto trklt = trackletsTRD[iTrklt];
    GPUTRDTrackletWord trkltLoad(trklt.getTrackletWord());
    if (mTracker->LoadTracklet(trkltLoad) > 0) {
      LOG(WARNING) << "Could not load tracklet " << iTrklt;
    }
    if (mUseTrackletTransform) {
      const CalibratedTracklet cTrklt = (cTrkltsPtr->data())[iTrklt];
      mTracker->SetInternalSpacePoint(iTrklt, cTrklt.getX(), cTrklt.getY(), cTrklt.getZ(), cTrklt.getDy());
    }
  }

  mTracker->SetTriggerRecordTimes(&(trdTriggerTimes[0]));
  mTracker->SetTriggerRecordIndices(&(trdTriggerIndices[0]));
  mTracker->SetNCollisions(nCollisions);
  //mTracker->DumpTracks();
  mTracker->ResetImpactAngleHistograms();
  mTracker->DoTracking(mChainTracking);
  //mTracker->DumpTracks();

  std::vector<GPUTRDTrack> tracksOutITSTPC(nTracksLoadedITSTPC);
  std::vector<GPUTRDTrack> tracksOutTPC(nTracksLoadedTPC);
  if (mTracker->NTracks() != nTracksLoadedITSTPC + nTracksLoadedTPC) {
    LOGF(FATAL, "Got %i matched tracks in total whereas %i ITS-TPC + %i TPC = %i tracks were loaded as input", mTracker->NTracks(), nTracksLoadedITSTPC, nTracksLoadedTPC, nTracksLoadedITSTPC + nTracksLoadedTPC);
  }

  // copy ITS-TPC matched tracks first
  std::copy(mTracker->Tracks(), mTracker->Tracks() + nTracksLoadedITSTPC, tracksOutITSTPC.begin());
  // and now the remaining TPC-only matches
  std::copy(mTracker->Tracks() + nTracksLoadedITSTPC, mTracker->Tracks() + mTracker->NTracks(), tracksOutTPC.begin());

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

DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, bool useTrkltTransf, GTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;

  dataRequestTRD.requestTracks(src, false);
  auto& inputs = dataRequestTRD.inputs;

  if (useTrkltTransf) {
    inputs.emplace_back("trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
  }
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
    AlgorithmSpec{adaptFromTask<TRDGlobalTracking>(useMC, useTrkltTransf)},
    Options{}};
}

} // namespace trd
} // namespace o2
