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

namespace o2
{
namespace trd
{

void TRDGlobalTracking::init(InitContext& ic)
{

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP(o2::base::NameConf::getGRPFileName());
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

  mRec->RegisterGPUProcessor(mTracker, false);
  mChainTracking->SetTRDGeometry(std::move(mFlatGeo));
  if (mRec->Init()) {
    LOG(FATAL) << "GPUReconstruction could not be initialized";
  }

  mTracker->PrintSettings();

  mTimer.Stop();
  mTimer.Reset();
}

void TRDGlobalTracking::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tpcitstrack");
  const auto trackletsTRD = pc.inputs().get<gsl::span<o2::trd::Tracklet64>>("trdtracklets");
  const auto triggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("trdtriggerrec");

  int nTracks = tracksITSTPC.size();
  int nCollisions = triggerRecords.size();
  int nTracklets = trackletsTRD.size();
  LOGF(INFO, "There are %i tracklets in total from %i trigger records", nTracklets, nCollisions);

  const gsl::span<const CalibratedTracklet>* cTrkltsPtr = nullptr;
  using cTrkltType = std::decay_t<decltype(pc.inputs().get<gsl::span<CalibratedTracklet>>(""))>;
  std::optional<cTrkltType> cTrklts;
  int nTrackletsCal = 0;

  if (mUseTrackletTransform) {
    cTrklts.emplace(pc.inputs().get<gsl::span<CalibratedTracklet>>("trdctracklets")); // MC labels associated to the input digits
    cTrkltsPtr = &cTrklts.value();
    nTrackletsCal = cTrkltsPtr->size();
    LOGF(INFO, "Got %i calibrated tracklets as input", nTrackletsCal);
    if (nTracklets != nTrackletsCal) {
      LOGF(ERROR, "Number of calibrated tracklets (%i) differs from the number of uncalibrated tracklets (%i)", nTrackletsCal, nTracklets);
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

  mChainTracking->mIOPtrs.nMergedTracks = nTracks;
  mChainTracking->mIOPtrs.nTRDTracklets = nTracklets;
  mChainTracking->AllocateIOMemory();
  mRec->PrepareEvent();
  mRec->SetupGPUProcessor(mTracker, true);

  LOG(DEBUG) << "Start loading input into TRD tracker";
  // load everything into the tracker
  int nTracksLoaded = 0;
  for (int iTrk = 0; iTrk < nTracks; ++iTrk) {
    const auto& match = tracksITSTPC[iTrk];
    const auto& trk = match.getParamOut();
    GPUTRDTrack trkLoad;
    trkLoad.setX(trk.getX());
    trkLoad.setAlpha(trk.getAlpha());
    for (int i = 0; i < 5; ++i) {
      trkLoad.setParam(trk.getParam(i), i);
    }
    for (int i = 0; i < 15; ++i) {
      trkLoad.setCov(trk.getCov()[i], i);
    }
    trkLoad.setTime(match.getTimeMUS().getTimeStamp());
    if (mTracker->LoadTrack(trkLoad)) {
      continue;
    }
    ++nTracksLoaded;
    LOGF(DEBUG, "Loaded track %i with time %f", nTracksLoaded, trkLoad.getTime());
  }

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
  mTracker->DoTracking(mChainTracking);
  //mTracker->DumpTracks();

  std::vector<GPUTRDTrack> tracksOut(mTracker->NTracks());
  std::copy(mTracker->Tracks(), mTracker->Tracks() + mTracker->NTracks(), tracksOut.begin());
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCHTRD", 0, Lifetime::Timeframe}, tracksOut);

  mTimer.Stop();
}

void TRDGlobalTracking::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TRD global tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, bool useTrkltTransf)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tpcitstrack", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useTrkltTransf) {
    inputs.emplace_back("trdctracklets", o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
  }
  inputs.emplace_back("trdtracklets", o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);

  if (useMC) {
    LOG(FATAL) << "MC usage must be disabled for this workflow, since it is not yet implemented";
  }

  outputs.emplace_back(o2::header::gDataOriginTRD, "MATCHTRD", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "trd-globaltracking",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDGlobalTracking>(useMC, useTrkltTransf)},
    Options{}};
}

} // namespace trd
} // namespace o2
