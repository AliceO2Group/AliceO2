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

/// @file  TPCInterpolationSpec.cxx

#include <vector>
#include <unordered_map>

#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace tpc
{

void TPCInterpolationDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mSlotLength = ic.options().get<uint32_t>("sec-per-slot");
  mProcessSeeds = ic.options().get<bool>("process-seeds");
  mMatCorr = ic.options().get<int>("matCorrType");
}

void TPCInterpolationDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // other init-once stuff
    const auto& param = SpacePointsCalibConfParam::Instance();
    mInterpolation.init(mSources);
    if (mProcessITSTPConly) {
      mInterpolation.setProcessITSTPConly();
    }
    mInterpolation.setSqrtS(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getSqrtS());
    int nTfs = mSlotLength / (o2::base::GRPGeomHelper::getNHBFPerTF() * o2::constants::lhc::LHCOrbitMUS * 1e-6);
    bool limitTracks = (param.maxTracksPerCalibSlot < 0) ? false : true;
    int nTracksPerTfMax = (nTfs > 0 && limitTracks) ? param.maxTracksPerCalibSlot / nTfs : -1;
    if (nTracksPerTfMax > 0) {
      LOGP(info, "We will stop processing tracks after validating {} tracks per TF, since we want to accumulate {} tracks for a slot with {} TFs",
           nTracksPerTfMax, param.maxTracksPerCalibSlot, nTfs);
      if (param.additionalTracksITSTPC > 0) {
        int nITSTPCadd = param.additionalTracksITSTPC / nTfs;
        LOGP(info, "In addition up to {} ITS-TPC tracks are processed per TF", nITSTPCadd);
        mInterpolation.setAddITSTPCTracksPerTF(nITSTPCadd);
      }
    } else if (nTracksPerTfMax < 0) {
      LOG(info) << "The number of processed tracks per TF is not limited";
    } else {
      LOG(error) << "No tracks will be processed. maxTracksPerCalibSlot must be greater than slot length in TFs";
    }
    mInterpolation.setMaxTracksPerTF(nTracksPerTfMax);
    mInterpolation.setMatCorr(static_cast<o2::base::Propagator::MatCorrType>(mMatCorr));
    if (mProcessSeeds) {
      mInterpolation.setProcessSeeds();
    }
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));
  }
  // we may have other params which need to be queried regularly
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mInterpolation.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
  }
  if (mDebugOutput) {
    mInterpolation.setDumpTrackPoints();
    mInterpolation.setITSClusterDictionary(mITSDict);
  }
}

void TPCInterpolationDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

void TPCInterpolationDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);
  mInterpolation.prepareInputTrackSample(recoData);
  mInterpolation.process();
  mTimer.Stop();
  LOGF(info, "TPC interpolation timing: Cpu: %.3e Real: %.3e s", mTimer.CpuTime(), mTimer.RealTime());
  if (SpacePointsCalibConfParam::Instance().writeUnfiltered) {
    // these are the residuals and tracks before outlier rejection; they are not used in production
    pc.outputs().snapshot(Output{"GLO", "TPCINT_RES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResidualsUnfiltered());
    if (mSendTrackData) {
      pc.outputs().snapshot(Output{"GLO", "TPCINT_TRK", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracksUnfiltered());
    }
  }
  pc.outputs().snapshot(Output{"GLO", "UNBINNEDRES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResiduals());
  pc.outputs().snapshot(Output{"GLO", "TRKREFS", 0, Lifetime::Timeframe}, mInterpolation.getTrackDataCompact());
  if (mSendTrackData) {
    pc.outputs().snapshot(Output{"GLO", "TRKDATA", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracks());
  }
  if (mDebugOutput) {
    pc.outputs().snapshot(Output{"GLO", "TRKDATAEXT", 0, Lifetime::Timeframe}, mInterpolation.getTrackDataExtended());
  }
  mInterpolation.reset();
}

void TPCInterpolationDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TPC residuals extraction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCInterpolationSpec(GTrackID::mask_t srcCls, GTrackID::mask_t srcVtx, GTrackID::mask_t srcTrk, bool useMC, bool processITSTPConly, bool sendTrackData, bool debugOutput)
{
  auto dataRequest = std::make_shared<DataRequest>();
  std::vector<OutputSpec> outputs;

  if (useMC) {
    LOG(fatal) << "MC usage must be disabled for this workflow, since it is not yet implemented";
  }

  dataRequest->requestTracks(srcVtx, useMC);
  dataRequest->requestClusters(srcCls, useMC);
  dataRequest->requestPrimaryVertertices(useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  if (SpacePointsCalibConfParam::Instance().writeUnfiltered) {
    outputs.emplace_back("GLO", "TPCINT_TRK", 0, Lifetime::Timeframe);
    if (sendTrackData) {
      outputs.emplace_back("GLO", "TPCINT_RES", 0, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back("GLO", "UNBINNEDRES", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TRKREFS", 0, Lifetime::Timeframe);
  if (sendTrackData) {
    outputs.emplace_back("GLO", "TRKDATA", 0, Lifetime::Timeframe);
  }
  if (debugOutput) {
    outputs.emplace_back("GLO", "TRKDATAEXT", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-interpolation",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCInterpolationDPL>(dataRequest, srcTrk, ggRequest, useMC, processITSTPConly, sendTrackData, debugOutput)},
    Options{
      {"matCorrType", VariantType::Int, 2, {"material correction type (definition in Propagator.h)"}},
      {"sec-per-slot", VariantType::UInt32, 600u, {"number of seconds per calibration time slot (put 0 for infinite slot length)"}},
      {"process-seeds", VariantType::Bool, false, {"do not remove duplicates, e.g. for ITS-TPC-TRD track also process its seeding ITS-TPC part"}}}};
}

} // namespace tpc
} // namespace o2
