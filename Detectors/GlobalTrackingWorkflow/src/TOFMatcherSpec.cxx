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

/// @file   TOFMatcherSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"

// from Tracks
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "DetectorsBase/GRPGeomHelper.h"

// from TOF
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"
#include "TOFBase/Utils.h"

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
// using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class TOFMatcherSpec : public Task
{
 public:
  TOFMatcherSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, bool useFIT, bool tpcRefit, bool strict, bool pushMatchable, int lanes = 1) : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC), mUseFIT(useFIT), mDoTPCRefit(tpcRefit), mStrict(strict), mPushMatchable(pushMatchable), mNlanes(lanes) {}
  ~TOFMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  bool mUseMC = true;
  bool mUseFIT = false;
  bool mDoTPCRefit = false;
  bool mStrict = false;
  bool mPushMatchable = false;
  float mExtraTolTRD = 0.;
  int mNlanes = 1;
  MatchTOF mMatcher; ///< Cluster finder
  TStopwatch mTimer;
};

void TOFMatcherSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  if (mStrict) {
    mMatcher.setHighPurity();
  }
  mTPCCorrMapsLoader.init(ic);
  mMatcher.storeMatchable(mPushMatchable);
  mMatcher.setExtraTimeToleranceTRD(mExtraTolTRD);
  mMatcher.setNlanes(mNlanes);
}

void TOFMatcherSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  mTPCCorrMapsLoader.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    const auto bcs = o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling().getFilledBCs();
    for (auto bc : bcs) {
      o2::tof::Utils::addInteractionBC(bc, true);
    }
    initOnceDone = true;
    // put here init-once stuff
  }
  // we may have other params which need to be queried regularly
  bool updateMaps = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mMatcher.setTPCCorrMaps(&mTPCCorrMapsLoader);
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mMatcher.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
  }
}

void TOFMatcherSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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
}

void TOFMatcherSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);
  auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation;

  LOG(debug) << "isTrackSourceLoaded: TPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPC);
  LOG(debug) << "isTrackSourceLoaded: ITSTPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPC);
  LOG(debug) << "isTrackSourceLoaded: TPCTRD -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD);
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTRD -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD);

  bool isTPCused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPC);
  bool isITSTPCused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPC);
  bool isTPCTRDused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD);
  bool isITSTPCTRDused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD);
  uint32_t ss = o2::globaltracking::getSubSpec(mStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  mMatcher.setFIT(mUseFIT);

  mMatcher.setTS(creationTime);

  mMatcher.run(recoData, pc.services().get<o2::framework::TimingInfo>().firstTForbit);
  static pmr::vector<o2::MCCompLabel> dummyMCLab;

  if (isTPCused) {
    auto& mtcInfo = pc.outputs().make<std::vector<o2::dataformats::MatchInfoTOF>>(Output{o2::header::gDataOriginTOF, "MTC_TPC", ss, Lifetime::Timeframe});
    auto& mclabels = mUseMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{o2::header::gDataOriginTOF, "MCMTC_TPC", ss, Lifetime::Timeframe}) : dummyMCLab;
    auto& tracksTPCTOF = pc.outputs().make<std::vector<o2::dataformats::TrackTPCTOF>>(OutputRef{"tpctofTracks", ss});
    auto nmatch = mMatcher.getMatchedTrackVector(o2::dataformats::MatchInfoTOFReco::TrackType::TPC).size();
    LOG(debug) << (mDoTPCRefit ? "Refitting " : "Shifting Z for ") << nmatch << " matched TPC tracks with TOF time info";
    mMatcher.makeConstrainedTPCTracks(mtcInfo, mclabels, tracksTPCTOF);
  }

  if (isITSTPCused) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MTC_ITSTPC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector(o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPC));
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMTC_ITSTPC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector(o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPC));
    }
  }

  if (isTPCTRDused) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MTC_TPCTRD", ss, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector(o2::dataformats::MatchInfoTOFReco::TrackType::TPCTRD));
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMTC_TPCTRD", ss, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector(o2::dataformats::MatchInfoTOFReco::TrackType::TPCTRD));
    }
  }

  if (isITSTPCTRDused) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MTC_ITSTPCTRD", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector(o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPCTRD));
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMTC_ITSTPCTRD", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector(o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPCTRD));
    }
  }

  // TODO: TRD-matched tracks
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe}, mMatcher.getCalibVector());

  if (mPushMatchable) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_0", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(0));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_1", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(1));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_2", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(2));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_3", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(3));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_4", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(4));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_5", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(5));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_6", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(6));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_7", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(7));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_8", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(8));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_9", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(9));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_10", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(10));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_11", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(11));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_12", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(12));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_13", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(13));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_14", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(14));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_15", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(15));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_16", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(16));
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHABLES_17", 0, Lifetime::Timeframe}, mMatcher.getMatchedTracksPair(17));
  }

  mTimer.Stop();
}

void TOFMatcherSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTOFMatcherSpec(GID::mask_t src, bool useMC, bool useFIT, bool tpcRefit, bool strict, float extratolerancetrd, bool pushMatchable, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts, int nlanes)
{
  uint32_t ss = o2::globaltracking::getSubSpec(strict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  Options opts;
  auto dataRequest = std::make_shared<DataRequest>();
  if (strict) {
    dataRequest->setMatchingInputStrict();
  }
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::TOF), useMC);
  if (tpcRefit && src[GID::TPC]) {
    dataRequest->requestClusters(GID::getSourceMask(GID::TPC), false);
  }
  if (useFIT) {
    dataRequest->requestFT0RecPoints(false);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);
  std::vector<OutputSpec> outputs;
  if (GID::includesSource(GID::TPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MTC_TPC", ss, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"tpctofTracks"}, o2::header::gDataOriginTOF, "TOFTRACKS_TPC", ss, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTOF, "MCMTC_TPC", ss, Lifetime::Timeframe);
    }
  }
  if (GID::includesSource(GID::ITSTPC, src)) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MTC_ITSTPC", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTOF, "MCMTC_ITSTPC", 0, Lifetime::Timeframe);
    }
  }
  if (GID::includesSource(GID::ITSTPCTRD, src)) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MTC_ITSTPCTRD", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTOF, "MCMTC_ITSTPCTRD", 0, Lifetime::Timeframe);
    }
  }
  if (GID::includesSource(GID::TPCTRD, src)) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MTC_TPCTRD", ss, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTOF, "MCMTC_TPCTRD", ss, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe);

  if (pushMatchable) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_0", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_1", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_2", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_3", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_4", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_5", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_6", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_7", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_8", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_9", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_10", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_11", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_12", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_13", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_14", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_15", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_16", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHABLES_17", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tof-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFMatcherSpec>(dataRequest, ggRequest, useMC, useFIT, tpcRefit, strict, pushMatchable, nlanes)},
    opts};
}

} // namespace globaltracking
} // namespace o2
