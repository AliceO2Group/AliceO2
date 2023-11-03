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

/// @file   HMPMatcherSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "DetectorsBase/GRPGeomHelper.h"

// from Tracks
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"

// from HMPID
#include "DataFormatsHMP/Cluster.h"
#include "GlobalTracking/MatchHMP.h"
#include "GlobalTrackingWorkflow/HMPMatcherSpec.h"

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
// using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoHMP>;
using GID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class HMPMatcherSpec : public Task
{
 public:
  HMPMatcherSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~HMPMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC = false; // true
  bool mUseFIT = false;
  bool mDoTPCRefit = false;
  bool mStrict = false;
  MatchHMP mMatcher; ///< Cluster finder
  TStopwatch mTimer;
};

void HMPMatcherSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void HMPMatcherSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

void HMPMatcherSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);

  auto creationTime = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;

  // LOG(debug) << "isTrackSourceLoaded: TPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPC);  LOG(debug) << "isTrackSourceLoaded: ITSTPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPC);

  LOG(debug) << "isTrackSourceLoaded: TPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPC);

  //:
  LOG(debug) << "isTrackSourceLoaded: TPCTOF -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF);

  LOG(debug) << "isTrackSourceLoaded: ITSTPC -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPC);
  LOG(debug) << "isTrackSourceLoaded: TPCTRD -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD);
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTRD -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD);
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTOF -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF);
  LOG(debug) << "isTrackSourceLoaded: TPCTRDTOF -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF);
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTRDTOF -> " << recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF);

  bool isTPCused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPC);
  bool isITSTPCused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPC);
  bool isTPCTRDused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD);

  bool isTPCTOFused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF);

  bool isITSTPCTRDused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD);
  bool isITSTPCTOFused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF);
  bool isTPCTRDTOFused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF);
  bool isITSTPCTRDTOFused = recoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF);
  //  uint32_t ss = o2::globaltracking::getSubSpec(mStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);

  mMatcher.run(recoData);

  if (isITSTPCused || isTPCTRDused || isTPCTOFused || isITSTPCTRDused || isTPCTRDTOFused || isITSTPCTOFused || isITSTPCTRDTOFused) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "MATCHES", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector(o2::globaltracking::MatchHMP::trackType::CONSTR));
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "MCLABELS", 0, Lifetime::Timeframe}, mMatcher.getMatchedHMPLabelsVector(o2::globaltracking::MatchHMP::trackType::CONSTR));
    }
  }

  mTimer.Stop();
}

void HMPMatcherSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "HMP matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getHMPMatcherSpec(GID::mask_t src, bool useMC, float extratolerancetrd, float extratolerancetof)
{
  // uint32_t ss = o2::globaltracking::getSubSpec(strict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  auto dataRequest = std::make_shared<DataRequest>();
  /* if (strict) {
     dataRequest->setMatchingInputStrict();
   }*/

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::HMP), useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  std::vector<OutputSpec> outputs;

  if (GID::includesSource(GID::ITSTPC, src) || GID::includesSource(GID::TPCTRD, src) || GID::includesSource(GID::TPCTOF, src) || GID::includesSource(GID::ITSTPCTRD, src) || GID::includesSource(GID::ITSTPCTOF, src) || GID::includesSource(GID::TPCTRDTOF, src) || GID::includesSource(GID::ITSTPCTRDTOF, src)) {
    outputs.emplace_back(o2::header::gDataOriginHMP, "MATCHES", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginHMP, "MCLABELS", 0, Lifetime::Timeframe);
    }
  }

  return DataProcessorSpec{
    "hmp-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<HMPMatcherSpec>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
