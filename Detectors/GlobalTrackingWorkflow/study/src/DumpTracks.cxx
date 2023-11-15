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

#include <vector>
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TrackingStudy.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterDetector.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "TPCCalibration/VDriftHelper.h"
#include "DetectorsVertexing/PVertexerParams.h"

namespace o2::trackstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class DumpTracksSpec : public Task
{
 public:
  DumpTracksSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC) {}
  ~DumpTracksSpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false}; ///< MC flag
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  GTrackID::mask_t mTracksSrc{};

  std::vector<long> mGlobalBC;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  int mBCToler = 0; // tolerance in BC for globalBC selections
  int mVerbose = 0;
  float mITSROFrameLengthMUS = 0; ///< ITS RO frame in mus
  float mMFTROFrameLengthMUS = 0; ///< MFT RO frame in mus
  float mMaxTPCDriftTimeMUS = 0;
  float mTPCTDriftOffset = 0.f;
  float mTPCBin2MUS = 0;
};

void DumpTracksSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("trackStudy.root", "recreate");

  mBCToler = ic.options().get<int>("bc-margin");
  mVerbose = ic.options().get<int>("dump-verbosity");
  auto bcstr = ic.options().get<std::string>("sel-bc");
  auto bctok = o2::utils::Str::tokenize(bcstr, ',');
  if (bctok.empty()) {
    LOG(error) << "empty BC list is provided " << bcstr;
  }
  for (auto& bcs : bctok) {
    try {
      long bcglo = std::stol(bcs);
      mGlobalBC.push_back(bcglo);
      LOGP(info, "adding {} to global BCs to dump", bcglo);
    } catch (...) {
      LOGP(fatal, "failed to extract global BC from {}", bcstr);
    }
  }
}

void DumpTracksSpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void DumpTracksSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // put here init-once stuff
    const auto& alpParamsITS = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mITSROFrameLengthMUS = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParamsITS.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsITS.roFrameLengthTrig * 1.e-3;
    const auto& alpParamsMFT = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    mMFTROFrameLengthMUS = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::MFT) ? alpParamsMFT.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsMFT.roFrameLengthTrig * 1.e-3;
    LOGP(info, "VertexTrackMatcher ITSROFrameLengthMUS:{} MFTROFrameLengthMUS:{}", mITSROFrameLengthMUS, mMFTROFrameLengthMUS);
  }
  // we may have other params which need to be queried regularly
  // VDrift may change from time to time
  if (mTPCVDriftHelper.isUpdated()) {
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    auto& detParam = o2::tpc::ParameterDetector::Instance();
    mTPCBin2MUS = elParam.ZbinWidth;
    auto& vd = mTPCVDriftHelper.getVDriftObject();
    mMaxTPCDriftTimeMUS = detParam.TPClength / (vd.refVDrift * vd.corrFact);
    mTPCTDriftOffset = vd.getTimeOffset();
  }
}

void DumpTracksSpec::process(o2::globaltracking::RecoContainer& recoData)
{
  std::vector<TBracket> selBCTF;
  auto irMin = recoData.startIR;
  auto irMax = irMin + o2::base::GRPGeomHelper::instance().getNHBFPerTF() * o2::constants::lhc::LHCMaxBunches;
  float tBCErr = mBCToler * o2::constants::lhc::LHCBunchSpacingMUS;
  LOGP(info, "TF dump for {}:{}", irMin.asString(), irMax.asString());

  for (const auto& bc : mGlobalBC) {
    if (bc >= irMin.toLong() && bc < irMax.toLong()) {
      o2::InteractionRecord bcir(bc % o2::constants::lhc::LHCMaxBunches, bc / o2::constants::lhc::LHCMaxBunches);
      float t = (bc - irMin.toLong()) * o2::constants::lhc::LHCBunchSpacingMUS;
      LOGP(info, "Selected BC {}({})  -> {}({}) : {}({}) mus", bc, bcir.asString(),
           t - tBCErr, (bcir - mBCToler).asString(),
           t + tBCErr, (bcir + mBCToler).asString());
      selBCTF.emplace_back(t - tBCErr, t + tBCErr);
    }
  }
  if (selBCTF.empty()) {
    LOGP(info, "No selections for {}:{}", irMin.asString(), irMax.asString());
    return;
  }

  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs

  float itsBias = 0.5 * mITSROFrameLengthMUS + o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS time is supplied in \mus as beginning of ROF
  float mftBias = 0.5 * mMFTROFrameLengthMUS + o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance().roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // MFT time is supplied in \mus as beginning of ROF

  GTrackID::Source prevSrc = GTrackID::Source(-1);
  auto creator = [this, &selBCTF, itsBias, mftBias, &recoData, &prevSrc](auto& _tr, GTrackID _origID, float t0, float terr) {
    const auto& PVParams = o2::vertexing::PVertexerParams::Instance();
    if constexpr (isTPCTrack<decltype(_tr)>()) {
      // unconstrained TPC track, with t0 = TrackTPC.getTime0+0.5*(DeltaFwd-DeltaBwd) and terr = 0.5*(DeltaFwd+DeltaBwd) in TimeBins
      t0 *= this->mTPCBin2MUS;
      t0 -= this->mTPCTDriftOffset;
      terr *= this->mTPCBin2MUS;
    } else if constexpr (isITSTrack<decltype(_tr)>()) {
      t0 += itsBias;
      terr *= this->mITSROFrameLengthMUS;               // error is supplied as a half-ROF duration, convert to \mus
    } else if constexpr (isMFTTrack<decltype(_tr)>()) { // Same for MFT
      t0 += mftBias;
      terr *= this->mMFTROFrameLengthMUS;
    } else if constexpr (!(isMCHTrack<decltype(_tr)>() || isGlobalFwdTrack<decltype(_tr)>())) {
      // for all other tracks the time is in \mus with gaussian error
      terr *= PVParams.nSigmaTimeTrack; // gaussian errors must be scaled by requested n-sigma
    }

    terr += PVParams.timeMarginTrackTime;
    TBracket tb{t0 - terr, t0 + terr};
    for (const auto& stb : selBCTF) {
      if (tb.isOutside(stb)) {
        return false;
      }
    }
    auto curSrc = GTrackID::Source(_origID.getSource());
    if (prevSrc != curSrc) {
      prevSrc = curSrc;
      LOGP(info, "Dumping {} tracks", GTrackID::getSourceName(prevSrc));
    }

    std::string outs;
    if constexpr (isGlobalFwdTrack<decltype(_tr)>() || isMFTTrack<decltype(_tr)>() || isMCHTrack<decltype(_tr)>() || isMIDTrack<decltype(_tr)>()) {
      outs = fmt::format("{:>15} {:8.3f}/{:5.3f} -> ", _origID.asString(), t0, terr);
    } else {
      outs = fmt::format("{:>15} {:8.3f}/{:5.3f} |{}| -> ", _origID.asString(), t0, terr, ((o2::track::TrackPar)_tr).asString());
    }

    // contributions
    auto refs = recoData.getSingleDetectorRefs(_origID);
    for (auto r : refs) {
      if (r.isSourceSet()) {
        outs += fmt::format(" {}", r.asString());
      }
    }
    LOG(info) << outs;
    return false; // allow redundancy
  };

  recoData.createTracksVariadic(creator);

  // print matching vertices
  int pvcnt = 0;
  for (const auto& pv : pvvec) {
    TBracket pbv{pv.getIRMin().differenceInBCMUS(irMin), pv.getIRMax().differenceInBCMUS(irMin)};
    for (const auto& stb : selBCTF) {
      if (!stb.isOutside(pbv)) {
        LOG(info) << "#" << pvcnt << " " << pv;
        LOG(info) << "References: " << vtxRefs[pvcnt];
        for (int is = 0; is < VTIndex::NSources; is++) {
          int ncontrib = 0, nambig = 0;
          int idMin = vtxRefs[pvcnt].getFirstEntryOfSource(is), idMax = idMin + vtxRefs[pvcnt].getEntriesOfSource(is);
          for (int i = idMin; i < idMax; i++) {
            if (trackIndex[i].isPVContributor()) {
              ncontrib++;
            } else if (trackIndex[i].isAmbiguous()) {
              nambig++;
            }
          }
          if (vtxRefs[pvcnt].getEntriesOfSource(is)) {
            LOGP(info, "{} : total attached: {}, contributors: {}, ambiguous: {}", VTIndex::getSourceName(is), vtxRefs[pvcnt].getEntriesOfSource(is), ncontrib, nambig);
          }
          if (mVerbose < 2) {
            continue;
          }
          std::string trIDs;
          int cntT = 0;
          for (int i = idMin; i < idMax; i++) {
            if (mVerbose > 2 || trackIndex[i].isPVContributor()) {
              trIDs += trackIndex[i].asString() + " ";
              if (!((++cntT) % 15)) {
                LOG(info) << trIDs;
                trIDs = "";
              }
            }
          }
          if (!trIDs.empty()) {
            LOG(info) << trIDs;
          }
        }
      }
      pvcnt++;
    }
  }
}

void DumpTracksSpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void DumpTracksSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "ALPIDEPARAM", 0)) {
    LOG(info) << "MFT Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    par.printKeyValues();
    return;
  }
}

DataProcessorSpec getDumpTracksSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);

  return DataProcessorSpec{
    "tracks-dump",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DumpTracksSpec>(dataRequest, ggRequest, srcTracks, useMC)},
    Options{
      {"sel-bc", VariantType::String, "", {"Dump tracks compatible with global BC list"}},
      {"bc-margin", VariantType::Int, 0, {"Apply margin in BC to selected global BCs list"}},
      {"dump-verbosity", VariantType::Int, 0, {"Dump verbosity level"}}}};
}

} // namespace o2::trackstudy
