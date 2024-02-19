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
#include "TPCCalibration/VDriftHelper.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TrackMCStudy.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "TPCBase/ParameterElectronics.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Steer/MCKinematicsReader.h"
#include "MathUtils/fit.h"
#include <map>
#include <array>
#include <utility>

// workflow to study relation of reco tracks to MCTruth
// o2-trackmc-study-workflow --device-verbosity 3 -b --run

namespace o2::trackstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using VTIndexV = std::pair<int, o2::dataformats::VtxTrackIndex>;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class TrackMCStudy : public Task
{
 public:
  TrackMCStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool checkMatching)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mCheckMatching(checkMatching) {}
  ~TrackMCStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);

 private:
  void prepareITSData(o2::globaltracking::RecoContainer& recoData);
  bool propagateToRefX(o2::track::TrackParCov& trcTPC, o2::track::TrackParCov& trcITS);
  void updateTimeDependentParams(ProcessingContext& pc);
  float getDCAYCut(float pt) const;

  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  int mVerbose = 0;
  bool mCheckMatching = false;
  float mITSTimeBiasMUS = 0.f;
  float mITSROFrameLengthMUS = 0.f; ///< ITS RO frame in mus
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds

  float mTPCDCAYCut = 2.;
  float mTPCDCAZCut = 2.;
  float mMinX = 6.;
  float mMaxEta = 0.8;
  float mMinPt = 0.03;
  int mMinTPCClusters = 10;
  std::string mDCAYFormula = "0.0105 + 0.0350 / pow(x, 1.1)";

  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
  std::vector<int> mITSROF;
  std::vector<TBracket> mITSROFBracket;

  static constexpr float MaxSnp = 0.9; // max snp of ITS or TPC track at xRef to be matched
};

void TrackMCStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mcReader.initFromDigitContext("collisioncontext.root");

  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("trackMCStudy.root", "recreate");
  mVerbose = ic.options().get<int>("device-verbosity");
  mTPCDCAYCut = ic.options().get<float>("max-tpc-dcay");
  mTPCDCAZCut = ic.options().get<float>("max-tpc-dcaz");
  mMinX = ic.options().get<float>("min-x-prop");
  mMaxEta = ic.options().get<float>("max-eta");
  mMinPt = ic.options().get<float>("min-pt");
  mMinTPCClusters = ic.options().get<int>("min-tpc-clusters");
  mDCAYFormula = ic.options().get<std::string>("dcay-vs-pt");
}

void TrackMCStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void TrackMCStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mTPCVDriftHelper.acknowledgeUpdate();
  }
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    const auto& alpParamsITS = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mITSROFrameLengthMUS = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParamsITS.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsITS.roFrameLengthTrig * 1.e-3;
    LOGP(info, "VertexTrackMatcher ITSROFrameLengthMUS:{}", mITSROFrameLengthMUS);

    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    mTPCTBinMUS = elParam.ZbinWidth;
  }
}

void TrackMCStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto prop = o2::base::Propagator::Instance();
  int nv = vtxRefs.size();
  float vdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth;                                                         // VDrift expressed in cm/TimeBin
  float itsBias = 0.5 * mITSROFrameLengthMUS + o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS time is supplied in \mus as beginning of ROF

  if (mCheckMatching) {
    prepareITSData(recoData);
  }

  auto getLowestPadrow = [&recoData](const o2::tpc::TrackTPC& trc) {
    if (recoData.inputsTPCclusters) {
      uint8_t clSect = 0, clRow = 0;
      uint32_t clIdx = 0;
      const auto clRefs = recoData.getTPCTracksClusterRefs();
      trc.getClusterReference(clRefs, trc.getNClusterReferences() - 1, clSect, clRow, clIdx);
      return int(clRow);
    }
    return -1;
  };

  std::map<o2::MCCompLabel, std::vector<VTIndexV>> MCTRMap;

  for (int iv = 0; iv < nv; iv++) {
    if (mVerbose > 0) {
      LOGP(info, "processing PV {} of {}", iv, nv);
    }
    const auto& vtref = vtxRefs[iv];
    for (int is = GTrackID::NSources; is--;) {
      DetID::mask_t dm = GTrackID::getSourceDetectorsMask(is);
      if (!mTracksSrc[is] || !recoData.isTrackSourceLoaded(is) || !(dm[DetID::ITS] || dm[DetID::TPC])) {
        continue;
      }
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        const auto& trc = recoData.getTrackParam(vid);
        if (trc.getPt() < mMinPt) {
          continue;
        }
        auto lbl = recoData.getTrackMCLabel(vid);
        if (lbl.isValid()) {
          lbl.setFakeFlag(false);
          auto& vvids = MCTRMap[lbl];
          if (vid.isAmbiguous() || vvids.empty()) { // do not repeat ambiguous tracks
            bool skip = false;
            for (const auto& va : vvids) {
              if (va.second == vid) {
                skip = true;
                break;
              }
            }
            if (skip) {
              continue;
            }
          }
          vvids.emplace_back(iv, vid);
        }
      }
    }
  }
  o2::track::TrackParCov dummyTrackParCov;
  o2::track::TrackPar dummyTrackPar;
  dummyTrackParCov.invalidate();
  dummyTrackPar.invalidate();

  const std::vector<o2::MCTrack>* mcTracks = nullptr;
  o2::MCCompLabel prevLbl;
  std::vector<o2::track::TrackParCov> recTracks;
  std::vector<VTIndex> recGIDs;
  std::vector<bool> recFakes;
  std::vector<int16_t> lowestPadrows;

  for (auto ent : MCTRMap) {
    auto lbl = ent.first;
    if (lbl.getEventID() != prevLbl.getEventID() || lbl.getSourceID() != prevLbl.getSourceID()) {
      mcTracks = &mcReader.getTracks(lbl.getSourceID(), lbl.getEventID());
      prevLbl = lbl;
    }
    const auto& mcPart = (*mcTracks)[lbl.getTrackID()];
    int pdg = mcPart.GetPdgCode(), pdgParent = 0;
    std::array<float, 3> xyz{(float)mcPart.GetStartVertexCoordinatesX(), (float)mcPart.GetStartVertexCoordinatesY(), (float)mcPart.GetStartVertexCoordinatesZ()};
    std::array<float, 3> pxyz{(float)mcPart.GetStartVertexMomentumX(), (float)mcPart.GetStartVertexMomentumY(), (float)mcPart.GetStartVertexMomentumZ()};
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(pdg);
    if (!pPDG) {
      LOGP(error, "Unknown particle {}, skip", pdg);
      continue;
    }
    o2::track::TrackPar mctrO2(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);

    bool primary = mcPart.isPrimary();
    auto parID = primary ? -1 : mcPart.getMotherTrackId();
    if (parID >= 0) {
      const auto& mcPartPar = (*mcTracks)[parID];
      pdgParent = mcPartPar.GetPdgCode();
    }
    auto& vgids = ent.second;
    // make sure the more global tracks come 1st
    if (vgids.size() > 1) {
      std::sort(vgids.begin(), vgids.end(), [](VTIndexV& lhs, VTIndexV& rhs) { return lhs.second.getSource() > rhs.second.getSource(); });
    }
    recTracks.clear();
    recGIDs.clear();
    recFakes.clear();
    lowestPadrows.clear();
    if (mVerbose > 1) {
      LOGP(info, "[{}] Lbl:{} PDG:{:+5d} (par: {:+5d}) | MC: {}", vgids.size(), lbl.asString(), pdg, pdgParent, mctrO2.asString());
    }
    bool itstpcMatch = false;
    int entITS = -1, entTPC = -1, entITSTPC = -1;
    for (size_t i = 0; i < vgids.size(); i++) {
      auto vid = vgids[i].second;
      auto lbl = recoData.getTrackMCLabel(vid);
      const auto& trc = recoData.getTrackParam(vid);
      if (mVerbose > 1) {
        LOGP(info, "       :{} {:22} | [{}] {}", lbl.asString(), vid.asString(), i, ((const o2::track::TrackPar&)trc).asString());
      }
      recTracks.push_back(trc);
      recGIDs.push_back(vid);
      recFakes.push_back(recoData.getTrackMCLabel(vid).isFake());
      if (mCheckMatching) {
        lowestPadrows.push_back(-1);
        auto msk = vid.getSourceDetectorsMask();
        if (msk[DetID::TPC]) {
          lowestPadrows.back() = getLowestPadrow(recoData.getTPCTrack(recoData.getTPCContributorGID(vid)));
        }
        if (msk[DetID::ITS] && msk[DetID::TPC]) {
          itstpcMatch = true;
          entITSTPC = i;
        } else {
          if (vid.getSource() == VTIndex::ITS) {
            entITS = i;
          } else {
            if (msk[DetID::TPC]) {
              entTPC = i;
            }
          }
        }
      }
    }
    (*mDBGOut) << "tracks"
               << "lbl=" << lbl
               << "mcTr=" << mctrO2
               << "pdg=" << pdg
               << "pdgPar=" << pdgParent
               << "recTr=" << recTracks
               << "recGID=" << recGIDs
               << "recFake=" << recFakes;
    if (mCheckMatching) {
      (*mDBGOut) << "tracks"
                 << "lowestPadRow=" << lowestPadrows;
    }
    (*mDBGOut) << "tracks"
               << "\n";

    // special ITS-TPC matching failure output
    while (mCheckMatching) {
      if (!itstpcMatch && entITS > -1 && entTPC > -1) { // ITS and TPC were found but matching failed
        auto vidITS = vgids[entITS].second;
        auto vidTPC = recoData.getTPCContributorGID(vgids[entTPC].second); // might be TPC match to outer detector, extract TPC
        auto trcTPC = recoData.getTrackParam(vidTPC);
        auto trcITS = recoData.getTrackParam(vidITS);
        if (!propagateToRefX(trcTPC, trcITS)) {
          break;
        }
        const auto& trcTPCOrig = recoData.getTPCTrack(vidTPC);
        const auto& trcITSOrig = recoData.getITSTrack(vidITS);
        int lowestTPCRow = lowestPadrows[entTPC];
        float tpcT0 = trcTPCOrig.getTime0(), tF = trcTPCOrig.getDeltaTFwd(), tB = trcTPCOrig.getDeltaTBwd();
        TBracket tpcBr((tpcT0 - tB) * mTPCTBinMUS, (tpcT0 + tF) * mTPCTBinMUS);

        (*mDBGOut) << "failMatch"
                   << "mcTr=" << mctrO2
                   << "pdg=" << pdg
                   << "pdgPar=" << pdgParent
                   << "labelITS=" << recoData.getTrackMCLabel(vidITS)
                   << "labelTPC=" << recoData.getTrackMCLabel(vidTPC)
                   << "gidITS=" << vidITS
                   << "gidTPC=" << vidTPC
                   << "itsBracket=" << mITSROFBracket[mITSROF[vidITS.getIndex()]]
                   << "tpcBracket=" << tpcBr
                   << "itsRef=" << trcITS
                   << "tpcRef=" << trcTPC
                   << "itsOrig=" << trcITSOrig
                   << "tpcOrig=" << trcTPCOrig
                   << "tpcLowestRow=" << lowestTPCRow
                   << "\n";
      } else if (itstpcMatch) { // match was found
        auto contribIDs = recoData.getSingleDetectorRefs(vgids[entITSTPC].second);
        auto vidMatch = contribIDs[VTIndex::ITSTPC];
        auto vidTPC = contribIDs[VTIndex::TPC];
        auto vidITS = contribIDs[VTIndex::ITSAB].isSourceSet() ? contribIDs[VTIndex::ITSAB] : contribIDs[VTIndex::ITS];
        const auto& trcTPCOrig = recoData.getTPCTrack(vidTPC);
        o2::MCCompLabel itsLb;
        int nITScl = 0;
        if (vidITS.getSource() == VTIndex::ITS) {
          itsLb = recoData.getTrackMCLabel(vidITS);
          nITScl = recoData.getITSTrack(vidITS).getNClusters();
        } else {
          itsLb = recoData.getITSABMCLabels()[vidITS];
          nITScl = recoData.getITSABRefs()[vidITS].getNClusters();
        }
        int lowestTPCRow = lowestPadrows[entITSTPC];
        const auto& trackITSTPC = recoData.getTPCITSTrack(vidMatch);
        float timeTB = trackITSTPC.getTimeMUS().getTimeStamp() / o2::constants::lhc::LHCBunchSpacingMUS / 8; // ITS-TPC time in TPC timebins

        (*mDBGOut) << "match"
                   << "mcTr=" << mctrO2
                   << "pdg=" << pdg
                   << "pdgPar=" << pdgParent
                   << "labelMatch=" << recoData.getTrackMCLabel(vidMatch)
                   << "labelTPC=" << recoData.getTrackMCLabel(vidTPC)
                   << "labelITS=" << itsLb
                   << "gidTPC=" << vidTPC
                   << "gidITS=" << vidITS
                   << "tpcOrig=" << trcTPCOrig
                   << "itstpc=" << ((o2::track::TrackParCov&)trackITSTPC)
                   << "timeTB=" << timeTB
                   << "tpcLowestRow=" << lowestTPCRow
                   << "\n";
      }
      break;
    }
  }
}

bool TrackMCStudy::propagateToRefX(o2::track::TrackParCov& trcTPC, o2::track::TrackParCov& trcITS)
{
  bool refReached = false;
  constexpr float TgHalfSector = 0.17632698f;
  const auto& par = o2::globaltracking::MatchTPCITSParams::Instance();
  int trialsLeft = 2;
  while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trcTPC, par.XMatchingRef, MaxSnp, 2., par.matCorr)) {
    if (refReached) {
      break;
    }
    // make sure the track is indeed within the sector defined by alpha
    if (fabs(trcTPC.getY()) < par.XMatchingRef * TgHalfSector) {
      refReached = true;
      break; // ok, within
    }
    if (!trialsLeft--) {
      break;
    }
    auto alphaNew = o2::math_utils::angle2Alpha(trcTPC.getPhiPos());
    if (!trcTPC.rotate(alphaNew) != 0) {
      break; // failed (RS: check effect on matching tracks to neighbouring sector)
    }
  }
  if (!refReached) {
    return false;
  }
  refReached = false;
  float alp = trcTPC.getAlpha();
  if (!trcITS.rotate(alp) != 0 || !o2::base::Propagator::Instance()->PropagateToXBxByBz(trcITS, par.XMatchingRef, MaxSnp, 2., par.matCorr)) {
    return false;
  }
  return true;
}

void TrackMCStudy::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void TrackMCStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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
    mITSTimeBiasMUS = par.roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
    mITSROFrameLengthMUS = par.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
    return;
  }
}

//_____________________________________________________
void TrackMCStudy::prepareITSData(o2::globaltracking::RecoContainer& recoData)
{
  auto ITSTracksArray = recoData.getITSTracks();
  auto ITSTrackROFRec = recoData.getITSTracksROFRecords();
  int nROFs = ITSTrackROFRec.size();
  mITSROF.clear();
  mITSROFBracket.clear();
  mITSROF.reserve(ITSTracksArray.size());
  mITSROFBracket.reserve(ITSTracksArray.size());
  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = ITSTrackROFRec[irof];
    long nBC = rofRec.getBCData().differenceInBC(recoData.startIR);
    float tMin = nBC * o2::constants::lhc::LHCBunchSpacingMUS + mITSTimeBiasMUS;
    float tMax = tMin + mITSROFrameLengthMUS;
    mITSROFBracket.emplace_back(tMin, tMax);
    for (int it = 0; it < rofRec.getNEntries(); it++) {
      mITSROF.push_back(irof);
    }
  }
}

float TrackMCStudy::getDCAYCut(float pt) const
{
  static TF1 fun("dcayvspt", mDCAYFormula.c_str(), 0, 20);
  return fun.Eval(pt);
}

DataProcessorSpec getTrackMCStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool checkMatching)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  bool useMC = true;
  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "track-mc-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackMCStudy>(dataRequest, ggRequest, srcTracks, checkMatching)},
    Options{
      {"device-verbosity", VariantType::Int, 0, {"Verbosity level"}},
      {"dcay-vs-pt", VariantType::String, "0.0105 + 0.0350 / pow(x, 1.1)", {"Formula for global tracks DCAy vs pT cut"}},
      {"min-tpc-clusters", VariantType::Int, 60, {"Cut on TPC clusters"}},
      {"max-tpc-dcay", VariantType::Float, 2.f, {"Cut on TPC dcaY"}},
      {"max-tpc-dcaz", VariantType::Float, 2.f, {"Cut on TPC dcaZ"}},
      {"max-eta", VariantType::Float, 1.5f, {"Cut on track eta"}},
      {"min-pt", VariantType::Float, 0.02f, {"Cut on track pT"}},
      {"min-x-prop", VariantType::Float, 6.f, {"track should be propagated to this X at least"}},
    }};
}

} // namespace o2::trackstudy
