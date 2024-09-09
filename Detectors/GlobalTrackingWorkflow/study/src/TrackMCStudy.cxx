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
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
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
#include "GlobalTrackingStudy/TrackMCStudyConfig.h"
#include "GlobalTrackingStudy/TrackMCStudyTypes.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "TPCBase/ParameterElectronics.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Steer/MCKinematicsReader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "GPUO2InterfaceRefit.h"
#include "GPUParam.h"
#include "GPUParam.inc"
#include "MathUtils/fit.h"
#include <map>
#include <unordered_map>
#include <array>
#include <utility>
#include <gsl/span>

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
  TrackMCStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src) {}
  ~TrackMCStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(const o2::globaltracking::RecoContainer& recoData);

 private:
  void loadTPCOccMap(const o2::globaltracking::RecoContainer& recoData);
  void fillMCClusterInfo(const o2::globaltracking::RecoContainer& recoData);
  void prepareITSData(const o2::globaltracking::RecoContainer& recoData);
  bool processMCParticle(int src, int ev, int trid);
  bool addMCParticle(const MCTrack& mctr, const o2::MCCompLabel& lb, TParticlePDG* pPDG = nullptr);
  bool acceptMCCharged(const MCTrack& tr, const o2::MCCompLabel& lb, int followDec = -1);
  bool propagateToRefX(o2::track::TrackParCov& trcTPC, o2::track::TrackParCov& trcITS);
  void updateTimeDependentParams(ProcessingContext& pc);
  float getDCAYCut(float pt) const;

  gsl::span<const MCTrack> mCurrMCTracks;
  TVector3 mCurrMCVertex;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::vector<float> mTBinClOcc; ///< TPC occupancy histo: i-th entry is the integrated occupancy for ~1 orbit starting from the TB = i*mNTPCOccBinLength
  std::vector<long> mIntBC;      ///< interaction global BC wrt TF start
  std::vector<float> mTPCOcc;    ///< TPC occupancy for this interaction time
  int mNTPCOccBinLength = 0;     ///< TPC occ. histo bin length in TBs
  float mNTPCOccBinLengthInv;
  int mVerbose = 0;
  float mITSTimeBiasMUS = 0.f;
  float mITSROFrameLengthMUS = 0.f; ///< ITS RO frame in mus
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds

  float mTPCDCAYCut = 2.;
  float mTPCDCAZCut = 2.;
  float mMinX = 6.;
  int mMinTPCClusters = 10;
  int mNCheckDecays = 0;
  std::string mDCAYFormula = "0.0105 + 0.0350 / pow(x, 1.1)";

  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
  std::vector<int> mITSROF;
  std::vector<TBracket> mITSROFBracket;
  std::vector<o2::MCCompLabel> mDecProdLblPool; // labels of decay products to watch, added to MC map
  struct DecayRef {
    o2::MCCompLabel mother{};
    o2::track::TrackPar parent{};
    int pdg = 0;
    int daughterFirst = -1;
    int daughterLast = -1;
  };
  std::vector<std::vector<DecayRef>> mDecaysMaps; // for every parent particle to watch store its label and entries of 1st/last decay product labels in mDecProdLblPool
  std::unordered_map<o2::MCCompLabel, TrackFamily> mSelMCTracks;

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
  mMinTPCClusters = ic.options().get<int>("min-tpc-clusters");
  mDCAYFormula = ic.options().get<std::string>("dcay-vs-pt");

  const auto& params = o2::trackstudy::TrackMCStudyConfig::Instance();
  for (int id = 0; id < sizeof(params.decayPDG) / sizeof(int); id++) {
    if (params.decayPDG[id] < 0) {
      break;
    }
    mNCheckDecays++;
  }
  mDecaysMaps.resize(mNCheckDecays);
}

void TrackMCStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  for (int i = 0; i < mNCheckDecays; i++) {
    mDecaysMaps[i].clear();
  }
  mDecProdLblPool.clear();
  mCurrMCTracks = {};

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

void TrackMCStudy::process(const o2::globaltracking::RecoContainer& recoData)
{
  const auto& params = o2::trackstudy::TrackMCStudyConfig::Instance();
  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto prop = o2::base::Propagator::Instance();
  int nv = vtxRefs.size();
  float vdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth;                                                         // VDrift expressed in cm/TimeBin
  float itsBias = 0.5 * mITSROFrameLengthMUS + o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS time is supplied in \mus as beginning of ROF

  prepareITSData(recoData);
  loadTPCOccMap(recoData);
  auto getITSPatt = [&](GTrackID gid, uint8_t& ncl) {
    int8_t patt = 0;
    if (gid.getSource() == VTIndex::ITSAB) {
      const auto& itsTrf = recoData.getITSABRefs()[gid];
      ncl = itsTrf.getNClusters();
      for (int il = 0; il < 7; il++) {
        if (itsTrf.hasHitOnLayer(il)) {
          patt |= 0x1 << il;
        }
      }
      patt |= 0x1 << 7;
    } else {
      const auto& itsTr = recoData.getITSTrack(gid);
      for (int il = 0; il < 7; il++) {
        if (itsTr.hasHitOnLayer(il)) {
          patt |= 0x1 << il;
          ncl++;
        }
      }
    }
    return patt;
  };

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

  const auto* digconst = mcReader.getDigitizationContext();
  const auto& mcEvRecords = digconst->getEventRecords(false);
  for (const auto& mcIR : mcEvRecords) {
    long tbc = mcIR.differenceInBC(recoData.startIR);
    mIntBC.push_back(tbc);
    int occBin = tbc / 8 * mNTPCOccBinLengthInv;
    mTPCOcc.push_back(occBin < 0 ? mTBinClOcc[0] : (occBin >= mTBinClOcc.size() ? mTBinClOcc.back() : mTBinClOcc[occBin]));
  }

  // collect interesting MC particle (tracks and parents)
  int curSrcMC = 0, curEvMC = 0;
  for (curSrcMC = 0; curSrcMC < (int)mcReader.getNSources(); curSrcMC++) {
    if (mVerbose > 1) {
      LOGP(info, "Source {}", curSrcMC);
    }
    for (curEvMC = 0; curEvMC < (int)mcReader.getNEvents(curSrcMC); curEvMC++) {
      if (mVerbose > 1) {
        LOGP(info, "Event {}", curEvMC);
      }
      const auto& mt = mcReader.getTracks(curSrcMC, curEvMC);
      mCurrMCTracks = gsl::span<const MCTrack>(mt.data(), mt.size());
      const_cast<o2::dataformats::MCEventHeader&>(mcReader.getMCEventHeader(curSrcMC, curEvMC)).GetVertex(mCurrMCVertex);
      for (int itr = 0; itr < mCurrMCTracks.size(); itr++) {
        processMCParticle(curSrcMC, curEvMC, itr);
      }
    }
  }
  if (mVerbose > 0) {
    for (int id = 0; id < mNCheckDecays; id++) {
      LOGP(info, "Decay PDG={} : {} entries", params.decayPDG[id], mDecaysMaps[id].size());
    }
  }

  // add reconstruction info to MC particles. If MC particle was not selected before but was reconstrected, account MC info
  for (int iv = 0; iv < nv; iv++) {
    if (mVerbose > 1) {
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
        if (trc.getPt() < params.minPt || std::abs(trc.getTgl()) > params.maxTgl) {
          continue;
        }
        auto lbl = recoData.getTrackMCLabel(vid);
        if (lbl.isValid()) {
          lbl.setFakeFlag(false);
          auto entry = mSelMCTracks.find(lbl);
          if (entry == mSelMCTracks.end()) { // add the track which was not added during MC scan
            if (lbl.getSourceID() != curSrcMC || lbl.getEventID() != curEvMC) {
              curSrcMC = lbl.getSourceID();
              curEvMC = lbl.getEventID();
              const auto& mt = mcReader.getTracks(curSrcMC, curEvMC);
              mCurrMCTracks = gsl::span<const MCTrack>(mt.data(), mt.size());
              const_cast<o2::dataformats::MCEventHeader&>(mcReader.getMCEventHeader(curSrcMC, curEvMC)).GetVertex(mCurrMCVertex);
            }
            if (!acceptMCCharged(mCurrMCTracks[lbl.getTrackID()], lbl)) {
              continue;
            }
            entry = mSelMCTracks.find(lbl);
          }
          auto& trackFamily = entry->second;
          if (vid.isAmbiguous()) { // do not repeat ambiguous tracks
            if (trackFamily.contains(vid)) {
              continue;
            }
          }
          auto& trf = trackFamily.recTracks.emplace_back();
          trf.gid = vid; //  account(iv, vid);
          if (mVerbose > 1) {
            LOGP(info, "Matched rec track {} to MC track {}", vid.asString(), entry->first.asString());
          }
        } else {
          continue;
        }
      }
    }
  }

  // collect ITS/TPC cluster info for selected MC particles
  fillMCClusterInfo(recoData);

  LOGP(info, "collected {} MC tracks", mSelMCTracks.size());
  int mcnt = 0;
  for (auto& entry : mSelMCTracks) {
    auto& trackFam = entry.second;
    auto& tracks = trackFam.recTracks;
    mcnt++;
    if (tracks.empty()) {
      continue;
    }
    if (mVerbose > 1) {
      LOGP(info, "Processing MC track#{} {} -> {} reconstructed tracks", mcnt - 1, entry.first.asString(), tracks.size());
    }
    // sort according to the gid complexity (in principle, should be already sorted due to the backwards loop over NSources above
    std::sort(tracks.begin(), tracks.end(), [](RecTrack& lhs, RecTrack& rhs) {
      const auto mskL = lhs.gid.getSourceDetectorsMask();
      const auto mskR = rhs.gid.getSourceDetectorsMask();
      bool itstpcL = mskL[DetID::ITS] && mskL[DetID::TPC], itstpcR = mskR[DetID::ITS] && mskR[DetID::TPC];
      if (itstpcL && !itstpcR) { // to avoid TPC/TRD or TPC/TOF shadowing ITS/TPC
        return true;
      }
      return lhs.gid.getSource() > rhs.gid.getSource();
    });
    // fill track params
    int tcnt = 0;
    for (auto& tref : tracks) {
      if (tref.gid.isSourceSet()) {
        tref.track = recoData.getTrackParam(tref.gid);
        tref.isFake = recoData.getTrackMCLabel(tref.gid).isFake();
        auto msk = tref.gid.getSourceDetectorsMask();
        if (msk[DetID::ITS]) {
          auto gidITS = recoData.getITSContributorGID(tref.gid);
          tref.pattITS = getITSPatt(gidITS, tref.nClITS);
          if (gidITS.getSource() == VTIndex::ITS && trackFam.entITS < 0) { // has ITS track rather than AB tracklet
            trackFam.entITS = tcnt;
          }
          if (msk[DetID::TPC] && trackFam.entITSTPC < 0) { // has both ITS and TPC contribution
            trackFam.entITSTPC = tcnt;
          }
        }
        if (msk[DetID::TPC]) {
          if (trackFam.entTPC < 0) {
            trackFam.entTPC = tcnt;
          }
          auto gidTPC = recoData.getTPCContributorGID(tref.gid);
          const auto& trtpc = recoData.getTPCTrack(gidTPC);
          tref.nClTPC = trtpc.getNClusters();
          tref.lowestPadRow = getLowestPadrow(recoData.getTPCTrack(gidTPC));
        }
      } else {
        LOGP(info, "Invalid entry {} of {} getTrackMCLabel {}", tcnt, tracks.size(), tref.gid.asString());
      }
      tcnt++;
    }
    if (trackFam.entITSTPC < 0 && trackFam.entITS > -1 && trackFam.entTPC > -1) { // ITS and TPC were found but matching failed
      auto vidITS = tracks[trackFam.entITS].gid;
      auto vidTPC = tracks[trackFam.entTPC].gid;
      auto trcTPC = recoData.getTrackParam(vidTPC);
      auto trcITS = recoData.getTrackParamOut(vidITS);
      if (!propagateToRefX(trcTPC, trcITS)) {
        // break;
      }
    }
  }

  // single tracks
  for (auto& entry : mSelMCTracks) {
    auto& trackFam = entry.second;
    (*mDBGOut) << "tracks" << "tr=" << trackFam << "\n";
  }
  // decays
  std::vector<TrackFamily> decFam;
  for (int id = 0; id < mNCheckDecays; id++) {
    std::string decTreeName = fmt::format("dec{}", params.decayPDG[id]);
    for (const auto& dec : mDecaysMaps[id]) {
      decFam.clear();
      for (int idd = dec.daughterFirst; idd <= dec.daughterLast; idd++) {
        auto dtLbl = mDecProdLblPool[idd]; // daughter MC label
        const auto& dtFamily = mSelMCTracks[dtLbl];
        if (dtFamily.mcTrackInfo.pdgParent != dec.pdg) {
          LOGP(error, "{}-th decay (pdg={}): {} in {}:{} range   refers to MC track with pdgParent = {}", id, params.decayPDG[id], idd, dec.daughterFirst, dec.daughterLast, dtFamily.mcTrackInfo.pdgParent);
          continue;
        }
        decFam.push_back(dtFamily);
      }
      (*mDBGOut) << decTreeName.c_str() << "pdgPar=" << dec.pdg << "trPar=" << dec.parent << "prod=" << decFam << "\n";
    }
  }
}

void TrackMCStudy::fillMCClusterInfo(const o2::globaltracking::RecoContainer& recoData)
{
  // TPC clusters info
  const auto& TPCClusterIdxStruct = recoData.inputsTPCclusters->clusterIndex;
  const auto* TPCClMClab = recoData.inputsTPCclusters->clusterIndex.clustersMCTruth;
  for (int sector = 0; sector < 36; sector++) {
    for (int row = 0; row < 152; row++) {
      unsigned int offs = TPCClusterIdxStruct.clusterOffset[sector][row];
      for (unsigned int icl0 = 0; icl0 < TPCClusterIdxStruct.nClusters[sector][row]; icl0++) {
        const auto labels = TPCClMClab->getLabels(icl0 + offs);
        for (const auto& lbl : labels) {
          auto entry = mSelMCTracks.find(lbl);
          if (entry == mSelMCTracks.end()) { // not selected
            continue;
          }
          auto& mctr = entry->second.mcTrackInfo;
          mctr.nTPCCl++;
          if (mctr.maxTPCRow < row) {
            mctr.maxTPCRow = row;
          }
          if (mctr.minTPCRow > row) {
            mctr.minTPCRow = row;
          }
          if (labels.size() > 1) {
            mctr.nTPCClShared++;
          }
        }
      }
    }
  }
  // fill ITS cluster info
  const auto* mcITSClusters = recoData.getITSClustersMCLabels();
  const auto& ITSClusters = recoData.getITSClusters();
  for (unsigned int icl = 0; icl < ITSClusters.size(); icl++) {
    const auto labels = mcITSClusters->getLabels(icl);
    for (const auto& lbl : labels) {
      auto entry = mSelMCTracks.find(lbl);
      if (entry == mSelMCTracks.end()) { // not selected
        continue;
      }
      auto& mctr = entry->second.mcTrackInfo;
      mctr.nITSCl++;
      mctr.pattITSCl |= 0x1 << o2::itsmft::ChipMappingITS::getLayer(ITSClusters[icl].getChipID());
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
void TrackMCStudy::prepareITSData(const o2::globaltracking::RecoContainer& recoData)
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

bool TrackMCStudy::processMCParticle(int src, int ev, int trid)
{
  const auto& mcPart = mCurrMCTracks[trid];
  int pdg = mcPart.GetPdgCode();
  bool res = false;
  while (true) {
    auto lbl = o2::MCCompLabel(trid, ev, src);
    int decay = -1; // is this decay to watch?
    const auto& params = o2::trackstudy::TrackMCStudyConfig::Instance();
    if (mcPart.T() < params.decayMotherMaxT) {
      for (int id = 0; id < mNCheckDecays; id++) {
        if (params.decayPDG[id] == std::abs(pdg)) {
          decay = id;
          break;
        }
      }
      if (decay >= 0) {                                                                      // check if decay and kinematics is acceptable
        int idd0 = mcPart.getFirstDaughterTrackId(), idd1 = mcPart.getLastDaughterTrackId(); // we want only charged and trackable daughters
        int dtStart = mDecProdLblPool.size(), dtEnd = -1;
        if (idd0 < 0) {
          break;
        }
        for (int idd = idd0; idd <= idd1; idd++) {
          const auto& product = mCurrMCTracks[idd];
          auto lbld = o2::MCCompLabel(idd, ev, src);
          if (!acceptMCCharged(product, lbld, decay)) {
            decay = -1; // discard decay
            mDecProdLblPool.resize(dtStart);
            break;
          }
          mDecProdLblPool.push_back(lbld); // register prong entry and label
        }
        if (decay >= 0) {
          // account decay
          dtEnd = mDecProdLblPool.size() - 1;
          std::array<float, 3> xyz{(float)mcPart.GetStartVertexCoordinatesX(), (float)mcPart.GetStartVertexCoordinatesY(), (float)mcPart.GetStartVertexCoordinatesZ()};
          std::array<float, 3> pxyz{(float)mcPart.GetStartVertexMomentumX(), (float)mcPart.GetStartVertexMomentumY(), (float)mcPart.GetStartVertexMomentumZ()};
          mDecaysMaps[decay].emplace_back(DecayRef{lbl,
                                                   o2::track::TrackPar(xyz, pxyz, TMath::Nint(O2DatabasePDG::Instance()->GetParticle(mcPart.GetPdgCode())->Charge() / 3), false),
                                                   mcPart.GetPdgCode(), dtStart, dtEnd});
          if (mVerbose > 1) {
            LOGP(info, "Adding MC parent pdg={} {}, with prongs in {}:{} range", pdg, lbl.asString(), dtStart, dtEnd);
          }
          res = true; // Accept!
        }
        break;
      }
    }
    // check if this is a charged which should be processed but was not accounted as a decay product
    if (mSelMCTracks.find(lbl) == mSelMCTracks.end()) {
      res = acceptMCCharged(mcPart, lbl);
    }
    break;
  }
  return res;
}

bool TrackMCStudy::acceptMCCharged(const MCTrack& tr, const o2::MCCompLabel& lb, int followDecay)
{
  const auto& params = o2::trackstudy::TrackMCStudyConfig::Instance();
  if (tr.GetPt() < params.minPtMC ||
      std::abs(tr.GetTgl()) > params.maxTglMC ||
      tr.R2() > params.minRMC * params.minRMC) {
    if (mVerbose > 1 && followDecay > -1) {
      LOGP(info, "rejecting decay {} prong : pdg={}, pT={}, tgL={}, r={}", followDecay, tr.GetPdgCode(), tr.GetPt(), tr.GetTgl(), std::sqrt(tr.R2()));
    }
    return false;
  }
  float dx = tr.GetStartVertexCoordinatesX() - mCurrMCVertex.X(), dy = tr.GetStartVertexCoordinatesY() - mCurrMCVertex.Y(), dz = tr.GetStartVertexCoordinatesZ() - mCurrMCVertex.Z();
  float r2 = dx * dx + dy * dy;
  float posTgl2 = r2 > 1 && std::abs(dz) < 20 ? dz * dz / r2 : 0;
  if (posTgl2 > params.maxPosTglMC * params.maxPosTglMC) {
    if (mVerbose > 1 && followDecay > -1) {
      LOGP(info, "rejecting decay {} prong : pdg={}, pT={}, tgL={}, dr={}, dz={} r={}, z={}, posTgl={}", followDecay, tr.GetPdgCode(), tr.GetPt(), tr.GetTgl(), std::sqrt(r2), dz, std::sqrt(tr.R2()), tr.GetStartVertexCoordinatesZ(), std::sqrt(posTgl2));
    }
    return false;
  }
  if (params.requireITSorTPCTrackRefs) {
    auto trspan = mcReader.getTrackRefs(lb.getSourceID(), lb.getEventID(), lb.getTrackID());
    bool ok = false;
    for (const auto& trf : trspan) {
      if (trf.getDetectorId() == DetID::ITS || trf.getDetectorId() == DetID::TPC) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      return false;
    }
  }
  TParticlePDG* pPDG = O2DatabasePDG::Instance()->GetParticle(tr.GetPdgCode());
  if (!pPDG) {
    LOGP(debug, "Unknown particle {}", tr.GetPdgCode());
    return false;
  }
  if (pPDG->Charge() == 0.) {
    return false;
  }
  return addMCParticle(tr, lb, pPDG);
}

bool TrackMCStudy::addMCParticle(const MCTrack& mcPart, const o2::MCCompLabel& lb, TParticlePDG* pPDG)
{
  std::array<float, 3> xyz{(float)mcPart.GetStartVertexCoordinatesX(), (float)mcPart.GetStartVertexCoordinatesY(), (float)mcPart.GetStartVertexCoordinatesZ()};
  std::array<float, 3> pxyz{(float)mcPart.GetStartVertexMomentumX(), (float)mcPart.GetStartVertexMomentumY(), (float)mcPart.GetStartVertexMomentumZ()};
  if (!pPDG && !(pPDG = O2DatabasePDG::Instance()->GetParticle(mcPart.GetPdgCode()))) {
    LOGP(debug, "Unknown particle {}", mcPart.GetPdgCode());
    return false;
  }
  auto& mcEntry = mSelMCTracks[lb];
  mcEntry.mcTrackInfo.pdg = mcPart.GetPdgCode();
  mcEntry.mcTrackInfo.track = o2::track::TrackPar(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
  mcEntry.mcTrackInfo.label = lb;
  mcEntry.mcTrackInfo.bcInTF = mIntBC[lb.getEventID()];
  mcEntry.mcTrackInfo.occTPC = mTPCOcc[lb.getEventID()];
  int moth = -1;
  o2::MCCompLabel mclbPar;
  if (!mcPart.isPrimary() && (moth = mcPart.getMotherTrackId()) >= 0) {
    const auto& mcPartPar = mCurrMCTracks[moth];
    mcEntry.mcTrackInfo.pdgParent = mcPartPar.GetPdgCode();
  }
  if (mVerbose > 1) {
    LOGP(info, "Adding charged MC pdg={} {} ", mcPart.GetPdgCode(), lb.asString());
  }
  return true;
}

void TrackMCStudy::loadTPCOccMap(const o2::globaltracking::RecoContainer& recoData)
{
  auto NHBPerTF = o2::base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF();
  o2::tpc::CorrectionMapsLoader TPCCorrMapsLoader{};
  const auto& TPCOccMap = recoData.occupancyMapTPC;
  auto prop = o2::base::Propagator::Instance();
  auto TPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(&recoData.inputsTPCclusters->clusterIndex, &TPCCorrMapsLoader, prop->getNominalBz(),
                                                                    recoData.getTPCTracksClusterRefs().data(), 0, recoData.clusterShMapTPC.data(), TPCOccMap.data(), TPCOccMap.size(), nullptr, prop);
  mNTPCOccBinLength = TPCRefitter->getParam()->rec.tpc.occupancyMapTimeBins;
  mTBinClOcc.clear();
  if (mNTPCOccBinLength > 1 && TPCOccMap.size()) {
    mNTPCOccBinLengthInv = 1. / mNTPCOccBinLength;
    int nTPCBins = NHBPerTF * o2::constants::lhc::LHCMaxBunches / 8, ninteg = 0;
    int nTPCOccBins = nTPCBins * mNTPCOccBinLengthInv, sumBins = std::max(1, int(o2::constants::lhc::LHCMaxBunches / 8 * mNTPCOccBinLengthInv));
    mTBinClOcc.resize(nTPCOccBins);
    std::vector<float> mltHistTB(nTPCOccBins);
    float sm = 0., tb = 0.5 * mNTPCOccBinLength;
    for (int i = 0; i < nTPCOccBins; i++) {
      mltHistTB[i] = TPCRefitter->getParam()->GetUnscaledMult(tb);
      tb += mNTPCOccBinLength;
    }
    for (int i = nTPCOccBins; i--;) {
      sm += mltHistTB[i];
      if (i + sumBins < nTPCOccBins) {
        sm -= mltHistTB[i + sumBins];
      }
      mTBinClOcc[i] = sm;
    }
  } else {
    mTBinClOcc.resize(1);
  }
}

DataProcessorSpec getTrackMCStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  bool useMC = true;
  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertices(useMC);
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
    AlgorithmSpec{adaptFromTask<TrackMCStudy>(dataRequest, ggRequest, srcTracks)},
    Options{
      {"device-verbosity", VariantType::Int, 0, {"Verbosity level"}},
      {"dcay-vs-pt", VariantType::String, "0.0105 + 0.0350 / pow(x, 1.1)", {"Formula for global tracks DCAy vs pT cut"}},
      {"min-tpc-clusters", VariantType::Int, 60, {"Cut on TPC clusters"}},
      {"max-tpc-dcay", VariantType::Float, 2.f, {"Cut on TPC dcaY"}},
      {"max-tpc-dcaz", VariantType::Float, 2.f, {"Cut on TPC dcaZ"}},
      {"min-x-prop", VariantType::Float, 6.f, {"track should be propagated to this X at least"}},
    }};
}

} // namespace o2::trackstudy
