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
#include <cmath>

#include <TStopwatch.h>
#include <TF1.h>

#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTSimulation/Hit.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DetectorsVertexing/PVertexer.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Reconstruction/IOUtils.h"
#include "ITS3TrackingStudy/ITS3TrackingStudyParam.h"
#include "ITS3TrackingStudy/ParticleInfoExt.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "Steer/MCKinematicsReader.h"

namespace o2::its3::study
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;
using PVertex = o2::dataformats::PrimaryVertex;
using GTrackID = o2::dataformats::GlobalTrackID;
using VtxTrackID = o2::dataformats::VtxTrackIndex;
using T2VMap = std::unordered_map<GTrackID, size_t>;

class TrackingStudySpec : public Task
{
 public:
  TrackingStudySpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC) {}
  ~TrackingStudySpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void process(o2::globaltracking::RecoContainer& recoData);
  void updateTimeDependentParams(ProcessingContext& pc);
  std::vector<o2::BaseCluster<float>> prepareITSClusters(const o2::globaltracking::RecoContainer& data) const;
  bool selectTrack(GTrackID trkID, o2::globaltracking::RecoContainer& recoData, bool checkMCTruth = true) const;
  T2VMap buildT2V(o2::globaltracking::RecoContainer& recoData, bool includeCont = false, bool requireMCMatch = true) const;
  bool refitITSPVTrack(o2::globaltracking::RecoContainer& recoData, o2::track::TrackParCov& trFit, GTrackID gidx);
  void doDCAStudy(o2::globaltracking::RecoContainer& recoData);
  void doDCARefitStudy(o2::globaltracking::RecoContainer& recoData);
  void doPullStudy(o2::globaltracking::RecoContainer& recoData);
  void doMCStudy(o2::globaltracking::RecoContainer& recoData);

  struct TrackCounter {
    TrackCounter() = default;

    void operator+=(int src)
    {
      if (src >= 0 && src < static_cast<int>(mSuccess.size())) {
        ++mSuccess[src];
      }
    }

    void operator-=(int src)
    {
      if (src >= 0 && src < static_cast<int>(mirrors.size())) {
        ++mirrors[src];
      }
    }

    void operator&=(int src)
    {
      if (src >= 0 && src < static_cast<int>(mRejected.size())) {
        ++mRejected[src];
      }
    }

    void print() const
    {
      LOGP(info, "\t\t\tSuccess / Error / Rejected");
      for (int cis = 0; cis < GTrackID::NSources; ++cis) {
        const auto cdm = GTrackID::getSourceDetectorsMask(cis);
        if (cdm[DetID::ITS]) {
          LOGP(info, "\t{:{}}\t{} / {} / {}", GTrackID::getSourceName(cis), 15, mSuccess[cis], mirrors[cis], mRejected[cis]);
        }
      }
    }

    void reset()
    {
      mSuccess.fill(0);
      mirrors.fill(0);
      mRejected.fill(0);
    }

    std::array<size_t, GTrackID::NSources> mSuccess{};
    std::array<size_t, GTrackID::NSources> mirrors{};
    std::array<size_t, GTrackID::NSources> mRejected{};
  };
  TrackCounter mTrackCounter;

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false};
  GTrackID::mask_t mTracksSrc;
  o2::vertexing::PVertexer mVertexer;
  o2::steer::MCKinematicsReader mcReader;                 // reader of MC information
  const o2::its3::TopologyDictionary* mITSDict = nullptr; // cluster patterns dictionary
};

void TrackingStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  int lane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  int maxLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
  std::string dbgnm = maxLanes == 1 ? "its3TrackStudy.root" : fmt::format("its3TrackStudy_{}.root", lane);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(dbgnm.c_str(), "recreate");

  if (mUseMC && !mcReader.initFromDigitContext(o2::base::NameConf::getCollisionContextFileName())) {
    LOGP(fatal, "initialization of MCKinematicsReader failed");
  }
}

void TrackingStudySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc);
  process(recoData);
}

void TrackingStudySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (static bool initOnceDone{false}; !initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    mVertexer.init();
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2G));
  }
}

void TrackingStudySpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void TrackingStudySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("IT3", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::its3::TopologyDictionary*)obj;
    return;
  }
}

void TrackingStudySpec::process(o2::globaltracking::RecoContainer& recoData)
{
  const auto& conf = ITS3TrackingStudyParam::Instance();
  if (conf.doDCA) {
    doDCAStudy(recoData);
  }
  if (conf.doDCARefit) {
    doDCARefitStudy(recoData);
  }
  if (mUseMC && conf.doPull) {
    doPullStudy(recoData);
  }
  if (mUseMC && conf.doMC) {
    doMCStudy(recoData);
  }
}

std::vector<o2::BaseCluster<float>> TrackingStudySpec::prepareITSClusters(const o2::globaltracking::RecoContainer& data) const
{
  std::vector<o2::BaseCluster<float>> itscl;
  const auto& clusITS = data.getITSClusters();
  if (clusITS.size()) {
    const auto& patterns = data.getITSClustersPatterns();
    itscl.reserve(clusITS.size());
    auto pattIt = patterns.begin();
    o2::its3::ioutils::convertCompactClusters(clusITS, pattIt, itscl, mITSDict);
  }
  return std::move(itscl);
}

bool TrackingStudySpec::selectTrack(GTrackID trkID, o2::globaltracking::RecoContainer& recoData, bool checkMCTruth) const
{
  const auto& conf = ITS3TrackingStudyParam::Instance();
  if (!trkID.includesDet(GTrackID::ITS)) {
    return false;
  }
  if (!recoData.isTrackSourceLoaded(trkID.getSource())) {
    return false;
  }
  auto contributorsGID = recoData.getSingleDetectorRefs(trkID);
  if (!contributorsGID[GTrackID::ITS].isIndexSet()) { // we need of course ITS
    return false;
  }
  // ITS specific
  const auto& itsTrk = recoData.getITSTrack(contributorsGID[GTrackID::ITS]);
  if (itsTrk.getChi2() > conf.maxChi2 || itsTrk.getNClusters() < conf.minITSCls) {
    return false;
  }
  // TPC specific
  if (contributorsGID[GTrackID::TPC].isIndexSet()) {
    const auto& tpcTrk = recoData.getTPCTrack(contributorsGID[GTrackID::TPC]);
    if (tpcTrk.getNClusters() < conf.minTPCCls) {
      return false;
    }
  }
  // general
  const auto& gTrk = recoData.getTrackParam(trkID);
  if (gTrk.getPt() < conf.minPt || gTrk.getPt() > conf.maxPt) {
    return false;
  }
  if (std::abs(gTrk.getEta()) > conf.maxEta) {
    return false;
  }
  if (mUseMC && checkMCTruth) {
    const auto& itsLbl = recoData.getTrackMCLabel(contributorsGID[GTrackID::ITS]);
    if (!itsLbl.isValid()) {
      return false;
    }
    if (contributorsGID[GTrackID::TPC].isIndexSet()) {
      const auto& tpcLbl = recoData.getTrackMCLabel(contributorsGID[GTrackID::TPC]);
      if (itsLbl != tpcLbl) {
        return false;
      }
    }
    if (contributorsGID[GTrackID::TRD].isIndexSet()) {
      // TODO
    }
    if (contributorsGID[GTrackID::TOF].isIndexSet()) {
      const auto& tofLbls = recoData.getTOFClustersMCLabels()->getLabels(contributorsGID[GTrackID::TOF]);
      for (const auto& lbl : tofLbls) {
        if (lbl.isValid()) {
          return true;
        }
      }
    }
  }
  return true;
}

T2VMap TrackingStudySpec::buildT2V(o2::globaltracking::RecoContainer& recoData, bool includeCont, bool requireMCMatch) const
{
  // build track->vertex assoc., maybe including contributor tracks
  const auto& conf = ITS3TrackingStudyParam::Instance();
  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                               // last entry is for unassigned tracks, ignore them
  T2VMap t2v;
  for (size_t iv = 0; iv < nv; ++iv) {
    const auto& pv = pvvec[iv];
    if (pv.getNContributors() - 1 < conf.minPVCont) {
      continue;
    }
    if (requireMCMatch) {
      auto pvl = recoData.getPrimaryVertexMCLabel(iv);
    }
    const auto& vtxRef = vtxRefs[iv];
    int it = vtxRef.getFirstEntry(), itLim = it + vtxRef.getEntries();
    for (; it < itLim; it++) {
      const auto& tvid = trackIndex[it];
      if (tvid.isAmbiguous()) {
        continue;
      }
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        continue;
      }
      if (mUseMC && requireMCMatch) {
        const auto& pvlbl = recoData.getPrimaryVertexMCLabel(iv);
        if (pvlbl.getEventID() != recoData.getTrackMCLabel(tvid).getEventID()) {
          continue;
        }
      }
      t2v[tvid] = iv;
      if (includeCont) {
        auto contributorsGID = recoData.getSingleDetectorRefs(tvid);
        for (int cis = 0; cis < GTrackID::NSources; cis++) {
          const auto cdm = GTrackID::getSourceDetectorsMask(cis);
          if (!recoData.isTrackSourceLoaded(cis) || !cdm[DetID::ITS] || !contributorsGID[cis].isIndexSet()) {
            continue;
          }
          if (mUseMC && requireMCMatch) {
            const auto& pvlbl = recoData.getPrimaryVertexMCLabel(iv);
            if (pvlbl.getEventID() != recoData.getTrackMCLabel(contributorsGID[cis]).getEventID()) {
              continue;
            }
          }
          t2v[contributorsGID[cis]] = iv;
        }
      }
    }
  }
  return std::move(t2v);
}

bool TrackingStudySpec::refitITSPVTrack(o2::globaltracking::RecoContainer& recoData, o2::track::TrackParCov& trFit, GTrackID gidx)
{
  if (gidx.getSource() != GTrackID::ITS) {
    return false;
  }
  static auto pvvec = recoData.getPrimaryVertices();
  static auto t2v = buildT2V(recoData, true, true);
  static const auto itsClusters = prepareITSClusters(recoData);
  static std::vector<unsigned int> itsTracksROF;
  if (static bool done{false}; !done) {
    done = true;
    const auto& itsTracksROFRec = recoData.getITSTracksROFRecords();
    itsTracksROF.resize(recoData.getITSTracks().size());
    for (unsigned irf = 0, cnt = 0; irf < itsTracksROFRec.size(); irf++) {
      int ntr = itsTracksROFRec[irf].getNEntries();
      for (int itr = 0; itr < ntr; itr++) {
        itsTracksROF[cnt++] = irf;
      }
    }
  }
  auto prop = o2::base::Propagator::Instance();
  const auto& conf = ITS3TrackingStudyParam::Instance();
  std::array<o2::BaseCluster<float>, 8> clArr{};
  std::array<float, 8> clAlpha{};
  const auto trkIn = recoData.getTrackParam(gidx);
  const auto trkOut = recoData.getTrackParamOut(gidx);
  const auto& itsTrOrig = recoData.getITSTrack(gidx);
  int ncl = itsTrOrig.getNumberOfClusters(), rof = itsTracksROF[gidx.getIndex()];
  const auto& itsTrackClusRefs = recoData.getITSTracksClusterRefs();
  int clEntry = itsTrOrig.getFirstClusterEntry();
  const auto propagator = o2::base::Propagator::Instance();
  // convert PV to a fake cluster in the track DCA frame
  const auto& pv = pvvec[t2v[gidx]];
  auto trkPV = trkIn;
  if (!prop->propagateToDCA(pv, trkPV, prop->getNominalBz(), 2.0, conf.CorrType)) {
    mTrackCounter -= gidx.getSource();
    return false;
  }
  // create base cluster from the PV, with the alpha corresponding to the track at DCA
  float cosAlp = NAN, sinAlp = NAN;
  o2::math_utils::sincos(trkPV.getAlpha(), sinAlp, cosAlp);
  // vertex position rotated to track frame
  clArr[0].setXYZ(pv.getX() * cosAlp + pv.getY() * sinAlp, -pv.getX() * sinAlp + pv.getY() * cosAlp, pv.getZ());
  clArr[0].setSigmaY2(0.5 * (pv.getSigmaX2() + pv.getSigmaY2()));
  clArr[0].setSigmaZ2(pv.getSigmaZ2());
  clAlpha[0] = trkPV.getAlpha();
  for (int icl = 0; icl < ncl; ++icl) { // ITS clusters are referred in layer decreasing order
    clArr[ncl - icl] = itsClusters[itsTrackClusRefs[clEntry + icl]];
    clAlpha[ncl - icl] = o2::its::GeometryTGeo::Instance()->getSensorRefAlpha(clArr[ncl - icl].getSensorID());
  }
  // start refit
  trFit = trkOut;
  trFit.resetCovariance(1'000);
  float chi2{0};
  for (int icl = ncl; icl >= 0; --icl) { // go backwards
    if (!trFit.rotate(clAlpha[icl]) || !prop->propagateToX(trFit, clArr[icl].getX(), prop->getNominalBz(), 0.85, 2.0, conf.CorrType)) {
      mTrackCounter -= gidx.getSource();
      return false;
    }
    chi2 += trFit.getPredictedChi2(clArr[icl]);
    if (!trFit.update(clArr[icl])) {
      mTrackCounter -= gidx.getSource();
      return false;
    }
  }
  // chi2 < conf.maxChi2; should I cut here?
  return true;
};

void TrackingStudySpec::doDCAStudy(o2::globaltracking::RecoContainer& recoData)
{
  /// analyse DCA of impact parameter for different track types
  LOGP(info, "Doing DCA study");
  mTrackCounter.reset();
  const auto& conf = ITS3TrackingStudyParam::Instance();
  auto prop = o2::base::Propagator::Instance();
  TStopwatch sw;
  sw.Start();
  int nDCAFits{0}, nDCAFitsFail{0};
  auto pvvec = recoData.getPrimaryVertices();
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                               // last entry is for unassigned tracks, ignore them
  auto& stream = (*mDBGOut) << "dca";
  for (int iv = 0; iv < nv; iv++) {
    const auto& pv = pvvec[iv];
    const auto& vtref = vtxRefs[iv];
    for (int is = 0; is < GTrackID::NSources; is++) {
      const auto dm = GTrackID::getSourceDetectorsMask(is);
      if (!recoData.isTrackSourceLoaded(is) || !dm[DetID::ITS]) {
        mTrackCounter &= is;
        continue;
      }
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        const auto vid = trackIndex[i];
        if (!vid.isPVContributor()) {
          mTrackCounter &= vid.getSource();
          continue;
        }

        // we fit each different sub-track type, that include ITS, e.g.
        // ITS,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF
        auto contributorsGID = recoData.getSingleDetectorRefs(vid);
        for (int cis = 0; cis < GTrackID::NSources && cis <= is; cis++) {
          const auto cdm = GTrackID::getSourceDetectorsMask(cis);
          if (!recoData.isTrackSourceLoaded(cis) || !cdm[DetID::ITS] || !contributorsGID[cis].isIndexSet()) {
            mTrackCounter &= cis;
            continue;
          }
          if (!selectTrack(contributorsGID[cis], recoData)) {
            mTrackCounter &= vid.getSource();
            continue;
          }

          o2::dataformats::DCA dcaInfo;
          const auto& trk = recoData.getTrackParam(contributorsGID[cis]);
          auto trkRefit = trk;
          // for ITS standalone tracks instead of having the trk at the pv we refit with the pv
          if (conf.refitITS && cis == GTrackID::ITS && !refitITSPVTrack(recoData, trkRefit, contributorsGID[cis])) {
            mTrackCounter -= cis;
            continue;
          } else {
            trkRefit.invalidate();
          };

          auto trkDCA = trk;
          if (!prop->propagateToDCABxByBz(pv, trkDCA, 2.f, conf.CorrType, &dcaInfo)) {
            mTrackCounter -= cis;
            ++nDCAFitsFail;
            continue;
          }

          stream << "src=" << cis
                 << "pv=" << pv
                 << "trk=" << trk
                 << "trkRefit=" << trkRefit
                 << "trkAtPV=" << trkDCA
                 << "dca=" << dcaInfo;

          if (mUseMC) {
            const auto& lbl = recoData.getTrackMCLabel(contributorsGID[cis]);
            lbl.print();
            o2::dataformats::DCA dcaInfoMC;
            const auto& eve = mcReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
            o2::dataformats::VertexBase mcEve;
            mcEve.setPos({(float)eve.GetX(), (float)eve.GetY(), (float)eve.GetZ()});
            auto trkC = trk;
            if (!prop->propagateToDCABxByBz(mcEve, trkC, 2.f, conf.CorrType, &dcaInfoMC)) {
              mTrackCounter -= cis;
              ++nDCAFitsFail;
              continue;
            }
            const auto& mcTrk = mcReader.getTrack(lbl);
            if (mcTrk == nullptr) {
              LOGP(fatal, "mcTrk is null did selection fail?");
            }
            stream << "mcTrk=" << *mcTrk
                   << "dca2MC=" << dcaInfoMC
                   << "lbl=" << lbl;
          }
          stream << "\n";

          ++nDCAFits;
          mTrackCounter += cis;
        }
      }
    }
  }
  sw.Stop();
  LOGP(info, "doDCAStudy: accepted {} fits, failed {} (in {:.2f} seconds)", nDCAFits, nDCAFitsFail, sw.RealTime());
  mTrackCounter.print();
}

void TrackingStudySpec::doDCARefitStudy(o2::globaltracking::RecoContainer& recoData)
{
  /// analyse DCA of impact parameter for different track types while refitting the PV without the cand track
  LOGP(info, "Doing DCARefit study");
  mTrackCounter.reset();
  const auto& conf = ITS3TrackingStudyParam::Instance();
  auto prop = o2::base::Propagator::Instance();
  TStopwatch sw;
  sw.Start();

  // build track->vertex assoc.
  auto pvvec = recoData.getPrimaryVertices();
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                               // last entry is for unassigned tracks, ignore them
  auto t2v = buildT2V(recoData);
  std::vector<std::vector<GTrackID>> v2t;
  v2t.resize(nv);
  auto creator = [&](const auto& trk, GTrackID trkID, float _t0, float terr) -> bool {
    if constexpr (!isBarrelTrack<decltype(trk)>()) {
      mTrackCounter &= trkID.getSource();
      return false;
    }
    if (!trkID.includesDet(GTrackID::ITS)) {
      mTrackCounter &= trkID.getSource();
      return false;
    }
    // general
    if constexpr (isBarrelTrack<decltype(trk)>()) {
      if (trk.getPt() < conf.minPt || trk.getPt() > conf.maxPt) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (std::abs(trk.getEta()) > conf.maxEta) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (!t2v.contains(trkID)) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (!selectTrack(trkID, recoData, mUseMC)) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
    }
    v2t[t2v[trkID]].push_back(trkID);
    return true;
  };
  recoData.createTracksVariadic(creator);

  int nDCAFits{0}, nDCAFitsFail{0};
  auto& stream = (*mDBGOut) << "dcaRefit";
  for (size_t iv = 0; iv < nv; ++iv) {
    const auto& pv = pvvec[iv];
    const auto& trkIDs = v2t[iv];
    if (trkIDs.size() - 1 < conf.minPVCont) {
      continue;
    }
    std::vector<o2::track::TrackParCov> trks;
    trks.reserve(trkIDs.size());
    for (const auto& trkID : trkIDs) {
      trks.push_back(recoData.getTrackParam(trkID));
    }

    if (!mVertexer.prepareVertexRefit(trks, pv)) {
      continue;
    }
    std::vector<bool> trkMask(trkIDs.size(), true);
    for (size_t it{0}; it < trkMask.size(); ++it) {
      trkMask[it] = false; // mask current track from pv refit
      if (it != 0) {
        trkMask[it - 1] = true; // unmask previoustrack from pv refit
      }
      auto pvRefit = mVertexer.refitVertex(trkMask, pv);
      if (pvRefit.getChi2() < 0) {
        trkMask[it] = true;
        continue;
      }

      // check DCA both for refitted and original PV
      o2::dataformats::DCA dcaInfo;
      auto trkC = trks[it];
      if (!prop->propagateToDCABxByBz(pv, trkC, 2.f, conf.CorrType, &dcaInfo)) {
        mTrackCounter -= trkIDs[it].getSource();
        ++nDCAFitsFail;
        continue;
      }
      o2::dataformats::DCA dcaInfoRefit;
      auto trkCRefit = trks[it];
      if (!prop->propagateToDCABxByBz(pv, trkCRefit, 2.f, conf.CorrType, &dcaInfoRefit)) {
        mTrackCounter -= trkIDs[it].getSource();
        ++nDCAFitsFail;
        continue;
      }

      stream << "src=" << trkIDs[it].getSource()
             << "pv=" << pv
             << "trkAtPV=" << trkC
             << "dca=" << dcaInfo
             << "pvRefit=" << pvRefit
             << "trkAtPVRefit=" << trkC
             << "dcaRefit=" << dcaInfoRefit;
      if (mUseMC) {
        const auto& mcTrk = mcReader.getTrack(recoData.getTrackMCLabel(trkIDs[it]));
        if (mcTrk == nullptr) {
          LOGP(fatal, "mcTrk is null did selection fail?");
        }
        stream << "mcTrk=" << *mcTrk;
      }
      stream << "\n";
      ++nDCAFits;
      mTrackCounter += trkIDs[it].getSource();
    }
  }
  sw.Stop();
  LOGP(info, "doDCARefitStudy: accepted {} fits, failed {} (in {:.2f} seconds)", nDCAFits, nDCAFitsFail, sw.RealTime());
  mTrackCounter.print();
}

void TrackingStudySpec::doPullStudy(o2::globaltracking::RecoContainer& recoData)
{
  // check track pulls compared to mc generation
  LOGP(info, "Doing Pull study");
  mTrackCounter.reset();
  TStopwatch sw;
  sw.Start();
  int nPulls{0}, nPullsFail{0};
  auto prop = o2::base::Propagator::Instance();
  const auto& conf = ITS3TrackingStudyParam::Instance();

  auto checkInTrack = [&](GTrackID trkID) {
    if (!selectTrack(trkID, recoData)) {
      mTrackCounter &= trkID.getSource();
      return;
    }
    const auto mcTrk = mcReader.getTrack(recoData.getTrackMCLabel(trkID));
    if (!mcTrk) {
      return;
    }
    auto trk = recoData.getTrackParam(trkID);

    // for ITS standalone tracks we add the PV as an additional measurement point
    if (conf.refitITS && trkID.getSource() == GTrackID::ITS && !refitITSPVTrack(recoData, trk, trkID)) {
      mTrackCounter -= trkID.getSource();
      ++nPullsFail;
      return;
    }

    std::array<float, 3> xyz{(float)mcTrk->GetStartVertexCoordinatesX(), (float)mcTrk->GetStartVertexCoordinatesY(), (float)mcTrk->GetStartVertexCoordinatesZ()},
      pxyz{(float)mcTrk->GetStartVertexMomentumX(), (float)mcTrk->GetStartVertexMomentumY(), (float)mcTrk->GetStartVertexMomentumZ()};
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrk->GetPdgCode());
    if (!pPDG) {
      mTrackCounter -= trkID.getSource();
      ++nPullsFail;
      return;
    }
    o2::track::TrackPar mcTrkO2(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
    // propagate it to the alpha/X of the reconstructed track
    if (!mcTrkO2.rotate(trk.getAlpha()) || !prop->PropagateToXBxByBz(mcTrkO2, trk.getX())) {
      mTrackCounter -= trkID.getSource();
      ++nPullsFail;
      return;
    }
    const auto contTrk = recoData.getSingleDetectorRefs(trkID);
    const auto& itsTrk = recoData.getITSTrack(contTrk[GTrackID::ITS]);

    (*mDBGOut)
      << "pull"
      << "src=" << trkID.getSource()
      << "itsTrk=" << itsTrk
      << "mcTrk=" << mcTrkO2
      << "mcPart=" << mcTrk
      << "trk=" << trk
      << "\n";
    ++nPulls;
    mTrackCounter += trkID.getSource();
  };

  for (size_t iTrk{0}; iTrk < recoData.getITSTracks().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITS));
  }
  for (size_t iTrk{0}; iTrk < recoData.getTPCITSTracks().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPC));
  }
  for (size_t iTrk{0}; iTrk < recoData.getITSTPCTRDTracksMCLabels().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTRD));
  }
  for (size_t iTrk{0}; iTrk < recoData.getITSTPCTOFMatches().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTOF));
  }
  for (size_t iTrk{0}; iTrk < recoData.getITSTPCTRDTOFMatches().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTRDTOF));
  }
  sw.Stop();
  LOGP(info, "doPullStudy: accepted {} pulls; rejected {} (in {:.2f} seconds)", nPulls, nPullsFail, sw.RealTime());
  mTrackCounter.print();
}

void TrackingStudySpec::doMCStudy(o2::globaltracking::RecoContainer& recoData)
{
  LOGP(info, "Doing MC study");
  mTrackCounter.reset();
  TStopwatch sw;
  sw.Start();
  int nTracks{0};

  const int iSrc{0};
  const int nev = mcReader.getNEvents(iSrc);
  std::unordered_map<o2::MCCompLabel, ParticleInfoExt> info;

  LOGP(info, "** Filling particle table ... ");
  for (int iEve{0}; iEve < nev; ++iEve) {
    const auto& mcTrks = mcReader.getTracks(iSrc, iEve);
    for (int iTrk{0}; iTrk < mcTrks.size(); ++iTrk) {
      const auto& mcTrk = mcTrks[iTrk];
      const auto pdg = mcTrk.GetPdgCode();
      if (o2::O2DatabasePDG::Instance()->GetParticle(pdg) == nullptr) {
        continue;
      }
      const auto apdg = std::abs(pdg);
      if (apdg != 11 && apdg != 211 && apdg != 321 && apdg != 2212) {
        continue;
      }
      o2::MCCompLabel lbl(iTrk, iEve, iSrc);
      auto& part = info[lbl];
      part.mcTrack = mcTrk;
    }
  }
  LOGP(info, "** Creating particle/clusters correspondence ... ");
  const auto& clusters = recoData.getITSClusters();
  const auto& clustersMCLCont = recoData.getITSClustersMCLabels();
  for (auto iCluster{0}; iCluster < clusters.size(); ++iCluster) {
    auto labs = clustersMCLCont->getLabels(iCluster);
    for (auto& lab : labs) {
      if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect()) {
        continue;
      }
      int trackID = 0, evID = 0, srcID = 0;
      bool fake = false;
      lab.get(trackID, evID, srcID, fake);
      auto& cluster = clusters[iCluster];
      auto layer = o2::its::GeometryTGeo::Instance()->getLayer(cluster.getSensorID());
      auto& part = info[{trackID, evID, srcID}];
      part.clusters |= (1 << layer);
      if (fake) {
        part.fakeClusters |= (1 << layer);
      }
    }
  }
  LOGP(info, "** Analysing tracks ... ");
  auto accountLbl = [&](const globaltracking::RecoContainer::GlobalIDSet& contributorsGID, DetID::ID det) {
    if (contributorsGID[det].isIndexSet()) {
      const auto& lbl = recoData.getTrackMCLabel(contributorsGID[det]);
      if (lbl.isValid()) {
        o2::MCCompLabel iLbl(lbl.getTrackID(), lbl.getEventID(), lbl.getSourceID());
        if (info.contains(iLbl)) {
          auto& part = info[iLbl];
          SETBIT(part.recoTracks, det);
          if (lbl.isFake()) {
            SETBIT(part.fakeTracks, det);
          }
        }
      }
    }
  };
  auto creator = [&](const auto& trk, GTrackID trkID, float _t0, float terr) -> bool {
    if constexpr (!isBarrelTrack<decltype(trk)>()) {
      return false;
    }
    if (!trkID.includesDet(GTrackID::ITS)) {
      return false;
    }
    // general
    auto contributorsGID = recoData.getSingleDetectorRefs(trkID);
    if (!contributorsGID[GTrackID::ITS].isIndexSet()) { // we need of course ITS
      return false;
    }
    const auto& gLbl = recoData.getTrackMCLabel(trkID);
    if (!gLbl.isValid()) {
      return false;
    }
    o2::MCCompLabel iLbl(gLbl.getTrackID(), gLbl.getEventID(), gLbl.getSourceID());
    if (!info.contains(iLbl)) {
      return false;
    }
    auto& part = info[iLbl];
    part.recoTrack = recoData.getTrackParam(trkID);

    accountLbl(contributorsGID, DetID::ITS);
    accountLbl(contributorsGID, DetID::TPC);
    accountLbl(contributorsGID, DetID::TRD);
    accountLbl(contributorsGID, DetID::TOF);

    ++nTracks;
    return true;
  };
  recoData.createTracksVariadic(creator);

  LOGP(info, "Streaming output to tree");
  for (const auto& [_, part] : info) {
    (*mDBGOut) << "mc"
               << "part=" << part
               << "\n";
  }

  sw.Stop();
  LOGP(info, "doMCStudy: accounted {} MCParticles and {} tracks (in {:.2f} seconds)", info.size(), nTracks, sw.RealTime());
}

DataProcessorSpec getTrackingStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestIT3Clusters(useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertices(useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    .name = "its3-track-study",
    .inputs = dataRequest->inputs,
    .outputs = outputs,
    .algorithm = AlgorithmSpec{adaptFromTask<TrackingStudySpec>(dataRequest, ggRequest, srcTracks, useMC)},
    .options = {}};
}

} // namespace o2::its3::study
