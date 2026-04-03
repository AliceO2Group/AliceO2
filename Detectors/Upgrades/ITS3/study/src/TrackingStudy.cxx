// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
#include <Eigen/Dense>

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
#include "ITS3Align/MisalignmentUtils.h"
#include "ITS3Align/TrackFit.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "Steer/MCKinematicsReader.h"
#include "Framework/Logger.h"

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
  TrackingStudySpec(const TrackingStudySpec&) = delete;
  TrackingStudySpec(TrackingStudySpec&&) = delete;
  TrackingStudySpec& operator=(const TrackingStudySpec&) = delete;
  TrackingStudySpec& operator=(TrackingStudySpec&&) = delete;
  TrackingStudySpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC) {}
  ~TrackingStudySpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void process();
  void updateTimeDependentParams(ProcessingContext& pc);
  void prepareITSClusters();
  bool selectTrack(GTrackID trkID, bool checkMCTruth = true) const;
  T2VMap buildT2V(bool includeCont = false, bool requireMCMatch = true) const;
  bool refitITSPVTrack(o2::track::TrackParCov& trFit, GTrackID gidx);

  void doDCAStudy();
  void doDCARefitStudy();
  void doPullStudy();
  void doMCStudy();
  void doResidStudy();
  void doMisalignmentStudy();

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

  using TrackingCluster = align::TrackingCluster<float>;
  std::vector<TrackingCluster> mITScl;
  std::span<const int> mITSclRef;

  const ITS3TrackingStudyParam* mParams{nullptr};
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false};
  GTrackID::mask_t mTracksSrc;
  o2::vertexing::PVertexer mVertexer;
  o2::steer::MCKinematicsReader mMCReader;                // reader of MC information
  const o2::its3::TopologyDictionary* mITSDict = nullptr; // cluster patterns dictionary
  o2::globaltracking::RecoContainer mRecoData;
  align::MisalignmentModel mMisalignment;
};

void TrackingStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  int lane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  int maxLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
  std::string dbgnm = maxLanes == 1 ? "its3TrackStudy.root" : fmt::format("its3TrackStudy_{}.root", lane);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(dbgnm.c_str(), "recreate");

  if (mUseMC && !mMCReader.initFromDigitContext(o2::base::NameConf::getCollisionContextFileName())) {
    LOGP(fatal, "initialization of MCKinematicsReader failed");
  }
}

void TrackingStudySpec::run(ProcessingContext& pc)
{
  mRecoData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc);
  process();
}

void TrackingStudySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (static bool initOnceDone{false}; !initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    mVertexer.init();
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2G));
    mParams = &ITS3TrackingStudyParam::Instance();
    if (mParams->doMisalignment) {
      mMisalignment = {};
      if (!mParams->misAlgJson.empty()) {
        mMisalignment = align::loadMisalignmentModel(mParams->misAlgJson);
      }
    }
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

void TrackingStudySpec::process()
{
  prepareITSClusters();
  if (mParams->doDCA) {
    doDCAStudy();
  }
  if (mParams->doDCARefit) {
    doDCARefitStudy();
  }
  if (mUseMC && mParams->doPull) {
    doPullStudy();
  }
  if (mUseMC && mParams->doMC) {
    doMCStudy();
  }
  if (mParams->doResid) {
    doResidStudy();
  }
  if (mUseMC && mParams->doMisalignment) {
    doMisalignmentStudy();
  }
}

void TrackingStudySpec::prepareITSClusters()
{
  const auto& clusITS = mRecoData.getITSClusters();
  LOGP(info, "Preparing {} measurments", clusITS.size());
  const auto& patterns = mRecoData.getITSClustersPatterns();
  mITScl.reserve(clusITS.size());
  auto pattIt = patterns.begin();
  auto geom = its::GeometryTGeo::Instance();
  mITSclRef = mRecoData.getITSTracksClusterRefs();
  mITScl.clear();
  mITScl.reserve(clusITS.size());
  for (const auto& cls : clusITS) {
    const auto sens = cls.getSensorID();
    float sigmaY2{0}, sigmaZ2{0};
    math_utils::Point3D<float> locXYZ = o2::its3::ioutils::extractClusterData(cls, pattIt, mITSDict, sigmaY2, sigmaZ2);
    // Transformation to the local --> global
    const auto gloXYZ = geom->getMatrixL2G(sens) * locXYZ;
    // Inverse transformation to the local --> tracking
    o2::math_utils::Point3D<float> trkXYZ = geom->getMatrixT2L(sens) ^ locXYZ;
    // Tracking alpha angle
    // We want that each cluster rotates its tracking frame to the clusters phi
    // that way the track linearization around the measurement is less biases to the arc
    // this means automatically that the measurement on the arc is at 0 for the curved layers
    float alpha = geom->getSensorRefAlpha(sens);
    if (constants::detID::isDetITS3(sens)) {
      trkXYZ.SetY(0.f);
      // alpha&x always have to be defined wrt to the global Z axis!
      trkXYZ.SetX(std::hypot(gloXYZ.x(), gloXYZ.y()));
      alpha = std::atan2(gloXYZ.y(), gloXYZ.x());
    }
    auto& cl3d = mITScl.emplace_back(sens, trkXYZ);
    cl3d.setErrors(sigmaY2, sigmaZ2, 0.f);
    cl3d.alpha = alpha;
    math_utils::detail::bringToPMPi(cl3d.alpha); // alpha is defined on -Pi,Pi
  }
}

bool TrackingStudySpec::selectTrack(GTrackID trkID, bool checkMCTruth) const
{
  if (!trkID.includesDet(GTrackID::ITS)) {
    return false;
  }
  if (!mRecoData.isTrackSourceLoaded(trkID.getSource())) {
    return false;
  }
  auto contributorsGID = mRecoData.getSingleDetectorRefs(trkID);
  if (!contributorsGID[GTrackID::ITS].isIndexSet()) { // we need of course ITS
    return false;
  }
  // ITS specific
  const auto& itsTrk = mRecoData.getITSTrack(contributorsGID[GTrackID::ITS]);
  if (itsTrk.getChi2() > mParams->maxChi2 || itsTrk.getNClusters() < mParams->minITSCls) {
    return false;
  }
  // TPC specific
  if (contributorsGID[GTrackID::TPC].isIndexSet()) {
    const auto& tpcTrk = mRecoData.getTPCTrack(contributorsGID[GTrackID::TPC]);
    if (tpcTrk.getNClusters() < mParams->minTPCCls) {
      return false;
    }
  }
  // general
  const auto& gTrk = mRecoData.getTrackParam(trkID);
  if (gTrk.getPt() < mParams->minPt || gTrk.getPt() > mParams->maxPt) {
    return false;
  }
  if (std::abs(gTrk.getEta()) > mParams->maxEta) {
    return false;
  }
  if (mUseMC && checkMCTruth) {
    const auto& itsLbl = mRecoData.getTrackMCLabel(contributorsGID[GTrackID::ITS]);
    if (!itsLbl.isValid()) {
      return false;
    }
    if (contributorsGID[GTrackID::TPC].isIndexSet()) {
      const auto& tpcLbl = mRecoData.getTrackMCLabel(contributorsGID[GTrackID::TPC]);
      if (itsLbl != tpcLbl) {
        return false;
      }
    }
    if (contributorsGID[GTrackID::TRD].isIndexSet()) {
      // TODO
    }
    if (contributorsGID[GTrackID::TOF].isIndexSet()) {
      const auto& tofLbls = mRecoData.getTOFClustersMCLabels()->getLabels(contributorsGID[GTrackID::TOF]);
      for (const auto& lbl : tofLbls) {
        if (lbl.isValid()) {
          return true;
        }
      }
    }
  }
  return true;
}

T2VMap TrackingStudySpec::buildT2V(bool includeCont, bool requireMCMatch) const
{
  // build track->vertex assoc., maybe including contributor tracks
  auto pvvec = mRecoData.getPrimaryVertices();
  auto trackIndex = mRecoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = mRecoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                                // last entry is for unassigned tracks, ignore them
  T2VMap t2v;
  for (size_t iv = 0; iv < nv; ++iv) {
    const auto& pv = pvvec[iv];
    if (pv.getNContributors() - 1 < mParams->minPVCont) {
      continue;
    }
    if (requireMCMatch) {
      auto pvl = mRecoData.getPrimaryVertexMCLabel(iv);
    }
    const auto& vtxRef = vtxRefs[iv];
    int it = vtxRef.getFirstEntry(), itLim = it + vtxRef.getEntries();
    for (; it < itLim; it++) {
      const auto& tvid = trackIndex[it];
      if (tvid.isAmbiguous()) {
        continue;
      }
      if (!mRecoData.isTrackSourceLoaded(tvid.getSource())) {
        continue;
      }
      if (mUseMC && requireMCMatch) {
        const auto& pvlbl = mRecoData.getPrimaryVertexMCLabel(iv);
        if (pvlbl.getEventID() != mRecoData.getTrackMCLabel(tvid).getEventID()) {
          continue;
        }
      }
      t2v[tvid] = iv;
      if (includeCont) {
        auto contributorsGID = mRecoData.getSingleDetectorRefs(tvid);
        for (int cis = 0; cis < GTrackID::NSources; cis++) {
          const auto cdm = GTrackID::getSourceDetectorsMask(cis);
          if (!mRecoData.isTrackSourceLoaded(cis) || !cdm[DetID::ITS] || !contributorsGID[cis].isIndexSet()) {
            continue;
          }
          if (mUseMC && requireMCMatch) {
            const auto& pvlbl = mRecoData.getPrimaryVertexMCLabel(iv);
            if (pvlbl.getEventID() != mRecoData.getTrackMCLabel(contributorsGID[cis]).getEventID()) {
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

bool TrackingStudySpec::refitITSPVTrack(o2::track::TrackParCov& trFit, GTrackID gidx)
{
  if (gidx.getSource() != GTrackID::ITS) {
    return false;
  }
  static auto pvvec = mRecoData.getPrimaryVertices();
  static auto t2v = buildT2V(true, true);
  static std::vector<unsigned int> itsTracksROF;
  if (static bool done{false}; !done) {
    done = true;
    const auto& itsTracksROFRec = mRecoData.getITSTracksROFRecords();
    itsTracksROF.resize(mRecoData.getITSTracks().size());
    for (unsigned irf = 0, cnt = 0; irf < itsTracksROFRec.size(); irf++) {
      int ntr = itsTracksROFRec[irf].getNEntries();
      for (int itr = 0; itr < ntr; itr++) {
        itsTracksROF[cnt++] = irf;
      }
    }
  }
  auto prop = o2::base::Propagator::Instance();
  std::array<const TrackingCluster*, 8> clArr{nullptr};
  const auto trkIn = mRecoData.getTrackParam(gidx);
  const auto trkOut = mRecoData.getTrackParamOut(gidx);
  const auto& itsTrOrig = mRecoData.getITSTrack(gidx);
  int ncl = itsTrOrig.getNumberOfClusters(), rof = itsTracksROF[gidx.getIndex()];
  const auto& itsTrackClusRefs = mRecoData.getITSTracksClusterRefs();
  int clEntry = itsTrOrig.getFirstClusterEntry();
  const auto propagator = o2::base::Propagator::Instance();
  // convert PV to a fake cluster in the track DCA frame
  const auto& pv = pvvec[t2v[gidx]];
  auto trkPV = trkIn;
  if (!prop->propagateToDCA(pv, trkPV, prop->getNominalBz(), 2.0, mParams->CorrType)) {
    mTrackCounter -= gidx.getSource();
    return false;
  }
  // create base cluster from the PV, with the alpha corresponding to the track at DCA
  float cosAlp = NAN, sinAlp = NAN;
  o2::math_utils::sincos(trkPV.getAlpha(), sinAlp, cosAlp);
  // vertex position rotated to track frame
  TrackingCluster pvCls;
  pvCls.setXYZ((pv.getX() * cosAlp) + (pv.getY() * sinAlp), (-pv.getX() * sinAlp) + (pv.getY() * cosAlp), pv.getZ());
  pvCls.setSigmaY2(0.5f * (pv.getSigmaX2() + pv.getSigmaY2()));
  pvCls.setSigmaZ2(pv.getSigmaZ2());
  clArr[0] = &pvCls;
  for (int icl = 0; icl < ncl; ++icl) { // ITS clusters are referred in layer decreasing order
    clArr[ncl - icl] = &mITScl[itsTrackClusRefs[clEntry + icl]];
  }
  // start refit
  trFit = trkOut;
  trFit.resetCovariance(1'000);
  float chi2{0};
  for (int icl = ncl; icl >= 0; --icl) { // go backwards
    if (!trFit.rotate(clArr[icl]->alpha) || !prop->propagateToX(trFit, clArr[icl]->getX(), prop->getNominalBz(), 0.85, 2.0, mParams->CorrType)) {
      mTrackCounter -= gidx.getSource();
      return false;
    }
    chi2 += trFit.getPredictedChi2(*clArr[icl]);
    if (!trFit.update(*clArr[icl])) {
      mTrackCounter -= gidx.getSource();
      return false;
    }
  }
  // chi2 < conf.maxChi2; should I cut here?
  return true;
};

void TrackingStudySpec::doDCAStudy()
{
  /// analyse DCA of impact parameter for different track types
  LOGP(info, "Doing DCA study");
  mTrackCounter.reset();
  auto prop = o2::base::Propagator::Instance();
  TStopwatch sw;
  sw.Start();
  int nDCAFits{0}, nDCAFitsFail{0};
  auto pvvec = mRecoData.getPrimaryVertices();
  auto trackIndex = mRecoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = mRecoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                                // last entry is for unassigned tracks, ignore them
  auto& stream = (*mDBGOut) << "dca";
  for (int iv = 0; iv < nv; iv++) {
    const auto& pv = pvvec[iv];
    const auto& vtref = vtxRefs[iv];
    for (int is = 0; is < GTrackID::NSources; is++) {
      const auto dm = GTrackID::getSourceDetectorsMask(is);
      if (!mRecoData.isTrackSourceLoaded(is) || !dm[DetID::ITS]) {
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
        auto contributorsGID = mRecoData.getSingleDetectorRefs(vid);
        for (int cis = 0; cis < GTrackID::NSources && cis <= is; cis++) {
          const auto cdm = GTrackID::getSourceDetectorsMask(cis);
          if (!mRecoData.isTrackSourceLoaded(cis) || !cdm[DetID::ITS] || !contributorsGID[cis].isIndexSet()) {
            mTrackCounter &= cis;
            continue;
          }
          if (!selectTrack(contributorsGID[cis])) {
            mTrackCounter &= vid.getSource();
            continue;
          }

          o2::dataformats::DCA dcaInfo;
          const auto& trk = mRecoData.getTrackParam(contributorsGID[cis]);
          auto trkRefit = trk;
          // for ITS standalone tracks instead of having the trk at the pv we refit with the pv
          if (mParams->refitITS && cis == GTrackID::ITS && !refitITSPVTrack(trkRefit, contributorsGID[cis])) {
            mTrackCounter -= cis;
            continue;
          } else {
            trkRefit.invalidate();
          };

          auto trkDCA = trk;
          if (!prop->propagateToDCABxByBz(pv, trkDCA, 2.f, mParams->CorrType, &dcaInfo)) {
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
            const auto& lbl = mRecoData.getTrackMCLabel(contributorsGID[cis]);
            lbl.print();
            o2::dataformats::DCA dcaInfoMC;
            const auto& eve = mMCReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
            o2::dataformats::VertexBase mcEve;
            mcEve.setPos({(float)eve.GetX(), (float)eve.GetY(), (float)eve.GetZ()});
            auto trkC = trk;
            if (!prop->propagateToDCABxByBz(mcEve, trkC, 2.f, mParams->CorrType, &dcaInfoMC)) {
              mTrackCounter -= cis;
              ++nDCAFitsFail;
              continue;
            }
            const auto& mcTrk = mMCReader.getTrack(lbl);
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

void TrackingStudySpec::doDCARefitStudy()
{
  /// analyse DCA of impact parameter for different track types while refitting the PV without the cand track
  LOGP(info, "Doing DCARefit study");
  mTrackCounter.reset();
  auto prop = o2::base::Propagator::Instance();
  TStopwatch sw;
  sw.Start();

  // build track->vertex assoc.
  auto pvvec = mRecoData.getPrimaryVertices();
  auto vtxRefs = mRecoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto nv = vtxRefs.size() - 1;                                // last entry is for unassigned tracks, ignore them
  auto t2v = buildT2V();
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
      if (trk.getPt() < mParams->minPt || trk.getPt() > mParams->maxPt) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (std::abs(trk.getEta()) > mParams->maxEta) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (!t2v.contains(trkID)) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
      if (!selectTrack(trkID, mUseMC)) {
        mTrackCounter &= trkID.getSource();
        return false;
      }
    }
    v2t[t2v[trkID]].push_back(trkID);
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  int nDCAFits{0}, nDCAFitsFail{0};
  auto& stream = (*mDBGOut) << "dcaRefit";
  for (size_t iv = 0; iv < nv; ++iv) {
    const auto& pv = pvvec[iv];
    const auto& trkIDs = v2t[iv];
    if (trkIDs.size() - 1 < mParams->minPVCont) {
      continue;
    }
    std::vector<o2::track::TrackParCov> trks;
    trks.reserve(trkIDs.size());
    for (const auto& trkID : trkIDs) {
      trks.push_back(mRecoData.getTrackParam(trkID));
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
      if (!prop->propagateToDCABxByBz(pv, trkC, 2.f, mParams->CorrType, &dcaInfo)) {
        mTrackCounter -= trkIDs[it].getSource();
        ++nDCAFitsFail;
        continue;
      }
      o2::dataformats::DCA dcaInfoRefit;
      auto trkCRefit = trks[it];
      if (!prop->propagateToDCABxByBz(pv, trkCRefit, 2.f, mParams->CorrType, &dcaInfoRefit)) {
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
        const auto& mcTrk = mMCReader.getTrack(mRecoData.getTrackMCLabel(trkIDs[it]));
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

void TrackingStudySpec::doPullStudy()
{
  // check track pulls compared to mc generation
  LOGP(info, "Doing Pull study");
  mTrackCounter.reset();
  TStopwatch sw;
  sw.Start();
  int nPulls{0}, nPullsFail{0};
  auto prop = o2::base::Propagator::Instance();

  auto checkInTrack = [&](GTrackID trkID) {
    if (!selectTrack(trkID)) {
      mTrackCounter &= trkID.getSource();
      return;
    }
    const auto mcTrk = mMCReader.getTrack(mRecoData.getTrackMCLabel(trkID));
    if (!mcTrk) {
      return;
    }
    auto trk = mRecoData.getTrackParam(trkID);

    // for ITS standalone tracks we add the PV as an additional measurement point
    if (mParams->refitITS && trkID.getSource() == GTrackID::ITS && !refitITSPVTrack(trk, trkID)) {
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
    const auto contTrk = mRecoData.getSingleDetectorRefs(trkID);
    const auto& itsTrk = mRecoData.getITSTrack(contTrk[GTrackID::ITS]);

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

  for (size_t iTrk{0}; iTrk < mRecoData.getITSTracks().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITS));
  }
  for (size_t iTrk{0}; iTrk < mRecoData.getTPCITSTracks().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPC));
  }
  for (size_t iTrk{0}; iTrk < mRecoData.getITSTPCTRDTracksMCLabels().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTRD));
  }
  for (size_t iTrk{0}; iTrk < mRecoData.getITSTPCTOFMatches().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTOF));
  }
  for (size_t iTrk{0}; iTrk < mRecoData.getITSTPCTRDTOFMatches().size(); ++iTrk) {
    checkInTrack(GTrackID(iTrk, GTrackID::ITSTPCTRDTOF));
  }
  sw.Stop();
  LOGP(info, "doPullStudy: accepted {} pulls; rejected {} (in {:.2f} seconds)", nPulls, nPullsFail, sw.RealTime());
  mTrackCounter.print();
}

void TrackingStudySpec::doMCStudy()
{
  LOGP(info, "Doing MC study");
  mTrackCounter.reset();
  TStopwatch sw;
  sw.Start();
  int nTracks{0};

  const int iSrc{0};
  const int nev = mMCReader.getNEvents(iSrc);
  std::unordered_map<o2::MCCompLabel, ParticleInfoExt> info;

  LOGP(info, "** Filling particle table ... ");
  for (int iEve{0}; iEve < nev; ++iEve) {
    const auto& mcTrks = mMCReader.getTracks(iSrc, iEve);
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
  const auto& clusters = mRecoData.getITSClusters();
  const auto& clustersMCLCont = mRecoData.getITSClustersMCLabels();
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
      const auto& lbl = mRecoData.getTrackMCLabel(contributorsGID[det]);
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
    auto contributorsGID = mRecoData.getSingleDetectorRefs(trkID);
    if (!contributorsGID[GTrackID::ITS].isIndexSet()) { // we need of course ITS
      return false;
    }
    const auto& gLbl = mRecoData.getTrackMCLabel(trkID);
    if (!gLbl.isValid()) {
      return false;
    }
    o2::MCCompLabel iLbl(gLbl.getTrackID(), gLbl.getEventID(), gLbl.getSourceID());
    if (!info.contains(iLbl)) {
      return false;
    }
    auto& part = info[iLbl];
    part.recoTrack = mRecoData.getTrackParam(trkID);

    accountLbl(contributorsGID, DetID::ITS);
    accountLbl(contributorsGID, DetID::TPC);
    accountLbl(contributorsGID, DetID::TRD);
    accountLbl(contributorsGID, DetID::TOF);

    ++nTracks;
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  LOGP(info, "Streaming output to tree");
  for (const auto& [_, part] : info) {
    (*mDBGOut) << "mc"
               << "part=" << part
               << "\n";
  }

  sw.Stop();
  LOGP(info, "doMCStudy: accounted {} MCParticles and {} tracks (in {:.2f} seconds)", info.size(), nTracks, sw.RealTime());
}

void TrackingStudySpec::doResidStudy()
{
  LOGP(info, "Doing residual study");
  const auto geom = o2::its::GeometryTGeo::Instance();
  const auto prop = o2::base::Propagator::Instance();
  const float bz = prop->getNominalBz();

  int goodRefit{0}, notPassedSel{0}, fitFail{0};

  auto doRefits = [&](const o2::its::TrackITS& iTrack, const o2::MCCompLabel& lbl) {
    std::array<TrackingCluster, 8> cl;
    std::array<const TrackingCluster*, 8> clArr{nullptr};
    if (mParams->addPVAsCluster) {
      const auto& eve = mMCReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
      dataformats::VertexBase pv;
      auto trFitOut = iTrack.getParamIn();
      pv.setXYZ(eve.GetX(), eve.GetY(), eve.GetZ());
      if (!prop->propagateToDCA(pv, trFitOut, bz, base::Propagator::MAX_STEP, mParams->CorrType)) {
        return;
      }
      pv.setSigmaX(20e-4f);
      pv.setSigmaY(20e-4f);
      pv.setSigmaZ(20e-4f);
      float cosAlp = NAN, sinAlp = NAN;
      o2::math_utils::sincos(trFitOut.getAlpha(), sinAlp, cosAlp);
      cl[0].alpha = trFitOut.getAlpha();
      cl[0].setXYZ((pv.getX() * cosAlp) + (pv.getY() * sinAlp), (-pv.getX() * sinAlp) + (pv.getY() * cosAlp), pv.getZ());
      cl[0].setSigmaY2(0.5f * (pv.getSigmaX2() + pv.getSigmaY2()));
      cl[0].setSigmaZ2(pv.getSigmaZ2());
      cl[0].setSensorID(-1);
      clArr[0] = &cl[0];
    }

    // collect track clusters into layer slots
    int nCl = iTrack.getNClusters();
    for (int i = 0; i < nCl; i++) {
      const auto& curClu = mITScl[mITSclRef[iTrack.getClusterEntry(i)]];
      int sens = curClu.getSensorID();
      int llr = geom->getLayer(sens);
      if (clArr[1 + llr]) {
        LOGP(fatal, "Cluster at lr {} was already assigned, old sens {}, new sens {}", llr, clArr[1 + llr]->getSensorID(), sens);
      }
      clArr[1 + llr] = &curClu;
    }

    std::array<o2::track::TrackParCov, 8> extrapOut, extrapInw;
    float chi2{0};
    if (!align::doBidirRefit(iTrack, clArr, extrapOut, extrapInw, chi2, mParams->useStableRef, mParams->CorrType)) {
      ++fitFail;
      return;
    }

    for (int i = 0; i <= 7; i++) {
      if (clArr[i]) {
        const auto tInt = align::interpolateTrackParCov(extrapInw[i], extrapOut[i]);
        if (!tInt.isValid()) {
          continue;
        }
        auto phi = i == 0 ? tInt.getPhi() : tInt.getPhiPos();
        o2::math_utils::bringTo02Pi(phi);
        (*mDBGOut) << "res"
                   << "dYInt=" << clArr[i]->getY() - tInt.getY()
                   << "dZInt=" << clArr[i]->getZ() - tInt.getZ()
                   << "dYIn=" << clArr[i]->getY() - extrapInw[i].getY()
                   << "dZIn=" << clArr[i]->getZ() - extrapInw[i].getZ()
                   << "dYOut=" << clArr[i]->getY() - extrapOut[i].getY()
                   << "dZOut=" << clArr[i]->getZ() - extrapOut[i].getZ()
                   << "chi2=" << chi2
                   << "clY=" << clArr[i]->getY()
                   << "clZ=" << clArr[i]->getZ()
                   << "clX=" << clArr[i]->getX()
                   << "alpha=" << clArr[i]->alpha
                   << "sens=" << clArr[i]->getSensorID()
                   << "phi=" << phi
                   << "pt=" << tInt.getPt()
                   << "chip=" << constants::detID::getSensorID(clArr[i]->getSensorID())
                   << "lay=" << i - 1
                   << "\n";
      }
    }
    ++goodRefit;
  };

  const auto itsTracks = mRecoData.getITSTracks();
  const auto itsMC = mRecoData.getITSTracksMCLabels();
  for (size_t iTrk{0}; iTrk < itsTracks.size(); ++iTrk) {
    const auto& iTrack = itsTracks[iTrk];
    const auto& lbl = itsMC[iTrk];
    const auto& mc = mMCReader.getTrack(lbl);
    if (std::abs(iTrack.getEta()) > mParams->maxEta || iTrack.getChi2() > mParams->maxChi2 || iTrack.getNClusters() < mParams->minITSCls || iTrack.getPt() < mParams->minPt || !lbl.isCorrect() || !mc->isPrimary()) {
      ++notPassedSel;
      continue;
    }
    doRefits(iTrack, lbl);
  }

  LOGP(info, "\trefitted {} out of {} tracks ({} !sel, {} !fit)", goodRefit, itsTracks.size(), notPassedSel, fitFail);
}

void TrackingStudySpec::doMisalignmentStudy()
{
  LOGP(info, "Doing misalignment study");
  const auto prop = o2::base::Propagator::Instance();
  const auto geom = o2::its::GeometryTGeo::Instance();

  int goodRefit{0}, notPassedSel{0}, fitFail{0}, fitFailMis{0};

  float chi2{0};
  auto writeTree = [&](const char* treeName,
                       const std::array<const TrackingCluster*, 8>& clArr,
                       const std::array<o2::track::TrackParCov, 8>& extrapOut,
                       const std::array<o2::track::TrackParCov, 8>& extrapInw,
                       const o2::MCCompLabel& lbl) {
    for (int i = 0; i <= 7; i++) {
      if (!clArr[i]) {
        continue;
      }
      // interpolated result
      auto tInt = align::interpolateTrackParCov(extrapInw[i], extrapOut[i]);
      if (!tInt.isValid()) {
        continue;
      }
      float dY = clArr[i]->getY() - tInt.getY();
      float dZ = clArr[i]->getZ() - tInt.getZ();
      // MC truth at same (alpha, x)
      o2::track::TrackPar mcTrkAtX;
      const auto mcTrk = mMCReader.getTrack(lbl);
      if (mcTrk) {
        std::array<float, 3> xyz{(float)mcTrk->GetStartVertexCoordinatesX(), (float)mcTrk->GetStartVertexCoordinatesY(), (float)mcTrk->GetStartVertexCoordinatesZ()};
        std::array<float, 3> pxyz{(float)mcTrk->GetStartVertexMomentumX(), (float)mcTrk->GetStartVertexMomentumY(), (float)mcTrk->GetStartVertexMomentumZ()};
        TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrk->GetPdgCode());
        if (pPDG) {
          mcTrkAtX = o2::track::TrackPar(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
          if (mcTrkAtX.rotate(tInt.getAlpha()) && prop->PropagateToXBxByBz(mcTrkAtX, tInt.getX())) {
            auto phi = i == 0 ? tInt.getPhi() : tInt.getPhiPos();
            o2::math_utils::bringTo02Pi(phi);
            (*mDBGOut) << treeName
                       << "trk=" << tInt
                       << "mcTrk=" << mcTrkAtX
                       << "chi2=" << chi2
                       << "dY=" << dY
                       << "dZ=" << dZ
                       << "phi=" << phi
                       << "eta=" << tInt.getEta()
                       << "lay=" << i - 1
                       << "\n";
          }
        }
      }
    }
  };

  const auto itsTracks = mRecoData.getITSTracks();
  const auto itsMC = mRecoData.getITSTracksMCLabels();
  for (size_t iTrk{0}; iTrk < itsTracks.size(); ++iTrk) {
    const auto& iTrack = itsTracks[iTrk];
    if (std::abs(iTrack.getEta()) > mParams->maxEta || iTrack.getChi2() > mParams->maxChi2 || iTrack.getNClusters() < mParams->minITSCls || iTrack.getPt() < mParams->minPt) {
      ++notPassedSel;
      continue;
    }
    const auto& lbl = itsMC[iTrk];
    if (!lbl.isCorrect() || !lbl.isValid()) {
      ++notPassedSel;
      continue;
    }
    const auto& mc = mMCReader.getTrack(lbl);
    if (!mc->isPrimary()) {
      ++notPassedSel;
      continue;
    }

    // ideal clusters
    std::array<TrackingCluster, 8> cl;
    std::array<const TrackingCluster*, 8> clArr{nullptr};
    if (mParams->addPVAsCluster) {
      const auto& eve = mMCReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
      dataformats::VertexBase pv;
      auto trFitOut = iTrack.getParamIn();
      pv.setXYZ(eve.GetX(), eve.GetY(), eve.GetZ());
      if (!prop->propagateToDCA(pv, trFitOut, prop->getNominalBz(), base::Propagator::MAX_STEP, mParams->CorrType)) {
        return;
      }
      pv.setSigmaX(20e-4f);
      pv.setSigmaY(20e-4f);
      pv.setSigmaZ(20e-4f);
      float cosAlp = NAN, sinAlp = NAN;
      o2::math_utils::sincos(trFitOut.getAlpha(), sinAlp, cosAlp);
      cl[0].alpha = trFitOut.getAlpha();
      cl[0].setXYZ((pv.getX() * cosAlp) + (pv.getY() * sinAlp), (-pv.getX() * sinAlp) + (pv.getY() * cosAlp), pv.getZ());
      cl[0].setSigmaY2(0.5f * (pv.getSigmaX2() + pv.getSigmaY2()));
      cl[0].setSigmaZ2(pv.getSigmaZ2());
      cl[0].setSensorID(-1);
      clArr[0] = &cl[0];
    }

    // collect track clusters into layer slots
    int nCl = iTrack.getNClusters();
    for (int i = 0; i < nCl; i++) {
      const auto& curClu = mITScl[mITSclRef[iTrack.getClusterEntry(i)]];
      int sens = curClu.getSensorID();
      int llr = geom->getLayer(sens);
      if (clArr[1 + llr]) {
        LOGP(fatal, "Cluster at lr {} was already assigned, old sens {}, new sens {}", llr, clArr[1 + llr]->getSensorID(), sens);
      }
      clArr[1 + llr] = &curClu;
    }
    std::array<o2::track::TrackParCov, 8> extrapOut, extrapInw;
    chi2 = 0;
    if (!align::doBidirRefit(iTrack, clArr, extrapOut, extrapInw, chi2, mParams->useStableRef, mParams->CorrType)) {
      ++fitFail;
      continue;
    }
    writeTree("idealRes", clArr, extrapOut, extrapInw, lbl);

    // Propagate MC truth to each cluster's (alpha, x) to get true track direction.
    // The shared misalignment evaluators then provide the tracking-frame dy/dz shift.
    const auto mcTrk = mMCReader.getTrack(lbl);
    if (!mcTrk) {
      continue;
    }
    std::array<float, 3> xyz{(float)mcTrk->GetStartVertexCoordinatesX(), (float)mcTrk->GetStartVertexCoordinatesY(), (float)mcTrk->GetStartVertexCoordinatesZ()};
    std::array<float, 3> pxyz{(float)mcTrk->GetStartVertexMomentumX(), (float)mcTrk->GetStartVertexMomentumY(), (float)mcTrk->GetStartVertexMomentumZ()};
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrk->GetPdgCode());
    if (!pPDG) {
      continue;
    }
    o2::track::TrackPar mcPar(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);

    std::array<TrackingCluster, 3> misClArr; // shifted copies for up to 3 IT3 layers
    std::array<const TrackingCluster*, 8> clArrMis{};
    for (int i = 0; i <= 7; i++) {
      clArrMis[i] = clArr[i]; // PV and OB clusters stay the same
    }
    for (int iLay = 0; iLay < 3; ++iLay) {
      if (!clArr[1 + iLay]) {
        continue;
      }
      const auto& orig = *clArr[1 + iLay];
      const int sens = orig.getSensorID();
      if (!constants::detID::isDetITS3(sens)) {
        continue;
      }
      const int sensorID = constants::detID::getSensorID(sens);
      const int layerID = constants::detID::getDetID2Layer(sens);
      const auto& sensorMis = mMisalignment[sensorID];

      // propagate MC track to cluster's tracking frame to get true slopes
      auto mcAtCl = mcPar;
      if (!mcAtCl.rotate(orig.alpha) || !prop->PropagateToXBxByBz(mcAtCl, orig.getX())) {
        clArrMis[1 + iLay] = nullptr; // can't compute slopes -> drop cluster
        continue;
      }
      const align::MisalignmentFrame misFrame{
        .sensorID = sensorID,
        .layerID = layerID,
        .x = orig.getX(),
        .alpha = orig.alpha,
        .z = orig.getZ()};
      const auto slopes = align::computeTrackSlopes(mcAtCl.getSnp(), mcAtCl.getTgl());

      align::MisalignmentShift totalShift;
      if (sensorMis.hasLegendre) {
        const auto shift = align::evaluateLegendreShift(sensorMis, misFrame, slopes);
        if (!shift.accepted) {
          clArrMis[1 + iLay] = nullptr; // shifted outside acceptance
          continue;
        }
        totalShift += shift;
      }
      if (sensorMis.hasInextensional) {
        totalShift += align::evaluateInextensionalShift(sensorMis, misFrame, slopes);
      }

      // create shifted copy: keep x=r (nominal), shift y and z
      misClArr[iLay] = orig;
      misClArr[iLay].setY(orig.getY() + totalShift.dy);
      misClArr[iLay].setZ(orig.getZ() + totalShift.dz);
      misClArr[iLay].setSigmaY2(orig.getSigmaY2() + (mParams->misAlgExtCY[sensorID] * mParams->misAlgExtCY[sensorID]));
      misClArr[iLay].setSigmaZ2(orig.getSigmaZ2() + (mParams->misAlgExtCZ[sensorID] * mParams->misAlgExtCZ[sensorID]));
      clArrMis[1 + iLay] = &misClArr[iLay];
    }

    // refit with shifted clusters
    chi2 = 0;
    if (!align::doBidirRefit(iTrack, clArrMis, extrapOut, extrapInw, chi2, mParams->useStableRef, mParams->CorrType)) {
      ++fitFailMis;
      ++goodRefit; // ideal still succeeded
      continue;
    }
    writeTree("misRes", clArrMis, extrapOut, extrapInw, lbl);

    ++goodRefit;
  }

  LOGP(info, "\tdoMisalignmentStudy: refitted {} out of {} tracks ({} !sel, {} !fit, {} !fitMis)", goodRefit, itsTracks.size(), notPassedSel, fitFail, fitFailMis);
}

DataProcessorSpec getTrackingStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC, bool withPV)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestIT3Clusters(useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  if (withPV) {
    dataRequest->requestPrimaryVertices(useMC);
  }
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
