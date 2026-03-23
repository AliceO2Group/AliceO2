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

#include "GlobalTrackingStudy/CheckResid.h"
#include "GlobalTrackingStudy/CheckResidTypes.h"
#include "GlobalTrackingStudy/CheckResidConfig.h"
#include <vector>
#include "ReconstructionDataFormats/Track.h"
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "CommonUtils/NameConf.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DeviceSpec.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "DetectorsVertexing/PVertexer.h"
#ifdef WITH_OPENMP
#include <omp.h>
#endif

// Attention: in case the residuals are checked with geometry different from the one used for initial reconstruction,
// pass a --configKeyValues option for vertex refit as:
// ;pvertexer.useMeanVertexConstraint=false;pvertexer.meanVertexExtraErrSelection=0.2;pvertexer.iniScale2=100;pvertexer.acceptableScale2=10.;
// In any case, it is better to pass ;pvertexer.useMeanVertexConstraint=false;

namespace o2::checkresid
{
using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class CheckResidSpec : public Task
{
 public:
  CheckResidSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC /*, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts*/)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC)
  {
    /*
    mTPCCorrMapsLoader.setLumiScaleType(sclOpts.lumiType);
    mTPCCorrMapsLoader.setLumiScaleMode(sclOpts.lumiMode);
    mTPCCorrMapsLoader.setCheckCTPIDCConsistency(sclOpts.checkCTPIDCconsistency);
    */
  }
  ~CheckResidSpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process();

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  bool refitPV(o2::dataformats::PrimaryVertex& pv, int vid);
  bool refitITStrack(o2::track::TrackParCov& track, GTrackID gid);
  bool processITSTrack(const o2::its::TrackITS& iTrack, const o2::dataformats::PrimaryVertex& pv, o2::checkresid::Track& resTrack);

  o2::globaltracking::RecoContainer* mRecoData = nullptr;
  int mNThreads = 1;
  bool mMeanVertexUpdated = false;
  float mITSROFrameLengthMUS = 0.f;
  o2::dataformats::MeanVertexObject mMeanVtx{};
  std::vector<o2::BaseCluster<float>> mITSClustersArray;    ///< ITS clusters created in run() method from compact clusters
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr; ///< cluster patterns dictionary
  o2::vertexing::PVertexer mVertexer;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false}; ///< MC flag
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  GTrackID::mask_t mTracksSrc{};
};

void CheckResidSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  int lane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  int maxLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
  std::string dbgnm = maxLanes == 1 ? "checkResid.root" : fmt::format("checkResid_t{}.root", lane);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(dbgnm.c_str(), "recreate");
  mNThreads = ic.options().get<int>("nthreads");
#ifndef WITH_OPENMP
  if (mNThreads > 1) {
    LOGP(warn, "No OpenMP");
  }
  mNThreads = 1;
#endif
  // mTPCCorrMapsLoader.init(ic);
}

void CheckResidSpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  mRecoData = &recoData;
  mRecoData->collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  mRecoData = &recoData;
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process();
  mRecoData = nullptr;
}

void CheckResidSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
  // mTPCVDriftHelper.extractCCDBInputs(pc);
  // mTPCCorrMapsLoader.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    const auto& params = o2::checkresid::CheckResidConfig::Instance();
    initOnceDone = true;
    // Note: reading of the ITS AlpideParam needed for ITS timing is done by the RecoContainer
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (!grp->isDetContinuousReadOut(DetID::ITS)) {
      mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3; // ITS ROFrame duration in \mus
    } else {
      mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
    }
    auto geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2G));
    o2::conf::ConfigurableParam::updateFromString("pvertexer.useTimeInChi2=false;");
    mVertexer.init();
  }
  if (mMeanVertexUpdated) {
    mMeanVertexUpdated = false;
    mVertexer.setMeanVertex(&mMeanVtx);
    mVertexer.initMeanVertexConstraint();
  }
  bool updateMaps = false;
  /*
  if (mTPCCorrMapsLoader.isUpdated()) {
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
  }
  */
}

void CheckResidSpec::process()
{
  if (!mITSDict) {
    LOGP(fatal, "ITS data is not loaded");
  }
  const auto itsTracks = mRecoData->getITSTracks();
  //  const auto itsLbls = mRecoData->getITSTracksMCLabels();
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  const auto clusITS = mRecoData->getITSClusters();
  const auto patterns = mRecoData->getITSClustersPatterns();
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  auto pattIt = patterns.begin();
  mITSClustersArray.clear();
  mITSClustersArray.reserve(clusITS.size());

  o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, mITSDict);

  auto pvvec = mRecoData->getPrimaryVertices();
  auto trackIndex = mRecoData->getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = mRecoData->getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto prop = o2::base::Propagator::Instance();
  static int TFCount = 0;
  int nv = vtxRefs.size() - 1;
  std::vector<std::vector<checkresid::Track>> slots;
  slots.resize(mNThreads);
  int nvGood = 0, nvUse = 0, nvRefFail = 0;
  long pvFitDuration{};
  for (int iv = 0; iv < nv; iv++) {
    const auto& vtref = vtxRefs[iv];
    auto pve = pvvec[iv];
    if (pve.getNContributors() < params.minPVContributors) {
      continue;
    }
    nvGood++;
    if (params.refitPV) {
      LOGP(debug, "Refitting PV#{} of {} tracks", iv, pve.getNContributors());
      auto tStartPVF = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      bool res = refitPV(pve, iv);
      pvFitDuration += std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count() - tStartPVF;
      if (!res) {
        nvRefFail++;
        continue;
      }
    }
    nvUse++;
    for (int is = 0; is < GTrackID::NSources; is++) {
      if (!mTracksSrc[is] || !mRecoData->isTrackSourceLoaded(is)) {
        continue;
      }
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      DetID::mask_t dm = GTrackID::getSourceDetectorsMask(is);
      if (!dm[DetID::ITS]) {
        continue;
      }
      if (dm[DetID::TPC] && params.minTPCCl > 0 && !mRecoData->isTrackSourceLoaded(GTrackID::TPC)) {
        LOGP(fatal, "Cut on TPC tracks is requested by they are not loaded");
      }
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        bool pvCont = vid.isPVContributor();
        if (!pvCont && params.pvcontribOnly) {
          continue;
        }
        if (dm[DetID::TPC] && params.minTPCCl > 0 && mRecoData->getTPCTrack(mRecoData->getTPCContributorGID(vid)).getNClusters() < params.minTPCCl) {
          continue;
        }
        auto gidITS = mRecoData->getITSContributorGID(vid);
        if (gidITS.getSource() != GTrackID::ITS) {
          continue;
        }
        const auto& trc = mRecoData->getTrackParam(vid);
        auto pt = trc.getPt();
        if (pt < params.minPt || pt > params.maxPt) {
          continue;
        }
        const auto& itsTrack = mRecoData->getITSTrack(gidITS);
        if (itsTrack.getNClusters() < params.minITSCl) {
          continue;
        }
#ifdef WITH_OPENMP
        auto& accum = slots[omp_get_thread_num()];
#else
        auto& accum = slots[0];
#endif
        auto& resTrack = accum.emplace_back();
        resTrack.gid = vid;
        if (!processITSTrack(itsTrack, pve, resTrack)) {
          accum.pop_back();
          continue;
        }
      }
    }
  }
  // output
  for (const auto& accum : slots) {
    for (const auto& tr : accum) {
      (*mDBGOut) << "res" << "tr=" << tr << "\n";
    }
  }
  LOGP(info, "processed {} PVs out of {} good vertices (out of {} in total), PV refits took {} mus, {} refits failed", nvUse, nvGood, nv, pvFitDuration, nvRefFail);
  TFCount++;
}

bool CheckResidSpec::processITSTrack(const o2::its::TrackITS& iTrack, const o2::dataformats::PrimaryVertex& pv, o2::checkresid::Track& resTrack)
{
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  auto trFitInw = iTrack.getParamOut(); // seed for inward refit
  auto trFitOut = iTrack.getParamIn();  // seed for outward refit
  auto prop = o2::base::Propagator::Instance();
  auto geom = o2::its::GeometryTGeo::Instance();
  float pvAlpha = 0;
  float bz = prop->getNominalBz();
  std::array<const o2::BaseCluster<float>*, 8> clArr{};
  const auto& params = CheckResidConfig::Instance();
  std::array<o2::track::TrackParCov, 8> extrapOut, extrapInw; // 2-way Kalman extrapolations, vertex + 7 layers

  auto rotateTrack = [bz](o2::track::TrackParCov& tr, float alpha, o2::track::TrackPar* refLin) {
    return refLin ? tr.rotate(alpha, *refLin, bz) : tr.rotate(alpha);
  };

  auto accountCluster = [&](int i, std::array<o2::track::TrackParCov, 8>& extrapDest, o2::track::TrackParCov& tr, o2::track::TrackPar* refLin) {
    if (clArr[i]) { // update with cluster
      if (!rotateTrack(tr, i == 0 ? pvAlpha : geom->getSensorRefAlpha(clArr[i]->getSensorID()), refLin) ||
          !prop->propagateTo(tr, refLin, clArr[i]->getX(), true)) {
        return 0;
      }
      extrapDest[i] = tr; // before update
      if (!tr.update(*clArr[i])) {
        return 0;
      }
    } else {
      extrapDest[i].invalidate();
      return -1;
    }
    return 1;
  };

  auto inv2d = [](float s00, float s11, float s01) -> std::array<float, 3> {
    auto det = s00 * s11 - s01 * s01;
    if (det < 1e-16) {
      return {0.f, 0.f, 0.f};
    }
    det = 1.f / det;
    return {s11 * det, s00 * det, -s01 * det};
  };

  resTrack.points.clear();
  if (!prop->propagateToDCA(pv, trFitOut, bz)) {
    LOGP(debug, "Failed to propagateToDCA, {}", trFitOut.asString());
    return false;
  }
  float cosAlp, sinAlp;
  pvAlpha = trFitOut.getAlpha();
  o2::math_utils::sincos(trFitOut.getAlpha(), sinAlp, cosAlp); // vertex position rotated to track frame
  o2::BaseCluster<float> bcPV;
  if (params.addPVAsCluster) {
    bcPV.setXYZ(pv.getX() * cosAlp + pv.getY() * sinAlp, -pv.getX() * sinAlp + pv.getY() * cosAlp, pv.getZ());
    bcPV.setSigmaY2(0.5 * (pv.getSigmaX2() + pv.getSigmaY2()));
    bcPV.setSigmaZ2(pv.getSigmaZ2());
    bcPV.setSensorID(-1);
    clArr[0] = &bcPV;
  }
  // collect all track clusters to array, placing them to layer+1 slot
  int nCl = iTrack.getNClusters();
  for (int i = 0; i < nCl; i++) { // clusters are ordered from the outermost to the innermost
    const auto& curClu = mITSClustersArray[itsClRefs[iTrack.getClusterEntry(i)]];

    int llr = geom->getLayer(curClu.getSensorID());
    if (clArr[1 + llr]) {
      LOGP(error, "Cluster at lr {} was already assigned, old sens {}, new sens {}", llr, clArr[1 + llr]->getSensorID(), curClu.getSensorID());
    }
    clArr[1 + geom->getLayer(curClu.getSensorID())] = &curClu;
  }
  o2::track::TrackPar refLinInw0, refLinOut0, *refLinOut = nullptr, *refLinInw = nullptr;
  o2::track::TrackPar refLinIBOut0, refLinOBInw0, *refLinOBInw = nullptr, *refLinIBOut = nullptr;
  if (params.useStableRef) {
    refLinOut = &(refLinOut0 = trFitOut);
    refLinInw = &(refLinInw0 = trFitInw);
  }
  trFitOut.resetCovariance();
  trFitOut.setCov(trFitOut.getQ2Pt() * trFitOut.getQ2Pt() * trFitOut.getCov()[14], 14);
  trFitInw.resetCovariance();
  trFitInw.setCov(trFitInw.getQ2Pt() * trFitInw.getQ2Pt() * trFitInw.getCov()[14], 14);
  // fit in inward and outward direction
  for (int i = 0; i <= 7; i++) {
    int resOut, resInw;
    // process resOut in ascending order (0-->7) and resInw in descending order (7-->0)
    if (!(resOut = accountCluster(i, extrapOut, trFitOut, refLinOut)) || !(resInw = accountCluster(7 - i, extrapInw, trFitInw, refLinInw))) {
      return false;
    }
    // at layer 3, find the IB track (trIBOut) and the OB track (trOBInw)
    // propagate both trcaks to a common radius, RCompIBOB (12cm), and rotates
    // them to the same reference frame for comparison
    if (i == 3 && resOut == 1 && resInw == 1 && params.doIBOB && nCl == 7) {
      resTrack.trIBOut = trFitOut; // outward track updated at outermost IB layer
      resTrack.trOBInw = trFitInw; // inward track updated at innermost OB layer
      o2::track::TrackPar refLinIBOut0, refLinIBIn0;
      if (refLinOut) {
        refLinIBOut = &(refLinIBOut0 = refLinOut0);
        refLinOBInw = &(refLinOBInw0 = refLinInw0);
      }
      float xRref;
      if (!resTrack.trOBInw.getXatLabR(params.rCompIBOB, xRref, bz) ||
          !prop->propagateTo(resTrack.trOBInw, refLinOBInw, xRref, true) ||
          !rotateTrack(resTrack.trOBInw, resTrack.trOBInw.getPhiPos(), refLinOBInw) || // propagate OB track to ref R and rotate
          !rotateTrack(resTrack.trIBOut, resTrack.trOBInw.getAlpha(), refLinIBOut) ||
          !prop->propagateTo(resTrack.trIBOut, refLinIBOut, resTrack.trOBInw.getX(), true)) { // rotate OB track to same frame and propagate to same X
                                                                                              // if any propagation or rotation steps fail, invalidate both tracks
        return false;
      }
    }
  }

  bool innerDone = false;
  if (params.doResid) {
    for (int i = 0; i <= 7; i++) {
      if (clArr[i]) {
        // calculate interpolation as a weighted mean of inward/outward extrapolations to this layer
        const auto &tInw = extrapInw[i], &tOut = extrapOut[i];
        auto wInw = inv2d(tInw.getSigmaY2(), tInw.getSigmaZ2(), tInw.getSigmaZY());
        auto wOut = inv2d(tOut.getSigmaY2(), tOut.getSigmaZ2(), tOut.getSigmaZY());
        if (wInw[0] == 0.f || wOut[0] == 0.f) {
          return -1;
        }
        std::array<float, 3> wTot = {wInw[0] + wOut[0], wInw[1] + wOut[1], wInw[2] + wOut[2]};
        auto cTot = inv2d(wTot[0], wTot[1], wTot[2]);
        auto ywi = wInw[0] * tInw.getY() + wInw[2] * tInw.getZ() + wOut[0] * tOut.getY() + wOut[2] * tOut.getZ();
        auto zwi = wInw[2] * tInw.getY() + wInw[1] * tInw.getZ() + wOut[2] * tOut.getY() + wOut[1] * tOut.getZ();
        auto yw = ywi * cTot[0] + zwi * cTot[2];
        auto zw = ywi * cTot[2] + zwi * cTot[1];
        // posCl.push_back(clArr[i]->getXYZGlo(*o2::its::GeometryTGeo::Instance()));
        auto phi = i == 0 ? tInw.getPhi() : tInw.getPhiPos();
        o2::math_utils::bringTo02Pi(phi);
        resTrack.points.emplace_back(clArr[i]->getY() - yw, clArr[i]->getZ() - zw, cTot[0] + clArr[i]->getSigmaY2(), cTot[1] + clArr[i]->getSigmaZ2(), phi, clArr[i]->getZ(), clArr[i]->getSensorID(), i - 1);
        if (!innerDone) {
          resTrack.track = tInw;
          innerDone = true;
        }
      } else {
        LOGP(debug, "No cluster on lr {}", i);
      }
    }
  }
  return true;
}

bool CheckResidSpec::refitPV(o2::dataformats::PrimaryVertex& pv, int vid)
{
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  std::vector<o2::track::TrackParCov> tracks;
  std::vector<bool> useTrack;
  std::vector<GTrackID> gidsITS;
  int ntr = pv.getNContributors(), ntrIni = ntr;
  tracks.reserve(ntr);
  useTrack.reserve(ntr);
  gidsITS.reserve(ntr);
  const auto& vtref = mRecoData->getPrimaryVertexMatchedTrackRefs()[vid];
  auto trackIndex = mRecoData->getPrimaryVertexMatchedTracks();
  int itr = vtref.getFirstEntry(), itLim = itr + vtref.getEntries();
  for (; itr < itLim; itr++) {
    auto vid = trackIndex[itr];
    if (vid.isPVContributor()) {
      tracks.emplace_back().setPID(mRecoData->getTrackParam(vid).getPID());
      gidsITS.push_back(mRecoData->getITSContributorGID(vid));
    }
  }
  ntr = tracks.size();
  useTrack.resize(ntr);
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
  for (int itr = 0; itr < ntr; itr++) {
    if (!(useTrack[itr] = refitITStrack(tracks[itr], gidsITS[itr]))) {
      tracks[itr] = mRecoData->getTrackParam(gidsITS[itr]); // this track will not be used but participates in prepareVertexRefit
    }
  }
  ntr = 0;
  for (auto v : useTrack) {
    ntr++;
  }
  if (ntr < params.minPVContributors || !mVertexer.prepareVertexRefit(tracks, pv)) {
    LOGP(warn, "Abandon vertex refit: NcontribNew = {} vs NcontribOld = {}", ntr, ntrIni);
    return false;
  }
  LOGP(debug, "Original vtx: Nc:{} {}, chi2={}", pv.getNContributors(), pv.asString(), pv.getChi2());
  auto pvSave = pv;
  pv = mVertexer.refitVertexFull(useTrack, pv);
  LOGP(debug, "Refitted vtx: Nc:{} {}, chi2={}", ntr, pv.asString(), pv.getChi2());
  if (pv.getChi2() < 0.f) {
    LOGP(warn, "Failed to refit PV {}", pvSave.asString());
    return false;
  }
  return true;
}

bool CheckResidSpec::refitITStrack(o2::track::TrackParCov& track, GTrackID gid)
{
  // destination tack might have non-default PID assigned
  const auto& trkITS = mRecoData->getITSTrack(gid);
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  const auto& params = CheckResidConfig::Instance();
  auto pid = track.getPID();
  track = trkITS.getParamOut();
  track.setPID(pid);
  auto nCl = trkITS.getNumberOfClusters();
  auto geom = o2::its::GeometryTGeo::Instance();
  auto prop = o2::base::Propagator::Instance();
  float bz = prop->getNominalBz();
  o2::track::TrackPar refLin{track};

  for (int iCl = 0; iCl < nCl; iCl++) { // clusters are stored from outer to inner layers
    const auto& cls = mITSClustersArray[itsClRefs[trkITS.getClusterEntry(iCl)]];
    auto alpha = geom->getSensorRefAlpha(cls.getSensorID());
    if (!(params.useStableRef ? track.rotate(alpha, refLin, bz) : track.rotate(alpha)) ||
        !prop->propagateTo(track, params.useStableRef ? &refLin : nullptr, cls.getX(), true)) {
      LOGP(debug, "refitITStrack failed on propagation to cl#{}, alpha={}, x={} | {}", iCl, alpha, cls.getX(), track.asString());
      return false;
    }
    if (!track.update(cls)) {
      LOGP(debug, "refitITStrack failed on update with cl#{}, | {}", iCl, track.asString());
      return false;
    }
  }
  return true;
}

void CheckResidSpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void CheckResidSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  /*
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    return;
  }
  */
  if (matcher == ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOG(info) << "Imposing new MeanVertex: " << ((const o2::dataformats::MeanVertexObject*)obj)->asString();
    mMeanVtx = *(const o2::dataformats::MeanVertexObject*)obj;
    mMeanVertexUpdated = true;
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

DataProcessorSpec getCheckResidSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC /*, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts*/)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracks, useMC);
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
  dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
  Options opts{
    {"nthreads", VariantType::Int, 1, {"number of threads"}},
  };
  //  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  //  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);

  return DataProcessorSpec{
    "check-resid",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CheckResidSpec>(dataRequest, ggRequest, srcTracks, useMC /*, sclOpts*/)},
    opts};
}

} // namespace o2::checkresid
