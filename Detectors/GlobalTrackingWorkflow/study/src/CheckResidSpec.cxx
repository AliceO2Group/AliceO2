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

#include "GlobalTrackingStudy/CheckResidSpec.h"
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
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DeviceSpec.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "DetectorsVertexing/PVertexer.h"
#include "GlobalTrackingStudy/HistoManager.h"
#include <TROOT.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLegendEntry.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TGraph.h>
#include <TF1.h>
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
  CheckResidSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool drawOnly)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mDrawOnly(drawOnly)
  {
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
  void bookHistos();
  void fillHistos(const o2::checkresid::Track& trc);
  void postProcessHistos();
  void drawHistos();

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
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  GTrackID::mask_t mTracksSrc{};

  bool mDrawOnly = false;
  bool mDraw = false;
  bool mFillHistos = true;
  bool mFillTree = true;
  std::vector<std::unique_ptr<o2::HistoManager>> mHManV{};
  o2::HistoManager* mHMan = nullptr;
};

void CheckResidSpec::init(InitContext& ic)
{
  mDraw = true;
  if (!mDrawOnly) {
    mDraw = ic.options().get<bool>("draw-report");
    mFillHistos = !ic.options().get<bool>("no-hist");
    mFillTree = !ic.options().get<bool>("no-tree");
    mNThreads = ic.options().get<int>("nthreads");
  }
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  int lane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  int maxLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
  std::string nm = params.outname;
  if (maxLanes > 1) {
    o2::conf::ConfigurableParam::updateFromString(fmt::format("checkresid.outname={}_{}", nm, lane));
  }
  if (mDraw) {
    mFillHistos = true;
  }
  if (!mDrawOnly && mFillHistos) {
    bookHistos();
  }
  if (!params.ext_hm_list.empty()) {
    auto vecNames = o2::utils::Str::tokenize(params.ext_hm_list, ',');
    auto vecLegends = o2::utils::Str::tokenize(params.ext_leg_list, ',');
    bool useLeg = true;
    if (vecNames.size() != vecLegends.size()) {
      LOGP(warn, "{} legend names provided for {} external histomanagers, will use file names as legends", vecLegends.size(), vecNames.size());
      useLeg = false;
    }
    int cntH = 0;
    for (const auto& vn : vecNames) {
      LOGP(info, "Loading external HistoManager {}", vn);
      mHManV.emplace_back() = std::make_unique<o2::HistoManager>("", vn, true);
      auto hm = mHManV.back().get();
      if (!hm) {
        LOGP(error, "Failed to load histograms from {}", vn);
        mHManV.pop_back();
      } else {
        hm->SetName(useLeg ? vecLegends[cntH].c_str() : vn.c_str());
      }
      cntH++;
    }
  }
  if (mDrawOnly) {
    return;
  }
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
#ifndef WITH_OPENMP
  if (mNThreads > 1) {
    LOGP(warn, "No OpenMP");
  }
  mNThreads = 1;
#endif
  if (mFillTree) {
    nm += ".root";
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(nm.c_str(), "recreate");
  }
}

void CheckResidSpec::run(ProcessingContext& pc)
{
  if (mDrawOnly) {
    drawHistos();
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }
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
        const auto& itsTrack = mRecoData->getITSTrack(gidITS);
        if (itsTrack.getNClusters() < params.minITSCl) {
          continue;
        }
        auto pt = trc.getPt();
        if (pt < params.minPt || pt > params.maxPt) {
          continue;
        }
        if (std::abs(trc.getTgl()) > params.maxTgl) {
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
      if (mDBGOut) {
        (*mDBGOut) << "res" << "tr=" << tr << "\n";
      }
      if (mHMan) {
        fillHistos(tr);
      }
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
      LOGP(error, "Singular det {}, input: {} {} {}", det, s00, s11, s01);
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
  o2::BaseCluster<float> bcPV;
  if (params.addPVAsCluster) {
    float cosAlp, sinAlp;
    pvAlpha = trFitOut.getAlpha();
    o2::math_utils::sincos(trFitOut.getAlpha(), sinAlp, cosAlp); // vertex position rotated to track frame
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
          return false;
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
    auto tid = trackIndex[itr];
    if (tid.isPVContributor() && mRecoData->isTrackSourceLoaded(tid.getSource())) {
      tracks.emplace_back().setPID(mRecoData->getTrackParam(tid).getPID());
      gidsITS.push_back(mRecoData->getITSContributorGID(tid));
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
  track.resetCovariance();
  track.setCov(track.getQ2Pt() * track.getQ2Pt() * track.getCov()[14], 14);
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

void CheckResidSpec::fillHistos(const o2::checkresid::Track& trc)
{
  const auto& params = CheckResidConfig::Instance();
  int np = trc.points.size();
  auto pt = trc.track.getPt();
  if (pt < params.minPt || pt > params.maxPt) {
    return;
  }
  for (int ip = 0; ip < np; ip++) {
    const auto& pnt = trc.points[ip];
    int il = pnt.lr >= 0 ? pnt.lr + 1 : 0;
    mHMan->getHisto2F(il * 10 + 0 * 100)->Fill(pnt.phi, pnt.dy);
    mHMan->getHisto2F(il * 10 + 0 * 100 + 1000)->Fill(pnt.z, pnt.dy);
    mHMan->getHisto2F(il * 10 + 0 * 100 + 2000)->Fill(pt, pnt.dy);
    mHMan->getHisto2F(il * 10 + 0 * 100 + 3000)->Fill(trc.track.getTgl(), pnt.dy);
    if (pnt.sig2y > 0) {
      auto pull = pnt.dy / std::sqrt(pnt.sig2y);
      mHMan->getHisto2F(il * 10 + 0 * 100 + 5)->Fill(pnt.phi, pull);
      mHMan->getHisto2F(il * 10 + 0 * 100 + 5 + 1000)->Fill(pnt.z, pull);
      mHMan->getHisto2F(il * 10 + 0 * 100 + 5 + 2000)->Fill(pt, pull);
      mHMan->getHisto2F(il * 10 + 0 * 100 + 5 + 3000)->Fill(trc.track.getTgl(), pull);
    }
    mHMan->getHisto2F(il * 10 + 1 * 100)->Fill(pnt.phi, pnt.dz);
    mHMan->getHisto2F(il * 10 + 1 * 100 + 1000)->Fill(pnt.z, pnt.dz);
    mHMan->getHisto2F(il * 10 + 1 * 100 + 2000)->Fill(pt, pnt.dz);
    mHMan->getHisto2F(il * 10 + 1 * 100 + 3000)->Fill(trc.track.getTgl(), pnt.dz);
    if (pnt.sig2z > 0) {
      auto pull = pnt.dz / std::sqrt(pnt.sig2z);
      mHMan->getHisto2F(il * 10 + 1 * 100 + 5)->Fill(pnt.phi, pull);
      mHMan->getHisto2F(il * 10 + 1 * 100 + 5 + 1000)->Fill(pnt.z, pull);
      mHMan->getHisto2F(il * 10 + 1 * 100 + 5 + 2000)->Fill(pt, pull);
      mHMan->getHisto2F(il * 10 + 1 * 100 + 5 + 3000)->Fill(trc.track.getTgl(), pull);
    }
  }
  //--------------
  if (trc.trIBOut.getX() > 1 && std::abs(trc.trIBOut.getX() - trc.trOBInw.getX()) < 0.1) {
    for (int ip = 0; ip < 5; ip++) {
      float d = trc.trIBOut.getParam(ip) - trc.trOBInw.getParam(ip);
      mHMan->getHisto2F(10000 + ip * 10)->Fill(trc.trIBOut.getPhiPos(), d);
      mHMan->getHisto2F(11000 + ip * 10)->Fill(trc.trIBOut.getZ(), d);
      mHMan->getHisto2F(12000 + ip * 10)->Fill(pt, d);
      mHMan->getHisto2F(13000 + ip * 10)->Fill(trc.track.getTgl(), d);
      float sg = trc.trIBOut.getCovarElem(ip, ip) + trc.trOBInw.getCovarElem(ip, ip);
      if (sg > 0) {
        auto pull = d / std::sqrt(sg);
        mHMan->getHisto2F(10000 + ip * 10 + 5)->Fill(trc.trIBOut.getPhiPos(), pull);
        mHMan->getHisto2F(11000 + ip * 10 + 5)->Fill(trc.trIBOut.getZ(), pull);
        mHMan->getHisto2F(12000 + ip * 10 + 5)->Fill(pt, pull);
        mHMan->getHisto2F(13000 + ip * 10 + 5)->Fill(trc.track.getTgl(), pull);
      }
    }
  }
}

void CheckResidSpec::bookHistos()
{
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  mHManV.emplace_back() = std::make_unique<o2::HistoManager>("", fmt::format("{}_hman.root", params.outname));
  mHMan = mHManV.back().get();
  mHMan->SetName(params.outname.c_str());
  auto defLogAxis = [](float xMn, float xMx, int nbin) { // get array for log axis
    if (xMn <= 0 || xMx <= xMn || nbin < 2) {
      LOGP(fatal, "Wrong log axis request: xmin = {} xmax = {} nbins = {}", xMn, xMx, nbin);
    }
    auto dx = std::log(xMx / xMn) / nbin;
    std::vector<double> xax(nbin + 1);
    for (int i = 0; i <= nbin; i++) {
      xax[i] = xMn * std::exp(dx * i);
    }
    return xax;
  };
  float minPt = std::max(0.1f, params.minPt), maxPt = std::min(50.f, params.maxPt);
  auto ptax = defLogAxis(minPt, maxPt, params.nBinsPt);

  for (int il = 0; il < 8; il++) {
    std::string lrName = il == 0 ? "Vtx" : fmt::format("Lr{}", il - 1);
    for (int iyz = 0; iyz < 2; iyz++) {
      std::string dname = iyz == 0 ? "dy" : "dz", dtit = iyz == 0 ? "#DeltaY" : "#DeltaZ";
      auto h2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "phi").c_str(), fmt::format("{}_{{{}}} vs {};#phi;{}", dtit, lrName, "#phi", dtit).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -params.maxDYZ[il], params.maxDYZ[il]);
      mHMan->addHisto(h2, il * 10 + iyz * 100);
      auto h2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "phi").c_str(), fmt::format("pull {}_{{{}}} vs {};#phi; pull{}", dtit, lrName, "phi", dtit).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -params.maxPull, params.maxPull);
      mHMan->addHisto(h2p, il * 10 + iyz * 100 + 5);

      auto hz2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "Z").c_str(), fmt::format("{}_{{{}}} vs {};Z;{}", dtit, lrName, "Z", dtit).c_str(), params.nBinsZ, -params.zranges[il], params.zranges[il], params.nBinsRes, -params.maxDYZ[il], params.maxDYZ[il]);
      mHMan->addHisto(hz2, il * 10 + iyz * 100 + 1000);
      auto hz2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "Z").c_str(), fmt::format("pull {}_{{{}}} vs {};Z; pull{}", dtit, lrName, "Z", dtit).c_str(), params.nBinsZ, -params.zranges[il], params.zranges[il], params.nBinsRes, -params.maxPull, params.maxPull);
      mHMan->addHisto(hz2p, il * 10 + iyz * 100 + 5 + 1000);

      auto hpt2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "Pt").c_str(), fmt::format("{}_{{{}}} vs {};p_{{T}};{}", dtit, lrName, "p_{T}", dtit).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -params.maxDYZ[il], params.maxDYZ[il]);
      mHMan->addHisto(hpt2, il * 10 + iyz * 100 + 2000);
      auto hpt2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "Pt").c_str(), fmt::format("pull {}_{{{}}} vs {};p_{{T}}; pull{}", dtit, lrName, "p_{T}", dtit).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -params.maxPull, params.maxPull);
      mHMan->addHisto(hpt2p, il * 10 + iyz * 100 + 5 + 2000);

      auto htgl2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "tgl").c_str(), fmt::format("{}_{{{}}} vs {};tg#lambda;{}", dtit, lrName, "tg#lambda", dtit).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -params.maxDYZ[il], params.maxDYZ[il]);
      mHMan->addHisto(htgl2, il * 10 + iyz * 100 + 3000);
      auto htgl2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "tgl").c_str(), fmt::format("pull {}_{{{}}} vs {};tg#lambda; pull{}", dtit, lrName, "tg#lambda", dtit).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -params.maxPull, params.maxPull);
      mHMan->addHisto(htgl2p, il * 10 + iyz * 100 + 5 + 3000);
    }
  }

  for (int ip = 0; ip < 5; ip++) {
    auto h2 = new TH2F(fmt::format("dPar{}_IBOBphi", ip).c_str(), fmt::format("#Delta par{} IB-OB vs phi;#phi;#Delta par{}", ip, ip).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -params.maxDPar[ip], params.maxDPar[ip]);
    mHMan->addHisto(h2, 10000 + ip * 10);
    auto h2p = new TH2F(fmt::format("dPar{}_IBOBphi_pull", ip).c_str(), fmt::format("pull #Delta par{} IB-OB vs phi;#phi;pull #Delta par{}", ip, ip).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -params.maxPull, params.maxPull);
    mHMan->addHisto(h2p, 10000 + ip * 10 + 5);

    auto hz2 = new TH2F(fmt::format("dPar{}_IBOBz", ip).c_str(), fmt::format("#Delta par{} IB-OB vs Z;Z;#Delta par{}", ip, ip).c_str(), params.nBinsZ, -20., 20., params.nBinsRes, -params.maxDPar[ip], params.maxDPar[ip]);
    mHMan->addHisto(hz2, 11000 + ip * 10);
    auto hz2p = new TH2F(fmt::format("dPar{}_IBOBz_pull", ip).c_str(), fmt::format("pull #Delta par{} IB-OB vs Z;Z;pull #Delta par{}", ip, ip).c_str(), params.nBinsZ, -20., 20., params.nBinsRes, -params.maxPull, params.maxPull);
    mHMan->addHisto(hz2p, 11000 + ip * 10 + 5);

    auto hpt2 = new TH2F(fmt::format("dPar{}_IBOBpt", ip).c_str(), fmt::format("#Delta par{} IB-OB vs pT;p_{{T}};#Delta par{}", ip, ip).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -params.maxDPar[ip], params.maxDPar[ip]);
    mHMan->addHisto(hpt2, 12000 + ip * 10);
    auto hpt2p = new TH2F(fmt::format("dPar{}_IBOBpt_pull", ip).c_str(), fmt::format("pull #Delta par{} IB-OB vs pT;p_{{T}};pull #Delta par{}", ip, ip).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -params.maxPull, params.maxPull);
    mHMan->addHisto(hpt2p, 12000 + ip * 10 + 5);

    auto htgl2 = new TH2F(fmt::format("dPar{}_IBOBtgl", ip).c_str(), fmt::format("#Delta par{} IB-OB vs tg#lambda;tg#lambda;#Delta par{}", ip, ip).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -params.maxDPar[ip], params.maxDPar[ip]);
    mHMan->addHisto(htgl2, 13000 + ip * 10);
    auto htgl2p = new TH2F(fmt::format("dPar{}_IBOBtgl_pull", ip).c_str(), fmt::format("pull #Delta par{} IB-OB vs tg#lambda;tg#lambda;pull #Delta par{}", ip, ip).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -params.maxPull, params.maxPull);
    mHMan->addHisto(htgl2p, 13000 + ip * 10 + 5);
  }
}

void CheckResidSpec::postProcessHistos()
{
  printf("Fitting histos\n");
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  auto gs = new TF1("gs", "gaus", -1, 1);
  TObjArray arr;
  auto* histm = mHMan;
  auto fitSlices = [&](int id) {
    auto h2 = histm->getHisto2F(id);
    if (!h2 || h2->GetEntries() < params.minHistoStat2Fit) {
      return;
    }
    h2->FitSlicesY(gs, 0, -1, 0, "QNR", &arr);
    arr.SetOwner(true);
    TH1* hmean = (TH1*)arr.RemoveAt(1);
    if (hmean) {
      hmean->SetTitle(Form("<%s>", h2->GetTitle()));
      histm->addHisto(hmean, id + 1);
    }
    TH1* hsig = (TH1*)arr.RemoveAt(2);
    if (hsig) {
      hsig->SetTitle(Form("#sigma(%s)", h2->GetTitle()));
      histm->addHisto(hsig, id + 2);
    }
  };
  for (int ioffs = 0; ioffs <= 3; ioffs++) { // vs phi, Z, pT, tgl
    int offs = ioffs * 1000;
    for (int iht = 0; iht < 2; iht++) { // resid, pull
      int offsV = iht == 0 ? 0 : 5;
      for (int il = 0; il < 8; il++) {
        for (int iyz = 0; iyz < 2; iyz++) {
          fitSlices(il * 10 + iyz * 100 + offsV + offs);
        }
      }
      for (int ip = 0; ip < 5; ip++) {
        fitSlices(10000 + ip * 10 + offsV + offs);
      }
    }
  }
  histm->write();
  delete gs;
}

void CheckResidSpec::drawHistos()
{
  gROOT->SetBatch(true);
  gStyle->SetTitleX(0.2);
  gStyle->SetTitleY(0.88);
  gStyle->SetTitleW(0.25);
  gStyle->SetOptStat(0);
  int nhm = mHManV.size();
  std::array<unsigned int, 3> hcol{EColor::kRed, EColor::kBlue, EColor::kGreen + 2};
  std::unique_ptr<TLegend> lg;
  lg = std::make_unique<TLegend>(0.12, 0.13, 0.9, 0.13 + std::min(0.5f, nhm * 0.2f / 3.f));
  lg->SetFillStyle(0);
  lg->SetBorderSize(0);
  for (int i = 0; i < nhm; i++) {
    auto hman = mHManV[i].get();
    if (!hman || hman->GetLast() < 1) {
      continue;
    }
    hman->setMarkerStyle(20 + i + (i % 2) * 4, 0.5);
    hman->setColor(hcol[i % hcol.size()]);
    auto le = lg->AddEntry(hman->getHisto(1), hman->GetName(), "lp");
    le->SetTextColor(hcol[i % hcol.size()]);
  }
  TCanvas cly("cly", "", 600, 800), clz("clz", "", 600, 800), clpar("clpar", "", 600, 800);
  TCanvas czly("czly", "", 600, 800), czlz("czlz", "", 600, 800), czlpar("czlpar", "", 600, 800);
  const auto& params = o2::checkresid::CheckResidConfig::Instance();

  auto AddLabel = [](const char* txt, float x = 0.1, float y = 0.9, int color = kBlack, float size = 0.04) {
    TLatex* lt = new TLatex(x, y, txt);
    lt->SetNDC();
    lt->SetTextColor(color);
    lt->SetTextSize(size);
    lt->Draw();
    return lt;
  };

  auto drawResLr = [this](TCanvas& canv, int offs, const float resMM[8], bool logX) {
    canv.Clear();
    canv.Divide(2, 4);
    int nh = this->mHManV.size();
    for (int i = 0; i < 8; i++) {
      canv.cd(i + 1);
      bool same = false;
      for (int j = 0; j < nh; j++) {
        auto hman = this->mHManV[j].get();
        if (!hman || hman->GetLast() < 1) {
          continue;
        }
        if (auto histo = hman->getHisto(10 * i + offs)) {
          histo->Draw(same ? "same" : "");
          if (!same) {
            histo->SetMinimum(-resMM[i]);
            histo->SetMaximum(resMM[i]);
            same = true;
          }
        }
      }
      gPad->SetGrid();
      gPad->SetLogx(logX);
    }
  };

  auto drawResPar = [this](TCanvas& canv, int offs, const float resMM[8], bool logX) {
    canv.Clear();
    canv.Divide(2, 3);
    int nh = this->mHManV.size();
    for (int i = 0; i < 5; i++) {
      canv.cd(i + 1);
      bool same = false;
      for (int j = 0; j < nh; j++) {
        auto hman = this->mHManV[j].get();
        if (!hman || hman->GetLast() < 1) {
          continue;
        }
        if (auto histo = hman->getHisto(10 * i + offs)) {
          histo->Draw(same ? "same" : "");
          if (!same) {
            histo->SetMinimum(-resMM[i]);
            histo->SetMaximum(resMM[i]);
            same = true;
          }
        }
      }
      gPad->SetGrid();
      gPad->SetLogx(logX);
    }
  };

  cly.Print(Form("%s_hman.pdf[", params.outname.c_str()));
  drawResLr(cly, 1, params.resMMLrY, false);
  cly.cd(2);
  lg->Draw();
  AddLabel("Y residuals", 0.1, 0.95);
  cly.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(clz, 101, params.resMMLrZ, false);
  clz.cd(2);
  lg->Draw();
  AddLabel("Z residuals", 0.1, 0.95);
  clz.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czly, 1001, params.resMMLrY, false);
  czly.cd(2);
  lg->Draw();
  AddLabel("Y residuals", 0.1, 0.95);
  czly.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czlz, 1101, params.resMMLrZ, false);
  czlz.cd(2);
  lg->Draw();
  AddLabel("Z residuals", 0.1, 0.95);
  czlz.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czly, 2001, params.resMMLrY, true);
  czly.cd(2);
  lg->Draw();
  AddLabel("Y residuals", 0.1, 0.95);
  czly.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czlz, 2101, params.resMMLrZ, true);
  czlz.cd(2);
  lg->Draw();
  AddLabel("Z residuals", 0.1, 0.95);
  czlz.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czly, 3001, params.resMMLrY, false);
  czly.cd(2);
  lg->Draw();
  AddLabel("Y residuals", 0.1, 0.95);
  czly.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResLr(czlz, 3101, params.resMMLrZ, false);
  czlz.cd(2);
  lg->Draw();
  AddLabel("Z residuals", 0.1, 0.95);
  czlz.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResPar(clpar, 10001, params.resMMPar, false);
  clpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  clpar.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResPar(czlpar, 11001, params.resMMPar, false);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResPar(czlpar, 12001, params.resMMPar, true);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(Form("%s_hman.pdf", params.outname.c_str()));

  drawResPar(czlpar, 13001, params.resMMPar, false);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(Form("%s_hman.pdf", params.outname.c_str()));

  cly.Print(Form("%s_hman.pdf]", params.outname.c_str()));
}

void CheckResidSpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
  if (mHManV.size()) {
    postProcessHistos();
  }
  if (mDraw) {
    drawHistos();
  }
}

void CheckResidSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
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

DataProcessorSpec getCheckResidSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool drawOnly)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  if (!drawOnly) {
    bool useMC = false;
    dataRequest->requestTracks(srcTracks, useMC);
    dataRequest->requestClusters(srcClusters, useMC);
    dataRequest->requestPrimaryVertices(useMC);
    dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
  }
  auto ggRequest = drawOnly ? std::make_shared<o2::base::GRPGeomRequest>(false, false, false, false, false, o2::base::GRPGeomRequest::None, dataRequest->inputs) : std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                                                                                                                                                                              true,                              // GRPECS=true
                                                                                                                                                                                                              true,                              // GRPLHCIF
                                                                                                                                                                                                              true,                              // GRPMagField
                                                                                                                                                                                                              true,                              // askMatLUT
                                                                                                                                                                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                                                                                                                                                                              dataRequest->inputs, true);
  Options opts;
  if (!drawOnly) {
    opts = Options{
      {"nthreads", VariantType::Int, 1, {"number of threads"}},
      {"no-tree", VariantType::Bool, false, {"do not fill residuals tree"}},
      {"no-hist", VariantType::Bool, false, {"do not fill residuals histograms"}},
      {"draw-report", VariantType::Bool, false, {"fill residuals report"}},
    };
  }

  return DataProcessorSpec{
    "check-resid",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CheckResidSpec>(dataRequest, ggRequest, srcTracks, drawOnly)},
    opts};
}

} // namespace o2::checkresid
