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
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>
#include "ReconstructionDataFormats/Track.h"
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsITSMFT/ClustersPerLayer.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
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
using ITSClusters = o2::itsmft::ClustersPerLayer<o2::BaseCluster<float>>;
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class CheckResidSpec final : public Task
{
 public:
  CheckResidSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool drawOnly, bool postProcOnly)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mDrawOnly(drawOnly), mPostProcOnly(postProcOnly)
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
  bool processITSTrack(const o2::its::TrackITS& iTrack, const o2::dataformats::PrimaryVertex& pv, o2::checkresid::Track& resTrack, o2::track::PID pid);
  void bookHistos();
  void fillHistos(const o2::checkresid::Track& trc);
  void postProcessHistos();
  void drawHistos();

  o2::itsmft::ChipMappingITS mChipMap;
  o2::globaltracking::RecoContainer* mRecoData = nullptr;
  int mNThreads = 1;
  bool mMeanVertexUpdated = false;
  float mITSROFrameLengthMUS = 0.f;
  o2::dataformats::MeanVertexObject mMeanVtx{};
  ITSClusters mITSClustersArray;                            ///< ITS clusters from compact clusters, by composed ID
  std::vector<int16_t> mITSClusterSizes;                    ///< number of pixels per converted cluster
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr; ///< cluster patterns dictionary
  o2::vertexing::PVertexer mVertexer;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  GTrackID::mask_t mTracksSrc{};

  bool mDrawOnly = false;
  bool mPostProcOnly = false;
  bool mDraw = false;
  bool mFillHistos = true;
  bool mFillTree = true;
  std::vector<std::unique_ptr<o2::HistoManager>> mHManV{};
  std::vector<o2::dataformats::PrimaryVertex> mPVUsed;
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
    o2::conf::ConfigurableParam::updateFromString(fmt::format("checkresid.outname={}_t{}", nm, lane));
  }
  if (mDraw) {
    mFillHistos = true;
  }
  if (!mDrawOnly && mFillHistos) {
    bookHistos();
  }
  if (!params.ext_hm_list.empty()) {
    auto vecNames = o2::utils::Str::tokenize(params.ext_hm_list, ",;");
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
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(fmt::format("{}.root", params.outname).c_str(), "recreate");
  }
}

void CheckResidSpec::run(ProcessingContext& pc)
{
  bool quit = false;
  if (mPostProcOnly) {

    postProcessHistos();
    quit = true;
  }
  if (mDrawOnly) {
    drawHistos();
    quit = true;
  }
  if (quit) {
    if (mPostProcOnly && o2::checkresid::CheckResidConfig::Instance().savePostprocessed) {
      for (const auto& hm : mHManV) {
        if (hm) {
          hm->write();
        }
      }
    }
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
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  int nLr = mDataRequest->getITSPerLayer() ? o2::globaltracking::MaxITSLayers : 1;
  mITSClustersArray.init(nLr);
  mITSClusterSizes.clear();
  for (int lr = 0; lr < nLr; lr++) { // with a single (monolithic) input all clusters are in the layer slot 0
    mITSClustersArray.beginLayer(lr);
    const auto clusITS = mRecoData->getITSClusters(lr);
    auto pattIt = mRecoData->getITSClustersPatterns(lr).begin();
    {
      auto sizePattIt = mRecoData->getITSClustersPatterns(lr).begin();
      mITSClustersArray.getClusters().reserve(mITSClustersArray.size() + clusITS.size());
      mITSClusterSizes.reserve(mITSClusterSizes.size() + clusITS.size());
      for (const auto& clu : clusITS) {
        const auto pattID = clu.getPatternID();
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mITSDict->isGroup(pattID)) {
          o2::itsmft::ClusterPattern patt(sizePattIt);
          mITSClusterSizes.emplace_back(patt.getNPixels());
        } else {
          mITSClusterSizes.emplace_back(mITSDict->getNpixels(pattID));
        }
      }
    }
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray.getClusters(), mITSDict);
  }
  mITSClustersArray.finalize();

  auto pvvec = mRecoData->getPrimaryVertices();
  auto trackIndex = mRecoData->getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = mRecoData->getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto prop = o2::base::Propagator::Instance();
  static int TFCount = 0;
  int nv = vtxRefs.size() - 1;
  std::vector<std::vector<checkresid::Track>> slots;
  mPVUsed.clear();
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
    mPVUsed.push_back(pve);
    const auto& itsContRefs = vtref.getITSGloContributors();
    int idMinITSGlo = 0, idMaxITSGlo = 0;
    if (params.useITSGloContributors) {
      if (itsContRefs.getFirstEntry() < vtref.getEntries() && itsContRefs.getEntries() == 0) {
        LOGP(fatal, "Usage of stored ITS global contributors is requested but they are missing");
      }
      idMinITSGlo = itsContRefs.getFirstEntry();
      idMaxITSGlo = idMinITSGlo + itsContRefs.getEntries();
    }
    int cntPVCont = 0;
    for (int is = 0; is < GTrackID::NSources; is++) {
      if (!params.useITSGloContributors && (!mTracksSrc[is] || !mRecoData->isTrackSourceLoaded(is))) {
        continue;
      }
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      DetID::mask_t dm = GTrackID::getSourceDetectorsMask(is);
      if (!dm[DetID::ITS]) {
        continue;
      }
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        bool pvCont = vid.isPVContributor();
        if (!pvCont && (params.pvcontribOnly || params.useITSGloContributors)) {
          continue;
        }
        GTrackID gidITS;
        o2::track::PID pidITS;
        if (params.useITSGloContributors) {
          int id = idMinITSGlo + cntPVCont;
          if (id >= idMaxITSGlo) {
            LOGP(fatal, "Calculated GlobalContributor ITS track index {} exceeds number of stored indices {}", id, itsContRefs.getEntries());
          }
          pidITS = trackIndex[id].getSource();
          gidITS = GTrackID(trackIndex[id].getIndex(), GTrackID::ITS);
          cntPVCont++;
        } else {
          gidITS = mRecoData->getITSContributorGID(vid);
          if (gidITS.getSource() != GTrackID::ITS) {
            continue;
          }
        }
        const auto& itsTrack = mRecoData->getITSTrack(gidITS);
        if (itsTrack.getNClusters() < params.minITSCl) {
          continue;
        }
        const auto& trc = params.useITSGloContributors ? ((o2::track::TrackParCov&)itsTrack) : mRecoData->getTrackParam(vid);
        if (!params.useITSGloContributors) {
          pidITS = trc.getPID();
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
        if (!processITSTrack(itsTrack, pve, resTrack, pidITS)) {
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
  if (mDBGOut) {
    (*mDBGOut) << "pvUsed" << "pv=" << mPVUsed << "\n";
  }
  if (mHMan) {
    for (const auto& pv : mPVUsed) {
      mHMan->getHisto(20000 + 0)->Fill(pv.getX());
      mHMan->getHisto(20000 + 1)->Fill(pv.getY());
      mHMan->getHisto(20000 + 2)->Fill(pv.getZ());
      mHMan->getHisto(20000 + 3)->Fill(pv.getNContributors());
    }
  }
  LOGP(info, "processed {} PVs out of {} good vertices (out of {} in total), PV refits took {} mus, {} refits failed", nvUse, nvGood, nv, pvFitDuration, nvRefFail);
  TFCount++;
}

bool CheckResidSpec::processITSTrack(const o2::its::TrackITS& iTrack, const o2::dataformats::PrimaryVertex& pv, o2::checkresid::Track& resTrack, o2::track::PID pid)
{
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  auto trFitInw = iTrack.getParamOut(); // seed for inward refit
  auto trFitOut = iTrack.getParamIn();  // seed for outward refit
  trFitInw.setPID(pid);
  trFitOut.setPID(pid);
  auto prop = o2::base::Propagator::Instance();
  auto geom = o2::its::GeometryTGeo::Instance();
  float pvAlpha = 0;
  float bz = prop->getNominalBz();
  std::array<const o2::BaseCluster<float>*, 8> clArr{};
  std::array<int16_t, 8> clSizeArr{};
  const auto& params = CheckResidConfig::Instance();
  std::array<o2::track::TrackParCov, 8> extrapOut, extrapInw; // 2-way Kalman extrapolations, vertex + 7 layers
  o2::track::TrackParCov trIBOutForIOB, trOBInwForIOB;        // IB/OB sub-tracks (pre-R12 state) used for IOB residuals

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
    clSizeArr[1 + geom->getLayer(curClu.getSensorID())] = mITSClusterSizes[itsClRefs[iTrack.getClusterEntry(i)]];
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
    if (i == 3 && resOut == 1 && resInw == 1 && (params.doIBOB || params.doIBOBresiduals) && nCl == 7) {
      if (params.doIBOBresiduals) { // keep the un-rotated IB/OB sub-tracks for the per-layer IOB residuals below
        trIBOutForIOB = trFitOut;
        trOBInwForIOB = trFitInw;
      }
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

  // extrapolate the IB(OB) sub-track, w/o intermediate updates, to the cluster (or PV pseudo-cluster) of an OB(IB) layer, for the IOB residuals
  auto extrapIOB = [&](const o2::track::TrackParCov& srcTr, int i) -> std::array<float, 4> { // {dy, dz, sig2y, sig2z}, sig2y<=0 : invalid
    o2::track::TrackParCov tr = srcTr;
    auto alpha = i == 0 ? pvAlpha : geom->getSensorRefAlpha(clArr[i]->getSensorID());
    if (!rotateTrack(tr, alpha, nullptr) || !prop->propagateTo(tr, nullptr, clArr[i]->getX(), true)) {
      return {0.f, 0.f, 0.f, 0.f};
    }
    return {clArr[i]->getY() - tr.getY(), clArr[i]->getZ() - tr.getZ(), tr.getSigmaY2() + clArr[i]->getSigmaY2(), tr.getSigmaZ2() + clArr[i]->getSigmaZ2()};
  };
  bool haveIOBTracks = params.doIBOBresiduals && nCl == 7 && trIBOutForIOB.getX() > 1 && trOBInwForIOB.getX() > 1;

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
        std::array<float, 4> iob{0.f, 0.f, 0.f, 0.f};
        if (haveIOBTracks) {
          int lr = i - 1; // -1 = vertex
          if (lr <= 2) {  // vertex or IB layer: extrapolate the OB sub-track to it (the vertex constrains the IB outward fit)
            iob = extrapIOB(trOBInwForIOB, i);
          } else { // OB layer: extrapolate the IB sub-track to it
            iob = extrapIOB(trIBOutForIOB, i);
          }
        }
        resTrack.points.emplace_back(clArr[i]->getY() - yw, clArr[i]->getZ() - zw, cTot[0] + clArr[i]->getSigmaY2(), cTot[1] + clArr[i]->getSigmaZ2(),
                                     iob[0], iob[1], iob[2], iob[3], phi, clArr[i]->getZ(), 0.5 * (tInw.getSnp() + tOut.getSnp()), clSizeArr[i], clArr[i]->getSensorID(), i - 1);
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
  const auto& itsContRefs = vtref.getITSGloContributors();
  if (params.useITSGloContributors && itsContRefs.getEntries()) {
    int itr = itsContRefs.getFirstEntry(), itLim = itr + itsContRefs.getEntries();
    for (; itr < itLim; itr++) {
      auto tid = trackIndex[itr]; // these are ITS tracks, the Source part is substituted by the PID!!!
      tracks.emplace_back().setPID(tid.getSource());
      gidsITS.emplace_back(tid.getIndex(), GTrackID::ITS);
    }
  } else {
    int itr = vtref.getFirstEntry(), itLim = itr + vtref.getEntries();
    for (; itr < itLim; itr++) {
      auto tid = trackIndex[itr];
      if (tid.isPVContributor() && mRecoData->isTrackSourceLoaded(tid.getSource())) {
        tracks.emplace_back().setPID(mRecoData->getTrackParam(tid).getPID());
        gidsITS.push_back(mRecoData->getITSContributorGID(tid));
      }
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
  auto geom = o2::its::GeometryTGeo::Instance();
  int np = trc.points.size();
  auto pt = trc.track.getPt();
  if (pt < params.minPt || pt > params.maxPt) {
    return;
  }
  auto tgl = trc.track.getTgl();
  // fills, for a given id offset, the dy/dz (+pull) vs phi/Z/pT/tgl histos of layer slot il
  auto fillLayerResid = [&](int idOffset, int il, float phi, float z, float dy, float dz, float sig2y, float sig2z) {
    mHMan->getHisto2F(idOffset + il * 10 + 0 * 100)->Fill(phi, dy);
    mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 1000)->Fill(z, dy);
    mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 2000)->Fill(pt, dy);
    mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 3000)->Fill(tgl, dy);
    if (sig2y > 0) {
      auto pull = dy / std::sqrt(sig2y);
      mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 5)->Fill(phi, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 5 + 1000)->Fill(z, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 5 + 2000)->Fill(pt, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 0 * 100 + 5 + 3000)->Fill(tgl, pull);
    }
    mHMan->getHisto2F(idOffset + il * 10 + 1 * 100)->Fill(phi, dz);
    mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 1000)->Fill(z, dz);
    mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 2000)->Fill(pt, dz);
    mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 3000)->Fill(tgl, dz);
    if (sig2z > 0) {
      auto pull = dz / std::sqrt(sig2z);
      mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 5)->Fill(phi, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 5 + 1000)->Fill(z, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 5 + 2000)->Fill(pt, pull);
      mHMan->getHisto2F(idOffset + il * 10 + 1 * 100 + 5 + 3000)->Fill(tgl, pull);
    }
  };
  // fills, for a given id offset, the per-chip dy/dz (+pull) histo of a layer point on sensor sens
  auto fillChipResid = [&](int idOffset, int il, int lr, int sens, float dy, float dz, float sig2y, float sig2z) {
    if (sens >= 0 && geom->getLayer(sens) == lr) {
      const int halfBarrel = geom->getHalfBarrel(sens);
      const int chip = geom->getFirstChipIndex(lr) + geom->getChipIdInLayer(sens);
      const int chipHistoID = idOffset + (2 * il + halfBarrel) * 100 + 15000;
      if (halfBarrel >= 0 && halfBarrel < 2) {
        mHMan->getHisto2F(chipHistoID)->Fill(chip, dy);
        mHMan->getHisto2F(chipHistoID + 10)->Fill(chip, dz);
        if (sig2y > 0) {
          mHMan->getHisto2F(chipHistoID + 5)->Fill(chip, dy / std::sqrt(sig2y));
        }
        if (sig2z > 0) {
          mHMan->getHisto2F(chipHistoID + 15)->Fill(chip, dz / std::sqrt(sig2z));
        }
      }
    }
  };

  for (int ip = 0; ip < np; ip++) {
    const auto& pnt = trc.points[ip];
    int il = pnt.lr >= 0 ? pnt.lr + 1 : 0;
    fillLayerResid(0, il, pnt.phi, pnt.z, pnt.dy, pnt.dz, pnt.sig2y, pnt.sig2z);
    if (pnt.lr >= 0) {
      fillChipResid(0, il, pnt.lr, pnt.sens, pnt.dy, pnt.dz, pnt.sig2y, pnt.sig2z);
    }
    if (pnt.sig2yIOB > 0) { // IOB residual was computed for this point
      fillLayerResid(100000, il, pnt.phi, pnt.z, pnt.dyIOB, pnt.dzIOB, pnt.sig2yIOB, pnt.sig2zIOB);
      if (pnt.lr >= 0) {
        fillChipResid(100000, il, pnt.lr, pnt.sens, pnt.dyIOB, pnt.dzIOB, pnt.sig2yIOB, pnt.sig2zIOB);
      }
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

  // books, for a given id offset (0 for the standard layer residuals, 100000 for the IOB ones) and name/title suffix,
  // the dy/dz (+pull) vs phi/Z/pT/tgl histos and the per-chip histos; bookVtx controls whether the vtx (il=0) slot is booked
  auto bookLayerResidHistos = [&](int idOffset, const std::string& suf, bool bookVtx) {
    const float dyzScale = idOffset ? params.IOBScaleFactor : 1.f; // IOB residuals have a much larger range than the standard ones
    for (int il = bookVtx ? 0 : 1; il < 8; il++) {
      std::string lrName = il == 0 ? "Vtx" : fmt::format("Lr{}", il - 1);
      const float maxDYZ = params.maxDYZ[il] * dyzScale;
      for (int iyz = 0; iyz < 2; iyz++) {
        std::string dname = fmt::format("{}{}", iyz == 0 ? "dy" : "dz", suf), dtit = fmt::format("{}{}", iyz == 0 ? "#DeltaY" : "#DeltaZ", suf.empty() ? "" : ("_" + suf));
        auto h2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "phi").c_str(), fmt::format("{}_{{{}}} vs {};#phi;{}", dtit, lrName, "#phi", dtit).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -maxDYZ, maxDYZ);
        mHMan->addHisto(h2, idOffset + il * 10 + iyz * 100);
        auto h2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "phi").c_str(), fmt::format("pull {}_{{{}}} vs {};#phi; pull{}", dtit, lrName, "phi", dtit).c_str(), params.nBinsPhi, 0, TMath::Pi() * 2, params.nBinsRes, -params.maxPull, params.maxPull);
        mHMan->addHisto(h2p, idOffset + il * 10 + iyz * 100 + 5);

        auto hz2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "Z").c_str(), fmt::format("{}_{{{}}} vs {};Z;{}", dtit, lrName, "Z", dtit).c_str(), params.nBinsZ, -params.zranges[il], params.zranges[il], params.nBinsRes, -maxDYZ, maxDYZ);
        mHMan->addHisto(hz2, idOffset + il * 10 + iyz * 100 + 1000);
        auto hz2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "Z").c_str(), fmt::format("pull {}_{{{}}} vs {};Z; pull{}", dtit, lrName, "Z", dtit).c_str(), params.nBinsZ, -params.zranges[il], params.zranges[il], params.nBinsRes, -params.maxPull, params.maxPull);
        mHMan->addHisto(hz2p, idOffset + il * 10 + iyz * 100 + 5 + 1000);

        auto hpt2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "Pt").c_str(), fmt::format("{}_{{{}}} vs {};p_{{T}};{}", dtit, lrName, "p_{T}", dtit).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -maxDYZ, maxDYZ);
        mHMan->addHisto(hpt2, idOffset + il * 10 + iyz * 100 + 2000);
        auto hpt2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "Pt").c_str(), fmt::format("pull {}_{{{}}} vs {};p_{{T}}; pull{}", dtit, lrName, "p_{T}", dtit).c_str(), params.nBinsPt, ptax.data(), params.nBinsRes, -params.maxPull, params.maxPull);
        mHMan->addHisto(hpt2p, idOffset + il * 10 + iyz * 100 + 5 + 2000);

        auto htgl2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, "tgl").c_str(), fmt::format("{}_{{{}}} vs {};tg#lambda;{}", dtit, lrName, "tg#lambda", dtit).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -maxDYZ, maxDYZ);
        mHMan->addHisto(htgl2, idOffset + il * 10 + iyz * 100 + 3000);
        auto htgl2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, "tgl").c_str(), fmt::format("pull {}_{{{}}} vs {};tg#lambda; pull{}", dtit, lrName, "tg#lambda", dtit).c_str(), params.nBinsTgl, -params.maxTgl, params.maxTgl, params.nBinsRes, -params.maxPull, params.maxPull);
        mHMan->addHisto(htgl2p, idOffset + il * 10 + iyz * 100 + 5 + 3000);

        if (il > 0) {
          const int ilr = il - 1;
          const int nChips = mChipMap.getNChipsOnLayer(ilr) / 2;
          for (int ib = 0; ib < 2; ++ib) { // chips per halfbarrel
            const int startChip = ib * nChips + mChipMap.getFirstChipsOnLayer(ilr);
            auto hc2 = new TH2F(fmt::format("{}_{}_{}", dname, lrName, ib).c_str(), fmt::format("{} Lr{} HB{};chip;{}", dname, lrName, ib, dtit).c_str(), nChips, startChip - 0.5, startChip - 0.5 + nChips, params.nBinsRes, -maxDYZ, maxDYZ);
            mHMan->addHisto(hc2, idOffset + (2 * il + ib) * 100 + iyz * 10 + 15000);
            auto hc2p = new TH2F(fmt::format("{}_{}_{}_pull", dname, lrName, ib).c_str(), fmt::format("{} Lr{} HB{} pull;chip;pull{}", dname, lrName, ib, dtit).c_str(), nChips, startChip - 0.5, startChip - 0.5 + nChips, params.nBinsRes, -params.maxPull, params.maxPull);
            mHMan->addHisto(hc2p, idOffset + (2 * il + ib) * 100 + iyz * 10 + 5 + 15000);
          }
        }
      }
    }
  };
  bookLayerResidHistos(0, "", true);
  if (params.doIBOBresiduals) { // the vtx slot is included: the PV constrains the IB outward fit, so it gets an OB-extrapolated IOB residual too
    bookLayerResidHistos(100000, "IOB", true);
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
  // used PV
  mHMan->addHisto(new TH1F("pvX", "PV X;x;", params.nBinsPVXYZ, -params.maxHPVXY, params.maxHPVXY), 20000 + 0);
  mHMan->addHisto(new TH1F("pvY", "PV Y;y;", params.nBinsPVXYZ, -params.maxHPVXY, params.maxHPVXY), 20000 + 1);
  mHMan->addHisto(new TH1F("pvZ", "PV Z;z;", params.nBinsPVXYZ, -params.maxHPVZ, params.maxHPVZ), 20000 + 2);
  mHMan->addHisto(new TH1F("pvN", "PV Contributors;Nc;", params.maxHPVN, 1.5, params.maxHPVN + 1.5), 20000 + 3);
}

void CheckResidSpec::postProcessHistos()
{
  printf("Fitting histos\n");
  if (!mHMan) {
    if (mHManV.empty()) {
      LOGP(warn, "nothing to process");
      return;
    }
    mHMan = mHManV[0].get();
  }
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  auto gs = new TF1("gs", "gaus", -1, 1);
  int maxH = mPostProcOnly ? mHManV.size() : 1;
  TObjArray arr;
  for (int ihm = 0; ihm < maxH; ihm++) {
    auto* histm = mHManV[ihm].get();
    if (!histm) {
      continue;
    }
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
        for (int idOffset : {0, 100000}) { // standard and IOB residuals; fitSlices is a no-op if the histo was not booked
          for (int il = 0; il < 8; il++) {
            for (int iyz = 0; iyz < 2; iyz++) {
              fitSlices(idOffset + il * 10 + iyz * 100 + offsV + offs);
            }
          }
        }
        for (int ip = 0; ip < 5; ip++) {
          fitSlices(10000 + ip * 10 + offsV + offs);
        }
      }
    }
    for (int idOffset : {0, 100000}) {
      for (int il = 1; il < 8; il++) {
        for (int ib = 0; ib < 2; ib++) {
          for (int iyz = 0; iyz < 2; iyz++) {
            for (int iht = 0; iht < 2; iht++) {
              const int offsV = iht == 0 ? 0 : 5;
              fitSlices(idOffset + (2 * il + ib) * 100 + iyz * 10 + offsV + 15000);
            }
          }
        }
      }
    }
  }
  delete gs;
}

void CheckResidSpec::drawHistos()
{
  // gROOT->SetBatch(true);
  gStyle->SetTitleX(0.3);
  gStyle->SetTitleY(0.88);
  gStyle->SetTitleW(0.25);
  gStyle->SetOptStat(0);
  const auto& params = o2::checkresid::CheckResidConfig::Instance();
  int nhm = mHManV.size();
  std::array<unsigned int, 3> hcol{EColor::kRed, EColor::kBlue, EColor::kGreen + 2};
  const auto hmNames = o2::utils::Str::tokenize(params.ext_hm_list, ",;");
  const auto colorNames = o2::utils::Str::tokenize(params.ext_color_list, ",;");
  const auto markerIDs = o2::utils::Str::tokenize(params.ext_marker_list, ",;");
  bool useColors = colorNames.size() == hmNames.size() && !colorNames.empty();
  bool useMarkers = markerIDs.size() == hmNames.size() && !markerIDs.empty();
  if (!useColors) {
    LOGP(warn, "{} color names provided for {} external histomanagers, will use automatic colors", colorNames.size(), hmNames.size());
  }
  if (!useMarkers) {
    LOGP(warn, "{} marker IDs provided for {} external histomanagers, will use automatic markers", markerIDs.size(), hmNames.size());
  }
  constexpr std::array<std::pair<std::string_view, int>, 15> rootColors{{
    {"kWhite", EColor::kWhite},
    {"kBlack", EColor::kBlack},
    {"kGray", EColor::kGray},
    {"kRed", EColor::kRed},
    {"kGreen", EColor::kGreen},
    {"kBlue", EColor::kBlue},
    {"kYellow", EColor::kYellow},
    {"kMagenta", EColor::kMagenta},
    {"kCyan", EColor::kCyan},
    {"kOrange", EColor::kOrange},
    {"kSpring", EColor::kSpring},
    {"kTeal", EColor::kTeal},
    {"kAzure", EColor::kAzure},
    {"kViolet", EColor::kViolet},
    {"kPink", EColor::kPink},
  }};
  auto parseColor = [&rootColors](const std::string& token) -> std::optional<int> {
    std::string normalized;
    normalized.reserve(token.size());
    for (unsigned char ch : token) {
      if (!std::isspace(ch)) {
        normalized.push_back(ch);
      }
    }
    for (const auto& [name, value] : rootColors) {
      if (normalized.compare(0, name.size(), name) != 0) {
        continue;
      }
      const auto increment = std::string_view(normalized).substr(name.size());
      if (increment.empty()) {
        return value;
      }
      if (increment.size() < 2 || (increment.front() != '+' && increment.front() != '-')) {
        return std::nullopt;
      }
      int offset = 0;
      const auto* begin = increment.data() + 1;
      const auto* end = increment.data() + increment.size();
      auto [parsedEnd, error] = std::from_chars(begin, end, offset);
      if (error != std::errc{} || parsedEnd != end) {
        return std::nullopt;
      }
      const auto signedOffset = increment.front() == '-' ? -static_cast<long long>(offset) : static_cast<long long>(offset);
      const auto color = static_cast<long long>(value) + signedOffset;
      if (color < std::numeric_limits<int>::min() || color > std::numeric_limits<int>::max()) {
        return std::nullopt;
      }
      return static_cast<int>(color);
    }
    return std::nullopt;
  };
  std::vector<int> colors;
  if (useColors) {
    for (const auto& colorName : colorNames) {
      auto color = parseColor(colorName);
      if (!color) {
        LOGP(warn, "Unrecognized EColor name '{}', will use automatic colors", colorName);
        useColors = false;
        break;
      }
      colors.push_back(*color);
    }
  }
  std::vector<int> markers;
  if (useMarkers) {
    for (const auto& markerID : markerIDs) {
      int marker = 0;
      const auto* begin = markerID.data();
      const auto* end = begin + markerID.size();
      auto [parsedEnd, error] = std::from_chars(begin, end, marker);
      if (error != std::errc{} || parsedEnd != end) {
        LOGP(warn, "Unrecognized marker ID '{}', will use automatic markers", markerID);
        useMarkers = false;
        break;
      }
      markers.push_back(marker);
    }
  }
  std::unique_ptr<TLegend> lg;
  lg = std::make_unique<TLegend>(0.12, 0.13, 0.9, 0.13 + std::min(0.5f, nhm * 0.2f / 3.f));
  lg->SetFillStyle(0);
  lg->SetBorderSize(0);
  for (int i = 0; i < nhm; i++) {
    auto hman = mHManV[i].get();
    if (!hman || hman->GetLast() < 1) {
      continue;
    }
    const int marker = useMarkers ? markers[i] : 20 + i + (i % 2) * 4;
    const int color = useColors ? colors[i] : hcol[i % hcol.size()];
    hman->setMarkerStyle(marker, 0.5);
    hman->setColor(color);
    auto le = lg->AddEntry(hman->getHisto(1), hman->GetName(), "lp");
    le->SetTextColor(color);
  }
  TCanvas cly("cly", "", 600, 800), clz("clz", "", 600, 800), clpar("clpar", "", 600, 800);
  TCanvas czly("czly", "", 600, 800), czlz("czlz", "", 600, 800), czlpar("czlpar", "", 600, 800);
  TCanvas cchip("cchip", "", 600, 800);
  constexpr std::array<const char*, 4> chipTitles{"#DeltaY", "pull #DeltaY", "#DeltaZ", "pull #DeltaZ"};
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

  // draws the 8 dy/dz-vs-phi/Z/pT/tgl pages (standard idOffset=0, or IOB idOffset=100000) into pdfName
  auto drawLrResidPages = [&](const std::string& pdfName, int idOffset) {
    const float dyzScale = idOffset ? params.IOBScaleFactor : 1.f; // IOB residuals have a much larger range than the standard ones
    std::array<float, 8> resMMY, resMMZ;
    for (int i = 0; i < 8; i++) {
      resMMY[i] = params.resMMLrY[i] * dyzScale;
      resMMZ[i] = params.resMMLrZ[i] * dyzScale;
    }
    drawResLr(cly, idOffset + 1, resMMY.data(), false);
    cly.cd(2);
    lg->Draw();
    AddLabel("Y residuals", 0.1, 0.95);
    cly.Print(pdfName.c_str());

    drawResLr(clz, idOffset + 101, resMMZ.data(), false);
    clz.cd(2);
    lg->Draw();
    AddLabel("Z residuals", 0.1, 0.95);
    clz.Print(pdfName.c_str());

    drawResLr(czly, idOffset + 1001, resMMY.data(), false);
    czly.cd(2);
    lg->Draw();
    AddLabel("Y residuals", 0.1, 0.95);
    czly.Print(pdfName.c_str());

    drawResLr(czlz, idOffset + 1101, resMMZ.data(), false);
    czlz.cd(2);
    lg->Draw();
    AddLabel("Z residuals", 0.1, 0.95);
    czlz.Print(pdfName.c_str());

    drawResLr(czly, idOffset + 2001, resMMY.data(), true);
    czly.cd(2);
    lg->Draw();
    AddLabel("Y residuals", 0.1, 0.95);
    czly.Print(pdfName.c_str());

    drawResLr(czlz, idOffset + 2101, resMMZ.data(), true);
    czlz.cd(2);
    lg->Draw();
    AddLabel("Z residuals", 0.1, 0.95);
    czlz.Print(pdfName.c_str());

    drawResLr(czly, idOffset + 3001, resMMY.data(), false);
    czly.cd(2);
    lg->Draw();
    AddLabel("Y residuals", 0.1, 0.95);
    czly.Print(pdfName.c_str());

    drawResLr(czlz, idOffset + 3101, resMMZ.data(), false);
    czlz.cd(2);
    lg->Draw();
    AddLabel("Z residuals", 0.1, 0.95);
    czlz.Print(pdfName.c_str());
  };

  const std::string mainPdf = fmt::format("{}_hman.pdf", params.outname);
  cly.Print(Form("%s[", mainPdf.c_str()));
  drawLrResidPages(mainPdf, 0);

  drawResPar(clpar, 10001, params.resMMPar, false);
  clpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  clpar.Print(mainPdf.c_str());

  drawResPar(czlpar, 11001, params.resMMPar, false);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(mainPdf.c_str());

  drawResPar(czlpar, 12001, params.resMMPar, true);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(mainPdf.c_str());

  drawResPar(czlpar, 13001, params.resMMPar, false);
  czlpar.cd(6);
  lg->Draw();
  AddLabel("IB-OB tracks params differences at R = 12 cm", 0.2, 0.8);
  czlpar.Print(mainPdf.c_str());

  auto getHMinMax = [this](int id, float marginFrac, float maxDev, int roundToMicrons) -> std::tuple<float, float, float> {
    float maxV = std::numeric_limits<float>::lowest(), minV = std::numeric_limits<float>::max();
    const int nh = this->mHManV.size();
    for (int j = 0; j < nh; j++) {
      if (!this->mHManV[j]) {
        continue;
      }
      if (auto histo = this->mHManV[j]->getHisto(id); histo != nullptr) {
        const int nb = histo->GetNbinsX();
        for (int ib = 1; ib <= nb; ib++) {
          float v = histo->GetBinContent(ib);
          maxV = std::max(maxV, v);
          minV = std::min(minV, v);
        }
      }
    }
    if (maxV < minV) {
      return {0.f, 1.f, 0.1f};
    }
    auto delta = maxV - minV;
    maxV += delta * marginFrac;
    minV -= delta * marginFrac;
    if (roundToMicrons > 0) {
      maxV = ((int(std::ceil(maxV * 1e4)) / roundToMicrons + 1) * roundToMicrons) * 1e-4;
      minV = ((int(std::floor(minV * 1e4)) / roundToMicrons - 1) * roundToMicrons) * 1e-4;
    }
    minV = std::max(minV, -maxDev);
    maxV = std::min(maxV, maxDev);
    return {minV, maxV, std::max(std::abs(minV), std::abs(maxV))};
  };
  auto collectSpread = [](TH1* hin, TH1* hout) {
    const int nb = hin->GetNbinsX();
    for (int ib = 1; ib <= nb; ib++) {
      if (hin->GetBinError(ib) != 0) {
        hout->Fill(hin->GetBinContent(ib));
      }
    }
    hout->SetLineColor(hin->GetLineColor());
    hout->SetMarkerColor(hin->GetMarkerColor());
    hout->SetMarkerStyle(hin->GetMarkerStyle());
  };
  int nh = this->mHManV.size();
  // Draw per-chip residual/pull means and sigmas, and their half-barrel and full-layer spreads.
  auto drawChipPages = [&](const std::string& pdfName, int idOffset) {
    for (int il = 1; il < 8; ++il) {
      for (int iht = 0; iht < 2; iht++) { // residual or pull
        for (int ib = 0; ib < 3; ib++) {  // two half-barrels, then the full layer
          if (iht == 1 && ib < 2) {
            continue; // Draw pulls only for the full layer.
          }
          for (int iyz = 0; iyz < 2; iyz++) {
            cchip.Clear();
            cchip.Divide(1, ib == 2 ? 2 : 4);
            bool havePage = false;
            for (int iv = 0; iv < 2; iv++) { // mean or sigma
              const float yRange = (iyz == 0 ? params.resMMLrY[il] : params.resMMLrZ[il]) * (idOffset ? params.IOBScaleFactor : 1.f);
              const int baseID = idOffset + 15000 + iyz * 10 + 2 * il * 100 + iht * 5 + 1 + iv;
              // Use common spread axes for both half-barrels and their full-layer sum.
              auto [minV, maxV, absMax] = getHMinMax(baseID, 0.15, (iv == 0 ? 2 : 8) * yRange, 10);
              auto [minV1, maxV1, absMax1] = getHMinMax(baseID + 100, 0.15, (iv == 0 ? 2 : 8) * yRange, 10);
              minV = std::min(minV, minV1);
              maxV = std::max(maxV, maxV1);
              absMax = std::max(absMax, absMax1);
              if (iht == 1) {
                minV = iv ? 0.f : -params.maxPull;
                maxV = absMax = params.maxPull;
              }
              bool drawn = false;
              TH1* firstAuto = nullptr;
              float maxAuto = 0.f;
              for (int j = 0; j < nh; j++) {
                auto hman = this->mHManV[j].get();
                if (!hman) {
                  continue;
                }
                TH1* hmeans = nullptr;
                for (int hb = ib == 2 ? 0 : ib; hb < (ib == 2 ? 2 : ib + 1); hb++) {
                  const int id = baseID + hb * 100;
                  auto histo = hman->getHisto(id);
                  if (!histo) {
                    continue;
                  }
                  if (!hmeans) {
                    if (ib == 2) {
                      hmeans = new TH1F(Form("%s_layer_spread", histo->GetName()), Form("Lr%d %s %s spread", il - 1, chipTitles[2 * iyz + iht], iv ? "sigma" : "mean"), params.nBinsRes, iv == 0 ? minV : std::max(0.f, minV), maxV);
                      hman->addHisto(hmeans, idOffset + 25000 + il * 100 + iyz * 10 + iht * 5 + iv);
                    } else {
                      hmeans = new TH1F(Form("%s_spread", histo->GetName()), Form("%s spread;#<Delta>", histo->GetTitle()), params.nBinsRes, iv == 0 ? minV : std::max(0.f, minV), maxV);
                      hman->addHisto(hmeans, id + 2);
                    }
                  }
                  if (iht == 0) {
                    hmeans->SetXTitle(iv ? "<#sigma>, cm" : "<#Delta>, cm");
                  }
                  collectSpread(histo, hmeans);
                  if (ib < 2) {
                    histo->SetMaximum(absMax);
                    histo->SetMinimum(iv ? 0 : -absMax);
                    if (il > 3) {
                      histo->SetMarkerSize(0.3);
                      histo->SetLineStyle(3);
                    }
                    cchip.cd(2 * iv + 1);
                    histo->Draw(drawn ? "same" : "");
                    gPad->SetGrid();
                  }
                }
                if (!hmeans) {
                  continue;
                }
                cchip.cd(ib == 2 ? iv + 1 : 2 * iv + 2);
                if (!drawn) {
                  firstAuto = hmeans;
                }
                maxAuto = std::max(maxAuto, (float)hmeans->GetMaximum());
                hmeans->Draw(drawn ? "same" : "");
                drawn = havePage = true;
              }
              if (firstAuto) {
                firstAuto->SetMaximum(maxAuto * 1.25);
                gPad->SetGrid();
                gPad->Modified();
              }
            }
            if (havePage) {
              cchip.Print(pdfName.c_str());
            }
          }
        }
      }
    }
  };
  drawChipPages(mainPdf, 0);
  cly.Print(Form("%s]", mainPdf.c_str()));

  // IOB residuals (dyIOB/dzIOB) are optional: only produce the extra PDF if at least one loaded HistoManager actually has them booked
  bool haveIOB = false;
  for (const auto& hm : mHManV) {
    if (hm && hm->getHisto2F(100000 + 1 * 10)) {
      haveIOB = true;
      break;
    }
  }
  if (haveIOB) {
    const std::string iobPdf = fmt::format("{}_iob_hman.pdf", params.outname);
    cly.Print(Form("%s[", iobPdf.c_str()));
    drawLrResidPages(iobPdf, 100000);
    drawChipPages(iobPdf, 100000);
    cly.Print(Form("%s]", iobPdf.c_str()));
  }
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
  if (auto nh = mHManV.size(); nh > 0 && !mPostProcOnly) {
    for (int ihm = 0; ihm < nh; ihm++) {
      if (mHManV[ihm]) {
        mHManV[ihm].get()->write();
      }
    }
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

DataProcessorSpec getCheckResidSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool drawOnly, bool postProcOnly, bool itsStag)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->setITSPerLayer(itsStag);
  if (!drawOnly && !postProcOnly) {
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
    AlgorithmSpec{adaptFromTask<CheckResidSpec>(dataRequest, ggRequest, srcTracks, drawOnly, postProcOnly)},
    opts};
}

} // namespace o2::checkresid
