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

/// \file SVertexer.cxx
/// \brief Secondary vertex finder
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/SVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include "CorrectionMapsHelper.h"
#include "Framework/ProcessingContext.h"
#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "CommonConstants/GeomConstants.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::vertexing;
namespace o2f = o2::framework;
using PID = o2::track::PID;
using TrackTPCITS = o2::dataformats::TrackTPCITS;
using TrackITS = o2::its::TrackITS;
using TrackTPC = o2::tpc::TrackTPC;

//__________________________________________________________________
void SVertexer::process(const o2::globaltracking::RecoContainer& recoData, o2::framework::ProcessingContext& pc)
{
  mRecoCont = &recoData;
  mNV0s = mNCascades = mN3Bodies = 0;
  updateTimeDependentParams(); // TODO RS: strictly speaking, one should do this only in case of the CCDB objects update
  mPVertices = recoData.getPrimaryVertices();
  buildT2V(recoData); // build track->vertex refs from vertex->track (if other workflow will need this, consider producing a message in the VertexTrackMatcher)
  int ntrP = mTracksPool[POS].size(), ntrN = mTracksPool[NEG].size();
  if (mStrTracker) {
    mStrTracker->loadData(recoData);
    mStrTracker->prepareITStracks();
  }
#ifdef WITH_OPENMP
  int dynGrp = std::min(4, std::max(1, mNThreads / 2));
#pragma omp parallel for schedule(dynamic, dynGrp) num_threads(mNThreads)
#endif
  for (int itp = 0; itp < ntrP; itp++) {
    auto& seedP = mTracksPool[POS][itp];
    int firstN = mVtxFirstTrack[NEG][seedP.vBracket.getMin()];
    if (firstN < 0) {
      LOG(debug) << "No partner is found for pos.track " << itp << " out of " << ntrP;
      continue;
    }
    for (int itn = firstN; itn < ntrN; itn++) { // start from the 1st negative track of lowest-ID vertex of positive
      auto& seedN = mTracksPool[NEG][itn];
      if (seedN.vBracket > seedP.vBracket) { // all vertices compatible with seedN are in future wrt that of seedP
        LOG(debug) << "Brackets do not match";
        break;
      }
      if (mSVParams->maxPVContributors < 2 && seedP.gid.isPVContributor() + seedN.gid.isPVContributor() > mSVParams->maxPVContributors) {
        continue;
      }
#ifdef WITH_OPENMP
      int iThread = omp_get_thread_num();
#else
      int iThread = 0;
#endif
      checkV0(seedP, seedN, itp, itn, iThread);
    }
  }

  // sort V0s and Cascades in vertex id
  struct vid {
    int thrID;
    int entry;
    int vtxID;
  };
  for (int ith = 0; ith < mNThreads; ith++) {
    mNV0s += mV0sIdxTmp[ith].size();
    mNCascades += mCascadesIdxTmp[ith].size();
    mN3Bodies += m3bodyIdxTmp[ith].size();
  }
  std::vector<vid> v0SortID, cascSortID, nbodySortID;
  v0SortID.reserve(mNV0s);
  cascSortID.reserve(mNCascades);
  nbodySortID.reserve(mN3Bodies);
  for (int ith = 0; ith < mNThreads; ith++) {
    for (int j = 0; j < (int)mV0sIdxTmp[ith].size(); j++) {
      v0SortID.emplace_back(vid{ith, j, mV0sIdxTmp[ith][j].getVertexID()});
    }
    for (int j = 0; j < (int)mCascadesIdxTmp[ith].size(); j++) {
      cascSortID.emplace_back(vid{ith, j, mCascadesIdxTmp[ith][j].getVertexID()});
    }
    for (int j = 0; j < (int)m3bodyIdxTmp[ith].size(); j++) {
      nbodySortID.emplace_back(vid{ith, j, m3bodyIdxTmp[ith][j].getVertexID()});
    }
  }
  std::sort(v0SortID.begin(), v0SortID.end(), [](const vid& a, const vid& b) { return a.vtxID < b.vtxID; });
  std::sort(cascSortID.begin(), cascSortID.end(), [](const vid& a, const vid& b) { return a.vtxID < b.vtxID; });
  std::sort(nbodySortID.begin(), nbodySortID.end(), [](const vid& a, const vid& b) { return a.vtxID < b.vtxID; });
  // sorted V0s

  auto& v0sIdx = pc.outputs().make<std::vector<V0Index>>(o2f::Output{"GLO", "V0S_IDX", 0, o2f::Lifetime::Timeframe});
  auto& cascsIdx = pc.outputs().make<std::vector<CascadeIndex>>(o2f::Output{"GLO", "CASCS_IDX", 0, o2f::Lifetime::Timeframe});
  auto& body3Idx = pc.outputs().make<std::vector<Decay3BodyIndex>>(o2f::Output{"GLO", "DECAYS3BODY_IDX", 0, o2f::Lifetime::Timeframe});
  auto& fullv0s = pc.outputs().make<std::vector<V0>>(o2f::Output{"GLO", "V0S", 0, o2f::Lifetime::Timeframe});
  auto& fullcascs = pc.outputs().make<std::vector<Cascade>>(o2f::Output{"GLO", "CASCS", 0, o2f::Lifetime::Timeframe});
  auto& full3body = pc.outputs().make<std::vector<Decay3Body>>(o2f::Output{"GLO", "DECAYS3BODY", 0, o2f::Lifetime::Timeframe});
  auto& v0Refs = pc.outputs().make<std::vector<RRef>>(o2f::Output{"GLO", "PVTX_V0REFS", 0, o2f::Lifetime::Timeframe});
  auto& cascRefs = pc.outputs().make<std::vector<RRef>>(o2f::Output{"GLO", "PVTX_CASCREFS", 0, o2f::Lifetime::Timeframe});
  auto& vtx3bodyRefs = pc.outputs().make<std::vector<RRef>>(o2f::Output{"GLO", "PVTX_3BODYREFS", 0, o2f::Lifetime::Timeframe});

  // sorted V0s
  v0sIdx.reserve(mNV0s);
  if (mSVParams->createFullV0s) {
    fullv0s.reserve(mNV0s);
  }
  // sorted Cascades
  cascsIdx.reserve(mNCascades);
  if (mSVParams->createFullCascades) {
    fullcascs.reserve(mNCascades);
  }
  // sorted 3 body decays
  body3Idx.reserve(mN3Bodies);
  if (mSVParams->createFull3Bodies) {
    full3body.reserve(mN3Bodies);
  }

  for (const auto& id : v0SortID) {
    auto& v0idx = mV0sIdxTmp[id.thrID][id.entry];
    int pos = v0sIdx.size();
    v0sIdx.push_back(v0idx);
    v0idx.setVertexID(pos); // this v0 copy will be discarded, use its vertexID to store the new position of final V0
    if (mSVParams->createFullV0s) {
      fullv0s.push_back(mV0sTmp[id.thrID][id.entry]);
    }
  }
  // since V0s were reshuffled, we need to correct the cascade -> V0 reference indices
  for (int ith = 0; ith < mNThreads; ith++) {                     // merge results of all threads
    for (size_t ic = 0; ic < mCascadesIdxTmp[ith].size(); ic++) { // before merging fix cascades references on v0
      auto& cidx = mCascadesIdxTmp[ith][ic];
      cidx.setV0ID(mV0sIdxTmp[ith][cidx.getV0ID()].getVertexID());
    }
  }
  int cascCnt = 0;
  for (const auto& id : cascSortID) {
    cascsIdx.push_back(mCascadesIdxTmp[id.thrID][id.entry]);
    mCascadesIdxTmp[id.thrID][id.entry].setVertexID(cascCnt++); // memorize new ID
    if (mSVParams->createFullCascades) {
      fullcascs.push_back(mCascadesTmp[id.thrID][id.entry]);
    }
  }
  int b3cnt = 0;
  for (const auto& id : nbodySortID) {
    body3Idx.push_back(m3bodyIdxTmp[id.thrID][id.entry]);
    m3bodyIdxTmp[id.thrID][id.entry].setVertexID(b3cnt++); // memorize new ID
    if (mSVParams->createFull3Bodies) {
      full3body.push_back(m3bodyTmp[id.thrID][id.entry]);
    }
  }
  if (mStrTracker) {
    mNStrangeTracks = 0;
    for (int ith = 0; ith < mNThreads; ith++) {
      mNStrangeTracks += mStrTracker->getNTracks(ith);
    }

    std::vector<o2::dataformats::StrangeTrack> strTracksTmp;
    std::vector<o2::strangeness_tracking::ClusAttachments> strClusTmp;
    std::vector<o2::MCCompLabel> mcLabTmp;
    strTracksTmp.reserve(mNStrangeTracks);
    strClusTmp.reserve(mNStrangeTracks);
    if (mStrTracker->getMCTruthOn()) {
      mcLabTmp.reserve(mNStrangeTracks);
    }

    for (int ith = 0; ith < mNThreads; ith++) { // merge results of all threads
      auto& strTracks = mStrTracker->getStrangeTrackVec(ith);
      auto& strClust = mStrTracker->getClusAttachments(ith);
      auto& stcTrMCLab = mStrTracker->getStrangeTrackLabels(ith);
      for (int i = 0; i < (int)strTracks.size(); i++) {
        auto& t = strTracks[i];
        if (t.mPartType == o2::dataformats::kStrkV0) {
          t.mDecayRef = mV0sIdxTmp[ith][t.mDecayRef].getVertexID(); // reassign merged V0 ID
        } else if (t.mPartType == o2::dataformats::kStrkCascade) {
          t.mDecayRef = mCascadesIdxTmp[ith][t.mDecayRef].getVertexID(); // reassign merged Cascase ID
        } else if (t.mPartType == o2::dataformats::kStrkThreeBody) {
          t.mDecayRef = m3bodyIdxTmp[ith][t.mDecayRef].getVertexID(); // reassign merged Cascase ID
        } else {
          LOGP(fatal, "Unknown strange track decay reference type {} for index {}", int(t.mPartType), t.mDecayRef);
        }

        strTracksTmp.push_back(t);
        strClusTmp.push_back(strClust[i]);
        if (mStrTracker->getMCTruthOn()) {
          mcLabTmp.push_back(stcTrMCLab[i]);
        }
      }
    }

    auto& strTracksOut = pc.outputs().make<std::vector<o2::dataformats::StrangeTrack>>(o2f::Output{"GLO", "STRANGETRACKS", 0, o2f::Lifetime::Timeframe});
    auto& strClustOut = pc.outputs().make<std::vector<o2::strangeness_tracking::ClusAttachments>>(o2f::Output{"GLO", "CLUSUPDATES", 0, o2f::Lifetime::Timeframe});
    o2::pmr::vector<o2::MCCompLabel> mcLabsOut;
    strTracksOut.resize(mNStrangeTracks);
    strClustOut.resize(mNStrangeTracks);
    if (mStrTracker->getMCTruthOn()) {
      mcLabsOut.resize(mNStrangeTracks);
    }

    std::vector<int> sortIdx(strTracksTmp.size());
    std::iota(sortIdx.begin(), sortIdx.end(), 0);
    // if mNTreads > 1 we need to sort tracks, clus and MCLabs by their mDecayRef
    if (mNThreads > 1 && mNStrangeTracks > 1) {
      std::sort(sortIdx.begin(), sortIdx.end(), [&strTracksTmp](int i1, int i2) { return strTracksTmp[i1].mDecayRef < strTracksTmp[i2].mDecayRef; });
    }

    for (int i = 0; i < (int)sortIdx.size(); i++) {
      strTracksOut[i] = strTracksTmp[sortIdx[i]];
      strClustOut[i] = strClusTmp[sortIdx[i]];
      if (mStrTracker->getMCTruthOn()) {
        mcLabsOut[i] = mcLabTmp[sortIdx[i]];
      }
    }

    if (mStrTracker->getMCTruthOn()) {
      auto& strTrMCLableOut = pc.outputs().make<std::vector<o2::MCCompLabel>>(o2f::Output{"GLO", "STRANGETRACKS_MC", 0, o2f::Lifetime::Timeframe});
      strTrMCLableOut.swap(mcLabsOut);
    }
  }
  //
  for (int ith = 0; ith < mNThreads; ith++) { // clean unneeded s.vertices
    mV0sTmp[ith].clear();
    mCascadesTmp[ith].clear();
    m3bodyTmp[ith].clear();
    mV0sIdxTmp[ith].clear();
    mCascadesIdxTmp[ith].clear();
    m3bodyIdxTmp[ith].clear();
  }

  extractPVReferences(v0sIdx, v0Refs, cascsIdx, cascRefs, body3Idx, vtx3bodyRefs);
}

//__________________________________________________________________
void SVertexer::init()
{
}

//__________________________________________________________________
void SVertexer::updateTimeDependentParams()
{
  // TODO RS: strictly speaking, one should do this only in case of the CCDB objects update
  static bool updatedOnce = false;
  if (!updatedOnce) {
    updatedOnce = true;
    mSVParams = &SVertexerParams::Instance();
    // precalculated selection cuts
    mMinR2ToMeanVertex = mSVParams->minRToMeanVertex * mSVParams->minRToMeanVertex;
    mMaxR2ToMeanVertexCascV0 = mSVParams->maxRToMeanVertexCascV0 * mSVParams->maxRToMeanVertexCascV0;
    mMaxDCAXY2ToMeanVertex = mSVParams->maxDCAXYToMeanVertex * mSVParams->maxDCAXYToMeanVertex;
    mMaxDCAXY2ToMeanVertexV0Casc = mSVParams->maxDCAXYToMeanVertexV0Casc * mSVParams->maxDCAXYToMeanVertexV0Casc;
    mMaxDCAXY2ToMeanVertex3bodyV0 = mSVParams->maxDCAXYToMeanVertex3bodyV0 * mSVParams->maxDCAXYToMeanVertex3bodyV0;
    mMinR2DiffV0Casc = mSVParams->minRDiffV0Casc * mSVParams->minRDiffV0Casc;
    mMinPt2V0 = mSVParams->minPtV0 * mSVParams->minPtV0;
    mMaxTgl2V0 = mSVParams->maxTglV0 * mSVParams->maxTglV0;
    mMinPt2Casc = mSVParams->minPtCasc * mSVParams->minPtCasc;
    mMaxTgl2Casc = mSVParams->maxTglCasc * mSVParams->maxTglCasc;
    mMinPt23Body = mSVParams->minPt3Body * mSVParams->minPt3Body;
    mMaxTgl23Body = mSVParams->maxTgl3Body * mSVParams->maxTgl3Body;
    setupThreads();
  }
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  mV0Hyps[HypV0::Photon].set(PID::Photon, PID::Electron, PID::Electron, mSVParams->pidCutsPhoton, bz);
  mV0Hyps[HypV0::K0].set(PID::K0, PID::Pion, PID::Pion, mSVParams->pidCutsK0, bz);
  mV0Hyps[HypV0::Lambda].set(PID::Lambda, PID::Proton, PID::Pion, mSVParams->pidCutsLambda, bz);
  mV0Hyps[HypV0::AntiLambda].set(PID::Lambda, PID::Pion, PID::Proton, mSVParams->pidCutsLambda, bz);
  mV0Hyps[HypV0::HyperTriton].set(PID::HyperTriton, PID::Helium3, PID::Pion, mSVParams->pidCutsHTriton, bz);
  mV0Hyps[HypV0::AntiHyperTriton].set(PID::HyperTriton, PID::Pion, PID::Helium3, mSVParams->pidCutsHTriton, bz);
  mV0Hyps[HypV0::Hyperhydrog4].set(PID::Hyperhydrog4, PID::Alpha, PID::Pion, mSVParams->pidCutsHhydrog4, bz);
  mV0Hyps[HypV0::AntiHyperhydrog4].set(PID::Hyperhydrog4, PID::Pion, PID::Alpha, mSVParams->pidCutsHhydrog4, bz);
  mCascHyps[HypCascade::XiMinus].set(PID::XiMinus, PID::Lambda, PID::Pion, mSVParams->pidCutsXiMinus, bz, mSVParams->maximalCascadeWidth);
  mCascHyps[HypCascade::OmegaMinus].set(PID::OmegaMinus, PID::Lambda, PID::Kaon, mSVParams->pidCutsOmegaMinus, bz, mSVParams->maximalCascadeWidth);

  m3bodyHyps[Hyp3body::H3L3body].set(PID::HyperTriton, PID::Proton, PID::Pion, PID::Deuteron, mSVParams->pidCutsH3L3body, bz);
  m3bodyHyps[Hyp3body::AntiH3L3body].set(PID::HyperTriton, PID::Pion, PID::Proton, PID::Deuteron, mSVParams->pidCutsH3L3body, bz);

  for (auto& ft : mFitterV0) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitterCasc) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitter3body) {
    ft.setBz(bz);
  }

  mPIDresponse.setBetheBlochParams(mSVParams->mBBpars);
}

//______________________________________________
void SVertexer::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  mTPCVDrift = v.refVDrift * v.corrFact;
  mTPCVDriftCorrFact = v.corrFact;
  mTPCVDriftRef = v.refVDrift;
  mTPCDriftTimeOffset = v.getTimeOffset();
  mTPCBin2Z = mTPCVDrift / mMUS2TPCBin;
}
//______________________________________________
void SVertexer::setTPCCorrMaps(o2::gpu::CorrectionMapsHelper* maph)
{
  mTPCCorrMapsHelper = maph;
}

//__________________________________________________________________
void SVertexer::setupThreads()
{
  if (!mV0sTmp.empty()) {
    return;
  }
  mV0sTmp.resize(mNThreads);
  mCascadesTmp.resize(mNThreads);
  m3bodyTmp.resize(mNThreads);
  mV0sIdxTmp.resize(mNThreads);
  mCascadesIdxTmp.resize(mNThreads);
  m3bodyIdxTmp.resize(mNThreads);
  mFitterV0.resize(mNThreads);
  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  int fitCounter = 0;
  for (auto& fitter : mFitterV0) {
    fitter.setFitterID(fitCounter++);
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIni);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType(mSVParams->matCorr));
    fitter.setUsePropagator(mSVParams->usePropagator);
    fitter.setRefitWithMatCorr(mSVParams->refitWithMatCorr);
    fitter.setMaxStep(mSVParams->maxStep);
    fitter.setMaxSnp(mSVParams->maxSnp);
    fitter.setMinXSeed(mSVParams->minXSeed);
  }
  mFitterCasc.resize(mNThreads);
  fitCounter = 1000;
  for (auto& fitter : mFitterCasc) {
    fitter.setFitterID(fitCounter++);
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIniCasc);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType(mSVParams->matCorr));
    fitter.setUsePropagator(mSVParams->usePropagator);
    fitter.setRefitWithMatCorr(mSVParams->refitWithMatCorr);
    fitter.setMaxStep(mSVParams->maxStep);
    fitter.setMaxSnp(mSVParams->maxSnp);
    fitter.setMinXSeed(mSVParams->minXSeed);
  }

  mFitter3body.resize(mNThreads);
  fitCounter = 2000;
  for (auto& fitter : mFitter3body) {
    fitter.setFitterID(fitCounter++);
    fitter.setBz(bz);
    fitter.setUseAbsDCA(mSVParams->useAbsDCA);
    fitter.setPropagateToPCA(false);
    fitter.setMaxR(mSVParams->maxRIni3body);
    fitter.setMinParamChange(mSVParams->minParamChange);
    fitter.setMinRelChi2Change(mSVParams->minRelChi2Change);
    fitter.setMaxDZIni(mSVParams->maxDZIni);
    fitter.setMaxChi2(mSVParams->maxChi2);
    fitter.setMatCorrType(o2::base::Propagator::MatCorrType(mSVParams->matCorr));
    fitter.setUsePropagator(mSVParams->usePropagator);
    fitter.setRefitWithMatCorr(mSVParams->refitWithMatCorr);
    fitter.setMaxStep(mSVParams->maxStep);
    fitter.setMaxSnp(mSVParams->maxSnp);
    fitter.setMinXSeed(mSVParams->minXSeed);
  }
}

//__________________________________________________________________
bool SVertexer::acceptTrack(GIndex gid, const o2::track::TrackParCov& trc) const
{
  if (gid.isPVContributor() && mSVParams->maxPVContributors < 1) {
    return false;
  }

  // DCA to mean vertex
  if (mSVParams->minDCAToPV > 0.f) {
    o2::track::TrackPar trp(trc);
    std::array<float, 2> dca;
    auto* prop = o2::base::Propagator::Instance();
    if (mSVParams->usePropagator) {
      if (trp.getX() > mSVParams->minRFor3DField && !prop->PropagateToXBxByBz(trp, mSVParams->minRFor3DField, mSVParams->maxSnp, mSVParams->maxStep, o2::base::Propagator::MatCorrType(mSVParams->matCorr))) {
        return true; // we don't need actually to propagate to the beam-line
      }
      if (!prop->propagateToDCA(mMeanVertex.getXYZ(), trp, prop->getNominalBz(), mSVParams->maxStep, o2::base::Propagator::MatCorrType(mSVParams->matCorr), &dca)) {
        return true;
      }
    } else {
      if (!trp.propagateParamToDCA(mMeanVertex.getXYZ(), prop->getNominalBz(), &dca)) {
        return true;
      }
    }
    if (std::abs(dca[0]) < mSVParams->minDCAToPV) {
      return false;
    }
  }
  return true;
}

//__________________________________________________________________
void SVertexer::buildT2V(const o2::globaltracking::RecoContainer& recoData) // accessor to various tracks
{
  // build track->vertices from vertices->tracks, rejecting vertex contributors if requested
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  bool isTPCloaded = recoData.isTrackSourceLoaded(GIndex::TPC);
  bool isITSloaded = recoData.isTrackSourceLoaded(GIndex::ITS);
  bool isITSTPCloaded = recoData.isTrackSourceLoaded(GIndex::ITSTPC);
  if (isTPCloaded && !mSVParams->mExcludeTPCtracks) {
    mTPCTracksArray = recoData.getTPCTracks();
    mTPCTrackClusIdx = recoData.getTPCTracksClusterRefs();
    mTPCClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
    mTPCRefitterShMap = recoData.clusterShMapTPC;
    mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, mTPCCorrMapsHelper, o2::base::Propagator::Instance()->getNominalBz(), mTPCTrackClusIdx.data(), mTPCRefitterShMap.data(), nullptr, o2::base::Propagator::Instance());
  }

  std::unordered_map<GIndex, std::pair<int, int>> tmap;
  std::unordered_map<GIndex, bool> rejmap;
  int nv = vtxRefs.size() - 1; // The last entry is for unassigned tracks, ignore them
  for (int i = 0; i < 2; i++) {
    mTracksPool[i].clear();
    mVtxFirstTrack[i].clear();
    mVtxFirstTrack[i].resize(nv, -1);
  }
  for (int iv = 0; iv < nv; iv++) {
    const auto& vtref = vtxRefs[iv];
    int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
    for (; it < itLim; it++) {
      auto tvid = trackIndex[it];
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        continue;
      }
      if (tvid.getSource() == GIndex::TPC) {
        if (mSVParams->mExcludeTPCtracks) {
          continue;
        }
        // unconstrained TPC tracks require special treatment: there is no point in checking DCA to mean vertex since it is not precise,
        // but we need to create a clone of TPC track constrained to this particular vertex time.
        if (processTPCTrack(mTPCTracksArray[tvid], tvid, iv)) {
          continue;
        }
      }

      if (tvid.isAmbiguous()) { // was this track already processed?
        auto tref = tmap.find(tvid);
        if (tref != tmap.end()) {
          mTracksPool[tref->second.second][tref->second.first].vBracket.setMax(iv); // this track was already processed with other vertex, account the latter
          continue;
        }
        // was it already rejected?
        if (rejmap.find(tvid) != rejmap.end()) {
          continue;
        }
      }
      const auto& trc = recoData.getTrackParam(tvid);

      bool hasTPC = false;
      bool heavyIonisingParticle = false;
      bool compatibleWithProton = mSVParams->mFractiondEdxforCascBaryons > 0.999f; // if 1 or above, accept all regardless of TPC
      auto tpcGID = recoData.getTPCContributorGID(tvid);
      if (tpcGID.isIndexSet() && isTPCloaded) {
        hasTPC = true;
        auto& tpcTrack = recoData.getTPCTrack(tpcGID);
        float dEdxTPC = tpcTrack.getdEdx().dEdxTotTPC;
        if (dEdxTPC > mSVParams->minTPCdEdx && trc.getP() > mSVParams->minMomTPCdEdx) // accept high dEdx tracks (He3, He4)
        {
          heavyIonisingParticle = true;
        }
        auto protonId = o2::track::PID::Proton;
        float dEdxExpected = mPIDresponse.getExpectedSignal(tpcTrack, protonId);
        float fracDevProton = std::abs((dEdxTPC - dEdxExpected) / dEdxExpected);
        if (fracDevProton < mSVParams->mFractiondEdxforCascBaryons) {
          compatibleWithProton = true;
        }
      }

      // get Nclusters in the ITS if available
      uint8_t nITSclu = -1;
      auto itsGID = recoData.getITSContributorGID(tvid);
      if (itsGID.getSource() == GIndex::ITS) {
        if (isITSloaded) {
          auto& itsTrack = recoData.getITSTrack(itsGID);
          nITSclu = itsTrack.getNumberOfClusters();
        }
      } else if (itsGID.getSource() == GIndex::ITSAB) {
        if (isITSTPCloaded) {
          auto& itsABTracklet = recoData.getITSABRef(itsGID);
          nITSclu = itsABTracklet.getNClusters();
        }
      }
      if (!acceptTrack(tvid, trc) && !heavyIonisingParticle) {
        if (tvid.isAmbiguous()) {
          rejmap[tvid] = true;
        }
        continue;
      }

      if (!hasTPC && nITSclu < mSVParams->mITSSAminNclu) {
        continue; // reject short ITS-only
      }

      int posneg = trc.getSign() < 0 ? 1 : 0;
      float r = std::sqrt(trc.getX() * trc.getX() + trc.getY() * trc.getY());
      mTracksPool[posneg].emplace_back(TrackCand{trc, tvid, {iv, iv}, r, hasTPC, nITSclu, compatibleWithProton});
      if (tvid.getSource() == GIndex::TPC) { // constrained TPC track?
        correctTPCTrack(mTracksPool[posneg].back(), mTPCTracksArray[tvid], -1, -1);
      }
      if (tvid.isAmbiguous()) { // track attached to >1 vertex, remember that it was already processed
        tmap[tvid] = {mTracksPool[posneg].size() - 1, posneg};
      }
    }
  }
  // register 1st track of each charge for each vertex

  for (int pn = 0; pn < 2; pn++) {
    auto& vtxFirstT = mVtxFirstTrack[pn];
    const auto& tracksPool = mTracksPool[pn];
    for (unsigned i = 0; i < tracksPool.size(); i++) {
      const auto& t = tracksPool[i];
      for (int j{t.vBracket.getMin()}; j <= t.vBracket.getMax(); ++j) {
        if (vtxFirstT[j] == -1) {
          vtxFirstT[j] = i;
        }
      }
    }
  }

  LOG(info) << "Collected " << mTracksPool[POS].size() << " positive and " << mTracksPool[NEG].size() << " negative seeds";
}

//__________________________________________________________________
bool SVertexer::checkV0(const TrackCand& seedP, const TrackCand& seedN, int iP, int iN, int ithread)
{
  auto& fitterV0 = mFitterV0[ithread];
  int nCand = fitterV0.process(seedP, seedN);
  if (nCand == 0) { // discard this pair
    return false;
  }
  const auto& v0XYZ = fitterV0.getPCACandidate();
  // validate V0 radial position
  // check closeness to the beam-line
  float dxv0 = v0XYZ[0] - mMeanVertex.getX(), dyv0 = v0XYZ[1] - mMeanVertex.getY(), r2v0 = dxv0 * dxv0 + dyv0 * dyv0;
  if (r2v0 < mMinR2ToMeanVertex) {
    return false;
  }
  float rv0 = std::sqrt(r2v0), drv0P = rv0 - seedP.minR, drv0N = rv0 - seedN.minR;
  if (drv0P > mSVParams->causalityRTolerance || drv0P < -mSVParams->maxV0ToProngsRDiff ||
      drv0N > mSVParams->causalityRTolerance || drv0N < -mSVParams->maxV0ToProngsRDiff) {
    LOG(debug) << "RejCausality " << drv0P << " " << drv0N;
    return false;
  }
  const int cand = 0;
  if (!fitterV0.isPropagateTracksToVertexDone(cand) && !fitterV0.propagateTracksToVertex(cand)) {
    return false;
  }
  auto& trPProp = fitterV0.getTrack(0, cand);
  auto& trNProp = fitterV0.getTrack(1, cand);
  std::array<float, 3> pP, pN;
  trPProp.getPxPyPzGlo(pP);
  trNProp.getPxPyPzGlo(pN);
  // estimate DCA of neutral V0 track to beamline: straight line with parametric equation
  // x = X0 + pV0[0]*t, y = Y0 + pV0[1]*t reaches DCA to beamline (Xv, Yv) at
  // t = -[ (x0-Xv)*pV0[0] + (y0-Yv)*pV0[1]) ] / ( pT(pV0)^2 )
  // Similar equation for 3D distance involving pV0[2]
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
  float pt2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1], prodXYv0 = dxv0 * pV0[0] + dyv0 * pV0[1], tDCAXY = prodXYv0 / pt2V0;
  if (pt2V0 < mMinPt2V0) { // pt cut
    LOG(debug) << "RejPt2 " << pt2V0;
    return false;
  }
  if (pV0[2] * pV0[2] / pt2V0 > mMaxTgl2V0) { // tgLambda cut
    LOG(debug) << "RejTgL " << pV0[2] * pV0[2] / pt2V0;
    return false;
  }
  float p2V0 = pt2V0 + pV0[2] * pV0[2], ptV0 = std::sqrt(pt2V0);
  // apply mass selections
  float p2Pos = pP[0] * pP[0] + pP[1] * pP[1] + pP[2] * pP[2], p2Neg = pN[0] * pN[0] + pN[1] * pN[1] + pN[2] * pN[2];

  bool goodHyp = false;
  std::array<bool, NHypV0> hypCheckStatus{};
  for (int ipid = 0; ipid < NHypV0; ipid++) {
    if (mV0Hyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
      goodHyp = hypCheckStatus[ipid] = true;
    }
  }
  // check tight lambda mass only
  bool goodLamForCascade = false, goodALamForCascade = false;
  bool usesTPCOnly = (seedP.hasTPC && seedP.nITSclu < 1) || (seedN.hasTPC && seedN.nITSclu < 1);
  bool usesShortITSOnly = (!seedP.hasTPC && seedP.nITSclu < mSVParams->mITSSAminNcluCascades) || (!seedN.hasTPC && seedN.nITSclu < mSVParams->mITSSAminNcluCascades);
  if (ptV0 > mSVParams->minPtV0FromCascade && (!mSVParams->mSkipTPCOnlyCascade || !usesTPCOnly) && !usesShortITSOnly) {
    if (mV0Hyps[Lambda].checkTight(p2Pos, p2Neg, p2V0, ptV0) && (!mSVParams->mRequireTPCforCascBaryons || seedP.hasTPC) && seedP.compatibleProton) {
      goodLamForCascade = true;
    }
    if (mV0Hyps[AntiLambda].checkTight(p2Pos, p2Neg, p2V0, ptV0) && (!mSVParams->mRequireTPCforCascBaryons || seedN.hasTPC) && seedN.compatibleProton) {
      goodALamForCascade = true;
    }
  }

  // apply mass selections for 3-body decay
  bool good3bodyV0Hyp = false;
  for (int ipid = 2; ipid < 4; ipid++) {
    float massForLambdaHyp = mV0Hyps[ipid].calcMass(p2Pos, p2Neg, p2V0);
    if (massForLambdaHyp - mV0Hyps[ipid].getMassV0Hyp() < mV0Hyps[ipid].getMargin(ptV0)) {
      good3bodyV0Hyp = true;
      break;
    }
  }

  // we want to reconstruct the 3 body decay of hypernuclei starting from the V0 of a proton and a pion (e.g. H3L->d + (p + pi-), or He4L->He3 + (p + pi-)))
  bool checkFor3BodyDecays = mEnable3BodyDecays && (!mSVParams->checkV0Hypothesis || good3bodyV0Hyp) && (pt2V0 > 0.5) && (!mSVParams->mSkipTPCOnly3Body || !usesTPCOnly);
  bool rejectAfter3BodyCheck = false; // To reject v0s which can be 3-body decay candidates but not cascade or v0
  bool checkForCascade = mEnableCascades && r2v0 < mMaxR2ToMeanVertexCascV0 && (!mSVParams->checkV0Hypothesis || (goodLamForCascade || goodALamForCascade));
  bool rejectIfNotCascade = false;

  if (!goodHyp && mSVParams->checkV0Hypothesis) {
    LOG(debug) << "RejHypo";
    if (!checkFor3BodyDecays && !checkForCascade) {
      return false;
    } else {
      rejectAfter3BodyCheck = true;
    }
  }

  float dcaX = dxv0 - pV0[0] * tDCAXY, dcaY = dyv0 - pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
  float cosPAXY = prodXYv0 / std::sqrt(r2v0 * pt2V0);

  if (checkForCascade) { // use looser cuts for cascade v0 candidates
    if (dca2 > mMaxDCAXY2ToMeanVertexV0Casc || cosPAXY < mSVParams->minCosPAXYMeanVertexCascV0) {
      LOG(debug) << "Rej for cascade DCAXY2: " << dca2 << " << cosPAXY: " << cosPAXY;
      if (!checkFor3BodyDecays) {
        return false;
      } else {
        rejectAfter3BodyCheck = true;
      }
    }
  }
  if (checkFor3BodyDecays) { // use looser cuts for 3-body decay candidates
    if (dca2 > mMaxDCAXY2ToMeanVertex3bodyV0 || cosPAXY < mSVParams->minCosPAXYMeanVertex3bodyV0) {
      LOG(debug) << "Rej for 3 body decays DCAXY2: " << dca2 << " << cosPAXY: " << cosPAXY;
      checkFor3BodyDecays = false;
    }
  }

  if (dca2 > mMaxDCAXY2ToMeanVertex || cosPAXY < mSVParams->minCosPAXYMeanVertex) {
    if (checkForCascade) {
      rejectIfNotCascade = true;
    } else if (checkFor3BodyDecays) {
      rejectAfter3BodyCheck = true;
    } else {
      return false;
    }
  }

  auto vlist = seedP.vBracket.getOverlap(seedN.vBracket); // indices of vertices shared by both seeds
  bool candFound = false;
  auto bestCosPA = checkForCascade ? mSVParams->minCosPACascV0 : mSVParams->minCosPA;
  bestCosPA = checkFor3BodyDecays ? std::min(mSVParams->minCosPA3bodyV0, bestCosPA) : bestCosPA;
  V0 v0new;
  V0Index v0Idxnew;

  for (int iv = vlist.getMin(); iv <= vlist.getMax(); iv++) {
    const auto& pv = mPVertices[iv];
    const auto v0XYZ = fitterV0.getPCACandidatePos(cand);
    // check cos of pointing angle
    float dx = v0XYZ[0] - pv.getX(), dy = v0XYZ[1] - pv.getY(), dz = v0XYZ[2] - pv.getZ(), prodXYZv0 = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
    float cosPA = prodXYZv0 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
    if (cosPA < bestCosPA) {
      LOG(debug) << "Rej. cosPA: " << cosPA;
      continue;
    }
    if (!candFound) {
      new (&v0new) V0(v0XYZ, pV0, fitterV0.calcPCACovMatrixFlat(cand), trPProp, trNProp);
      new (&v0Idxnew) V0Index(-1, seedP.gid, seedN.gid);
      v0new.setDCA(fitterV0.getChi2AtPCACandidate(cand));
      candFound = true;
    }
    v0new.setCosPA(cosPA);
    v0Idxnew.setVertexID(iv);
    bestCosPA = cosPA;
  }
  if (!candFound) {
    return false;
  }
  if (bestCosPA < mSVParams->minCosPACascV0) {
    rejectAfter3BodyCheck = true;
  }
  if (bestCosPA < mSVParams->minCosPA && checkForCascade) {
    rejectIfNotCascade = true;
  }
  int nV0Ini = mV0sIdxTmp[ithread].size();
  // check 3 body decays
  if (checkFor3BodyDecays) {
    int n3bodyDecays = 0;
    n3bodyDecays += check3bodyDecays(v0Idxnew, v0new, rv0, pV0, p2V0, iN, NEG, vlist, ithread);
    n3bodyDecays += check3bodyDecays(v0Idxnew, v0new, rv0, pV0, p2V0, iP, POS, vlist, ithread);
  }
  if (rejectAfter3BodyCheck) {
    return false;
  }

  // check cascades
  int nCascIni = mCascadesIdxTmp[ithread].size(), nV0Used = 0; // number of times this particular v0 (with assigned PV) was used (not counting using its clones with other PV)
  if (checkForCascade) {
    if (goodLamForCascade || !mSVParams->checkCascadeHypothesis) {
      nV0Used += checkCascades(v0Idxnew, v0new, rv0, pV0, p2V0, iN, NEG, vlist, ithread);
    }
    if (goodALamForCascade || !mSVParams->checkCascadeHypothesis) {
      nV0Used += checkCascades(v0Idxnew, v0new, rv0, pV0, p2V0, iP, POS, vlist, ithread);
    }
  }

  if (nV0Used) { // need to fix the index of V0 for the cascades using this v0
    for (unsigned int ic = nCascIni; ic < mCascadesIdxTmp[ithread].size(); ic++) {
      if (mCascadesIdxTmp[ithread][ic].getV0ID() == -1) {
        mCascadesIdxTmp[ithread][ic].setV0ID(nV0Ini);
      }
    }
  }

  if (nV0Used || !rejectIfNotCascade) { // need to add this v0
    mV0sIdxTmp[ithread].push_back(v0Idxnew);
    if (mSVParams->createFullV0s) {
      mV0sTmp[ithread].push_back(v0new);
    }
  }

  if (mStrTracker) {
    for (int iv = nV0Ini; iv < (int)mV0sIdxTmp[ithread].size(); iv++) {
      mStrTracker->processV0(iv, v0new, v0Idxnew, ithread);
    }
  }

  return mV0sIdxTmp[ithread].size() - nV0Ini != 0;
}

//__________________________________________________________________
int SVertexer::checkCascades(const V0Index& v0Idx, const V0& v0, float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread)
{

  // check last added V0 for belonging to cascade
  auto& fitterCasc = mFitterCasc[ithread];
  auto& tracks = mTracksPool[posneg];
  int nCascIni = mCascadesIdxTmp[ithread].size(), nv0use = 0;

  // check if a given PV has already been used in a cascade
  std::unordered_map<int, int> pvMap;

  // start from the 1st bachelor track compatible with earliest vertex in the v0vlist
  int firstTr = mVtxFirstTrack[posneg][v0vlist.getMin()], nTr = tracks.size();
  if (firstTr < 0) {
    firstTr = nTr;
  }
  for (int it = firstTr; it < nTr; it++) {
    if (it == avoidTrackID) {
      continue; // skip the track used by V0
    }
    auto& bach = tracks[it];

    if (!bach.hasTPC && bach.nITSclu < mSVParams->mITSSAminNcluCascades) {
      continue; // reject short ITS-only
    }

    if (bach.vBracket.getMin() > v0vlist.getMax()) {
      LOG(debug) << "Skipping";
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    auto cascVlist = v0vlist.getOverlap(bach.vBracket); // indices of vertices shared by V0 and bachelor
    if (mSVParams->selectBestV0) {
      // select only the best V0 candidate among the compatible ones
      if (v0Idx.getVertexID() < cascVlist.getMin() || v0Idx.getVertexID() > cascVlist.getMax()) {
        continue;
      }
      cascVlist.setMin(v0Idx.getVertexID());
      cascVlist.setMax(v0Idx.getVertexID());
    }

    int nCandC = fitterCasc.process(v0, bach);
    if (nCandC == 0) { // discard this pair
      continue;
    }
    int candC = 0;
    const auto& cascXYZ = fitterCasc.getPCACandidatePos(candC);

    // make sure the cascade radius is smaller than that of the mean vertex
    float dxc = cascXYZ[0] - mMeanVertex.getX(), dyc = cascXYZ[1] - mMeanVertex.getY(), r2casc = dxc * dxc + dyc * dyc;
    if (rv0 * rv0 - r2casc < mMinR2DiffV0Casc || r2casc < mMinR2ToMeanVertex) {
      continue;
    }
    // do we want to apply mass cut ?
    //
    if (!fitterCasc.isPropagateTracksToVertexDone(candC) && !fitterCasc.propagateTracksToVertex(candC)) {
      continue;
    }

    auto& trNeut = fitterCasc.getTrack(0, candC);
    auto& trBach = fitterCasc.getTrack(1, candC);
    trNeut.setPID(o2::track::PID::Lambda);
    trBach.setPID(o2::track::PID::Pion);
    std::array<float, 3> pNeut, pBach;
    trNeut.getPxPyPzGlo(pNeut);
    trBach.getPxPyPzGlo(pBach);
    std::array<float, 3> pCasc = {pNeut[0] + pBach[0], pNeut[1] + pBach[1], pNeut[2] + pBach[2]};

    float pt2Casc = pCasc[0] * pCasc[0] + pCasc[1] * pCasc[1], p2Casc = pt2Casc + pCasc[2] * pCasc[2];
    if (pt2Casc < mMinPt2Casc) { // pt cut
      LOG(debug) << "Casc pt too low";
      continue;
    }
    if (pCasc[2] * pCasc[2] / pt2Casc > mMaxTgl2Casc) { // tgLambda cut
      LOG(debug) << "Casc tgLambda too high";
      continue;
    }

    // compute primary vertex and cosPA of the cascade
    auto bestCosPA = mSVParams->minCosPACasc;
    auto cascVtxID = -1;

    for (int iv = cascVlist.getMin(); iv <= cascVlist.getMax(); iv++) {
      const auto& pv = mPVertices[iv];
      // check cos of pointing angle
      float dx = cascXYZ[0] - pv.getX(), dy = cascXYZ[1] - pv.getY(), dz = cascXYZ[2] - pv.getZ(), prodXYZcasc = dx * pCasc[0] + dy * pCasc[1] + dz * pCasc[2];
      float cosPA = prodXYZcasc / std::sqrt((dx * dx + dy * dy + dz * dz) * p2Casc);
      if (cosPA < bestCosPA) {
        LOG(debug) << "Rej. cosPA: " << cosPA;
        continue;
      }
      cascVtxID = iv;
      bestCosPA = cosPA;
    }
    if (cascVtxID == -1) {
      LOG(debug) << "Casc not compatible with any vertex";
      continue;
    }

    const auto& cascPv = mPVertices[cascVtxID];
    float dxCasc = cascXYZ[0] - cascPv.getX(), dyCasc = cascXYZ[1] - cascPv.getY(), dzCasc = cascXYZ[2] - cascPv.getZ();
    auto prodPPos = pV0[0] * dxCasc + pV0[1] * dyCasc + pV0[2] * dzCasc;
    if (prodPPos < 0.) { // causality cut
      LOG(debug) << "Casc not causally compatible";
      continue;
    }

    float p2Bach = pBach[0] * pBach[0] + pBach[1] * pBach[1] + pBach[2] * pBach[2];
    float ptCasc = std::sqrt(pt2Casc);
    bool goodHyp = false;
    for (int ipid = 0; ipid < NHypCascade; ipid++) {
      if (mCascHyps[ipid].check(p2V0, p2Bach, p2Casc, ptCasc)) {
        goodHyp = true;
        break;
      }
    }
    if (!goodHyp) {
      LOG(debug) << "Casc not compatible with any hypothesis";
      continue;
    }
    // note: at the moment the v0 was not added yet. If some cascade will use v0 (with its PV), the v0 will be added after checkCascades
    // but not necessarily at the and of current v0s vector, since meanwhile checkCascades may add v0 clones (with PV redefined).
    Cascade casc(cascXYZ, pCasc, fitterCasc.calcPCACovMatrixFlat(candC), trNeut, trBach);
    o2::track::TrackParCov trc = casc;
    o2::dataformats::DCA dca;
    if (!trc.propagateToDCA(cascPv, fitterCasc.getBz(), &dca, 5.) ||
        std::abs(dca.getY()) > mSVParams->maxDCAXYCasc || std::abs(dca.getZ()) > mSVParams->maxDCAZCasc) {
      LOG(debug) << "Casc not compatible with PV";
      LOG(debug) << "DCA: " << dca.getY() << " " << dca.getZ();
      continue;
    }
    CascadeIndex cascIdx(cascVtxID, -1, bach.gid); // the v0Idx was not yet added, this will be done after the checkCascades

    LOGP(debug, "cascade successfully validated");

    // clone the V0, set new cosPA and VerteXID, add it to the list of V0s
    if (cascVtxID != v0Idx.getVertexID()) {
      auto pvIdx = pvMap.find(cascVtxID);
      if (pvIdx != pvMap.end()) {
        cascIdx.setV0ID(pvIdx->second); // V0 already exists, add reference to the cascade
      } else {                          // add V0 clone for this cascade (may be used also by other cascades)
        const auto& pv = mPVertices[cascVtxID];
        cascIdx.setV0ID(mV0sIdxTmp[ithread].size()); // set the new V0 index in the cascade
        pvMap[cascVtxID] = mV0sTmp[ithread].size();  // add the new V0 index to the map
        mV0sIdxTmp[ithread].emplace_back(cascVtxID, v0Idx.getProngs());
        if (mSVParams->createFullV0s) {
          mV0sTmp[ithread].push_back(v0);
          float dx = v0.getX() - pv.getX(), dy = v0.getY() - pv.getY(), dz = v0.getZ() - pv.getZ(), prodXYZ = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
          mV0sTmp[ithread].back().setCosPA(prodXYZ / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0));
        }
      }
    } else {
      nv0use++; // original v0 was used
    }
    mCascadesIdxTmp[ithread].push_back(cascIdx);
    if (mSVParams->createFullCascades) {
      casc.setCosPA(bestCosPA);
      casc.setDCA(fitterCasc.getChi2AtPCACandidate(candC));
      mCascadesTmp[ithread].push_back(casc);
    }
    if (mStrTracker) {
      mStrTracker->processCascade(mCascadesIdxTmp[ithread].size() - 1, casc, cascIdx, v0, ithread);
    }
  }

  return nv0use;
}

//__________________________________________________________________
int SVertexer::check3bodyDecays(const V0Index& v0Idx, const V0& v0, float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread)
{
  // check last added V0 for belonging to cascade
  auto& fitter3body = mFitter3body[ithread];
  auto& tracks = mTracksPool[posneg];
  int n3BodyIni = m3bodyIdxTmp[ithread].size();

  // start from the 1st bachelor track compatible with earliest vertex in the v0vlist
  int firstTr = mVtxFirstTrack[posneg][v0vlist.getMin()], nTr = tracks.size();
  if (firstTr < 0) {
    firstTr = nTr;
  }

  // If the V0 is a pair of proton and pion, we should pair it with all positive particles, and the positive particle in the V0 is a proton.
  // Otherwise, we should pair it with all negative particles, and the negative particle in the V0 is a antiproton.

  // start from the 1st track compatible with V0's primary vertex
  for (int it = firstTr; it < nTr; it++) {
    if (it == avoidTrackID) {
      continue; // skip the track used by V0
    }
    auto& bach = tracks[it];
    if (bach.vBracket > v0vlist.getMax()) {
      LOG(debug) << "Skipping";
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    auto decay3bodyVlist = v0vlist.getOverlap(bach.vBracket); // indices of vertices shared by V0 and bachelor
    if (mSVParams->selectBestV0) {
      // select only the best V0 candidate among the compatible ones
      if (v0Idx.getVertexID() < decay3bodyVlist.getMin() || v0Idx.getVertexID() > decay3bodyVlist.getMax()) {
        continue;
      }
      decay3bodyVlist.setMin(v0Idx.getVertexID());
      decay3bodyVlist.setMax(v0Idx.getVertexID());
    }

    if (bach.getPt() < 0.6) {
      continue;
    }

    int n3bodyVtx = fitter3body.process(v0.getProng(0), v0.getProng(1), bach);
    if (n3bodyVtx == 0) { // discard this pair
      continue;
    }
    int cand3B = 0;
    const auto& vertexXYZ = fitter3body.getPCACandidatePos(cand3B);

    // make sure the 3 body vertex radius is close to that of the mean vertex
    float dxc = vertexXYZ[0] - mMeanVertex.getX(), dyc = vertexXYZ[1] - mMeanVertex.getY(), dzc = vertexXYZ[2] - mMeanVertex.getZ(), r2vertex = dxc * dxc + dyc * dyc;
    if (std::abs(rv0 - std::sqrt(r2vertex)) > mSVParams->maxRDiffV03body || r2vertex < mMinR2ToMeanVertex) {
      continue;
    }
    float drvtxBach = std::sqrt(r2vertex) - bach.minR;
    if (drvtxBach > mSVParams->causalityRTolerance || drvtxBach < -mSVParams->maxV0ToProngsRDiff) {
      LOG(debug) << "RejCausality " << drvtxBach;
    }
    //
    if (!fitter3body.isPropagateTracksToVertexDone() && !fitter3body.propagateTracksToVertex()) {
      continue;
    }

    auto& tr0 = fitter3body.getTrack(0, cand3B);
    auto& tr1 = fitter3body.getTrack(1, cand3B);
    auto& tr2 = fitter3body.getTrack(2, cand3B);
    std::array<float, 3> p0, p1, p2;
    tr0.getPxPyPzGlo(p0);
    tr1.getPxPyPzGlo(p1);
    tr2.getPxPyPzGlo(p2);
    std::array<float, 3> p3B = {p0[0] + p1[0] + p2[0], p0[1] + p1[1] + p2[1], p0[2] + p1[2] + p2[2]};

    float pt2candidate = p3B[0] * p3B[0] + p3B[1] * p3B[1], p2candidate = pt2candidate + p3B[2] * p3B[2];
    if (pt2candidate < mMinPt23Body) { // pt cut
      continue;
    }
    if (p3B[2] * p3B[2] / pt2candidate > mMaxTgl23Body) { // tgLambda cut
      continue;
    }

    // compute primary vertex and cosPA of the 3-body decay
    auto bestCosPA = mSVParams->minCosPA3body;
    auto decay3bodyVtxID = -1;

    for (int iv = decay3bodyVlist.getMin(); iv <= decay3bodyVlist.getMax(); iv++) {
      const auto& pv = mPVertices[iv];
      // check cos of pointing angle
      float dx = vertexXYZ[0] - pv.getX(), dy = vertexXYZ[1] - pv.getY(), dz = vertexXYZ[2] - pv.getZ(), prodXYZ3body = dx * p3B[0] + dy * p3B[1] + dz * p3B[2];
      float cosPA = prodXYZ3body / std::sqrt((dx * dx + dy * dy + dz * dz) * p2candidate);
      if (cosPA < bestCosPA) {
        LOG(debug) << "Rej. cosPA: " << cosPA;
        continue;
      }
      decay3bodyVtxID = iv;
      bestCosPA = cosPA;
    }
    if (decay3bodyVtxID == -1) {
      LOG(debug) << "3-body decay not compatible with any vertex";
      continue;
    }

    const auto& decay3bodyPv = mPVertices[decay3bodyVtxID];
    float sqP0 = p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2], sqP1 = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2], sqP2 = p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2];
    float pt3B = std::sqrt(pt2candidate);

    bool goodHyp = false;
    for (int ipid = 0; ipid < 2; ipid++) { // TODO: expand this loop to cover all the 3body cases if (m3bodyHyps[ipid].check(sqP0, sqP1, sqP2, sqPtot, pt3B))
      if (m3bodyHyps[ipid].check(sqP0, sqP1, sqP2, p2candidate, pt3B)) {
        goodHyp = true;
        break;
      }
    }
    if (!goodHyp) {
      continue;
    }
    Decay3Body candidate3B(PID::HyperTriton, vertexXYZ, p3B, fitter3body.calcPCACovMatrixFlat(cand3B), tr0, tr1, tr2);
    o2::track::TrackParCov trc = candidate3B;
    o2::dataformats::DCA dca;
    if (!trc.propagateToDCA(decay3bodyPv, fitter3body.getBz(), &dca, 5.) ||
        std::abs(dca.getY()) > mSVParams->maxDCAXY3Body || std::abs(dca.getZ()) > mSVParams->maxDCAZ3Body) {
      continue;
    }
    if (mSVParams->createFull3Bodies) {
      candidate3B.setCosPA(bestCosPA);
      candidate3B.setDCA(fitter3body.getChi2AtPCACandidate());
      m3bodyTmp[ithread].push_back(candidate3B);
    }
    m3bodyIdxTmp[ithread].emplace_back(decay3bodyVtxID, v0Idx.getProngID(0), v0Idx.getProngID(1), bach.gid);
  }
  return m3bodyIdxTmp[ithread].size() - n3BodyIni;
}

//__________________________________________________________________
template <class TVI, class TCI, class T3I, class TR>
void SVertexer::extractPVReferences(const TVI& v0s, TR& vtx2V0Refs, const TCI& cascades, TR& vtx2CascRefs, const T3I& vtx3, TR& vtx2body3Refs)
{
  // V0s, cascades and 3bodies are already sorted in PV ID
  vtx2V0Refs.clear();
  vtx2V0Refs.resize(mPVertices.size());
  vtx2CascRefs.clear();
  vtx2CascRefs.resize(mPVertices.size());
  vtx2body3Refs.clear();
  vtx2body3Refs.resize(mPVertices.size());
  int nv0 = v0s.size(), nCasc = cascades.size(), n3body = vtx3.size();

  // relate V0s to primary vertices
  int pvID = -1, nForPV = 0;
  for (int iv = 0; iv < nv0; iv++) {
    if (pvID < v0s[iv].getVertexID()) {
      if (pvID > -1) {
        vtx2V0Refs[pvID].setEntries(nForPV);
      }
      pvID = v0s[iv].getVertexID();
      vtx2V0Refs[pvID].setFirstEntry(iv);
      nForPV = 0;
    }
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2V0Refs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = nv0;
    for (int ip = vtx2V0Refs.size(); ip--;) {
      if (vtx2V0Refs[ip].getEntries()) {
        ent = vtx2V0Refs[ip].getFirstEntry();
      } else {
        vtx2V0Refs[ip].setFirstEntry(ent);
      }
    }
  }

  // relate Cascades to primary vertices
  pvID = -1;
  nForPV = 0;
  for (int iv = 0; iv < nCasc; iv++) {
    if (pvID < cascades[iv].getVertexID()) {
      if (pvID > -1) {
        vtx2CascRefs[pvID].setEntries(nForPV);
      }
      pvID = cascades[iv].getVertexID();
      vtx2CascRefs[pvID].setFirstEntry(iv);
      nForPV = 0;
    }
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2CascRefs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = nCasc;
    for (int ip = vtx2CascRefs.size(); ip--;) {
      if (vtx2CascRefs[ip].getEntries()) {
        ent = vtx2CascRefs[ip].getFirstEntry();
      } else {
        vtx2CascRefs[ip].setFirstEntry(ent);
      }
    }
  }

  // relate 3 body decays to primary vertices
  pvID = -1;
  nForPV = 0;
  for (int iv = 0; iv < n3body; iv++) {
    const auto& vertex3body = vtx3[iv];
    if (pvID < vertex3body.getVertexID()) {
      if (pvID > -1) {
        vtx2body3Refs[pvID].setEntries(nForPV);
      }
      pvID = vertex3body.getVertexID();
      vtx2body3Refs[pvID].setFirstEntry(iv);
      nForPV = 0;
    }
    nForPV++;
  }
  if (pvID != -1) { // finalize
    vtx2body3Refs[pvID].setEntries(nForPV);
    // fill empty slots
    int ent = n3body;
    for (int ip = vtx2body3Refs.size(); ip--;) {
      if (vtx2body3Refs[ip].getEntries()) {
        ent = vtx2body3Refs[ip].getFirstEntry();
      } else {
        vtx2body3Refs[ip].setFirstEntry(ent);
      }
    }
  }
}

//__________________________________________________________________
void SVertexer::setNThreads(int n)
{
#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  mNThreads = 1;
#endif
}

//______________________________________________
bool SVertexer::processTPCTrack(const o2::tpc::TrackTPC& trTPC, GIndex gid, int vtxid)
{
  if (mSVParams->mTPCTrackMaxX > 0. && trTPC.getX() > mSVParams->mTPCTrackMaxX) {
    return true;
  }
  // if TPC trackis unconstrained, try to create in the tracks pool a clone constrained to vtxid vertex time.
  if (trTPC.hasBothSidesClusters()) { // this is effectively constrained track
    return false;                     // let it be processed as such
  }
  const auto& vtx = mPVertices[vtxid];
  auto twe = vtx.getTimeStamp();
  int posneg = trTPC.getSign() < 0 ? 1 : 0;
  auto& trLoc = mTracksPool[posneg].emplace_back(TrackCand{trTPC, gid, {vtxid, vtxid}, 0.});
  auto err = correctTPCTrack(trLoc, trTPC, twe.getTimeStamp(), twe.getTimeStampError());
  if (err < 0) {
    mTracksPool[posneg].pop_back(); // discard
  }
  return true;
}

//______________________________________________
float SVertexer::correctTPCTrack(SVertexer::TrackCand& trc, const o2::tpc::TrackTPC& tTPC, float tmus, float tmusErr) const
{
  // Correct the track copy trc of the TPC track for the assumed interaction time
  // return extra uncertainty in Z due to the interaction time uncertainty
  // TODO: at the moment, apply simple shift, but with Z-dependent calibration we may
  // need to do corrections on TPC cluster level and refit
  // This is almosto clone of the MatchTPCITS::correctTPCTrack

  float tTB, tTBErr;
  if (tmusErr < 0) { // use track data
    tTB = tTPC.getTime0();
    tTBErr = 0.5 * (tTPC.getDeltaTBwd() + tTPC.getDeltaTFwd());
  } else {
    tTB = tmus * mMUS2TPCBin;
    tTBErr = tmusErr * mMUS2TPCBin;
  }
  float dDrift = (tTB - tTPC.getTime0()) * mTPCBin2Z;
  float driftErr = tTBErr * mTPCBin2Z;
  // eventually should be refitted, at the moment we simply shift...
  trc.setZ(tTPC.getZ() + (tTPC.hasASideClustersOnly() ? dDrift : -dDrift));
  trc.setCov(trc.getSigmaZ2() + driftErr * driftErr, o2::track::kSigZ2);
  uint8_t sector, row;
  auto cl = &tTPC.getCluster(mTPCTrackClusIdx, tTPC.getNClusters() - 1, *mTPCClusterIdxStruct, sector, row);
  float x = 0, y = 0, z = 0;
  mTPCCorrMapsHelper->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, tTB);
  if (x < o2::constants::geom::XTPCInnerRef) {
    x = o2::constants::geom::XTPCInnerRef;
  }
  trc.minR = std::sqrt(x * x + y * y);
  LOGP(debug, "set MinR = {} for row {}, x:{}, y:{}, z:{}", trc.minR, row, x, y, z);
  return driftErr;
}

//______________________________________________
std::array<size_t, 3> SVertexer::getNFitterCalls() const
{
  std::array<size_t, 3> calls{};
  for (int i = 0; i < mNThreads; i++) {
    calls[0] += mFitterV0[i].getCallID();
    calls[1] += mFitterCasc[i].getCallID();
    calls[2] += mFitter3body[i].getCallID();
  }
  return calls;
}
