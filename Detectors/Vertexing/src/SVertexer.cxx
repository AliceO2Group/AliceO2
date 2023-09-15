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
#include<iostream>
#include <fstream>
#include "DetectorsVertexing/SVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include "CorrectionMapsHelper.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "ReconstructionDataFormats/GlobalTrackID.h"
using namespace o2::vertexing;

using PID = o2::track::PID;
using TrackTPCITS = o2::dataformats::TrackTPCITS;
using TrackITS = o2::its::TrackITS;
using TrackTPC = o2::tpc::TrackTPC;
using VBracket = o2::math_utils::Bracket<int>;

std::ofstream log_file(
        "log_file.txt", std::ios_base::out | std::ios_base::app );
std::ofstream log_pT4Hlambdalambda(
        "log_pT4Hlambdalambda.txt", std::ios_base::out | std::ios_base::app );
std::ofstream log_pT4He3Lambda(
        "log_pT4He3Lambda.txt", std::ios_base::out | std::ios_base::app );
std::ofstream log_pTpion(
        "log_pTpion.txt", std::ios_base::out | std::ios_base::app );

std::ofstream log_pT3body(
        "log_pT3body.txt", std::ios_base::out | std::ios_base::app );

//__________________________________________________________________
void SVertexer::process(const o2::globaltracking::RecoContainer& recoData) // accessor to various reconstrucred data types
{
  updateTimeDependentParams(); // TODO RS: strictly speaking, one should do this only in case of the CCDB objects update
  mPVertices = recoData.getPrimaryVertices();
  buildT2V(recoData); // build track->vertex refs from vertex->track (if other workflow will need this, consider producing a message in the VertexTrackMatcher)
  int ntrP = mTracksPool[POS].size(), ntrN = mTracksPool[NEG].size(), iThread = 0;
  mV0sTmp[0].clear();
  mCascadesTmp[0].clear();
  mDoubleHypH4Tmp[0].clear();

  m3bodyTmp[0].clear();
  //  mCasc3bodyTmp[0].clear();
  
  //  mHypHe4Tmp[0].clear();
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
#ifdef WITH_OPENMP
      iThread = omp_get_thread_num();
#endif
      if (mSVParams->maxPVContributors < 2 && seedP.gid.isPVContributor() + seedN.gid.isPVContributor() > mSVParams->maxPVContributors) {
        continue;
      }
      checkV0(seedP, seedN, itp, itn, iThread);
    }
  }
  // sort V0s and Cascades in vertex id
  struct vid {
    int thrID;
    int entry;
    int vtxID;
  };
  size_t nv0 = 0, ncsc = 0, n3body = 0, nCasc3body =0, nHypHe4 =0, nDoubleHypeH4=0;
  for (int i = 0; i < mNThreads; i++) {
    nv0 += mV0sTmp[0].size();
    ncsc += mCascadesTmp[i].size();
    n3body += m3bodyTmp[i].size();
    nCasc3body += mCasc3bodyTmp[i].size();
    nDoubleHypeH4 += mDoubleHypH4Tmp[i].size();
    
    //   nHypHe4 += mHypHe4Tmp[i].size();
    
  }
  std::vector<vid> v0SortID, cascSortID, nbodySortID, nCascbodySortID, nHypHe4SortID,ndoubleHypH4SortID;
  v0SortID.reserve(nv0);
  cascSortID.reserve(ncsc);
  nbodySortID.reserve(n3body);
  ndoubleHypH4SortID.reserve(nDoubleHypeH4);
  
  //  nCascbodySortID.reserve(nCasc3body);
  //  nHypHe4SortID.reserve(nHypHe4);
  
  for (int i = 0; i < mNThreads; i++) {
    for (int j = 0; j < (int)mV0sTmp[i].size(); j++) {
      v0SortID.emplace_back(vid{i, j, mV0sTmp[i][j].getVertexID()});
    }
    for (int j = 0; j < (int)mCascadesTmp[i].size(); j++) {
      cascSortID.emplace_back(vid{i, j, mCascadesTmp[i][j].getVertexID()});
    }
    for (int j = 0; j < (int)mDoubleHypH4Tmp[i].size(); j++) {
      ndoubleHypH4SortID.emplace_back(vid{i, j, mDoubleHypH4Tmp[i][j].getVertexID()});
    }
    
    for (int j = 0; j < (int)m3bodyTmp[i].size(); j++) {
      nbodySortID.emplace_back(vid{i, j, m3bodyTmp[i][j].getVertexID()});
    }
    //    for (int j = 0; j < (int)mCasc3bodyTmp[i].size(); j++) {
    //      nCascbodySortID.emplace_back(vid{i, j, mCasc3bodyTmp[i][j].getVertexID()});
    //    }
    //    for (int j = 0; j < (int)mHypHe4Tmp[i].size(); j++) {
    //      nHypHe4SortID.emplace_back(vid{i, j, mCasc3bodyTmp[i][j].getVertexID()});
    //    }
    //
  }
  std::sort(v0SortID.begin(), v0SortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  std::sort(cascSortID.begin(), cascSortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  std::sort(nbodySortID.begin(), nbodySortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  std::sort(ndoubleHypH4SortID.begin(), ndoubleHypH4SortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  
  //  std::sort(nHypHe4SortID.begin(), nHypHe4SortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  
  //  std::sort(nCascbodySortID.begin(), nCascbodySortID.end(), [](const vid& a, const vid& b) { return a.vtxID > b.vtxID; });
  
  // sorted V0s
  std::vector<V0> bufV0;
  bufV0.reserve(nv0);
  for (const auto& id : v0SortID) {
    auto& v0 = mV0sTmp[id.thrID][id.entry];
    int pos = bufV0.size();
    bufV0.push_back(v0);
    v0.setVertexID(pos); // this v0 copy will be discarded, use its vertexID to store the new position of final V0
  }
  // since V0s were reshuffled, we need to correct the cascade -> V0 reference indices
  for (int i = 0; i < mNThreads; i++) {  // merge results of all threads
    for (auto& casc : mCascadesTmp[i]) { // before merging fix cascades references on v0
      casc.setV0ID(mV0sTmp[i][casc.getV0ID()].getVertexID());
    }
  }
  // sorted Cascades
  std::vector<Cascade> bufCasc;
  bufCasc.reserve(ncsc);
  for (const auto& id : cascSortID) {
    bufCasc.push_back(mCascadesTmp[id.thrID][id.entry]);
  }
  
  //I don't know but for the time being doing the same thing as for the cascade analysis
  // I thing here we are using the 3body candidates so we need to rehuffled that thing ->let's try what will be happen 
      	// since V0s were reshuffled, we need to correct the cascade -> V0 reference indices
  for (int i = 0; i < mNThreads; i++) {  // merge results of all threads
    for (auto& doubleH4 : mDoubleHypH4Tmp[i]) { // before merging fix cascades references on v0
      doubleH4.setV0ID(m3bodyTmp[i][doubleH4.getV0ID()].getVertexID());
    }
  }
  // sorted doubleHypeH4
  std::vector<Cascade> bufdoubleHypeH4;
  bufdoubleHypeH4.reserve(nDoubleHypeH4);
  for (const auto& id : ndoubleHypH4SortID ) {
    bufdoubleHypeH4.push_back(mDoubleHypH4Tmp[id.thrID][id.entry]);
  }
  

  
  // sorted 3 body decays
  std::vector<DecayNbody> buf3body;
  buf3body.reserve(n3body);
  for (const auto& id : nbodySortID) {
    auto& decay3body = m3bodyTmp[id.thrID][id.entry];
    int pos = bufV0.size();
    buf3body.push_back(decay3body);
  }
  // sorted cascade to 3 body decays
  std::vector<DecayNbody> bufCasc3body;
  bufCasc3body.reserve(nCasc3body);
  for (const auto& id : nCascbodySortID) {
    auto& decayCasc3body = mCasc3bodyTmp[id.thrID][id.entry];
    int pos = bufCasc3body.size();
    bufCasc3body.push_back(decayCasc3body);
  }
  // sorted HypHe4 to 3 body decays
  std::vector<DecayNbody> bufHypHe43body;
  bufHypHe43body.reserve(nHypHe4);
  for (const auto& id : nHypHe4SortID) {
    auto& decayHypHe43body = mHypHe4Tmp[id.thrID][id.entry];
    int pos = bufHypHe43body.size();
    bufHypHe43body.push_back(decayHypHe43body);
  }
  
  //
  mV0sTmp[0].swap(bufV0);               // the final result is fetched from here
  mCascadesTmp[0].swap(bufCasc);        // the final result is fetched from here
  mDoubleHypH4Tmp[0].swap(bufdoubleHypeH4);        // the final result is fetched from here
  
  m3bodyTmp[0].swap(buf3body);          // the final result is fetched from here
  //  mCasc3bodyTmp[0].swap(bufCasc3body);          // the final result is fetched from here
  //  mHypHe4Tmp[0].swap(bufHypHe43body);          // the final result is fetched from here
  
  for (int i = 1; i < mNThreads; i++) { // clean unneeded s.vertices
    mV0sTmp[i].clear();
    mCascadesTmp[i].clear();
    mDoubleHypH4Tmp[i].clear();
    m3bodyTmp[i].clear();
    //   mCasc3bodyTmp[i].clear();
    //   mHypHe4Tmp[i].clear();
    
  }
  LOG(debug) << "DONE : " << mV0sTmp[0].size() << " " << mCascadesTmp[0].size() << " " << m3bodyTmp[0].size()<< mCasc3bodyTmp[0].size()<<""  <<mHypHe4Tmp[0].size()<< mDoubleHypH4Tmp[0].size() ;
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
  mCascHyps[HypCascade::XiMinus].set(PID::XiMinus, PID::Lambda, PID::Pion, mSVParams->pidCutsXiMinus, bz);
  mCascHyps[HypCascade::OmegaMinus].set(PID::OmegaMinus, PID::Lambda, PID::Kaon, mSVParams->pidCutsOmegaMinus, bz);
  mDoubleHyps[DoubleHyp3B::DoubleHyperhydrogen4].set(PID::DoubleHyperhydrogen4, PID::Hyperhelium4, PID::Pion, mSVParams->pidCutsXiMinus, bz);
  mDoubleHyps[DoubleHyp3B::DoubleAntiHyperhydrogen4].set(PID::DoubleHyperhydrogen4, PID::Hyperhelium4, PID::Pion, mSVParams->pidCutsXiMinus, bz);


  m3bodyHyps[Hyp3body::H3L3body].set(PID::HyperTriton, PID::Proton, PID::Pion, PID::Deuteron, mSVParams->pidCutsH3L3body, bz);
  m3bodyHyps[Hyp3body::AntiH3L3body].set(PID::HyperTriton, PID::Pion, PID::Proton, PID::Deuteron, mSVParams->pidCutsH3L3body, bz);
 m3bodyHyps[Hyp3body::He4L3body].set(PID::Hyperhelium4, PID::Proton, PID::Pion, PID::Deuteron, mSVParams->pidCutsHe4L3body, bz);
  m3bodyHyps[Hyp3body::AntiHe4L3body].set(PID::Hyperhelium4, PID::Pion, PID::Proton, PID::Deuteron, mSVParams->pidCutsHe4L3body, bz);
  for (auto& ft : mFitterV0) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitterCasc) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitter3body) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitterCasc3body) {
    ft.setBz(bz);
  }
  for (auto& ft : mFitterdoubleHypH4) {
    ft.setBz(bz);
  }

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
  // to be used with refitter as
  // mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, mTPCCorrMapsHelper, mBz, mTPCTrackClusIdx.data(), mTPCRefitterShMap.data(), nullptr, o2::base::Propagator::Instance());
}

//__________________________________________________________________
void SVertexer::setupThreads()
{
  if (!mV0sTmp.empty()) {
    return;
  }
  mV0sTmp.resize(mNThreads);
  mCascadesTmp.resize(mNThreads);
  mCasc3bodyTmp.resize(mNThreads);
  m3bodyTmp.resize(mNThreads);
  mFitterV0.resize(mNThreads);
  mDoubleHypH4Tmp.resize(mNThreads);

  auto bz = o2::base::Propagator::Instance()->getNominalBz();
  for (auto& fitter : mFitterV0) {
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
  for (auto& fitter : mFitterCasc) {
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
  for (auto& fitter : mFitter3body) {
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
  mFitterCasc3body.resize(mNThreads);
  for (auto& fitter : mFitterCasc3body) {
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
  mFitterdoubleHypH4.resize(mNThreads);
  for (auto& fitter : mFitterdoubleHypH4) {
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
      // unconstrained TPC tracks require special treatment: there is no point in checking DCA to mean vertex since it is not precise,
      // but we need to create a clone of TPC track constrained to this particular vertex time.
      if (tvid.getSource() == GIndex::TPC && processTPCTrack(recoData.getTPCTrack(tvid), tvid, iv)) { // processTPCTrack may decide that this track does not need special treatment (e.g. it is constrained...)
        continue;
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
      if (!acceptTrack(tvid, trc)) {
        if (tvid.isAmbiguous()) {
          rejmap[tvid] = true;
        }
        continue;
      }
      int posneg = trc.getSign() < 0 ? 1 : 0;
      float r = std::sqrt(trc.getX() * trc.getX() + trc.getY() * trc.getY());
      mTracksPool[posneg].emplace_back(TrackCand{trc, tvid, {iv, iv}, r});
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
  int nCand = fitterV0.process(seedP, seedN);	//This needed to emplement in doubleHyperHydrogen4
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
  float rv0 = std::sqrt(r2v0), drv0P = rv0 - seedP.minR, drv0N = rv0 - seedN.minR;	//I don't know why he done all these things
  if (drv0P > mSVParams->causalityRTolerance || drv0P < -mSVParams->maxV0ToProngsRDiff ||
      drv0N > mSVParams->causalityRTolerance || drv0N < -mSVParams->maxV0ToProngsRDiff) {
    LOG(debug) << "RejCausality " << drv0P << " " << drv0N;
    return false;
  }
  //  LOG(info)<<"The value of seedP.minR is "<<seedP.minR <<" and the value of seedN.minR is "<<seedN.minR << " the value of rv0 "<< rv0 << " and the value of mSVParams->causalityRTolerance is "<< mSVParams->causalityRTolerance << " and the value of -mSVParams->maxV0ToProngsRDiff is " << -mSVParams->maxV0ToProngsRDiff;

  if (!fitterV0.isPropagateTracksToVertexDone() && !fitterV0.propagateTracksToVertex()) {
    return false;
  }
  int cand = 0;
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
  
  //   LOG(info) <<"The momentum of 1st particle is "<< std::sqrt(p2Pos) << " and the momentum of second particle is "<<std::sqrt(p2Neg) ;
  
  //can you please tell me what is this v0 Hypothesis check ?
  bool goodHyp = false;
  std::array<bool, NHypV0> hypCheckStatus{};
  for (int ipid = 0; ipid < NHypV0; ipid++) {
    if (mV0Hyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
      goodHyp = hypCheckStatus[ipid] = true;
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
  bool checkFor3BodyDecays = mEnable3BodyDecays && (!mSVParams->checkV0Hypothesis || good3bodyV0Hyp) && (pt2V0 > 0.5);
  bool rejectAfter3BodyCheck = false; // To reject v0s which can be 3-body decay candidates but not cascade or v0
  bool checkForCascade = mEnableCascades && r2v0 < mMaxR2ToMeanVertexCascV0 && (!mSVParams->checkV0Hypothesis || (hypCheckStatus[HypV0::Lambda] || hypCheckStatus[HypV0::AntiLambda]));
  bool rejectIfNotCascade = false;
  
  //  bool checkFordoubleHypH4 = mEnableCascades && r2v0 < mMaxR2ToMeanVertexCascV0 && (!mSVParams->checkV0Hypothesis || (hypCheckStatus[DoubleHyp3B::DoubleHyperhydrogen4] || hypCheckStatus[DoubleHyp3B::DoubleHyperhydrogen4]));
  //  bool rejectIfNotDoubeHypeH4 = false;

  
  
  if (!goodHyp && mSVParams->checkV0Hypothesis) {
    LOG(debug) << "RejHypo";
    if (!checkFor3BodyDecays && !checkForCascade ) {
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
  bool added = false;
  auto bestCosPA = checkForCascade ? mSVParams->minCosPACascV0 : mSVParams->minCosPA;
  bestCosPA = checkFor3BodyDecays ? std::min(mSVParams->minCosPA3bodyV0, bestCosPA) : bestCosPA;

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
    if (!added) {
      auto& v0new = mV0sTmp[ithread].emplace_back(v0XYZ, pV0, fitterV0.calcPCACovMatrixFlat(cand), trPProp, trNProp, seedP.gid, seedN.gid);
      v0new.setDCA(fitterV0.getChi2AtPCACandidate());
      added = true;
    }
    auto& v0 = mV0sTmp[ithread].back();
    v0.setCosPA(cosPA);
    v0.setVertexID(iv);
    bestCosPA = cosPA;
  }
  if (!added) {
    return false;
  }
  if (bestCosPA < mSVParams->minCosPACascV0) {
    rejectAfter3BodyCheck = true;
  }
  if (bestCosPA < mSVParams->minCosPA && checkForCascade) {
    rejectIfNotCascade = true;
  }

  // check 3 body decays
  if (checkFor3BodyDecays) {
    int n3bodyDecays = 0;
    n3bodyDecays += check3bodyDecays(rv0, pV0, p2V0, iN, NEG, vlist, ithread);
    //    n3bodyDecays += check3bodyDecays(rv0, pV0, p2V0, iP, POS, vlist, ithread);
  }
  if (rejectAfter3BodyCheck) {
    mV0sTmp[ithread].pop_back();
    return false;
  }
  
  // check cascades
  if (checkForCascade) {
    int nCascAdded = 0;
    if (hypCheckStatus[HypV0::Lambda] || !mSVParams->checkCascadeHypothesis) {
      nCascAdded += checkCascades(rv0, pV0, p2V0, iN, NEG, vlist, ithread);
    }
    if (hypCheckStatus[HypV0::AntiLambda] || !mSVParams->checkCascadeHypothesis) {
      nCascAdded += checkCascades(rv0, pV0, p2V0, iP, POS, vlist, ithread);
    }
    if (!nCascAdded && rejectIfNotCascade) { // v0 would be accepted only if it creates a cascade
      mV0sTmp[ithread].pop_back();
      return false;
    }
  }

  return true;
}

//__________________________________________________________________
int SVertexer::checkCascades(float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread)
{

  // check last added V0 for belonging to cascade
  auto& fitterCasc = mFitterCasc[ithread];
  const auto v0Id = mV0sTmp[ithread].size() - 1; // we check the last V0 but some cascades may add V0 clones
  auto& tracks = mTracksPool[posneg];
  int nCascIni = mCascadesTmp[ithread].size();

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
    if (bach.vBracket.getMin() > v0vlist.getMax()) {
      LOG(debug) << "Skipping";
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    const auto& v0 = mV0sTmp[ithread][v0Id];
    auto cascVlist = v0vlist.getOverlap(bach.vBracket); // indices of vertices shared by V0 and bachelor
    if (mSVParams->selectBestV0) {
      // select only the best V0 candidate among the compatible ones
      if (v0.getVertexID() < cascVlist.getMin() || v0.getVertexID() > cascVlist.getMax()) {
        continue;
      }
      cascVlist.setMin(v0.getVertexID());
      cascVlist.setMax(v0.getVertexID());
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
    if (!fitterCasc.isPropagateTracksToVertexDone() && !fitterCasc.propagateTracksToVertex()) {
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
    auto& casc = mCascadesTmp[ithread].emplace_back(cascXYZ, pCasc, fitterCasc.calcPCACovMatrixFlat(candC), trNeut, trBach, v0Id, bach.gid);
    //auto& doubleHyperHypdrogen4 = mDoubleHypH4Tmp[ithread].emplace_back(doubleHypH4XYZ, pdoubleHypH4, fitterdoubleHypH4.calcPCACovMatrixFlat(nCanddoubleH4), trHypHe4, trBach, body3Id,bach.gid);


    o2::track::TrackParCov trc = casc;
    o2::dataformats::DCA dca;
    if (!trc.propagateToDCA(cascPv, fitterCasc.getBz(), &dca, 5.) ||
        std::abs(dca.getY()) > mSVParams->maxDCAXYCasc || std::abs(dca.getZ()) > mSVParams->maxDCAZCasc) {
      LOG(debug) << "Casc not compatible with PV";
      LOG(debug) << "DCA: " << dca.getY() << " " << dca.getZ();
      mCascadesTmp[ithread].pop_back();
      continue;
    }

    LOG(debug) << "Casc successfully added";
    casc.setCosPA(bestCosPA);
    casc.setVertexID(cascVtxID);
    casc.setDCA(fitterCasc.getChi2AtPCACandidate());

    // clone the V0, set new cosPA and VerteXID, add it to the list of V0s
    if (cascVtxID != v0.getVertexID()) {
      auto v0clone = v0;
      const auto& pv = mPVertices[cascVtxID];

      float dx = v0.getX() - pv.getX(), dy = v0.getY() - pv.getY(), dz = v0.getZ() - pv.getZ(), prodXYZ = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
      float cosPA = prodXYZ / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
      v0clone.setCosPA(cosPA);
      v0clone.setVertexID(cascVtxID);

      auto pvIdx = pvMap.find(cascVtxID);
      if (pvIdx != pvMap.end()) {
        casc.setV0ID(pvIdx->second); // V0 already exists, add reference to the cascade
      } else {
        mV0sTmp[ithread].push_back(v0clone);
        casc.setV0ID(mV0sTmp[ithread].size() - 1);      // set the new V0 index in the cascade
        pvMap[cascVtxID] = mV0sTmp[ithread].size() - 1; // add the new V0 index to the map
      }
    }
  }

  return mCascadesTmp[ithread].size() - nCascIni;
}

//__________________________________________________________________
int SVertexer::checkdoubleHypH4(float r3body, std::array<float, 3> p3body, float p23body, int avoidTrackID, int posneg, VBracket decay3bodylist, int ithread)
{

	LOG(info) << "check for the doubleHyperHydrogen4 is started now ";

  auto& fitterdoubleHypH4 = mFitterdoubleHypH4[ithread];

  // I just want to create an object which can store the doubleHyperHelium4
  const auto body3Id = m3bodyTmp[ithread].size() - 1; // we check the last added 3body

  //  const auto body3Id = m3bodyTmp[ithread].size() - 1; // we check the last added 3body
  auto& tracks = mTracksPool[posneg];
  int ndoubleHypH4Ini = mDoubleHypH4Tmp[ithread].size();
  //
  //  // check if a given PV has already been used in a HypeHe4
  //  std::unordered_map<int, int> pvMap;
  //
  // start from the 1st bachelor track compatible with earliest vertex in the v0vlist	//Here I need decay3bodylist but I am not getting any getter for it but lets try
  int firstTr = mVtxFirstTrack[posneg][decay3bodylist.getMin()], nTr = tracks.size();
  if (firstTr < 0) {
    firstTr = nTr;
  }
  for (int it = firstTr; it < nTr; it++) {
    if (it == avoidTrackID) { //from where you know to which trackID should be avoid becay in V0 this is simpke because we have only two tracks if positive or negative.      Need to work on it (but my problem is solved since in 3body reconstruction they use the proton and pions for the V0 candidates not the 3He or anything else).
      continue; // skip the track used by 3bodyDecay
    }
    auto& bach = tracks[it];
    if (bach.vBracket.getMin() > decay3bodylist.getMax()) {
      LOG(debug) << "Skipping";
      break; // all other bachelor candidates will be also not compatible with this PV
    }
    const auto& body3 = m3bodyTmp[ithread].back();      //from my side this is the 3body reconstructed track but why I am getting error in this case

    //    const auto& body3 = m3bodyTmp[ithread][body3Id];      //from my side this is the 3body reconstructed track but why I am getting error in this case
    auto doubleHypH4Vlist = decay3bodylist.getOverlap(bach.vBracket); // indices of vertices shared by V0 and bachelor
    if (mSVParams->selectBestV0) {
      // select only the best V0 candidate among the compatible ones
      if (body3.getVertexID() < doubleHypH4Vlist.getMin() || body3.getVertexID() > doubleHypH4Vlist.getMax()) {
        continue;
      }
      doubleHypH4Vlist.setMin(body3.getVertexID());
      doubleHypH4Vlist.setMax(body3.getVertexID());
    }
    //
    int nCanddoubleH4 = fitterdoubleHypH4.process(body3, bach);
    if (nCanddoubleH4 == 0) { // discard this pair
      continue;
    }

    int canddoubleH4 = 0;
    const auto& doubleHypH4XYZ = fitterdoubleHypH4.getPCACandidatePos(canddoubleH4);	//need one this variable for emplace_back
    
    // make sure the doubleHypH4 radius is smaller than that of the mean vertex
    float dxc = doubleHypH4XYZ[0] - mMeanVertex.getX(), dyc = doubleHypH4XYZ[1] - mMeanVertex.getY(), r2doubleHypH4 = dxc * dxc + dyc * dyc;
    if (r3body * r3body - r2doubleHypH4 < mMinR2DiffV0Casc || r2doubleHypH4 < mMinR2ToMeanVertex) {
      continue;
    }
    // do we want to apply mass cut ?
    //
    if (!fitterdoubleHypH4.isPropagateTracksToVertexDone() && !fitterdoubleHypH4.propagateTracksToVertex()) {
      continue;
    }
    auto& trHypHe4 = fitterdoubleHypH4.getTrack(0, canddoubleH4);
    auto& trBach = fitterdoubleHypH4.getTrack(1, canddoubleH4);
    trHypHe4.setPID(o2::track::PID::Hyperhelium4);
    trBach.setPID(o2::track::PID::Pion);
    std::array<float, 3> pHypHe4, pBach;
    trHypHe4.getPxPyPzGlo(pHypHe4);
    trBach.getPxPyPzGlo(pBach);	
    std::array<float, 3> pdoubleHypH4 = {pHypHe4[0] + pBach[0], pHypHe4[1] + pBach[1], pHypHe4[2] + pBach[2]};


    float pt2doubleHypH4 = pdoubleHypH4[0] * pdoubleHypH4[0] + pdoubleHypH4[1] * pdoubleHypH4[1], p2doubleHypH4 = pt2doubleHypH4 + pdoubleHypH4[2] * pdoubleHypH4[2];
    //Now I have the information of momentum of the two candidates can you please tell me how to calculate the mass from here:
	float p2HyperHe4 = pHypHe4[0]*pHypHe4[0] + pHypHe4[1]* pHypHe4[1] + pHypHe4[2]*pHypHe4[2];
	float p2Bach = pBach[0]*pBach[0] + pBach[1]*pBach[1] +pBach[2]*pBach[2];

	float pTHyperHe4 = std::sqrt(pHypHe4[0]*pHypHe4[0] + pHypHe4[1]* pHypHe4[1]);
	float pTBach = std::sqrt(pBach[0]*pBach[0] + pBach[1]*pBach[1]);


    if (pt2doubleHypH4 < mMinPt2Casc) { // pt cut
      LOG(debug) << "doubleHypH4 pt too low";
      continue;
    }
    if (pdoubleHypH4[2] * pdoubleHypH4[2] / pt2doubleHypH4 > mMaxTgl2Casc) { // I don't know about this cut but similar cut is applied in the cascades
      LOG(debug) << "Casc tgLambda too high";
      continue;
    }

    // compute primary vertex and cosPA of the cascade
    auto bestCosPA1 = mSVParams->minCosPACasc;
    auto doubleHypeH4VtxID = -1;

   for (int iv = doubleHypH4Vlist.getMin(); iv <= doubleHypH4Vlist.getMax(); iv++) {
      const auto& pv = mPVertices[iv];
      // check cos of pointing angle
      float dx = doubleHypH4XYZ[0] - pv.getX(), dy = doubleHypH4XYZ[1] - pv.getY(), dz = doubleHypH4XYZ[2] - pv.getZ(), prodXYZdoubleHyperH4 = dx * pdoubleHypH4[0] + dy * pdoubleHypH4[1] + dz * pdoubleHypH4[2];
      float cosPA = prodXYZdoubleHyperH4 / std::sqrt((dx * dx + dy * dy + dz * dz) * pt2doubleHypH4);
      if (cosPA < bestCosPA1) {
        LOG(debug) << "Rej. cosPA: " << cosPA;
        continue;
      }
      doubleHypeH4VtxID = iv;
      bestCosPA1 = cosPA;
    }
    if (doubleHypeH4VtxID == -1) {
      LOG(debug) << "Casc not compatible with any vertex";
      continue;
    }

   const auto& doubleHyperH4Pv = mPVertices[doubleHypeH4VtxID];
    float dxdoubleHypeH4 = doubleHypH4XYZ[0] - doubleHyperH4Pv.getX(), dydoubleHyperH4 = doubleHypH4XYZ[1] - doubleHyperH4Pv.getY(), dzdoubleHyperH4 = doubleHypH4XYZ[2] - doubleHyperH4Pv.getZ();
    auto prodPPos = p3body[0] * dxdoubleHypeH4 + p3body[1] * dxdoubleHypeH4 + p3body[2] * dzdoubleHyperH4;
    if (prodPPos < 0.) { // causality cut
      LOG(debug) << "Casc not causally compatible";
      continue;
    }
    // compute primary vertex and cosPA of the cascade
    auto bestCosPA = mSVParams->minCosPACasc;
    auto doubleHypH4VtxID = -1;

    auto& doubleHyperHypdrogen4 = mDoubleHypH4Tmp[ithread].emplace_back(doubleHypH4XYZ, pdoubleHypH4, fitterdoubleHypH4.calcPCACovMatrixFlat(nCanddoubleH4), trHypHe4, trBach, body3Id,bach.gid);
    // apply mass selections for 3-body decay
    //  bool good3bodyV0Hyp = false;
    //  for (int ipid = 0; ipid < 1; ipid++) {
    //    float massForDoubleHypH4 = mDoubleHyps[ipid].calcMass(p2HyperHe4, p2Bach, p2doubleHypH4);
    ////    if (massForLambdaHyp - mV0Hyps[ipid].getMassV0Hyp() < mV0Hyps[ipid].getMargin(ptV0)) {
    ////      good3bodyV0Hyp = true;
    ////      break;
    ////    }
    //   LOG(info)<<"The mass of double HyperHydrogen4 is "<< massForDoubleHypH4;
    //   log_file<< massForDoubleHypH4 <<std::endl;
    //
    //   log_pT4Hlambdalambda<< std::sqrt(pt2doubleHypH4) <<std::endl;
    //
    //   log_pT4He3Lambda<< pTHyperHe4 <<std::endl;
    //
    //   log_pTpion<< pTBach <<std::endl;
    //  }
    //
    
    //Other cuts or checks also needed similar as for the cascades like
    // a) cascade not compatible with any vertex
    // b) cascade not casually compatible
    // c) check for good hypothisis
    
  }
  
  
  //    const auto& body3 = m3bodyTmp[ithread][body3Id];
  //    auto doubleHypH4list = decay3bodylist.getOverlap(bach.vBracket); // indices of vertices shared by V0 and bachelor
  //    if (mSVParams->selectBestV0) {
  //	    //Now here is the catch how will you select the best candidate compatible with 3body parmeter ->Lets do something 
  //      // select only the best V0 candidate among the compatible ones
  //      if (body3.getVertexID() < decay3bodylist.getMin() || body3.getVertexID() > decay3bodylist.getMax()) {
  //        continue;
  //      }
  //      doubleHypH4list.setMin(body3.getVertexID());
  //      doubleHypH4list.setMax(body3.getVertexID());
  //    }
  //
  //    int nCanddoubleH4 = fitterdoubleHypH4.process(body3, bach);
  //    if (nCanddoubleH4 == 0) { // discard this pair
  //      continue;
  //    }
  //    int canddoubleH4 = 0;
  //    const auto& doubleHypH4XYZ = fitterdoubleHypH4.getPCACandidatePos(canddoubleH4);
  //
  //    // make sure the cascade radius is smaller than that of the mean vertex
  //    float dxc = doubleHypH4XYZ[0] - mMeanVertex.getX(), dyc = doubleHypH4XYZ[1] - mMeanVertex.getY(), r2doubleHypH4 = dxc * dxc + dyc * dyc;
  //    //Now the main prblem is that from where we can get the rdoubleHypH4 since in v0 we simply get it but lets try to hack something.
  //    if (rdoubleHypH4 * rdoubleHypH4 - r2doubleHypH4 < mMinR2DiffV0Casc || r2doubleHypH4 < mMinR2ToMeanVertex) {
  //      continue;
  //    }
  //    // do we want to apply mass cut ?
  //    //
  //    if (!fitterdoubleHypH4.isPropagateTracksToVertexDone() && !fitterdoubleHypH4.propagateTracksToVertex()) {
  //      continue;
  //    }
  //
  //    auto& trHypHe4 = fitterdoubleHypH4.getTrack(0, canddoubleH4);
  //    auto& trBach = fitterdoubleHypH4.getTrack(1, canddoubleH4);
  //    trHypHe4.setPID(o2::track::PID::Hyperhelium4);
  //    trBach.setPID(o2::track::PID::Pion);
  //    std::array<float, 3> pHypHe4, pBach;
  //    trHypHe4.getPxPyPzGlo(pHypHe4);
  //    trBach.getPxPyPzGlo(pBach);
  //    std::array<float, 3> pdoubleHypH4 = {pHypHe4[0] + pBach[0], pHypHe4[1] + pBach[1], pHypHe4[2] + pBach[2]};
  //
  //    float pt2doubleHypH4 = pdoubleHypH4[0] * pdoubleHypH4[0] + pdoubleHypH4[1] * pdoubleHypH4[1], p2doubleHypH4 = pt2doubleHypH4 + pdoubleHypH4[2] * pdoubleHypH4[2];
  //    if (pt2doubleHypH4 < mMinPt2Casc) { // pt cut  //for the time being use the same cut as for the cascades
  //      LOG(debug) << "Casc pt too low";
  //      continue;
  //    }
  //    if (pdoubleHypH4[2] * pdoubleHypH4[2] / pt2doubleHypH4 > mMaxTgl2Casc) { // tgLambda cut
  //      LOG(debug) << "Casc tgLambda too high";
  //      continue;
  //    }
  //
  //    // compute primary vertex and cosPA of the cascade
  //    auto bestCosPA = mSVParams->minCosPACasc;
  //    auto doubleHypH4VtxID = -1;
  //
  //    for (int iv = doubleHypH4list.getMin(); iv <= doubleHypH4list.getMax(); iv++) {
  //      const auto& pv = mPVertices[iv];
  //      // check cos of pointing angle
  //      float dx = doubleHypH4XYZ[0] - pv.getX(), dy = doubleHypH4XYZ[1] - pv.getY(), dz = doubleHypH4XYZ[2] - pv.getZ(), prodXYZdoubleHypH4 = dx * pdoubleHypH4[0] + dy * pdoubleHypH4[1] + dz * pdoubleHypH4[2];
  //      float cosPA = prodXYZdoubleHypH4 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2doubleHypH4);
  //      if (cosPA < bestCosPA) {
  //        LOG(debug) << "Rej. cosPA: " << cosPA;
  //        continue;
  //      }
  //      doubleHypH4VtxID = iv;
  //      bestCosPA = cosPA;
  //    }
  //    if (doubleHypH4VtxID == -1) {
  //      LOG(debug) << "Casc not compatible with any vertex";
  //      continue;
  //    }
  //
  //    const auto& doubleHypH4Pv = mPVertices[doubleHypH4VtxID];
  //    float dxdoubleHypH4 = doubleHypH4XYZ[0] - doubleHypH4Pv.getX(), dydoubleHypH4 = doubleHypH4XYZ[1] - doubleHypH4Pv.getY(), dzdoubleHypH4 = doubleHypH4XYZ[2] - doubleHypH4Pv.getZ();
  //    auto prodPPos = p3body[0] * dxdoubleHypH4 + p3body[1] * dydoubleHypH4 + p3body[2] * dzdoubleHypH4;
  //    if (prodPPos < 0.) { // causality cut
  //      LOG(debug) << "Casc not causally compatible";
  //      continue;
  //    }
  //
  //    float p2Bach = pBach[0] * pBach[0] + pBach[1] * pBach[1] + pBach[2] * pBach[2];
  //    float ptdoubleHypH4 = std::sqrt(pt2doubleHypH4);
  //    bool goodHyp = false;
  //    for (int ipid = 0; ipid < NHypCascade; ipid++) {
  //      if (mCascHyps[ipid].check(p23body, p2Bach, p2doubleHypH4, pt2doubleHypH4)) {
  //        goodHyp = true;
  //        break;
  //      }
  //    }
  //    if (!goodHyp) {
  //      LOG(debug) << "doubleHypH4 not compatible with any hypothesis";
  //      continue;
  //    }
  //    auto& doubleHypH4 = mDoubleHypH4Tmp[ithread].emplace_back(doubleHypH4XYZ, pdoubleHypH4, fitterdoubleHypH4.calcPCACovMatrixFlat(nCanddoubleH4), trHypHe4, trBach, body3Id, bach.gid);
  //    o2::track::TrackParCov trc = doubleHypH4;
  //    o2::dataformats::DCA dca;
  //    if (!trc.propagateToDCA(doubleHypH4Pv, fitterdoubleHypH4.getBz(), &dca, 5.) ||
  //        std::abs(dca.getY()) > mSVParams->maxDCAXYCasc || std::abs(dca.getZ()) > mSVParams->maxDCAZCasc) {
  //      LOG(debug) << "doubleHypH4 not compatible with PV";
  //      LOG(debug) << "DCA: " << dca.getY() << " " << dca.getZ();
  //      mDoubleHypHe4Tmp[ithread].pop_back();
  //      continue;
  //    }
  //
  //    LOG(debug) << "doubleHypH4 successfully added";
  //    doubleHypH4.setCosPA(bestCosPA);
  //    doubleHypH4.setVertexID(doubleHypH4VtxID);
  //    doubleHypH4.setDCA(fitterdoubleHypH4.getChi2AtPCACandidate());
  //
  //    // clone the V0, set new cosPA and VerteXID, add it to the list of V0s
  //    if (doubleHypH4VtxID != body3.getVertexID()) {
  //      auto body3clone = body3;
  //      const auto& pv = mPVertices[doubleHypH4VtxID];
  //
  //      float dx = body3.getX() - pv.getX(), dy = body3.getY() - pv.getY(), dz = body3.getZ() - pv.getZ(), prodXYZ = dx * pdoubleHypH4[0] + dy * pdoubleHypH4[1] + dz * pdoubleHypH4[2];
  //      float cosPA = prodXYZ / std::sqrt((dx * dx + dy * dy + dz * dz) * p2doubleHypH4);
  //      body3clone.setCosPA(cosPA);
  //      body3clone.setVertexID(doubleHypH4VtxID);
  //
  //      auto pvIdx = pvMap.find(doubleHypH4VtxID);
  //      if (pvIdx != pvMap.end()) {
  //        doubleHypH4.setV0ID(pvIdx->second); // V0 already exists, add reference to the cascade
  //      } else {
  //        mHypHe4Tmp[ithread].push_back(body3clone);
  //        doubleHypH4.setV0ID(m3bodyTmp[ithread].size() - 1);      // set the new V0 index in the cascade
  //        pvMap[doubleHypH4VtxID] = m3bodyTmp[ithread].size() - 1; // add the new V0 index to the map
  //      }
  //    }
  //  }
  //
  return mDoubleHypH4Tmp[ithread].size() - ndoubleHypH4Ini;
  //return 1;
}



//__________________________________________________________________
int SVertexer::check3bodyDecays(float rv0, std::array<float, 3> pV0, float p2V0, int avoidTrackID, int posneg, VBracket v0vlist, int ithread)
{
  
  // check last added V0 for belonging to cascade
  auto& fitter3body = mFitter3body[ithread];
  const auto& v0 = mV0sTmp[ithread].back();	//to get the last added v0 candidate
  auto& tracks = mTracksPool[posneg];
  int n3BodyIni = m3bodyTmp[ithread].size();
  
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
      if (v0.getVertexID() < decay3bodyVlist.getMin() || v0.getVertexID() > decay3bodyVlist.getMax()) {
        continue;
      }
      decay3bodyVlist.setMin(v0.getVertexID());
      decay3bodyVlist.setMax(v0.getVertexID());
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
    
    //        LOG(info)<< "The momentum of particle 1 again is " << std::sqrt(p0[0]*p0[0] + p0[1]*p0[1] + p0[2]*p0[2]) << " and the 2nd particle is "<<std::sqrt( p1[0]*p1[0]+p1[1]*p1[1]+p1[2]*p1[2]) << " and for the 3rd particle is " <<std::sqrt( p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2])  << " and the total momentum of the 3 particle is  "<<std::sqrt( p2candidate);
    
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
    for (int ipid = 2; ipid < 4; ipid++) { // TODO: expand this loop to cover all the 3body cases if (m3bodyHyps[ipid].check(sqP0, sqP1, sqP2, sqPtot, pt3B))
      if (m3bodyHyps[ipid].check(sqP0, sqP1, sqP2, p2candidate, pt3B)) {
        goodHyp = true;
        break;
      }
    }
    if (!goodHyp) {
      continue;
    }

    
    //I just want to look at the ptdistribution of 3body befor doing anything
	log_pT3body << pt3B <<std::endl;
	
	//consider Hypetrition 
	//    auto& candidate3B = m3bodyTmp[ithread].emplace_back(PID::HyperTriton, vertexXYZ, p3B, fitter3body.calcPCACovMatrixFlat(cand3B), tr0, tr1, tr2, v0.getProngID(0), v0.getProngID(1), bach.gid);
	auto& candidate3B = m3bodyTmp[ithread].emplace_back(PID::Hyperhelium4, vertexXYZ, p3B, fitter3body.calcPCACovMatrixFlat(cand3B), tr0, tr1, tr2, v0.getProngID(0), v0.getProngID(1), bach.gid);
	
	o2::track::TrackParCov trc = candidate3B;
	o2::dataformats::DCA dca;
	if (!trc.propagateToDCA(decay3bodyPv, fitter3body.getBz(), &dca, 5.) ||
	    std::abs(dca.getY()) > mSVParams->maxDCAXY3Body || std::abs(dca.getZ()) > mSVParams->maxDCAZ3Body) {
	  m3bodyTmp[ithread].pop_back();
	  continue;
	}
	
	candidate3B.setCosPA(bestCosPA);
	candidate3B.setVertexID(decay3bodyVtxID);
	candidate3B.setDCA(fitter3body.getChi2AtPCACandidate());

	// check doubleHypH4
	//  bool goodHyp = false;
	//  std::array<bool, NDoubleHyp3B> hypCheckStatus{};
	//  for (int ipid = 0; ipid < NHypV0; ipid++) {
	//    if (m3bodyHyps[ipid].check(p2Pos, p2Neg, p2V0, ptV0)) {
	//      goodHyp = hypCheckStatus[ipid] = true;
	//    }
	//  }
	//  if (!goodHyp && mSVParams->checkV0Hypothesis) {
	//    LOG(debug) << "RejHypo";
	//    return false;
	//  }
  
	bool  checkFordoubleHypH4 = true;
	bool rejectIfNotDoubleHypeH4 = false;


	if (checkFordoubleHypH4) {
	  int ndoubleHypH4Added = 0;
	  //    if (hypCheckStatus[DoubleHyp3B::DoubleHyperhydrogen4] || !mSVParams->checkCascadeHypothesis) {
	  ndoubleHypH4Added += checkdoubleHypH4(r2vertex, p3B, p2candidate, it, NEG, decay3bodyVlist, ithread);
	  //    }
	  //    if (hypCheckStatus[DoubleHyp3B::DoubleAntiHyperhydrogen4] || !mSVParams->checkCascadeHypothesis) {
	  //      ndoubleHypH4Added += checkdoubleHypH4(r2vertex, p3B, p2candidate, it, POS, decay3bodyVlist, ithread);
	  //    }
	  //    if (!ndoubleHypH4Added && rejectIfNotDoubleHypeH4) { // v0 would be accepted only if it creates a cascade
	  //      m3bodyTmp[ithread].pop_back();
	  //      return false;
	  //    }
	}
	
	

	//        // Consider Hyperhelium4
	//    auto& candidateHyHe4 = m3bodyTmp[ithread].emplace_back(PID::Hyperhelium4, vertexXYZ, p3B, fitter3body.calcPCACovMatrixFlat(cand3B), tr0, tr1, tr2, v0.getProngID(0), v0.getProngID(1), bach.gid);
	//    o2::track::TrackParCov trcHyHe4 = candidateHyHe4;
	//    if (!trcHyHe4.propagateToDCA(decay3bodyPv, fitter3body.getBz(), &dca, 5.) ||
	//        std::abs(dca.getY()) > mSVParams->maxDCAXY3Body || std::abs(dca.getZ()) > mSVParams->maxDCAZ3Body) {
	//      m3bodyTmp[ithread].pop_back();
	//      continue;
	//    }
	//    candidateHyHe4.setCosPA(bestCosPA);
	//    candidateHyHe4.setVertexID(decay3bodyVtxID);
	//    candidateHyHe4.setDCA(fitter3body.getChi2AtPCACandidate());
  }
  return m3bodyTmp[ithread].size() - n3BodyIni;
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
  // if TPC trackis unconstrained, try to create in the tracks pool a clone constrained to vtxid vertex time.
  if (trTPC.hasBothSidesClusters()) { // this is effectively constrained track
    return false;                     // let it be processed as such
  }
  const auto& vtx = mPVertices[vtxid];
  auto twe = vtx.getTimeStamp();
  int posneg = trTPC.getSign() < 0 ? 1 : 0;
  auto trLoc = mTracksPool[posneg].emplace_back(TrackCand{trTPC, gid, {vtxid, vtxid}, 0.});
  auto err = correctTPCTrack(trLoc, trTPC, twe.getTimeStamp(), twe.getTimeStampError());
  if (err < 0) {
    mTracksPool[posneg].pop_back(); // discard
  }
  trLoc.minR = std::sqrt(trLoc.getX() * trLoc.getX() + trLoc.getY() * trLoc.getY());
  return true;
}

//______________________________________________
float SVertexer::correctTPCTrack(o2::track::TrackParCov& trc, const o2::tpc::TrackTPC tTPC, float tmus, float tmusErr) const
{
  // Correct the track copy trc of the TPC track for the assumed interaction time
  // return extra uncertainty in Z due to the interaction time uncertainty
  // TODO: at the moment, apply simple shift, but with Z-dependent calibration we may
  // need to do corrections on TPC cluster level and refit
  // This is a clone of MatchTPCITS::correctTPCTrack
  float dDrift = ((tmus + mTPCDriftTimeOffset) * mMUS2TPCBin - tTPC.getTime0()) * mTPCBin2Z;
  float driftErr = tmusErr * mMUS2TPCBin * mTPCBin2Z;
  // eventually should be refitted, at the moment we simply shift...
  trc.setZ(tTPC.getZ() + (tTPC.hasASideClustersOnly() ? dDrift : -dDrift));
  trc.setCov(trc.getSigmaZ2() + driftErr * driftErr, o2::track::kSigZ2);
  
  return driftErr;
}
