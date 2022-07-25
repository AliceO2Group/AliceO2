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

/// \file PVertexer.cxx
/// \brief Primary vertex finder
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/PVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <unordered_map>
#include <TStopwatch.h>
#include "CommonUtils/StringUtils.h" // RS REM
#include <TH2F.h>

using namespace o2::vertexing;

constexpr float PVertexer::kAlmost0F;
constexpr double PVertexer::kAlmost0D;
constexpr float PVertexer::kHugeF;

//___________________________________________________________________
int PVertexer::runVertexing(const gsl::span<o2d::GlobalTrackID> gids, const gsl::span<o2::InteractionRecord> bcData,
                            std::vector<PVertex>& vertices, std::vector<o2d::VtxTrackIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                            gsl::span<const o2::MCCompLabel> lblTracks, std::vector<o2::MCEventLabel>& lblVtx)
{
#ifdef _PV_DEBUG_TREE_
  doDBGPoolDump(lblTracks);
#endif
  dbscan_clusterize();
  std::vector<PVertex> verticesLoc;
  std::vector<uint32_t> trackIDs;
  std::vector<V2TRef> v2tRefsLoc;
  std::vector<float> validationTimes;
  std::vector<o2::MCEventLabel> lblVtxLoc;
  for (auto tc : mTimeZClusters) {
    VertexingInput inp;
    inp.idRange = gsl::span<int>(tc.trackIDs);
    inp.scaleSigma2 = mPVParams->iniScale2;
    inp.timeEst = tc.timeEst;
#ifdef _PV_DEBUG_TREE_
    doDBScanDump(inp, lblTracks);
#endif
    findVertices(inp, verticesLoc, trackIDs, v2tRefsLoc);
  }
  // sort in time
  std::vector<int> vtTimeSortID(verticesLoc.size());
  std::iota(vtTimeSortID.begin(), vtTimeSortID.end(), 0);
  std::sort(vtTimeSortID.begin(), vtTimeSortID.end(), [&verticesLoc](int i, int j) {
    return verticesLoc[i].getTimeStamp().getTimeStamp() < verticesLoc[j].getTimeStamp().getTimeStamp();
  });

#ifdef _PV_DEBUG_TREE_
  if (lblTracks.size()) { // at this stage labels are needed just for the debug output
    createMCLabels(lblTracks, trackIDs, v2tRefsLoc, lblVtxLoc);
  }
#endif

  if (mPVParams->applyDebrisReduction) {
    reduceDebris(verticesLoc, vtTimeSortID, lblVtxLoc);
  }
  if (mPVParams->applyReattachment) {
    reAttach(verticesLoc, vtTimeSortID, trackIDs, v2tRefsLoc);
  }

  if (lblTracks.size()) { // at this stage labels are needed just for the debug output
    createMCLabels(lblTracks, trackIDs, v2tRefsLoc, lblVtxLoc);
  }

#ifdef _PV_DEBUG_TREE_
  doVtxDump(verticesLoc, trackIDs, v2tRefsLoc, lblTracks);
#endif

  vertices.clear();
  v2tRefs.clear();
  vertexTrackIDs.clear();
  vertices.reserve(verticesLoc.size());
  v2tRefs.reserve(v2tRefsLoc.size());
  vertexTrackIDs.reserve(trackIDs.size());

  int trCopied = 0, count = 0, vtimeID = 0;
  for (auto i : vtTimeSortID) {
    if (i < 0) {
      continue; // vertex was suppressed
    }
    auto& vtx = verticesLoc[i];

    bool irSet = setCompatibleIR(vtx);
    if (!irSet) {
      continue;
    }
    // do we need to validate by Int. records ?
    if (mValidateWithIR) {
      auto bestMatch = getBestIR(vtx, bcData, vtimeID);
      if (bestMatch.first >= 0) {
        vtx.setFlags(PVertex::TimeValidated);
        if (bestMatch.second == 1) {
          vtx.setIR(bcData[bestMatch.first]);
        }
        LOG(debug) << "Validated with t0 " << bestMatch.first << " with " << bestMatch.second << " candidates";
      } else if (vtx.getNContributors() >= mPVParams->minNContributorsForIRcut) {
        LOG(debug) << "Discarding " << vtx;
        continue; // reject
      }
    }
    vertices.push_back(vtx);
    if (lblVtxLoc.size()) {
      lblVtx.push_back(lblVtxLoc[i]);
    }
    int it = v2tRefsLoc[i].getFirstEntry(), itEnd = it + v2tRefsLoc[i].getEntries(), dest0 = vertexTrackIDs.size();
    for (; it < itEnd; it++) {
      const auto& trc = mTracksPool[trackIDs[it]];
      auto& gid = vertexTrackIDs.emplace_back(trc.gid); // assign global track ID
      gid.setPVContributor();
    }
    v2tRefs.emplace_back(dest0, v2tRefsLoc[i].getEntries());
    LOG(debug) << "#" << count++ << " " << vertices.back() << " | " << v2tRefs.back().getEntries() << " indices from " << v2tRefs.back().getFirstEntry(); // RS REM
  }

  return vertices.size();
}

//______________________________________________
int PVertexer::findVertices(const VertexingInput& input, std::vector<PVertex>& vertices, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs)
{
  // find vertices using tracks with indices (sorted in time) from idRange from "tracks" pool. The pool may containt arbitrary number of tracks,
  // only those which are in the idRange and have canUse()==true, will be used.
  // Results are placed in vertices and v2tRefs vectors

  int nfound = 0, ntr = 0;
  auto seedHistoTZ = buildHistoTZ(input); // histo for seeding peak finding

#ifdef _PV_DEBUG_TREE_
  static int dbsCount = -1;
  dbsCount++;
  int trialCount = 0; // TODO REM
  auto hh = seedHistoTZ.createTH2F(o2::utils::Str::concat_string("htz", std::to_string(dbsCount)));
  hh->SetDirectory(nullptr);
  mDebugDumpFile->cd();
  hh->Write();
#endif

  int nTrials = 0;
  while (nfound < mPVParams->maxVerticesPerCluster && nTrials < mPVParams->maxTrialsPerCluster) {
    int peakBin = seedHistoTZ.findPeakBin();
    if (!seedHistoTZ.isValidBin(peakBin)) {
      break;
    }
    int peakBinT = seedHistoTZ.getXBin(peakBin), peakBinZ = seedHistoTZ.getYBin(peakBin);
    float tv = seedHistoTZ.getBinXCenter(peakBinT);
    float zv = seedHistoTZ.getBinYCenter(peakBinZ);
    LOG(debug) << "Seeding with T=" << tv << " Z=" << zv << " bin " << peakBin << " on trial " << nTrials << " for vertex " << nfound;

    PVertex vtx;
    vtx.setXYZ(mMeanVertex.getX(), mMeanVertex.getY(), zv);
    vtx.setTimeStamp({tv, 0.f});
    if (findVertex(input, vtx)) {
      finalizeVertex(input, vtx, vertices, v2tRefs, trackIDs, &seedHistoTZ);
      nfound++;
      nTrials = 0;
    } else {                                                                    // suppress failed seeding bin and its proximities
      seedHistoTZ.setBinContent(peakBin, -1);
    }
    nTrials++;

#ifdef _PV_DEBUG_TREE_
    auto hh1 = seedHistoTZ.createTH2F(o2::utils::Str::concat_string("htz", std::to_string(dbsCount), "_", std::to_string(trialCount++)));
    hh1->SetDirectory(nullptr);
    mDebugDumpFile->cd();
    hh1->Write();
#endif
  }
  return nfound;
}

//______________________________________________
bool PVertexer::findVertex(const VertexingInput& input, PVertex& vtx)
{
  // fit vertex taking provided vertex as a seed
  // tracks pool may contain arbitrary number of tracks, only those which are in
  // the idRange (indices of tracks sorted in time) will be used.

  int ntr = input.idRange.size(); // RSREM

  VertexSeed vtxSeed(vtx);
  vtxSeed.setScale(input.scaleSigma2, mTukey2I);
  vtxSeed.scaleSigma2Prev = input.scaleSigma2;
  //  vtxSeed.setTimeStamp( timeEstimate(input) );
  LOG(debug) << "Start time guess: " << vtxSeed.getTimeStamp();
  vtx.setChi2(1.e30);
  //
  FitStatus result = FitStatus::IterateFurther;
  bool found = false;
  while (result == FitStatus::IterateFurther) {
    vtxSeed.resetForNewIteration();
    vtxSeed.nIterations++;
    LOG(debug) << "iter " << vtxSeed.nIterations << " with scale=" << vtxSeed.scaleSigma2 << " prevScale=" << vtxSeed.scaleSigma2Prev
               << " ntr=" << ntr << " Zv=" << vtxSeed.getZ() << " Tv=" << vtxSeed.getTimeStamp().getTimeStamp();
    result = fitIteration(input, vtxSeed);

    if (result == FitStatus::OK) {
      result = evalIterations(vtxSeed, vtx);
    } else if (result == FitStatus::NotEnoughTracks) {
      if (vtxSeed.nIterations <= mPVParams->maxIterations && upscaleSigma(vtxSeed)) {
        LOG(debug) << "Upscaling scale to " << vtxSeed.scaleSigma2;
        result = FitStatus::IterateFurther;
        continue; // redo with stronger rescaling
      } else {
        break;
      }
    } else if (result == FitStatus::PoolEmpty || result == FitStatus::Failure) {
      break;
    } else {
      LOG(fatal) << "Unknown fit status " << int(result);
    }
  }
  LOG(debug) << "Stopped with scale=" << vtxSeed.scaleSigma2 << " prevScale=" << vtxSeed.scaleSigma2Prev << " result = " << int(result) << " ntr=" << ntr;

  if (result != FitStatus::OK) {
    vtx.setChi2(vtxSeed.maxScaleSigma2Tested);
    return false;
  } else {
    return true;
  }
}

//___________________________________________________________________
PVertexer::FitStatus PVertexer::fitIteration(const VertexingInput& input, VertexSeed& vtxSeed)
{
  int nTested = 0;
  for (int i : input.idRange) {
    if (mTracksPool[i].canUse()) {
      accountTrack(mTracksPool[i], vtxSeed);
      //      printf("#%d z:%f t:%f te:%f w:%f wh:%f\n", nTested, mTracksPool[i].z, mTracksPool[i].timeEst.getTimeStamp(),
      //             mTracksPool[i].timeEst.getTimeStampError(), mTracksPool[i].wgh, mTracksPool[i].wghHisto);
      nTested++;
    }
  }

  vtxSeed.maxScaleSigma2Tested = vtxSeed.scaleSigma2;
  if (vtxSeed.getNContributors() < mPVParams->minTracksPerVtx) {
    return nTested < mPVParams->minTracksPerVtx ? FitStatus::PoolEmpty : FitStatus::NotEnoughTracks;
  }
  if (mPVParams->useMeanVertexConstraint) {
    applyConstraint(vtxSeed);
  }
  if (!solveVertex(vtxSeed)) {
    return FitStatus::Failure;
  }
  return FitStatus::OK;
}

//___________________________________________________________________
void PVertexer::accountTrack(TrackVF& trc, VertexSeed& vtxSeed) const
{
  // deltas defined as track - vertex
  bool useTime = vtxSeed.getTimeStamp().getTimeStampError() >= 0.f;
  auto chi2T = trc.evalChi2ToVertex(vtxSeed, useTime && mPVParams->useTimeInChi2);
  float wghT = (1.f - chi2T * vtxSeed.scaleSig2ITuk2I); // weighted distance to vertex
  if (wghT < kAlmost0F) {
    trc.wgh = 0.f;
    return;
  }
  wghT *= wghT;
  float syyI(trc.sig2YI), szzI(trc.sig2ZI), syzI(trc.sigYZI);

  auto timeErrorFromTB = [&trc]() {
    // decide if the time error is from the time bracket rather than gaussian error
    return trc.gid.getSource() == GTrackID::ITS;
  };

  //
  vtxSeed.wghSum += wghT;
  vtxSeed.wghChi2 += wghT * chi2T;
  //
  syyI *= wghT;
  syzI *= wghT;
  szzI *= wghT;
  trc.wgh = wghT;
  //
  // aux variables
  double tmpSP = trc.sinAlp * trc.tgP, tmpCP = trc.cosAlp * trc.tgP,
         tmpSC = trc.sinAlp + tmpCP, tmpCS = -trc.cosAlp + tmpSP,
         tmpCL = trc.cosAlp * trc.tgL, tmpSL = trc.sinAlp * trc.tgL,
         tmpYXP = trc.y - trc.tgP * trc.x, tmpZXL = trc.z - trc.tgL * trc.x,
         tmpCLzz = tmpCL * szzI, tmpSLzz = tmpSL * szzI, tmpSCyz = tmpSC * syzI,
         tmpCSyz = tmpCS * syzI, tmpCSyy = tmpCS * syyI, tmpSCyy = tmpSC * syyI,
         tmpSLyz = tmpSL * syzI, tmpCLyz = tmpCL * syzI;
  //
  // symmetric matrix equation
  vtxSeed.cxx += tmpCL * (tmpCLzz + tmpSCyz + tmpSCyz) + tmpSC * tmpSCyy;         // dchi^2/dx/dx
  vtxSeed.cxy += tmpCL * (tmpSLzz + tmpCSyz) + tmpSL * tmpSCyz + tmpSC * tmpCSyy; // dchi^2/dx/dy
  vtxSeed.cxz += -trc.sinAlp * syzI - tmpCLzz - tmpCP * syzI;                     // dchi^2/dx/dz
  vtxSeed.cx0 += -(tmpCLyz + tmpSCyy) * tmpYXP - (tmpCLzz + tmpSCyz) * tmpZXL;    // RHS
  //
  vtxSeed.cyy += tmpSL * (tmpSLzz + tmpCSyz + tmpCSyz) + tmpCS * tmpCSyy;      // dchi^2/dy/dy
  vtxSeed.cyz += -(tmpCSyz + tmpSLzz);                                         // dchi^2/dy/dz
  vtxSeed.cy0 += -tmpYXP * (tmpCSyy + tmpSLyz) - tmpZXL * (tmpCSyz + tmpSLzz); // RHS
  //
  vtxSeed.czz += szzI;                          // dchi^2/dz/dz
  vtxSeed.cz0 += tmpZXL * szzI + tmpYXP * syzI; // RHS
  //
  if (useTime) {
    float trErr2I = wghT / (trc.timeEst.getTimeStampError() * trc.timeEst.getTimeStampError());
    if (timeErrorFromTB()) {
      vtxSeed.tMeanAccTB += trc.timeEst.getTimeStamp() * trErr2I;
      vtxSeed.tMeanAccErrTB += trErr2I;
      vtxSeed.nContributorsTB++;
      vtxSeed.wghSumTB += wghT;
    } else {
      vtxSeed.tMeanAcc += trc.timeEst.getTimeStamp() * trErr2I;
      vtxSeed.tMeanAccErr += trErr2I;
    }
  }
  vtxSeed.addContributor();
}

//___________________________________________________________________
bool PVertexer::solveVertex(VertexSeed& vtxSeed) const
{
  ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> mat;
  mat(0, 0) = vtxSeed.cxx;
  mat(0, 1) = vtxSeed.cxy;
  mat(0, 2) = vtxSeed.cxz;
  mat(1, 1) = vtxSeed.cyy;
  mat(1, 2) = vtxSeed.cyz;
  mat(2, 2) = vtxSeed.czz;
  if (!mat.InvertFast()) {
    LOG(error) << "Failed to invert matrix" << mat;
    return false;
  }
  ROOT::Math::SVector<double, 3> rhs(vtxSeed.cx0, vtxSeed.cy0, vtxSeed.cz0);
  auto sol = mat * rhs;
  vtxSeed.setXYZ(sol(0), sol(1), sol(2));
  vtxSeed.setCov(mat(0, 0), mat(1, 0), mat(1, 1), mat(2, 0), mat(2, 1), mat(2, 2));
  if (vtxSeed.tMeanAccErr + vtxSeed.tMeanAccErrTB > 0.) {
    // since the time error from the ITS measurements does not improve with statistics, we downscale it with number of such tracks
    auto t = vtxSeed.tMeanAcc;
    auto e2i = vtxSeed.tMeanAccErr;
    if (vtxSeed.wghSumTB > 0.) {
      t += vtxSeed.tMeanAccTB / vtxSeed.wghSumTB;
      e2i += vtxSeed.tMeanAccErrTB / vtxSeed.wghSumTB;
    }
    auto err2 = 1. / e2i;
    vtxSeed.setTimeStamp({float(t * err2), float(std::sqrt(err2))});
  }

  vtxSeed.setChi2((vtxSeed.getNContributors() - vtxSeed.wghSum) / vtxSeed.scaleSig2ITuk2I); // calculate chi^2
  auto newScale = vtxSeed.wghChi2 / vtxSeed.wghSum;
  LOG(debug) << "Solve: wghChi2=" << vtxSeed.wghChi2 << " wghSum=" << vtxSeed.wghSum << " -> scale= " << newScale << " old scale " << vtxSeed.scaleSigma2 << " prevScale: " << vtxSeed.scaleSigma2Prev;
  vtxSeed.setScale(newScale < mPVParams->minScale2 ? mPVParams->minScale2 : newScale, mTukey2I);
  return true;
}

//___________________________________________________________________
PVertexer::FitStatus PVertexer::evalIterations(VertexSeed& vtxSeed, PVertex& vtx) const
{
  // decide if new iteration should be done, prepare next one if needed
  // if scaleSigma2 reached its lower limit stop
  PVertexer::FitStatus result = PVertexer::FitStatus::IterateFurther;

  if (vtxSeed.nIterations > mPVParams->maxIterations) {
    result = PVertexer::FitStatus::Failure;
  } else if (vtxSeed.scaleSigma2Prev <= mPVParams->minScale2 + kAlmost0F) {
    result = PVertexer::FitStatus::OK;
    LOG(debug) << "stop on simga :" << vtxSeed.scaleSigma2 << " prev: " << vtxSeed.scaleSigma2Prev;
  }
  if (fair::Logger::Logging(fair::Severity::debug)) {
    auto dchi = (vtx.getChi2() - vtxSeed.getChi2()) / vtxSeed.getChi2();
    auto dx = vtxSeed.getX() - vtx.getX(), dy = vtxSeed.getY() - vtx.getY(), dz = vtxSeed.getZ() - vtx.getZ();
    auto dst = std::sqrt(dx * dx + dy * dy + dz * dz);

    LOG(debug) << "dChi:" << vtx.getChi2() << "->" << vtxSeed.getChi2() << " :-> " << dchi;
    LOG(debug) << "dx: " << dx << " dy: " << dy << " dz: " << dz << " -> " << dst;
  }

  vtx = reinterpret_cast<const PVertex&>(vtxSeed);

  if (result == PVertexer::FitStatus::OK) {
    auto chi2Mean = vtxSeed.getChi2() / vtxSeed.getNContributors();
    if (chi2Mean > mPVParams->maxChi2Mean) {
      result = PVertexer::FitStatus::Failure;
      LOG(debug) << "Rejecting at iteration " << vtxSeed.nIterations << " and ScalePrev " << vtxSeed.scaleSigma2Prev << " with meanChi2 = " << chi2Mean;
    } else {
      return result;
    }
  }

  if (vtxSeed.scaleSigma2 > vtxSeed.scaleSigma2Prev) {
    if (++vtxSeed.nScaleIncrease > mPVParams->maxNScaleIncreased) {
      result = PVertexer::FitStatus::Failure;
      LOG(debug) << "Rejecting at iteration " << vtxSeed.nIterations << " with NScaleIncreased " << vtxSeed.nScaleIncrease;
    }
  } else if (vtxSeed.scaleSigma2 > mPVParams->slowConvergenceFactor * vtxSeed.scaleSigma2Prev) {
    if (++vtxSeed.nScaleSlowConvergence > mPVParams->maxNScaleSlowConvergence) {
      if (vtxSeed.scaleSigma2 < mPVParams->acceptableScale2) {
        vtxSeed.setScale(mPVParams->minScale2, mTukey2I);
        LOG(debug) << "Forcing scale2 to " << mPVParams->minScale2;
        result = PVertexer::FitStatus::IterateFurther;
      } else {
        result = PVertexer::FitStatus::Failure;
        LOG(debug) << "Rejecting at iteration " << vtxSeed.nIterations << " with NScaleSlowConvergence " << vtxSeed.nScaleSlowConvergence;
      }
    }
  } else {
    vtxSeed.nScaleSlowConvergence = 0;
  }

  return result;
}

//___________________________________________________________________
void PVertexer::reAttach(std::vector<PVertex>& vertices, std::vector<int>& timeSort, std::vector<uint32_t>& trackIDs, std::vector<V2TRef>& v2tRefs)
{
  float tRange = 0.5 * std::max(mITSROFrameLengthMUS, mPVParams->dbscanDeltaT) + mPVParams->timeMarginReattach; // consider only vertices in this proximity to tracks
  std::vector<std::pair<int, TimeEst>> vtvec;                                                                   // valid vertex times and indices
  int nvtOrig = vertices.size();
  vtvec.reserve(nvtOrig);
  mTimeZClusters.resize(nvtOrig);
  for (int ivt = 0; ivt < nvtOrig; ivt++) {
    mTimeZClusters[ivt].trackIDs.clear();
    mTimeZClusters[ivt].trackIDs.reserve(int(vertices[ivt].getNContributors() * 1.2));
  }
  for (auto ivt : timeSort) {
    if (ivt >= 0) {
      vtvec.emplace_back(ivt, vertices[ivt].getTimeStamp());
    }
  }
  int ntr = mTracksPool.size(), nvt = vtvec.size();
  int vtStart = 0;
  for (int itr = 0; itr < ntr; itr++) {
    auto& trc = mTracksPool[itr];
    trc.vtxID = TrackVF::kNoVtx;
    trc.wgh = kAlmost0F;
    for (int ivt = vtStart; ivt < nvt; ivt++) {
      auto dt = vtvec[ivt].second.getTimeStamp() - trc.timeEst.getTimeStamp();
      if (dt < -tRange) { // all following tracks will have higher time than the vertex vtStart, move it
        vtStart = ivt + 1;
        continue;
      }
      if (dt > tRange) { // all following vertices will be have higher time than this track, stop checking
        break;
      }
      bool useTime = trc.gid.getSource() != GTrackID::ITS && mPVParams->useTimeInChi2; // TODO Check if it is ok to not account time error for ITS tracks
      float wgh = 1.f - mTukey2I * trc.evalChi2ToVertex(vertices[vtvec[ivt].first], useTime);
      if (wgh > trc.wgh) {
        trc.wgh = wgh;
        trc.vtxID = vtvec[ivt].first;
      }
    }
    if (trc.vtxID != TrackVF::kNoVtx) {
      mTimeZClusters[trc.vtxID].trackIDs.push_back(itr);
      trc.vtxID = TrackVF::kNoVtx; // to enable for using in the fit
      trc.bin = -1;
    }
  }
  // refit vertices with reattached tracks
  v2tRefs.clear();
  trackIDs.clear();
  std::vector<PVertex> verticesUpd;
  for (int ivt = 0; ivt < nvtOrig; ivt++) {
    auto& clusZT = mTimeZClusters[ivt];
    auto& vtx = vertices[ivt];
    if (clusZT.trackIDs.size() < mPVParams->minTracksPerVtx) {
      continue;
    }
    VertexingInput inp;
    inp.idRange = gsl::span<int>(clusZT.trackIDs);
    inp.scaleSigma2 = 1.;
    inp.timeEst = vtx.getTimeStamp();
    if (!findVertex(inp, vtx)) {
      vtx.setNContributors(0);
      continue;
    }
    finalizeVertex(inp, vtx, verticesUpd, v2tRefs, trackIDs);
  }
  // reorder in time since the time-stamp of vertices might have been changed
  vertices.swap(verticesUpd);
  timeSort.resize(vertices.size());
  std::iota(timeSort.begin(), timeSort.end(), 0);
  std::sort(timeSort.begin(), timeSort.end(), [&vertices](int i, int j) {
    return vertices[i].getTimeStamp().getTimeStamp() < vertices[j].getTimeStamp().getTimeStamp();
  });
}

//___________________________________________________________________
void PVertexer::reduceDebris(std::vector<PVertex>& vertices, std::vector<int>& timeSort, const std::vector<o2::MCEventLabel>& lblVtx)
{
  // eliminate low multiplicity vertices in the close proximity of high mult ones, assuming that these are their debries
  // The timeSort vector indicates the time ordering of the vertices
  int nv = vertices.size();
  std::vector<int> multSort(nv); // sort time indices in multiplicity
  std::iota(multSort.begin(), multSort.end(), 0);
  std::sort(multSort.begin(), multSort.end(), [&timeSort, vertices](int i, int j) {
    return vertices[timeSort[i]].getNContributors() > vertices[timeSort[j]].getNContributors();
  });

  // suppress vertex pointed by j if needed, if time difference between i and j is too large, return true to stop checking
  // pairs starting with i.
  auto checkPair = [&vertices, &timeSort, &lblVtx, this](int i, int j) {
    auto &vtI = vertices[timeSort[i]], &vtJ = vertices[timeSort[j]];
    auto tDiff = std::abs(vtI.getTimeStamp().getTimeStamp() - vtJ.getTimeStamp().getTimeStamp());
    if (tDiff > this->mPVParams->maxTDiffDebris) {
      return true; // don't continue checking other neighbours in time
    }
    if (vtI.getNContributors() < vtJ.getNContributors()) {
      return false; // comparison goes from higher to lower mult vtx
    }
    bool rej = true;
    float zDiff = std::abs(vtI.getZ() - vtJ.getZ());
    if (zDiff > this->mPVParams->maxZDiffDebris) { // cannot be reduced as too far in Z
#ifndef _PV_DEBUG_TREE_
      return false;
#endif
      rej = false;
    }
    float multRat = float(vtJ.getNContributors()) / float(vtI.getNContributors());
    if (multRat > this->mPVParams->maxMultRatDebris) {
#ifndef _PV_DEBUG_TREE_
      return false;
#endif
      rej = false;
    }
    float tiE = vtI.getTimeStamp().getTimeStampError(), tjE = vtJ.getTimeStamp().getTimeStampError();
    float chi2z = zDiff * zDiff / (vtI.getSigmaZ2() + vtJ.getSigmaZ2() + this->mPVParams->addZSigma2Debris);
    float chi2t = tDiff * tDiff / (tiE * tiE + tjE * tjE + this->mPVParams->addTimeSigma2Debris);
    if (chi2z + chi2t > this->mPVParams->maxChi2TZDebris) {
#ifndef _PV_DEBUG_TREE_
      return false;
#endif
      rej = false;
    }
    // all veto cuts passed, declare as fake!
#ifdef _PV_DEBUG_TREE_
    o2::MCEventLabel dummyLbl;
    this->mDebugDumpPVComp.emplace_back(PVtxCompDump{vtI, vtJ, chi2z, chi2t, rej});
    if (!lblVtx.empty()) {
      this->mDebugDumpPVCompLbl0.push_back(lblVtx[timeSort[i]]);
      this->mDebugDumpPVCompLbl1.push_back(lblVtx[timeSort[j]]);
    }
#endif
    if (rej) {
      timeSort[j] = -1;
      vtJ.setNContributors(0);
    }
    return false;
  };

  for (int im = 0; im < nv; im++) { // loop from highest multiplicity to lowest one
    int it = multSort[im];
    if (timeSort[it] < 0) { // if <0, the vertex was already discarded
      continue;
    }

    int itL = it; // look for vertices with smaller time
    while (itL) {
      if (timeSort[--itL] >= 0) { // if <0, the vertex was already discarded
        if (checkPair(it, itL)) { // if too far in time, don't compare further
          break;
        }
      }
    }             // itL loop
    int itH = it; // look for vertices with higher time
    while (++itH < nv) {
      if (timeSort[itH] >= 0) {   // if <0, the vertex was already discarded
        if (checkPair(it, itH)) { // if too far in time, don't compare further
          break;
        }
      }
    } // itH loop
  }
#ifdef _PV_DEBUG_TREE_
  mDebugVtxCompTree->Fill();
  mDebugDumpPVComp.clear();
  mDebugDumpPVCompLbl0.clear();
  mDebugDumpPVCompLbl1.clear();
#endif
}

//___________________________________________________________________
void PVertexer::initMeanVertexConstraint()
{
  // set mean vertex constraint and its errors
  double det = mMeanVertex.getSigmaX2() * mMeanVertex.getSigmaY2() - mMeanVertex.getSigmaXY() * mMeanVertex.getSigmaXY();
  if (det <= kAlmost0D || mMeanVertex.getSigmaY2() < kAlmost0D || mMeanVertex.getSigmaY2() < kAlmost0D) {
    throw std::runtime_error(fmt::format("Singular matrix for vertex constraint: sxx={:+.4e} syy={:+.4e} sxy={:+.4e}",
                                         mMeanVertex.getSigmaX2(), mMeanVertex.getSigmaY2(), mMeanVertex.getSigmaXY()));
  }
  mXYConstraintInvErr[0] = mMeanVertex.getSigmaY2() / det;
  mXYConstraintInvErr[1] = -mMeanVertex.getSigmaXY() / det;
  mXYConstraintInvErr[2] = mMeanVertex.getSigmaX2() / det;
}

//______________________________________________
float PVertexer::getTukey() const
{
  // convert 1/tukey^2 to tukey
  return sqrtf(1. / mTukey2I);
}

//___________________________________________________________________
TimeEst PVertexer::timeEstimate(const VertexingInput& input) const
{
  o2::math_utils::StatAccumulator test;
  for (int i : input.idRange) {
    if (mTracksPool[i].canUse()) {
      const auto& timeT = mTracksPool[i].timeEst;
      auto trErr2I = 1. / (timeT.getTimeStampError() * timeT.getTimeStampError());
      test.add(timeT.getTimeStamp(), trErr2I);
    }
  }

  const auto [t, te2] = test.getMeanRMS2<float>();
  return {t, te2};
}

//___________________________________________________________________
void PVertexer::init()
{
  mPVParams = &PVertexerParams::Instance();
  setTukey(mPVParams->tukey);
  initMeanVertexConstraint();
  auto* prop = o2::base::Propagator::Instance();
  setBz(prop->getNominalBz());

#ifdef _PV_DEBUG_TREE_
  mDebugDumpFile = std::make_unique<TFile>("pvtxDebug.root", "recreate");

  mDebugPoolTree = std::make_unique<TTree>("pvtxTrackPool", "PVertexer tracks pool debug output");
  mDebugPoolTree->Branch("trc", &mDebugDumpDBSTrc);
  mDebugPoolTree->Branch("gid", &mDebugDumpDBSGID);
  mDebugPoolTree->Branch("mc", &mDebugDumpDBSTrcMC);

  mDebugDBScanTree = std::make_unique<TTree>("pvtxDBScan", "PVertexer DBScan debug output");
  mDebugDBScanTree->Branch("trc", &mDebugDumpDBSTrc);
  mDebugDBScanTree->Branch("gid", &mDebugDumpDBSGID);
  mDebugDBScanTree->Branch("mc", &mDebugDumpDBSTrcMC);

  mDebugVtxTree = std::make_unique<TTree>("pvtx", "final PVertexer debug output");
  mDebugVtxTree->Branch("vtx", &mDebugDumpVtx);
  mDebugVtxTree->Branch("trc", &mDebugDumpVtxTrc);
  mDebugVtxTree->Branch("gid", &mDebugDumpVtxGID);
  mDebugVtxTree->Branch("mc", &mDebugDumpVtxTrcMC);

  mDebugVtxCompTree = std::make_unique<TTree>("pvtxComp", "PVertexer neighbouring vertices debud output");
  mDebugVtxCompTree->Branch("vtxComp", &mDebugDumpPVComp);
  mDebugVtxCompTree->Branch("vtxCompLbl0", &mDebugDumpPVCompLbl0);
  mDebugVtxCompTree->Branch("vtxCompLbl1", &mDebugDumpPVCompLbl1);
#endif
}

//___________________________________________________________________
void PVertexer::end()
{
#ifdef _PV_DEBUG_TREE_
  mDebugPoolTree->Write();
  mDebugDBScanTree->Write();
  mDebugVtxCompTree->Write();
  mDebugVtxTree->Write();
  mDebugDBScanTree.reset();
  mDebugDBScanTree.reset();
  mDebugVtxCompTree.reset();
  mDebugDumpFile->Close();
  mDebugDumpFile.reset();
#endif
}

//___________________________________________________________________
void PVertexer::finalizeVertex(const VertexingInput& input, const PVertex& vtx,
                               std::vector<PVertex>& vertices, std::vector<V2TRef>& v2tRefs, std::vector<uint32_t>& trackIDs,
                               SeedHistoTZ* histo)
{
  int lastID = vertices.size();
  vertices.emplace_back(vtx);
  auto& ref = v2tRefs.emplace_back(trackIDs.size(), 0);
  for (int i : input.idRange) {
    auto& trc = mTracksPool[i];
    if (trc.canAssign()) {
      trackIDs.push_back(i);
      trc.vtxID = lastID;
      if (trc.bin >= 0 && histo) {
        histo->setBinContent(trc.bin, -1.f); // discard used bin
      }
    }
  }
  ref.setEntries(trackIDs.size() - ref.getFirstEntry());
}

//___________________________________________________________________
void PVertexer::createMCLabels(gsl::span<const o2::MCCompLabel> lblTracks,
                               const std::vector<uint32_t>& trackIDs, const std::vector<V2TRef>& v2tRefs,
                               std::vector<o2::MCEventLabel>& lblVtx)
{
  lblVtx.clear();
  if (!lblTracks.size()) {
    LOG(error) << "Track labels are not provided";
    return;
  }
  std::unordered_map<o2::MCEventLabel, int> labelOccurence;

  auto bestLbl = [](std::unordered_map<o2::MCEventLabel, int> mp, int norm) -> o2::MCEventLabel {
    o2::MCEventLabel best;
    int bestCount = 0;
    for (auto [lbl, cnt] : mp) {
      if (cnt > bestCount && lbl.isSet()) {
        bestCount = cnt;
        best = lbl;
      }
    }
    if (bestCount && norm) {
      best.setCorrWeight(float(bestCount) / norm);
    }
    return best;
  };

  for (const auto& v2t : v2tRefs) {
    int tref = v2t.getFirstEntry(), last = tref + v2t.getEntries();
    labelOccurence.clear();
    o2::MCEventLabel winner; // unset at the moment
    for (; tref < last; tref++) {
      const auto& lbl = lblTracks[mTracksPool[trackIDs[tref]].entry];
      if (!lbl.isSet()) {
        break;
      }
      // if (!lbl.isFake()) { // RS account all labels, not only correct ones
      labelOccurence[{lbl.getEventID(), lbl.getSourceID(), 0.}]++;
      // }
    }
    if (labelOccurence.size()) {
      winner = bestLbl(labelOccurence, v2t.getEntries());
    }
    lblVtx.push_back(winner);
  }
}


//___________________________________________________________________
void PVertexer::setBunchFilling(const o2::BunchFilling& bf)
{
  mBunchFilling = bf;
  // find closest (from above) filled bunch
  int minBC = bf.getFirstFilledBC(), maxBC = bf.getLastFilledBC();
  if (minBC < 0) {
    LOG(error) << "Empty bunch filling is provided, all vertices will be rejected";
    mClosestBunchAbove[0] = mClosestBunchBelow[0] = -1;
    return;
  }
  int bcAbove = minBC;
  for (int i = o2::constants::lhc::LHCMaxBunches; i--;) {
    if (bf.testBC(i)) {
      bcAbove = i;
    }
    mClosestBunchAbove[i] = bcAbove;
  }
  int bcBelow = maxBC;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (bf.testBC(i)) {
      bcBelow = i;
    }
    mClosestBunchBelow[i] = bcBelow;
  }
}

//___________________________________________________________________
bool PVertexer::setCompatibleIR(PVertex& vtx)
{
  // assign compatible IRs accounting for the bunch filling scheme
  if (mClosestBunchAbove[0] < 0) { // empty or no BF was provided
    return false;
  }
  const auto& vtxT = vtx.getTimeStamp();
  o2::InteractionRecord irMin(mStartIR), irMax(mStartIR);
  auto rangeT = std::max(mPVParams->minTError, mPVParams->nSigmaTimeCut * std::min(mPVParams->maxTError, vtxT.getTimeStampError())) + mPVParams->timeMarginVertexTime;
  float t = vtxT.getTimeStamp() + mPVParams->timeBiasMS;
  if (t > rangeT) {
    irMin += o2::InteractionRecord(1.e3 * (t - rangeT));
  }
  if (t < -rangeT) {
    return false; // discard vertex at negative time
  }
  irMax += o2::InteractionRecord(1.e3 * (t + rangeT)); // RS TODO: make sure that irMax does not exceed TF edge
  irMax++; // to account for rounding
  // restrict using bunch filling
  int bc = mClosestBunchAbove[irMin.bc];
  LOG(debug) << "irMin.bc = " << irMin.bc << " bcAbove = " << bc;
  if (bc < irMin.bc) {
    irMin.orbit++;
  }
  irMin.bc = bc;
  bc = mClosestBunchBelow[irMax.bc];
  LOG(debug) << "irMax.bc = " << irMax.bc << " bcBelow = " << bc;
  if (bc > irMax.bc) {
    if (irMax.orbit == 0) {
      return false;
    }
    irMax.orbit--;
  }
  irMax.bc = bc;
  vtx.setIRMin(irMin);
  vtx.setIRMax(irMax);
  if (irMin > irMax) {
    LOG(debug) << "Reject VTX " << vtx.asString() << " trange = " << rangeT;
  }
  return irMax >= irMin;
}

//___________________________________________________________________
int PVertexer::dbscan_RangeQuery(int id, std::vector<int>& cand, std::vector<int>& status)
{
  // find neighbours for dbscan cluster core point candidate
  // Since we use asymmetric distance definition, is it bit more complex than simple search within chi2 proximity
  int nFound = 0;
  const auto& tI = mTracksPool[id];
  int ntr = mTracksPool.size();

  auto procPnt = [this, &tI, &status, &cand, &nFound, id](int idN) {
    const auto& tL = this->mTracksPool[idN];
    if (std::abs(tI.timeEst.getTimeStamp() - tL.timeEst.getTimeStamp()) > this->mPVParams->dbscanDeltaT) {
      return -1;
    }
    auto statN = status[idN], stat = status[id];
    if (statN >= 0 && (stat < 0 || (stat >= 0 && statN != stat))) { // do not consider as a neighbour if already added to other cluster
      return 0;
    }
    auto dist2 = tL.getDist2(tI);
    if (dist2 < this->mPVParams->dbscanMaxDist2) {
      nFound++;
      if (statN < 0 && statN > DBS_INCHECK) { // no point in adding for check already assigned point, or which is already in the list (i.e. < INCHECK)
        cand.push_back(idN);
        status[idN] += DBS_INCHECK; // flag that the track is in the candidates list (i.e. DBS_UDEF-10 = -12 or DPB_NOISE-10 = -11).
      }
    }
    return 1;
  };
  int idL = id;
  while (--idL >= 0) { // index in time decreasing direction
    if (procPnt(idL) < 0) {
      break;
    }
  }
  int idU = id;
  while (++idU < ntr) { // index in time increasing direction
    if (procPnt(idU) < 0) {
      break;
    }
  }
  return nFound;
}

//_____________________________________________________
void PVertexer::dbscan_clusterize()
{
  mTimeZClusters.clear();
  int ntr = mTracksPool.size();
  std::vector<int> status(ntr, DBS_UNDEF);
  TStopwatch timer;
  int clID = -1;

  std::vector<int> nbVec;
  for (int it = 0; it < ntr; it++) {
    if (status[it] != DBS_UNDEF) {
      continue;
    }
    nbVec.clear();
    auto nnb0 = dbscan_RangeQuery(it, nbVec, status);
    int minNeighbours = mPVParams->minTracksPerVtx - 1;
    if (nnb0 < minNeighbours) {
      status[it] = DBS_NOISE; // noise
      continue;
    }
    if (nnb0 > minNeighbours) {
      minNeighbours = std::max(minNeighbours, int(nnb0 * mPVParams->dbscanAdaptCoef));
    }
    status[it] = ++clID;
    auto& clusVec = mTimeZClusters.emplace_back().trackIDs; // new cluster
    clusVec.push_back(it);

    for (int j = 0; j < nnb0; j++) {
      int jt = nbVec[j];
      auto statjt = status[jt];
      if (statjt >= 0) {
        LOG(error) << "assigned track " << jt << " with status " << statjt << " head is " << it << " clID= " << clID;
        continue;
      }
      status[jt] = clID;
      clusVec.push_back(jt);
      if (statjt == DBS_NOISE + DBS_INCHECK) { // was border point, no check for being core point is needed
        continue;
      }
      int ncurr = nbVec.size();
      if (clusVec.size() > minNeighbours) {
        minNeighbours = std::max(minNeighbours, int(clusVec.size() * mPVParams->dbscanAdaptCoef));
      }
      auto nnb1 = dbscan_RangeQuery(jt, nbVec, status);
      if (nnb1 < minNeighbours) {
        for (unsigned k = ncurr; k < nbVec.size(); k++) {
          if (status[nbVec[k]] < DBS_INCHECK) {
            status[nbVec[k]] -= DBS_INCHECK; // remove from checks
          }
        }
        nbVec.resize(ncurr); // not a core point, reset the seeds pool to the state before RangeQuery
      } else {
        nnb0 = nbVec.size(); // core point, its neighbours need to be checked
      }
    }
  }

  for (auto& clus : mTimeZClusters) {
    if (clus.trackIDs.size() < mPVParams->minTracksPerVtx) {
      clus.trackIDs.clear();
      continue;
    }
    float tMean = 0;
    for (const auto tid : clus.trackIDs) {
      tMean += mTracksPool[tid].timeEst.getTimeStamp();
    }
    clus.timeEst.setTimeStamp(tMean / clus.trackIDs.size());
  }
  timer.Stop();
  LOG(info) << "Found " << mTimeZClusters.size() << " seeding clusters from DBSCAN in " << timer.CpuTime() << " CPU s";
}

//___________________________________________________________________
std::pair<int, int> PVertexer::getBestIR(const PVertex& vtx, const gsl::span<o2::InteractionRecord> bcData, int& currEntry) const
{
  // select best matching interaction record
  int best = -1, n = bcData.size();
  while (currEntry < n && bcData[currEntry] < vtx.getIRMin()) {
    currEntry++; // skip all times which have no chance to be matched
  }
  int i = currEntry, nCompatible = 0;
  float bestDf = 1e12;
  auto tVtxNS = (vtx.getTimeStamp().getTimeStamp() + mPVParams->timeBiasMS) * 1e3; // time in ns
  while (i < n) {
    if (bcData[i] > vtx.getIRMax()) {
      break;
    }
    nCompatible++;
    auto dfa = std::abs(bcData[i].differenceInBCNS(mStartIR) - tVtxNS);
    if (dfa <= bestDf) {
      bestDf = dfa;
      best = i;
    }
    i++;
  }
  return {best, nCompatible};
}

//___________________________________________________________________
SeedHistoTZ PVertexer::buildHistoTZ(const VertexingInput& input)
{
  // build histo for tracks time / z entries weigthed by their inverse error, to be used for seeding peak finding
  // estimat the range of TZ histo

  float hZMin = 1e9, hZMax = -1e8, hTMin = 1e9, hTMax = -1e9;
  for (int i : input.idRange) {
    const auto& trc = mTracksPool[i];
    if (trc.canUse()) {
      if (trc.z > hZMax) {
        hZMax = trc.z;
      }
      if (trc.z < hZMin) {
        hZMin = trc.z;
      }
      if (trc.timeEst.getTimeStamp() > hTMax) {
        hTMax = trc.timeEst.getTimeStamp();
      }
      if (trc.timeEst.getTimeStamp() < hTMin) {
        hTMin = trc.timeEst.getTimeStamp();
      }
    }
  }

  float dz = hZMax - hZMin, dt = hTMax - hTMin;
  int nbz = 1 + int((dz) / mPVParams->histoBinZSize), nbt = 1 + int((dt) / mPVParams->histoBinTSize);
  float dzh = 0.5f * (nbz * mPVParams->histoBinZSize - dz), dth = 0.5f * (nbt * mPVParams->histoBinTSize - dt);
  SeedHistoTZ seedHistoTZ(nbt, hTMin - dth, hTMax + dth, nbz, hZMin - dzh, hZMax + dzh);

  for (int i : input.idRange) {
    auto& trc = mTracksPool[i];
    if (trc.canUse()) {
      trc.bin = seedHistoTZ.fillAndFlagBin(trc.timeEst.getTimeStamp(), trc.z, trc.wghHisto);
    }
  }

  return std::move(seedHistoTZ);
}

//______________________________________________
bool PVertexer::relateTrackToMeanVertex(o2::track::TrackParCov& trc, float vtxErr2) const
{
  o2d::DCA dca;
  return o2::base::Propagator::Instance()->propagateToDCA(mMeanVertex, trc, mBz, 2.0f,
                                                          o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca, nullptr, 0, mPVParams->dcaTolerance) &&
         (dca.getY() * dca.getY() / (dca.getSigmaY2() + vtxErr2) < mPVParams->pullIniCut);
}

//______________________________________________
bool PVertexer::relateTrackToVertex(o2::track::TrackParCov& trc, const o2d::VertexBase& vtxSeed) const
{
  return o2::base::Propagator::Instance()->propagateToDCA(vtxSeed, trc, mBz, 2.0f, o2::base::Propagator::MatCorrType::USEMatCorrLUT);
}

//______________________________________________
void PVertexer::doDBScanDump(const VertexingInput& input, gsl::span<const o2::MCCompLabel> lblTracks)
{
  // dump tracks for T-Z clusters identified by the DBScan
#ifdef _PV_DEBUG_TREE_
  for (int i : input.idRange) {
    const auto& trc = mTracksPool[i];
    if (trc.canUse()) {
      mDebugDumpDBSTrc.emplace_back(TrackVFDump{trc.z, trc.sig2ZI, trc.timeEst.getTimeStamp(), trc.timeEst.getTimeStampError(), trc.wghHisto});
      mDebugDumpDBSGID.push_back(trc.gid);
      if (lblTracks.size()) {
        mDebugDumpDBSTrcMC.push_back(lblTracks[trc.entry]);
      }
    }
  }
  mDebugDBScanTree->Fill();
  mDebugDumpDBSTrc.clear();
  mDebugDumpDBSGID.clear();
  mDebugDumpDBSTrcMC.clear();
#endif
}

//______________________________________________
void PVertexer::doDBGPoolDump(gsl::span<const o2::MCCompLabel> lblTracks)
{
  // dump tracks of the pool
#ifdef _PV_DEBUG_TREE_
  for (const auto& trc : mTracksPool) {
    mDebugDumpDBSTrc.emplace_back(TrackVFDump{trc.z, trc.sig2ZI, trc.timeEst.getTimeStamp(), trc.timeEst.getTimeStampError(), trc.wghHisto});
    mDebugDumpDBSGID.push_back(trc.gid);
    if (lblTracks.size()) {
      mDebugDumpDBSTrcMC.push_back(lblTracks[trc.entry]);
    }
  }
  mDebugPoolTree->Fill();
  mDebugDumpDBSTrc.clear();
  mDebugDumpDBSGID.clear();
  mDebugDumpDBSTrcMC.clear();
#endif
}

//______________________________________________
void PVertexer::doVtxDump(std::vector<PVertex>& vertices, std::vector<uint32_t> trackIDsLoc, std::vector<V2TRef>& v2tRefsLoc,
                          gsl::span<const o2::MCCompLabel> lblTracks)
{
  // dump tracks for T-Z clusters identified by the DBScan
#ifdef _PV_DEBUG_TREE_
  int nv = vertices.size();
  for (int iv = 0; iv < nv; iv++) {
    mDebugDumpVtx = vertices[iv];
    if (mDebugDumpVtx.getNContributors() == 0) { // discarded
      continue;
    }
    int start = v2tRefsLoc[iv].getFirstEntry(), stop = start + v2tRefsLoc[iv].getEntries();
    for (int it = start; it < stop; it++) {
      const auto& trc = mTracksPool[trackIDsLoc[it]];
      mDebugDumpVtxTrc.emplace_back(TrackVFDump{trc.z, trc.sig2ZI, trc.timeEst.getTimeStamp(), trc.timeEst.getTimeStampError(), trc.wgh});
      mDebugDumpVtxGID.push_back(trc.gid);
      if (lblTracks.size()) {
        mDebugDumpVtxTrcMC.push_back(lblTracks[trc.entry]);
      }
    }
    mDebugVtxTree->Fill();
    mDebugDumpVtxTrc.clear();
    mDebugDumpVtxGID.clear();
    mDebugDumpVtxTrcMC.clear();
  }
#endif
}

//______________________________________________
PVertex PVertexer::refitVertex(const std::vector<bool> useTrack, const o2d::VertexBase& vtxSeed)
{
  // Refit the tracks prepared by the successful prepareVertexRefit, possible skipping those tracks wich have useTrack value false
  // (useTrack is ignored if empty).
  // The vtxSeed is the originally found vertex, must be the same as used for the prepareVertexRefit.
  // Refitted PrimaryVertex is returned, negative chi2 means failure of the refit.
  // ATTENTION: only the position is refitted, the vertex time and IRMin/IRMax info is dummy.

  if (vtxSeed != mVtxRefitOrig) {
    throw std::runtime_error("refitVertex must be preceded by successful prepareVertexRefit");
  }
  VertexingInput inp;
  inp.scaleSigma2 = mPVParams->minScale2;
  inp.idRange = gsl::span<int>(mRefitTrackIDs);
  if (useTrack.size()) {
    for (uint32_t i = 0; i < mTracksPool.size(); i++) {
      mTracksPool[i].vtxID = useTrack[mTracksPool[i].entry] ? TrackVF::kNoVtx : TrackVF::kDiscarded;
    }
  }
  VertexSeed vtxs;
  vtxs.VertexBase::operator=(vtxSeed);
  PVertex vtxRes;
  vtxs.setScale(inp.scaleSigma2, mTukey2I);
  vtxs.setTimeStamp({0.f, -1.}); // time is not refitter
  if (fitIteration(inp, vtxs) == FitStatus::OK) {
    vtxRes = vtxs;
  } else {
    vtxRes.setChi2(-1.);
  }
  return vtxRes;
}
