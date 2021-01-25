// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PVertexer.cxx
/// \brief Primary vertex finder
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/PVertexer.h"
#include "ReconstructionDataFormats/DCA.h"
#include "DetectorsBase/Propagator.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <numeric>
#include <unordered_map>
#include <TStopwatch.h>

using namespace o2::vertexing;

constexpr float PVertexer::kAlmost0F;
constexpr double PVertexer::kAlmost0D;
constexpr float PVertexer::kHugeF;

//___________________________________________________________________
int PVertexer::process(gsl::span<const o2d::TrackTPCITS> tracksITSTPC, gsl::span<const o2::ft0::RecPoints> ft0Data,
                       std::vector<PVertex>& vertices, std::vector<GIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs)
{
  createTracksPool(tracksITSTPC);
  dbscan_clusterize();

  std::vector<PVertex> verticesLoc;
  std::vector<GTrackID> vertexTrackIDsLoc;
  std::vector<V2TRef> v2tRefsLoc;
  std::vector<float> validationTimes;

  for (auto tc : mTimeZClusters) {
    VertexingInput inp;
    //    inp.idRange = gsl::span<int>((int*)&mSortedTrackID[tc.first], tc.count);
    inp.idRange = gsl::span<int>((int*)&mClusterTrackIDs[tc.first], tc.count);
    inp.scaleSigma2 = 3. * estimateScale2();
    inp.timeEst = tc.timeEst;
    findVertices(inp, verticesLoc, vertexTrackIDsLoc, v2tRefsLoc);
  }

  // sort in time
  std::vector<int> vtTimeSortID(verticesLoc.size());
  std::iota(vtTimeSortID.begin(), vtTimeSortID.end(), 0);
  std::sort(vtTimeSortID.begin(), vtTimeSortID.end(), [&verticesLoc](int i, int j) {
    return verticesLoc[i].getTimeStamp().getTimeStamp() < verticesLoc[j].getTimeStamp().getTimeStamp();
  });

  vertices.clear();
  v2tRefs.clear();
  vertexTrackIDs.clear();
  vertices.reserve(verticesLoc.size());
  v2tRefs.reserve(v2tRefsLoc.size());
  vertexTrackIDs.reserve(vertexTrackIDsLoc.size());

  int trCopied = 0, count = 0, vtimeID = 0;
  for (auto i : vtTimeSortID) {
    auto& vtx = verticesLoc[i];

    bool irSet = setCompatibleIR(vtx);
    if (!irSet) {
      continue;
    }
    // do we need to validate by FT0 ?
    if (mValidateWithFT0) {
      auto bestMatch = getBestFT0Trigger(vtx, ft0Data, vtimeID);
      if (bestMatch.first >= 0) {
        vtx.setFlags(PVertex::TimeValidated);
        if (bestMatch.second == 1) {
          vtx.setIR(ft0Data[bestMatch.first].getInteractionRecord());
        }
        LOG(DEBUG) << "Validated with t0 " << bestMatch.first << " with " << bestMatch.second << " candidates";
      } else if (vtx.getNContributors() >= mPVParams->minNContributorsForFT0cut) {
        LOG(DEBUG) << "Discarding " << vtx;
        continue; // reject
      }
    }
    vertices.push_back(vtx);
    int it = v2tRefsLoc[i].getFirstEntry(), itEnd = it + v2tRefsLoc[i].getEntries(), dest0 = vertexTrackIDs.size();
    for (; it < itEnd; it++) {
      auto& gid = vertexTrackIDs.emplace_back(vertexTrackIDsLoc[it]);
      gid.setPVContributor();
    }
    v2tRefs.emplace_back(dest0, v2tRefsLoc[i].getEntries());
    LOG(DEBUG) << "#" << count++ << " " << vertices.back() << " | " << v2tRefs.back().getEntries() << " indices from " << v2tRefs.back().getFirstEntry(); // RS REM
  }

  return vertices.size();
}

//___________________________________________________________________
int PVertexer::process(gsl::span<const o2d::TrackTPCITS> tracksITSTPC, gsl::span<const o2::ft0::RecPoints> ft0Data,
                       std::vector<PVertex>& vertices, std::vector<GIndex>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs,
                       gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC, std::vector<o2::MCEventLabel>& lblVtx)
{
  auto nv = process(tracksITSTPC, ft0Data, vertices, vertexTrackIDs, v2tRefs);
  if (lblITS.size() && lblTPC.size()) {
    createMCLabels(lblITS, lblTPC, vertices, vertexTrackIDs, v2tRefs, lblVtx);
  }
  return nv;
}

//______________________________________________
int PVertexer::findVertices(const VertexingInput& input, std::vector<PVertex>& vertices, std::vector<GTrackID>& vertexTrackIDs, std::vector<V2TRef>& v2tRefs)
{
  // find vertices using tracks with indices (sorted in time) from idRange from "tracks" pool. The pool may containt arbitrary number of tracks,
  // only those which are in the idRange and have canUse()==true, will be used.
  // Results are placed in vertices and v2tRefs vectors
  int nfound = 0, ntr = input.idRange.size();
  if (ntr < mPVParams->minTracksPerVtx) {
    return nfound;
  }
  //
  SeedHisto seedHisto(mPVParams->zHistoRange, mPVParams->zHistoBinSize);

  for (int i : input.idRange) {
    if (mTracksPool[i].canUse()) {
      mTracksPool[i].bin = seedHisto.findBin(mTracksPool[i].getZForXY(mMeanVertex.getX(), mMeanVertex.getY()));
      seedHisto.incrementBin(mTracksPool[i].bin);
    }
  }
  if (seedHisto.nFilled < mPVParams->minTracksPerVtx) {
    return nfound;
  }
  LOG(DEBUG) << "New cluster: times: " << mTracksPool[input.idRange[0]].timeEst.getTimeStamp() << " : " << mTracksPool[input.idRange[ntr - 1]].timeEst.getTimeStamp() << " nfound " << nfound << " of " << ntr; // RSTMP
  int nTrials = 0;
  while (nfound < mPVParams->maxVerticesPerCluster && nTrials < mPVParams->maxTrialsPerCluster) {
    int peakBin = seedHisto.findHighestPeakBin(); // find next seed
    if (!seedHisto.isValidBin(peakBin)) {
      break;
    }
    float zv = seedHisto.getBinCenter(peakBin);
    LOG(DEBUG) << "Seeding with Z=" << zv << " bin " << peakBin << " on trial " << nTrials << " for vertex " << nfound;
    PVertex vtx;
    vtx.setXYZ(mMeanVertex.getX(), mMeanVertex.getY(), zv);
    vtx.setTimeStamp(input.timeEst);
    if (findVertex(input, vtx)) {
      finalizeVertex(input, vtx, vertices, v2tRefs, vertexTrackIDs, seedHisto);
      nfound++;
      nTrials = 0;
    } else {                                                                    // suppress failed seeding bin and its proximities
      auto delta = std::sqrt(vtx.getChi2()) * mStatZErr.getMean() * getTukey(); // largest scale used will be transferred as chi2
      int proximity = delta * seedHisto.binSizeInv;
      int bmin = std::max(0, peakBin - proximity), bmax = std::min(peakBin + proximity + 1, seedHisto.size());
      LOG(DEBUG) << "suppress bins for delta=" << delta << " (" << std::sqrt(vtx.getChi2()) << "*" << mStatZErr.getMean() << "*" << getTukey() << ")"
                 << " bins " << bmin << " : " << bmax - 1;
      for (int i = bmin; i < bmax; i++) {
        seedHisto.discardBin(i);
      }
      nTrials++;
    }
  }
  return nfound;
}

//______________________________________________
bool PVertexer::findVertex(const VertexingInput& input, PVertex& vtx)
{
  // fit vertex taking provided vertex as a seed
  // tracks pool may contain arbitrary number of tracks, only those which are in
  // the idRange (indices of tracks sorted in time) will be used.

  VertexSeed vtxSeed(vtx, input.useConstraint, input.fillErrors);
  vtxSeed.setScale(input.scaleSigma2, mTukey2I);
  vtxSeed.scaleSigma2Prev = input.scaleSigma2;
  //  vtxSeed.setTimeStamp( timeEstimate(input) );
  LOG(DEBUG) << "Start time guess: " << vtxSeed.getTimeStamp();
  vtx.setChi2(1.e30);
  //
  FitStatus result = FitStatus::IterateFurther;
  bool found = false;
  while (result == FitStatus::IterateFurther) {
    vtxSeed.resetForNewIteration();
    vtxSeed.nIterations++;
    LOG(DEBUG) << "iter " << vtxSeed.nIterations << " with scale=" << vtxSeed.scaleSigma2 << " prevScale=" << vtxSeed.scaleSigma2Prev;
    result = fitIteration(input, vtxSeed);

    if (result == FitStatus::OK) {
      result = evalIterations(vtxSeed, vtx);
    } else if (result == FitStatus::NotEnoughTracks) {
      if (vtxSeed.nIterations <= mPVParams->maxIterations && upscaleSigma(vtxSeed)) {
        LOG(DEBUG) << "Upscaling scale to " << vtxSeed.scaleSigma2;
        result = FitStatus::IterateFurther;
        continue; // redo with stronger rescaling
      } else {
        break;
      }
    } else if (result == FitStatus::PoolEmpty || result == FitStatus::Failure) {
      break;
    } else {
      LOG(FATAL) << "Unknown fit status " << int(result);
    }
  }
  LOG(DEBUG) << "Stopped with scale=" << vtxSeed.scaleSigma2 << " prevScale=" << vtxSeed.scaleSigma2Prev << " result = " << int(result);

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
      nTested++;
      accountTrack(mTracksPool[i], vtxSeed);
    }
  }
  vtxSeed.maxScaleSigma2Tested = vtxSeed.scaleSigma2;
  if (vtxSeed.getNContributors() < mPVParams->minTracksPerVtx) {
    return nTested < mPVParams->minTracksPerVtx ? FitStatus::PoolEmpty : FitStatus::NotEnoughTracks;
  }
  if (vtxSeed.useConstraint) {
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
  float dt, trErr2I = 0, dy, dz, chi2T = trc.getResiduals(vtxSeed, dy, dz); // track-vertex residuals and chi2
  auto& timeV = vtxSeed.getTimeStamp();
  auto& timeT = trc.timeEst;
  float ndff = 1. / 2;
  bool noTime = false;
  if (timeV.getTimeStampError() < 0.) {
    noTime = true;
  } else {
    dt = timeT.getTimeStamp() - timeV.getTimeStamp();
    trErr2I = 1. / (timeT.getTimeStampError() * timeT.getTimeStampError());
    if (mPVParams->useTimeInChi2) {
      chi2T += dt * dt * trErr2I;
      ndff = 1. / 3.;
    }
  }
  chi2T *= ndff;
  float wghT = (1.f - chi2T * vtxSeed.scaleSig2ITuk2I); // weighted distance to vertex
  if (wghT < kAlmost0F) {
    trc.wgh = 0.f;
    return;
  }
  float syyI(trc.sig2YI), szzI(trc.sig2ZI), syzI(trc.sigYZI);

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
  if (!noTime) {
    trErr2I *= wghT;
    vtxSeed.tMeanAcc += timeT.getTimeStamp() * trErr2I;
    vtxSeed.tMeanAccErr += trErr2I;
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
    LOG(ERROR) << "Failed to invert matrix" << mat;
    return false;
  }
  ROOT::Math::SVector<double, 3> rhs(vtxSeed.cx0, vtxSeed.cy0, vtxSeed.cz0);
  auto sol = mat * rhs;
  vtxSeed.setXYZ(sol(0), sol(1), sol(2));
  if (vtxSeed.fillErrors) {
    vtxSeed.setCov(mat(0, 0), mat(1, 0), mat(1, 1), mat(2, 0), mat(2, 1), mat(2, 2));
  }
  if (vtxSeed.tMeanAccErr > 0.) {
    auto err2 = 1. / vtxSeed.tMeanAccErr;
    vtxSeed.setTimeStamp({float(vtxSeed.tMeanAcc * err2), float(std::sqrt(err2))});
  }

  vtxSeed.setChi2((vtxSeed.getNContributors() - vtxSeed.wghSum) / vtxSeed.scaleSig2ITuk2I); // calculate chi^2
  auto newScale = vtxSeed.wghChi2 / vtxSeed.wghSum;
  LOG(DEBUG) << "Solve: wghChi2=" << vtxSeed.wghChi2 << " wghSum=" << vtxSeed.wghSum << " -> scale= " << newScale << " old scale " << vtxSeed.scaleSigma2 << " prevScale: " << vtxSeed.scaleSigma2Prev;
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
    LOG(DEBUG) << "stop on simga :" << vtxSeed.scaleSigma2 << " prev: " << vtxSeed.scaleSigma2Prev;
  }
  if (fair::Logger::Logging(fair::Severity::debug)) {
    auto dchi = (vtx.getChi2() - vtxSeed.getChi2()) / vtxSeed.getChi2();
    auto dx = vtxSeed.getX() - vtx.getX(), dy = vtxSeed.getY() - vtx.getY(), dz = vtxSeed.getZ() - vtx.getZ();
    auto dst = std::sqrt(dx * dx + dy * dy + dz * dz);

    LOG(DEBUG) << "dChi:" << vtx.getChi2() << "->" << vtxSeed.getChi2() << " :-> " << dchi;
    LOG(DEBUG) << "dx: " << dx << " dy: " << dy << " dz: " << dz << " -> " << dst;
  }

  vtx = reinterpret_cast<const PVertex&>(vtxSeed);

  if (result == PVertexer::FitStatus::OK) {
    auto chi2Mean = vtxSeed.getChi2() / vtxSeed.getNContributors();
    if (chi2Mean > mPVParams->maxChi2Mean) {
      result = PVertexer::FitStatus::Failure;
      LOG(DEBUG) << "Rejecting at iteration " << vtxSeed.nIterations << " and ScalePrev " << vtxSeed.scaleSigma2Prev << " with meanChi2 = " << chi2Mean;
    } else {
      return result;
    }
  }

  if (vtxSeed.scaleSigma2 > vtxSeed.scaleSigma2Prev) {
    if (++vtxSeed.nScaleIncrease > mPVParams->maxNScaleIncreased) {
      result = PVertexer::FitStatus::Failure;
      LOG(DEBUG) << "Rejecting at iteration " << vtxSeed.nIterations << " with NScaleIncreased " << vtxSeed.nScaleIncrease;
    }
  } else if (vtxSeed.scaleSigma2 > mPVParams->slowConvergenceFactor * vtxSeed.scaleSigma2Prev) {
    if (++vtxSeed.nScaleSlowConvergence > mPVParams->maxNScaleSlowConvergence) {
      if (vtxSeed.scaleSigma2 < mPVParams->acceptableScale2) {
        vtxSeed.setScale(mPVParams->minScale2, mTukey2I);
        LOG(DEBUG) << "Forcing scale2 to " << mPVParams->minScale2;
        result = PVertexer::FitStatus::IterateFurther;
      } else {
        result = PVertexer::FitStatus::Failure;
        LOG(DEBUG) << "Rejecting at iteration " << vtxSeed.nIterations << " with NScaleSlowConvergence " << vtxSeed.nScaleSlowConvergence;
      }
    }
  } else {
    vtxSeed.nScaleSlowConvergence = 0;
  }

  return result;
}

//___________________________________________________________________
void PVertexer::initMeanVertexConstraint()
{
  // set mean vertex constraint and its errors
  double det = mMeanVertex.getSigmaY2() * mMeanVertex.getSigmaZ2() - mMeanVertex.getSigmaYZ() * mMeanVertex.getSigmaYZ();
  if (det <= kAlmost0D || mMeanVertex.getSigmaY2() < kAlmost0D || mMeanVertex.getSigmaZ2() < kAlmost0D) {
    throw std::runtime_error(fmt::format("Singular matrix for vertex constraint: syy={:+.4e} syz={:+.4e} szz={:+.4e}",
                                         mMeanVertex.getSigmaY2(), mMeanVertex.getSigmaYZ(), mMeanVertex.getSigmaZ2()));
  }
  mXYConstraintInvErr[0] = mMeanVertex.getSigmaZ2() / det;
  mXYConstraintInvErr[2] = mMeanVertex.getSigmaY2() / det;
  mXYConstraintInvErr[1] = -mMeanVertex.getSigmaYZ() / det;
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
  mFT0Params = &o2::ft0::InteractionTag::Instance();
  setTukey(mPVParams->tukey);
  initMeanVertexConstraint();

  auto* prop = o2::base::Propagator::Instance();
  setBz(prop->getNominalBz());
}

//___________________________________________________________________
void PVertexer::finalizeVertex(const VertexingInput& input, const PVertex& vtx,
                               std::vector<PVertex>& vertices, std::vector<V2TRef>& v2tRefs, std::vector<GTrackID>& vertexTrackIDs,
                               SeedHisto& histo)
{
  int lastID = vertices.size();
  vertices.emplace_back(vtx);
  auto& ref = v2tRefs.emplace_back(vertexTrackIDs.size(), 0);
  for (int i : input.idRange) {
    if (mTracksPool[i].canAssign()) {
      vertexTrackIDs.push_back(mTracksPool[i].entry);
      mTracksPool[i].vtxID = lastID;

      // remove track from ZSeeds histo
      histo.decrementBin(mTracksPool[i].bin);
    }
  }
  ref.setEntries(vertexTrackIDs.size() - ref.getFirstEntry());
}

//___________________________________________________________________
void PVertexer::createTracksPool(gsl::span<const o2d::TrackTPCITS> tracksITSTPC)
{
  // create pull of all candidate tracks in a global array ordered in time
  mTracksPool.clear();
  mSortedTrackID.clear();

  auto ntGlo = tracksITSTPC.size();
  mTracksPool.reserve(ntGlo);
  // check all containers
  float vtxErr2 = 0.5 * (mMeanVertex.getSigmaX2() + mMeanVertex.getSigmaY2());
  float pullIniCut = 9.;                          // RS FIXME  pullIniCut should be a parameter
  o2d::DCA dca;

  for (uint32_t i = 0; i < ntGlo; i++) {
    o2::track::TrackParCov trc = tracksITSTPC[i];
    if (!trc.propagateToDCA(mMeanVertex, mBz, &dca, mPVParams->dcaTolerance) ||
        dca.getY() * dca.getY() / (dca.getSigmaY2() + vtxErr2) > mPVParams->pullIniCut) {
      continue;
    }
    auto& tvf = mTracksPool.emplace_back(trc, tracksITSTPC[i].getTimeMUS(), GTrackID{i, o2::dataformats::GlobalTrackID::ITSTPC});
    mStatZErr.add(std::sqrt(trc.getSigmaZ2()));
    mStatTErr.add(tvf.timeEst.getTimeStampError());
  }
  // TODO: try to narrow timestamps using tof times
  auto [zerrMean, zerrRMS] = mStatZErr.getMeanRMS2<float>();

  auto [terrMean, terrRMS] = mStatTErr.getMeanRMS2<float>();

  if (mTracksPool.empty()) {
    return;
  }
  //
  mSortedTrackID.resize(mTracksPool.size());
  std::iota(mSortedTrackID.begin(), mSortedTrackID.end(), 0);

  std::sort(mSortedTrackID.begin(), mSortedTrackID.end(), [this](int i, int j) {
    return this->mTracksPool[i].timeEst.getTimeStamp() < this->mTracksPool[j].timeEst.getTimeStamp();
  });

  auto tMin = mTracksPool[mSortedTrackID.front()].timeEst.getTimeStamp();
  auto tMax = mTracksPool[mSortedTrackID.back()].timeEst.getTimeStamp();
}

//___________________________________________________________________
void PVertexer::createMCLabels(gsl::span<const o2::MCCompLabel> lblITS, gsl::span<const o2::MCCompLabel> lblTPC,
                               const std::vector<PVertex> vertices, const std::vector<o2::dataformats::VtxTrackIndex> vertexTrackIDs, const std::vector<V2TRef> v2tRefs,
                               std::vector<o2::MCEventLabel>& lblVtx)
{
  lblVtx.clear();
  int nv = vertices.size();
  if (lblITS.size() != lblITS.size() || !lblITS.size()) {
    LOG(ERROR) << "labels are not provided or incorrect";
    return;
  }
  std::unordered_map<o2::MCEventLabel, int> labelOccurenceCorr, labelOccurenceITS;

  auto bestLbl = [](std::unordered_map<o2::MCEventLabel, int> mp, int norm) -> o2::MCEventLabel {
    o2::MCEventLabel best;
    int bestCount = 0;
    for (auto [lbl, cnt] : mp) {
      if (cnt > bestCount) {
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
    labelOccurenceCorr.clear();
    labelOccurenceITS.clear();
    o2::MCEventLabel winner; // unset at the moment
    for (; tref < last; tref++) {
      int tid = vertexTrackIDs[tref].getIndex();
      const auto& lITS = lblITS[tid];
      const auto& lTPC = lblTPC[tid];
      if (!lITS.isSet() || !lTPC.isSet()) {
        break;
      }
      if (lITS.getTrackID() == lTPC.getTrackID() && lITS.getEventID() == lTPC.getEventID() && lITS.getSourceID() == lTPC.getSourceID()) {
        labelOccurenceCorr[{lITS.getEventID(), lITS.getSourceID(), 0.}]++;
      } else {
        labelOccurenceITS[{lITS.getEventID(), lITS.getSourceID(), 0.}]++;
      }
    }
    if (labelOccurenceCorr.size()) {
      winner = bestLbl(labelOccurenceCorr, v2t.getEntries());
    } else if (labelOccurenceITS.size()) {
      winner = bestLbl(labelOccurenceITS, 0); // in absence of correct matches, set the ITS only label but set its weight to 0
    }
    lblVtx.push_back(winner);
  }
}

//___________________________________________________________________
std::pair<int, int> PVertexer::getBestFT0Trigger(const PVertex& vtx, gsl::span<const o2::ft0::RecPoints> ft0Data, int& currEntry) const
{
  // select best matching FT0 recpoint
  int best = -1, n = ft0Data.size();
  while (currEntry < n && ft0Data[currEntry].getInteractionRecord() < vtx.getIRMin()) {
    currEntry++; // skip all times which have no chance to be matched
  }
  int i = currEntry, nCompatible = 0;
  float bestDf = 1e12;
  auto tVtxNS = (vtx.getTimeStamp().getTimeStamp() + mPVParams->timeBiasMS) * 1e3; // time in ns
  while (i < n) {
    if (ft0Data[i].getInteractionRecord() > vtx.getIRMax()) {
      break;
    }
    if (mFT0Params->isSelected(ft0Data[i])) {
      nCompatible++;
      auto dfa = std::abs(mFT0Params->getInteractionTimeNS(ft0Data[i], mStartIR) - tVtxNS);
      if (dfa <= bestDf) {
        bestDf = dfa;
        best = i;
      }
    }
    i++;
  }
  return {best, nCompatible};
}

//___________________________________________________________________
void PVertexer::setBunchFilling(const o2::BunchFilling& bf)
{
  mBunchFilling = bf;
  // find closest (from above) filled bunch
  int minBC = bf.getFirstFilledBC(), maxBC = bf.getLastFilledBC();
  if (minBC < 0) {
    throw std::runtime_error("Bunch filling is not set in PVertexer");
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
  const auto& vtxT = vtx.getTimeStamp();
  o2::InteractionRecord irMin(mStartIR), irMax(mStartIR);
  auto rangeT = mPVParams->nSigmaTimeCut * std::max(mPVParams->minTError, std::min(mPVParams->maxTError, vtxT.getTimeStampError()));
  float t = vtxT.getTimeStamp() + mPVParams->timeBiasMS;
  if (t > rangeT) {
    irMin += o2::InteractionRecord(1.e3 * (t - rangeT));
  }
  irMax += o2::InteractionRecord(1.e3 * (t + rangeT));
  irMax++; // to account for rounding
  // restrict using bunch filling
  int bc = mClosestBunchAbove[irMin.bc];
  if (bc < irMin.bc) {
    irMin.orbit++;
  }
  irMin.bc = bc;
  bc = mClosestBunchBelow[irMax.bc];
  if (bc > irMax.bc) {
    if (irMax.orbit == 0) {
      return false;
    }
    irMax.orbit--;
  }
  irMax.bc = bc;
  vtx.setIRMin(irMin);
  vtx.setIRMax(irMax);
  return irMax >= irMin;
}

//___________________________________________________________________
int PVertexer::dbscan_RangeQuery(int id, std::vector<int>& cand, const std::vector<int>& status)
{
  // find neighbours for dbscan cluster core point candidate
  // Since we use asymmetric distance definition, is it bit more complex than simple search within chi2 proximity
  int nFound = 0;
  const auto& tI = mTracksPool[mSortedTrackID[id]];
  int ntr = mTracksPool.size();

  auto procPnt = [this, &tI, &status, &cand, &nFound, id](int idN) {
    const auto& tL = this->mTracksPool[this->mSortedTrackID[idN]];
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
      if (statN < 0) {
        cand.push_back(idN); // no point in adding for check already assigned point
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
  const int UNDEF = -2, NOISE = -1;
  int ntr = mSortedTrackID.size();
  std::vector<std::vector<int>> clusters;
  std::vector<int> status(ntr, UNDEF);
  TStopwatch timer;
  int clID = -1;

  std::vector<int> nbVec;
  for (int it = 0; it < ntr; it++) {
    if (status[it] != UNDEF) {
      continue;
    }
    nbVec.clear();
    auto nnb0 = dbscan_RangeQuery(it, nbVec, status);
    int minNeighbours = mPVParams->minTracksPerVtx - 1;
    if (nnb0 < minNeighbours) {
      status[it] = NOISE; // noise
      continue;
    }
    if (nnb0 > minNeighbours) {
      minNeighbours = std::max(minNeighbours, int(nnb0 * mPVParams->dbscanAdaptCoef));
    }
    status[it] = ++clID;
    auto& clusVec = clusters.emplace_back(1, it); // new cluster

    for (int j = 0; j < nnb0; j++) {
      int jt = nbVec[j];
      auto statjt = status[jt];
      if (statjt >= 0) {
        continue;
      }
      status[jt] = clID;
      clusVec.push_back(jt);
      if (statjt == NOISE) { // was border point, no check for being core point is needed
        continue;
      }
      int ncurr = nbVec.size();
      if (clusVec.size() > minNeighbours) {
        minNeighbours = std::max(minNeighbours, int(clusVec.size() * mPVParams->dbscanAdaptCoef));
      }
      auto nnb1 = dbscan_RangeQuery(jt, nbVec, status);
      if (nnb1 < minNeighbours) {
        nbVec.resize(ncurr); // not a core point, reset the seeds pool to the state before RangeQuery
      } else {
        nnb0 = ncurr; // core point, its neighbours need to be checked
      }
    }
  }

  mTimeZClusters.clear();
  mClusterTrackIDs.reserve(ntr);
  for (const auto& clus : clusters) {
    if (clus.size() < mPVParams->minTracksPerVtx) {
      continue;
    }
    int first = mClusterTrackIDs.size();
    float tMean = 0;
    for (const auto tid : clus) {
      mClusterTrackIDs.push_back(mSortedTrackID[tid]);
      tMean += mTracksPool[mClusterTrackIDs.back()].timeEst.getTimeStamp();
    }
    int last = int(mClusterTrackIDs.size());
    int count = last - first;
    mTimeZClusters.emplace_back(TimeZCluster{{tMean / count, 0.}, first, --last, count});
  }
  timer.Stop();
  LOG(INFO) << "Found " << mTimeZClusters.size() << " clusters from DBSCAN out of " << clusters.size() << " seeds in " << timer.CpuTime() << " CPU s";
}
