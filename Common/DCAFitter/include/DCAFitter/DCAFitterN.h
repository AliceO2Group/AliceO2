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

/// \file DCAFitterN.h
/// \brief Defintions for N-prongs secondary vertex fit
/// \author ruben.shahoyan@cern.ch
/// For the formulae derivation see /afs/cern.ch/user/s/shahoian/public/O2/DCAFitter/DCAFitterN.pdf

#ifndef _ALICEO2_DCA_FITTERN_
#define _ALICEO2_DCA_FITTERN_

#include "DCAFitter/HelixHelper.h"
#include "DetectorsBase/Propagator.h"
#include "MathUtils/Cartesian.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace vertexing
{
///__________________________________________________________________________________
///< Inverse cov matrix (augmented by a dummy X error) of the point defined by the track
struct TrackCovI {
  float sxx, syy, syz, szz;

  GPUd() TrackCovI(const o2::track::TrackParCov& trc, float xerrFactor = 1.) { set(trc, xerrFactor); }

  GPUdDefault() TrackCovI() = default;

  GPUd() void set(const o2::track::TrackParCov& trc, float xerrFactor = 1)
  {
    // we assign Y error to X for DCA calculation
    // (otherwise for quazi-collinear tracks the X will not be constrained)
    float cyy = trc.getSigmaY2(), czz = trc.getSigmaZ2(), cyz = trc.getSigmaZY(), cxx = cyy * xerrFactor;
    float detYZ = cyy * czz - cyz * cyz;
    if (detYZ > 0.) {
      auto detYZI = 1. / detYZ;
      sxx = 1. / cxx;
      syy = czz * detYZI;
      syz = -cyz * detYZI;
      szz = cyy * detYZI;
    } else {
#ifndef GPUCA_GPUCODE
      throw std::runtime_error("invalid track covariance");
#else
      printf("invalid track covariance\n");
#endif
    }
  }
};

///__________________________________________________________________________
///< Derivative (up to 2) of the TrackParam position over its running param X
struct TrackDeriv {
  float dydx, dzdx, d2ydx2, d2zdx2;
  GPUdDefault() TrackDeriv() = default;
  GPUd() TrackDeriv(const o2::track::TrackPar& trc, float bz) { set(trc, bz); }
  GPUd() void set(const o2::track::TrackPar& trc, float bz)
  {
    float snp = trc.getSnp(), csp = o2::gpu::GPUCommonMath::Sqrt((1. - snp) * (1. + snp)), cspI = 1. / csp, crv2c = trc.getCurvature(bz) * cspI;
    dydx = snp * cspI;            // = snp/csp
    dzdx = trc.getTgl() * cspI;   // = tgl/csp
    d2ydx2 = crv2c * cspI * cspI; // = crv/csp^3
    d2zdx2 = crv2c * dzdx * dydx; // = crv*tgl*snp/csp^3
  }
};

template <int N, typename... Args>
class DCAFitterN
{
  static constexpr double NMin = 2;
  static constexpr double NMax = 4;
  static constexpr double NInv = 1. / N;
  static constexpr int MAXHYP = 2;
  static constexpr float XerrFactor = 5.; // factor for conversion of track covYY to dummy covXX
  using Track = o2::track::TrackParCov;
  using TrackAuxPar = o2::track::TrackAuxPar;
  using CrossInfo = o2::track::CrossInfo;

  using Vec3D = o2::math_utils::SVector<double, 3>;
  using VecND = o2::math_utils::SVector<double, N>;
  using MatSym3D = o2::math_utils::SMatrix<double, 3, 3, o2::math_utils::MatRepSym<double, 3>>;
  using MatStd3D = o2::math_utils::SMatrix<double, 3, 3, o2::math_utils::MatRepStd<double, 3>>;
  using MatSymND = o2::math_utils::SMatrix<double, N, N, o2::math_utils::MatRepSym<double, N>>;
  using MatStdND = o2::math_utils::SMatrix<double, N, N, o2::math_utils::MatRepStd<double, N>>;
  using TrackCoefVtx = MatStd3D;
  using ArrTrack = o2::gpu::gpustd::array<Track, N>;         // container for prongs (tracks) at single vertex cand.
  using ArrTrackCovI = o2::gpu::gpustd::array<TrackCovI, N>; // container for inv.cov.matrices at single vertex cand.
  using ArrTrCoef = o2::gpu::gpustd::array<TrackCoefVtx, N>; // container of TrackCoefVtx coefficients at single vertex cand.
  using ArrTrDer = o2::gpu::gpustd::array<TrackDeriv, N>;    // container of Track 1st and 2nd derivative over their X param
  using ArrTrPos = o2::gpu::gpustd::array<Vec3D, N>;         // container of Track positions

 public:
  static constexpr int getNProngs() { return N; }

  DCAFitterN() = default;
  DCAFitterN(float bz, bool useAbsDCA, bool prop2DCA) : mBz(bz), mUseAbsDCA(useAbsDCA), mPropagateToPCA(prop2DCA)
  {
    static_assert(N >= NMin && N <= NMax, "N prongs outside of allowed range");
  }

  //=========================================================================
  ///< return PCA candidate, by default best on is provided (no check for the index validity)
  GPUd() const Vec3D& getPCACandidate(int cand = 0) const { return mPCA[mOrder[cand]]; }
  GPUd() const auto getPCACandidatePos(int cand = 0) const
  {
    const auto& vd = mPCA[mOrder[cand]];
    return o2::gpu::gpustd::array<float, 3>{static_cast<float>(vd[0]), static_cast<float>(vd[1]), static_cast<float>(vd[2])};
  }

  ///< return position of quality-ordered candidate in the internal structures
  int getCandidatePosition(int cand = 0) const { return mOrder[cand]; }

  ///< return Chi2 at PCA candidate (no check for its validity)
  float getChi2AtPCACandidate(int cand = 0) const { return mChi2[mOrder[cand]]; }

  ///< prepare copies of tracks at the V0 candidate (no check for the candidate validity)
  ///  must be called before getTrack(i,cand) query
  GPUd() bool propagateTracksToVertex(int cand = 0);

  ///< check if propagation of tracks to candidate vertex was done
  GPUd() bool isPropagateTracksToVertexDone(int cand = 0) const { return mTrPropDone[mOrder[cand]]; }

  ///< check if propagation of tracks to candidate vertex was done
  bool isPropagationFailure(int cand = 0) const { return mPropFailed[mOrder[cand]]; }

  ///< track param propagated to V0 candidate (no check for the candidate validity)
  ///  propagateTracksToVertex must be called in advance
  Track& getTrack(int i, int cand = 0)
  {
    if (!mTrPropDone[mOrder[cand]]) {
#ifndef GPUCA_GPUCODE_DEVICE
      throw std::runtime_error("propagateTracksToVertex was not called yet");
#endif
    }
    return mCandTr[mOrder[cand]][i];
  }

  const Track& getTrack(int i, int cand = 0) const
  {
    if (!mTrPropDone[mOrder[cand]]) {
#ifndef GPUCA_GPUCODE_DEVICE
      throw std::runtime_error("propagateTracksToVertex was not called yet");
#endif
    }
    return mCandTr[mOrder[cand]][i];
  }

  ///< create parent track param with errors for decay vertex
  GPUd() o2::track::TrackParCov createParentTrackParCov(int cand = 0, bool sectorAlpha = true) const;

  ///< create parent track param w/o errors for decay vertex
  GPUd() o2::track::TrackPar createParentTrackPar(int cand = 0, bool sectorAlpha = true) const;

  ///< calculate on the fly track param (no cov mat) at candidate, check isValid to make sure propagation was successful
  GPUd() o2::track::TrackPar getTrackParamAtPCA(int i, int cand = 0);

  ///< recalculate PCA as a cov-matrix weighted mean, even if absDCA method was used
  GPUd() bool recalculatePCAWithErrors(int cand = 0);

  GPUd() MatSym3D calcPCACovMatrix(int cand = 0) const;

  o2::gpu::gpustd::array<float, 6> calcPCACovMatrixFlat(int cand = 0) const
  {
    auto m = calcPCACovMatrix(cand);
    return {static_cast<float>(m(0, 0)), static_cast<float>(m(1, 0)), static_cast<float>(m(1, 1)), static_cast<float>(m(2, 0)), static_cast<float>(m(2, 1)), static_cast<float>(m(2, 2))};
  }

  const Track* getOrigTrackPtr(int i) const { return mOrigTrPtr[i]; }

  ///< return number of iterations during minimization (no check for its validity)
  GPUdi() int getNIterations(int cand = 0) const { return mNIters[mOrder[cand]]; }
  GPUdi() void setPropagateToPCA(bool v = true) { mPropagateToPCA = v; }
  GPUdi() void setMaxIter(int n = 20) { mMaxIter = n > 2 ? n : 2; }
  GPUdi() void setMaxR(float r = 200.) { mMaxR2 = r * r; }
  GPUdi() void setMaxDZIni(float d = 4.) { mMaxDZIni = d; }
  GPUdi() void setMaxDXYIni(float d = 4.) { mMaxDXYIni = d > 0 ? d : 1e9; }
  GPUdi() void setMaxChi2(float chi2 = 999.) { mMaxChi2 = chi2; }
  GPUdi() void setBz(float bz) { mBz = o2::gpu::GPUCommonMath::Abs(bz) > o2::constants::math::Almost0 ? bz : 0.f; }
  GPUdi() void setMinParamChange(float x = 1e-3) { mMinParamChange = x > 1e-4 ? x : 1.e-4; }
  GPUdi() void setMinRelChi2Change(float r = 0.9) { mMinRelChi2Change = r > 0.1 ? r : 999.; }
  GPUdi() void setUseAbsDCA(bool v) { mUseAbsDCA = v; }
  GPUdi() void setWeightedFinalPCA(bool v) { mWeightedFinalPCA = v; }
  GPUdi() void setMaxDistance2ToMerge(float v) { mMaxDist2ToMergeSeeds = v; }
  GPUdi() void setMatCorrType(o2::base::Propagator::MatCorrType m = o2::base::Propagator::MatCorrType::USEMatCorrLUT) { mMatCorr = m; }
  GPUdi() void setUsePropagator(bool v) { mUsePropagator = v; }
  GPUdi() void setRefitWithMatCorr(bool v) { mRefitWithMatCorr = v; }
  GPUdi() void setMaxSnp(float s) { mMaxSnp = s; }
  GPUdi() void setMaxStep(float s) { mMaxStep = s; }
  GPUdi() void setMinXSeed(float x) { mMinXSeed = x; }
  GPUdi() void setCollinear(bool isCollinear) { mIsCollinear = isCollinear; }

  GPUdi() int getNCandidates() const { return mCurHyp; }
  GPUdi() int getMaxIter() const { return mMaxIter; }
  GPUdi() float getMaxR() const { return o2::gpu::GPUCommonMath::Sqrt(mMaxR2); }
  GPUdi() float getMaxDZIni() const { return mMaxDZIni; }
  GPUdi() float getMaxDXYIni() const { return mMaxDXYIni; }
  GPUdi() float getMaxChi2() const { return mMaxChi2; }
  GPUdi() float getMinParamChange() const { return mMinParamChange; }
  GPUdi() float getBz() const { return mBz; }
  GPUdi() float getMaxDistance2ToMerge() const { return mMaxDist2ToMergeSeeds; }
  GPUdi() bool getUseAbsDCA() const { return mUseAbsDCA; }
  GPUdi() bool getWeightedFinalPCA() const { return mWeightedFinalPCA; }
  GPUdi() bool getPropagateToPCA() const { return mPropagateToPCA; }
  GPUdi() o2::base::Propagator::MatCorrType getMatCorrType() const { return mMatCorr; }
  GPUdi() bool getUsePropagator() const { return mUsePropagator; }
  GPUdi() bool getRefitWithMatCorr() const { return mRefitWithMatCorr; }
  GPUdi() float getMaxSnp() const { return mMaxSnp; }
  GPUdi() float getMasStep() const { return mMaxStep; }
  GPUdi() float getMinXSeed() const { return mMinXSeed; }

  template <class... Tr>
  GPUd() int process(const Tr&... args);
  GPUd() void print() const;

  GPUdi() int getFitterID() const { return mFitterID; }
  GPUdi() void setFitterID(int i) { mFitterID = i; }
  GPUdi() size_t getCallID() const { return mCallID; }

 protected:
  GPUd() bool calcPCACoefs();
  GPUd() bool calcInverseWeight();
  GPUd() void calcResidDerivatives();
  GPUd() void calcResidDerivativesNoErr();
  GPUd() void calcRMatrices();
  GPUd() void calcChi2Derivatives();
  GPUd() void calcChi2DerivativesNoErr();
  GPUd() void calcPCA();
  GPUd() void calcPCANoErr();
  GPUd() void calcTrackResiduals();
  GPUd() void calcTrackDerivatives();
  GPUd() double calcChi2() const;
  GPUd() double calcChi2NoErr() const;
  GPUd() bool correctTracks(const VecND& corrX);
  GPUd() bool minimizeChi2();
  GPUd() bool minimizeChi2NoErr();
  GPUd() bool roughDZCut() const;
  GPUd() bool closerToAlternative() const;
  GPUd() bool propagateToX(o2::track::TrackParCov& t, float x);
  GPUd() bool propagateParamToX(o2::track::TrackPar& t, float x);

  GPUd() static double getAbsMax(const VecND& v);
  ///< track param positions at V0 candidate (no check for the candidate validity)
  GPUdi() const Vec3D& getTrackPos(int i, int cand = 0) const { return mTrPos[mOrder[cand]][i]; }

  ///< track X-param at V0 candidate (no check for the candidate validity)
  GPUd() float getTrackX(int i, int cand = 0) const { return getTrackPos(i, cand)[0]; }

  GPUd() MatStd3D getTrackRotMatrix(int i) const // generate 3D matrix for track rotation to global frame
  {
    MatStd3D mat;
    mat(2, 2) = 1;
    mat(0, 0) = mat(1, 1) = mTrAux[i].c;
    mat(0, 1) = -mTrAux[i].s;
    mat(1, 0) = mTrAux[i].s;
    return mat;
  }

  GPUd() MatSym3D getTrackCovMatrix(int i, int cand = 0) const // generate covariance matrix of track position, adding fake X error
  {
    const auto& trc = mCandTr[mOrder[cand]][i];
    MatSym3D mat;
    mat(0, 0) = trc.getSigmaY2() * XerrFactor;
    mat(1, 1) = trc.getSigmaY2();
    mat(2, 2) = trc.getSigmaZ2();
    mat(2, 1) = trc.getSigmaZY();
    return mat;
  }

  GPUd() void assign(int) {}
  template <class T, class... Tr>
  GPUd() void assign(int i, const T& t, const Tr&... args)
  {
#ifndef GPUCA_GPUCODE_DEVICE
    static_assert(std::is_convertible<T, Track>(), "Wrong track type");
#endif
    mOrigTrPtr[i] = &t;
    assign(i + 1, args...);
  }

  GPUdi() void clear()
  {
    mCurHyp = 0;
    mAllowAltPreference = true;
  }

  GPUdi() static void setTrackPos(Vec3D& pnt, const Track& tr)
  {
    pnt[0] = tr.getX();
    pnt[1] = tr.getY();
    pnt[2] = tr.getZ();
  }

 private:
  // vectors of 1st derivatives of track local residuals over X parameters
  o2::gpu::gpustd::array<o2::gpu::gpustd::array<Vec3D, N>, N> mDResidDx;
  // vectors of 1nd derivatives of track local residuals over X parameters
  // (cross-derivatives DR/(dx_j*dx_k) = 0 for j!=k, therefore the hessian is diagonal)
  o2::gpu::gpustd::array<o2::gpu::gpustd::array<Vec3D, N>, N> mD2ResidDx2;
  VecND mDChi2Dx;      // 1st derivatives of chi2 over tracks X params
  MatSymND mD2Chi2Dx2; // 2nd derivatives of chi2 over tracks X params (symmetric matrix)
  MatSymND mCosDif;    // matrix with cos(alp_j-alp_i) for j<i
  MatSymND mSinDif;    // matrix with sin(alp_j-alp_i) for j<i
  o2::gpu::gpustd::array<const Track*, N> mOrigTrPtr;
  o2::gpu::gpustd::array<TrackAuxPar, N> mTrAux; // Aux track info for each track at each cand. vertex
  CrossInfo mCrossings;                          // info on track crossing

  o2::gpu::gpustd::array<ArrTrackCovI, MAXHYP> mTrcEInv; // errors for each track at each cand. vertex
  o2::gpu::gpustd::array<ArrTrack, MAXHYP> mCandTr;      // tracks at each cond. vertex (Note: Errors are at seed XY point)
  o2::gpu::gpustd::array<ArrTrCoef, MAXHYP> mTrCFVT;     // TrackCoefVtx for each track at each cand. vertex
  o2::gpu::gpustd::array<ArrTrDer, MAXHYP> mTrDer;       // Track derivativse
  o2::gpu::gpustd::array<ArrTrPos, MAXHYP> mTrPos;       // Track positions
  o2::gpu::gpustd::array<ArrTrPos, MAXHYP> mTrRes;       // Track residuals
  o2::gpu::gpustd::array<Vec3D, MAXHYP> mPCA;            // PCA for each vertex candidate
  o2::gpu::gpustd::array<float, MAXHYP> mChi2 = {0};     // Chi2 at PCA candidate
  o2::gpu::gpustd::array<int, MAXHYP> mNIters;           // number of iterations for each seed
  o2::gpu::gpustd::array<bool, MAXHYP> mTrPropDone{};    // Flag that the tracks are fully propagated to PCA
  o2::gpu::gpustd::array<bool, MAXHYP> mPropFailed{};    // Flag that some propagation failed for this PCA candidate
  MatSym3D mWeightInv;                                   // inverse weight of single track, [sum{M^T E M}]^-1 in EQ.T
  o2::gpu::gpustd::array<int, MAXHYP> mOrder{0};
  int mCurHyp = 0;
  int mCrossIDCur = 0;
  int mCrossIDAlt = -1;
  bool mAllowAltPreference = true;                                                                // if the fit converges to alternative PCA seed, abandon the current one
  bool mUseAbsDCA = false;                                                                        // use abs. distance minimization rather than chi2
  bool mWeightedFinalPCA = false;                                                                 // recalculate PCA as a cov-matrix weighted mean, even if absDCA method was used
  bool mPropagateToPCA = true;                                                                    // create tracks version propagated to PCA
  bool mUsePropagator = false;                                                                    // use propagator with 3D B-field, set automatically if material correction is requested
  bool mRefitWithMatCorr = false;                                                                 // when doing propagateTracksToVertex, propagate tracks to V0 with material corrections and rerun minimization again
  bool mIsCollinear = false;                                                                      // use collinear fits when there 2 crossing points
  o2::base::Propagator::MatCorrType mMatCorr = o2::base::Propagator::MatCorrType::USEMatCorrNONE; // material corrections type
  int mMaxIter = 20;                                                                              // max number of iterations
  float mBz = 0;                                                                                  // bz field, to be set by user
  float mMaxR2 = 200. * 200.;                                                                     // reject PCA's above this radius
  float mMinXSeed = -50.;                                                                         // reject seed if it corresponds to X-param < mMinXSeed for one of candidates (e.g. X becomes strongly negative)
  float mMaxDZIni = 4.;                                                                           // reject (if>0) PCA candidate if tracks DZ exceeds threshold
  float mMaxDXYIni = 4.;                                                                          // reject (if>0) PCA candidate if tracks dXY exceeds threshold
  float mMinParamChange = 1e-3;                                                                   // stop iterations if largest change of any X is smaller than this
  float mMinRelChi2Change = 0.9;                                                                  // stop iterations is chi2/chi2old > this
  float mMaxChi2 = 100;                                                                           // abs cut on chi2 or abs distance
  float mMaxDist2ToMergeSeeds = 1.;                                                               // merge 2 seeds to their average if their distance^2 is below the threshold
  float mMaxSnp = 0.95;                                                                           // Max snp for propagation with Propagator
  float mMaxStep = 2.0;                                                                           // Max step for propagation with Propagator
  int mFitterID = 0;                                                                              // locat fitter ID (mostly for debugging)
  size_t mCallID = 0;
  ClassDefNV(DCAFitterN, 2);
};

///_________________________________________________________________________
template <int N, typename... Args>
template <class... Tr>
GPUd() int DCAFitterN<N, Args...>::process(const Tr&... args)
{
  // This is a main entry point: fit PCA of N tracks
  mCallID++;
  static_assert(sizeof...(args) == N, "incorrect number of input tracks");
  assign(0, args...);
  clear();
  for (int i = 0; i < N; i++) {
    mTrAux[i].set(*mOrigTrPtr[i], mBz);
  }
  if (!mCrossings.set(mTrAux[0], *mOrigTrPtr[0], mTrAux[1], *mOrigTrPtr[1], mMaxDXYIni, mIsCollinear)) { // even for N>2 it should be enough to test just 1 loop
    return 0;                                                                                            // no crossing
  }
  for (int ih = 0; ih < MAXHYP; ih++) {
    mPropFailed[ih] = false;
  }
  if (mUseAbsDCA) {
    calcRMatrices(); // needed for fast residuals derivatives calculation in case of abs. distance minimization
  }
  if (mCrossings.nDCA == MAXHYP) { // if there are 2 candidates and they are too close, chose their mean as a starting point
    auto dst2 = (mCrossings.xDCA[0] - mCrossings.xDCA[1]) * (mCrossings.xDCA[0] - mCrossings.xDCA[1]) +
                (mCrossings.yDCA[0] - mCrossings.yDCA[1]) * (mCrossings.yDCA[0] - mCrossings.yDCA[1]);
    if (dst2 < mMaxDist2ToMergeSeeds) {
      mCrossings.nDCA = 1;
      mCrossings.xDCA[0] = 0.5 * (mCrossings.xDCA[0] + mCrossings.xDCA[1]);
      mCrossings.yDCA[0] = 0.5 * (mCrossings.yDCA[0] + mCrossings.yDCA[1]);
    }
  }
  // check all crossings
  for (int ic = 0; ic < mCrossings.nDCA; ic++) {
    // check if radius is acceptable
    if (mCrossings.xDCA[ic] * mCrossings.xDCA[ic] + mCrossings.yDCA[ic] * mCrossings.yDCA[ic] > mMaxR2) {
      continue;
    }
    mCrossIDCur = ic;
    mCrossIDAlt = (mCrossings.nDCA == 2 && mAllowAltPreference) ? 1 - ic : -1; // works for max 2 crossings
    mNIters[mCurHyp] = 0;
    mTrPropDone[mCurHyp] = false;
    mChi2[mCurHyp] = -1.;
    mPCA[mCurHyp][0] = mCrossings.xDCA[ic];
    mPCA[mCurHyp][1] = mCrossings.yDCA[ic];

    if (mUseAbsDCA ? minimizeChi2NoErr() : minimizeChi2()) {
      mOrder[mCurHyp] = mCurHyp;
      if (mPropagateToPCA && !propagateTracksToVertex(mCurHyp)) {
        continue; // discard candidate if failed to propagate to it
      }
      mCurHyp++;
    }
  }

  for (int i = mCurHyp; i--;) { // order in quality
    for (int j = i; j--;) {
      if (mChi2[mOrder[i]] < mChi2[mOrder[j]]) {
        o2::gpu::GPUCommonMath::Swap(mOrder[i], mOrder[j]);
      }
    }
  }
  if (mUseAbsDCA && mWeightedFinalPCA) {
    for (int i = mCurHyp; i--;) {
      recalculatePCAWithErrors(i);
    }
  }
  return mCurHyp;
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::calcPCACoefs()
{
  //< calculate Ti matrices for global vertex decomposition to V = sum_{0<i<N} Ti pi, see EQ.T in the ref
  if (!calcInverseWeight()) {
    return false;
  }
  for (int i = N; i--;) { // build Mi*Ei matrix
    const auto& taux = mTrAux[i];
    const auto& tcov = mTrcEInv[mCurHyp][i];
    MatStd3D miei;
    miei[0][0] = taux.c * tcov.sxx;
    miei[0][1] = -taux.s * tcov.syy;
    miei[0][2] = -taux.s * tcov.syz;
    miei[1][0] = taux.s * tcov.sxx;
    miei[1][1] = taux.c * tcov.syy;
    miei[1][2] = taux.c * tcov.syz;
    miei[2][0] = 0;
    miei[2][1] = tcov.syz;
    miei[2][2] = tcov.szz;
    mTrCFVT[mCurHyp][i] = mWeightInv * miei;
  }
  return true;
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::calcInverseWeight()
{
  //< calculate [sum_{0<j<N} M_j*E_j*M_j^T]^-1 used for Ti matrices, see EQ.T
  auto* arrmat = mWeightInv.Array();
  memset(arrmat, 0, sizeof(mWeightInv));
  enum { XX,
         XY,
         YY,
         XZ,
         YZ,
         ZZ };
  for (int i = N; i--;) {
    const auto& taux = mTrAux[i];
    const auto& tcov = mTrcEInv[mCurHyp][i];
    arrmat[XX] += taux.cc * tcov.sxx + taux.ss * tcov.syy;
    arrmat[XY] += taux.cs * (tcov.sxx - tcov.syy);
    arrmat[XZ] += -taux.s * tcov.syz;
    arrmat[YY] += taux.cc * tcov.syy + taux.ss * tcov.sxx;
    arrmat[YZ] += taux.c * tcov.syz;
    arrmat[ZZ] += tcov.szz;
  }
  // invert 3x3 symmetrix matrix
  return mWeightInv.Invert();
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcResidDerivatives()
{
  //< calculate matrix of derivatives for weighted chi2: residual i vs parameter X of track j
  MatStd3D matMT;
  for (int i = N; i--;) { // residual being differentiated
    const auto& taux = mTrAux[i];
    for (int j = N; j--;) {                   // track over which we differentiate
      const auto& matT = mTrCFVT[mCurHyp][j]; // coefficient matrix for track J
      const auto& trDx = mTrDer[mCurHyp][j];  // track point derivs over track X param
      auto& dr1 = mDResidDx[i][j];
      auto& dr2 = mD2ResidDx2[i][j];
      // calculate M_i^tr * T_j
      matMT[0][0] = taux.c * matT[0][0] + taux.s * matT[1][0];
      matMT[0][1] = taux.c * matT[0][1] + taux.s * matT[1][1];
      matMT[0][2] = taux.c * matT[0][2] + taux.s * matT[1][2];
      matMT[1][0] = -taux.s * matT[0][0] + taux.c * matT[1][0];
      matMT[1][1] = -taux.s * matT[0][1] + taux.c * matT[1][1];
      matMT[1][2] = -taux.s * matT[0][2] + taux.c * matT[1][2];
      matMT[2][0] = matT[2][0];
      matMT[2][1] = matT[2][1];
      matMT[2][2] = matT[2][2];

      // calculate DResid_i/Dx_j = (delta_ij - M_i^tr * T_j) * DTrack_k/Dx_k
      dr1[0] = -(matMT[0][0] + matMT[0][1] * trDx.dydx + matMT[0][2] * trDx.dzdx);
      dr1[1] = -(matMT[1][0] + matMT[1][1] * trDx.dydx + matMT[1][2] * trDx.dzdx);
      dr1[2] = -(matMT[2][0] + matMT[2][1] * trDx.dydx + matMT[2][2] * trDx.dzdx);

      // calculate D2Resid_I/(Dx_J Dx_K) = (delta_ijk - M_i^tr * T_j * delta_jk) * D2Track_k/dx_k^2
      dr2[0] = -(matMT[0][1] * trDx.d2ydx2 + matMT[0][2] * trDx.d2zdx2);
      dr2[1] = -(matMT[1][1] * trDx.d2ydx2 + matMT[1][2] * trDx.d2zdx2);
      dr2[2] = -(matMT[2][1] * trDx.d2ydx2 + matMT[2][2] * trDx.d2zdx2);

      if (i == j) {
        dr1[0] += 1.;
        dr1[1] += trDx.dydx;
        dr1[2] += trDx.dzdx;

        dr2[1] += trDx.d2ydx2;
        dr2[2] += trDx.d2zdx2;
      }
    } // track over which we differentiate
  } // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcResidDerivativesNoErr()
{
  //< calculate matrix of derivatives for absolute distance chi2: residual i vs parameter X of track j
  constexpr double NInv1 = 1. - NInv;       // profit from Rii = I/Ninv
  for (int i = N; i--;) {                   // residual being differentiated
    const auto& trDxi = mTrDer[mCurHyp][i]; // track point derivs over track X param
    auto& dr1ii = mDResidDx[i][i];
    auto& dr2ii = mD2ResidDx2[i][i];
    dr1ii[0] = NInv1;
    dr1ii[1] = NInv1 * trDxi.dydx;
    dr1ii[2] = NInv1 * trDxi.dzdx;

    dr2ii[0] = 0;
    dr2ii[1] = NInv1 * trDxi.d2ydx2;
    dr2ii[2] = NInv1 * trDxi.d2zdx2;

    for (int j = i; j--;) { // track over which we differentiate
      auto& dr1ij = mDResidDx[i][j];
      auto& dr1ji = mDResidDx[j][i];
      const auto& trDxj = mTrDer[mCurHyp][j];        // track point derivs over track X param
      auto cij = mCosDif[i][j], sij = mSinDif[i][j]; // M_i^T*M_j / N matrices non-trivial elements = {ci*cj+si*sj , si*cj-ci*sj }, see 5 in ref.

      // calculate DResid_i/Dx_j = (delta_ij - R_ij) * DTrack_j/Dx_j  for j<i
      dr1ij[0] = -(cij + sij * trDxj.dydx);
      dr1ij[1] = -(-sij + cij * trDxj.dydx);
      dr1ij[2] = -trDxj.dzdx * NInv;

      // calculate DResid_j/Dx_i = (delta_ij - R_ji) * DTrack_i/Dx_i  for j<i
      dr1ji[0] = -(cij - sij * trDxi.dydx);
      dr1ji[1] = -(sij + cij * trDxi.dydx);
      dr1ji[2] = -trDxi.dzdx * NInv;

      auto& dr2ij = mD2ResidDx2[i][j];
      auto& dr2ji = mD2ResidDx2[j][i];
      // calculate D2Resid_I/(Dx_J Dx_K) = (delta_ij - Rij) * D2Track_j/dx_j^2 * delta_jk for j<i
      dr2ij[0] = -sij * trDxj.d2ydx2;
      dr2ij[1] = -cij * trDxj.d2ydx2;
      dr2ij[2] = -trDxj.d2zdx2 * NInv;

      // calculate D2Resid_j/(Dx_i Dx_k) = (delta_ij - Rji) * D2Track_i/dx_i^2 * delta_ik for j<i
      dr2ji[0] = sij * trDxi.d2ydx2;
      dr2ji[1] = -cij * trDxi.d2ydx2;
      dr2ji[2] = -trDxi.d2zdx2 * NInv;

    } // track over which we differentiate
  } // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcRMatrices()
{
  //< calculate Rij = 1/N M_i^T * M_j matrices (rotation from j-th track to i-th track frame)
  for (int i = N; i--;) {
    const auto& mi = mTrAux[i];
    for (int j = i; j--;) {
      const auto& mj = mTrAux[j];
      mCosDif[i][j] = (mi.c * mj.c + mi.s * mj.s) * NInv; // cos(alp_i-alp_j) / N
      mSinDif[i][j] = (mi.s * mj.c - mi.c * mj.s) * NInv; // sin(alp_i-alp_j) / N
    }
  }
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcChi2Derivatives()
{
  //< calculate 1st and 2nd derivatives of wighted DCA (chi2) over track parameters X, see EQ.Chi2 in the ref
  o2::gpu::gpustd::array<o2::gpu::gpustd::array<Vec3D, N>, N> covIDrDx; // tempory vectors of covI_j * dres_j/dx_i

  // chi2 1st derivative
  for (int i = N; i--;) {
    auto& dchi1 = mDChi2Dx[i]; // DChi2/Dx_i = sum_j { res_j * covI_j * Dres_j/Dx_i }
    dchi1 = 0;
    for (int j = N; j--;) {
      const auto& res = mTrRes[mCurHyp][j];    // vector of residuals of track j
      const auto& covI = mTrcEInv[mCurHyp][j]; // inverse cov matrix of track j
      const auto& dr1 = mDResidDx[j][i];       // vector of j-th residuals 1st derivative over X param of track i
      auto& cidr = covIDrDx[i][j];             // vector covI_j * dres_j/dx_i, save for 2nd derivative calculation
      cidr[0] = covI.sxx * dr1[0];
      cidr[1] = covI.syy * dr1[1] + covI.syz * dr1[2];
      cidr[2] = covI.syz * dr1[1] + covI.szz * dr1[2];
      // calculate res_i * covI_j * dres_j/dx_i
      dchi1 += o2::math_utils::Dot(res, cidr);
    }
  }
  // chi2 2nd derivative
  for (int i = N; i--;) {
    for (int j = i + 1; j--;) {       // symmetric matrix
      auto& dchi2 = mD2Chi2Dx2[i][j]; // D2Chi2/Dx_i/Dx_j = sum_k { Dres_k/Dx_j * covI_k * Dres_k/Dx_i + res_k * covI_k * D2res_k/Dx_i/Dx_j }
      dchi2 = 0;
      for (int k = N; k--;) {
        const auto& dr1j = mDResidDx[k][j];  // vector of k-th residuals 1st derivative over X param of track j
        const auto& cidrkj = covIDrDx[i][k]; // vector covI_k * dres_k/dx_i
        dchi2 += o2::math_utils::Dot(dr1j, cidrkj);
        if (k == j) {
          const auto& res = mTrRes[mCurHyp][k];    // vector of residuals of track k
          const auto& covI = mTrcEInv[mCurHyp][k]; // inverse cov matrix of track k
          const auto& dr2ij = mD2ResidDx2[k][j];   // vector of k-th residuals 2nd derivative over X params of track j
          dchi2 += res[0] * covI.sxx * dr2ij[0] + res[1] * (covI.syy * dr2ij[1] + covI.syz * dr2ij[2]) + res[2] * (covI.syz * dr2ij[1] + covI.szz * dr2ij[2]);
        }
      }
    }
  }
}

//__________________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcChi2DerivativesNoErr()
{
  //< calculate 1st and 2nd derivatives of abs DCA (chi2) over track parameters X, see (6) in the ref
  for (int i = N; i--;) {
    auto& dchi1 = mDChi2Dx[i]; // DChi2/Dx_i = sum_j { res_j * Dres_j/Dx_i }
    dchi1 = 0;                 // chi2 1st derivative
    for (int j = N; j--;) {
      const auto& res = mTrRes[mCurHyp][j]; // vector of residuals of track j
      const auto& dr1 = mDResidDx[j][i];    // vector of j-th residuals 1st derivative over X param of track i
      dchi1 += o2::math_utils::Dot(res, dr1);
      if (i >= j) { // symmetrix matrix
        // chi2 2nd derivative
        auto& dchi2 = mD2Chi2Dx2[i][j]; // D2Chi2/Dx_i/Dx_j = sum_k { Dres_k/Dx_j * covI_k * Dres_k/Dx_i + res_k * covI_k * D2res_k/Dx_i/Dx_j }
        dchi2 = o2::math_utils::Dot(mTrRes[mCurHyp][i], mD2ResidDx2[i][j]);
        for (int k = N; k--;) {
          dchi2 += o2::math_utils::Dot(mDResidDx[k][i], mDResidDx[k][j]);
        }
      }
    }
  }
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcPCA()
{
  // calculate point of closest approach for N prongs
  mPCA[mCurHyp] = mTrCFVT[mCurHyp][N - 1] * mTrPos[mCurHyp][N - 1];
  for (int i = N - 1; i--;) {
    mPCA[mCurHyp] += mTrCFVT[mCurHyp][i] * mTrPos[mCurHyp][i];
  }
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::recalculatePCAWithErrors(int cand)
{
  // recalculate PCA as a cov-matrix weighted mean, even if absDCA method was used
  if (isPropagateTracksToVertexDone(cand) && !propagateTracksToVertex(cand)) {
    return false;
  }
  int saveCurHyp = mCurHyp;
  mCurHyp = mOrder[cand];
  if (mUseAbsDCA) {
    for (int i = N; i--;) {
      mTrcEInv[mCurHyp][i].set(mCandTr[mCurHyp][i], XerrFactor); // prepare inverse cov.matrices at starting point
    }
    if (!calcPCACoefs()) {
      mCurHyp = saveCurHyp;
      return false;
    }
  }
  auto oldPCA = mPCA[mOrder[cand]];
  calcPCA();
  mCurHyp = saveCurHyp;
  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcPCANoErr()
{
  // calculate point of closest approach for N prongs w/o errors
  auto& pca = mPCA[mCurHyp];
  o2::math_utils::rotateZd(mTrPos[mCurHyp][N - 1][0], mTrPos[mCurHyp][N - 1][1], pca[0], pca[1], mTrAux[N - 1].s, mTrAux[N - 1].c);
  // RRRR    mTrAux[N-1].loc2glo(mTrPos[mCurHyp][N-1][0], mTrPos[mCurHyp][N-1][1], pca[0], pca[1] );
  pca[2] = mTrPos[mCurHyp][N - 1][2];
  for (int i = N - 1; i--;) {
    double x, y;
    o2::math_utils::rotateZd(mTrPos[mCurHyp][i][0], mTrPos[mCurHyp][i][1], x, y, mTrAux[i].s, mTrAux[i].c);
    // RRRR mTrAux[i].loc2glo(mTrPos[mCurHyp][i][0], mTrPos[mCurHyp][i][1], x, y );
    pca[0] += x;
    pca[1] += y;
    pca[2] += mTrPos[mCurHyp][i][2];
  }
  pca[0] *= NInv;
  pca[1] *= NInv;
  pca[2] *= NInv;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() o2::math_utils::SMatrix<double, 3, 3, o2::math_utils::MatRepSym<double, 3>> DCAFitterN<N, Args...>::calcPCACovMatrix(int cand) const
{
  // calculate covariance matrix for the point of closest approach
  MatSym3D covm;
  int nAdded = 0;
  for (int i = N; i--;) { // calculate sum of inverses
    // MatSym3D covTr = o2::math_utils::Similarity(mUseAbsDCA ? getTrackRotMatrix(i) : mTrCFVT[mOrder[cand]][i], getTrackCovMatrix(i, cand));
    // RS by using Similarity(mTrCFVT[mOrder[cand]][i], getTrackCovMatrix(i, cand)) we underestimate the error, use simple rotation
    MatSym3D covTr = o2::math_utils::Similarity(getTrackRotMatrix(i), getTrackCovMatrix(i, cand));
    if (covTr.Invert()) {
      covm += covTr;
      nAdded++;
    }
  }
  if (nAdded && covm.Invert()) {
    return covm;
  }
  // correct way has failed, use simple sum
  MatSym3D covmSum;
  for (int i = N; i--;) {
    MatSym3D covTr = o2::math_utils::Similarity(getTrackRotMatrix(i), getTrackCovMatrix(i, cand));
  }
  return covmSum;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::calcTrackResiduals()
{
  // calculate residuals
  Vec3D vtxLoc;
  for (int i = N; i--;) {
    mTrRes[mCurHyp][i] = mTrPos[mCurHyp][i];
    vtxLoc = mPCA[mCurHyp];
    o2::math_utils::rotateZInvd(vtxLoc[0], vtxLoc[1], vtxLoc[0], vtxLoc[1], mTrAux[i].s, mTrAux[i].c); // glo->loc
    mTrRes[mCurHyp][i] -= vtxLoc;
  }
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() void DCAFitterN<N, Args...>::calcTrackDerivatives()
{
  // calculate track derivatives over X param
  for (int i = N; i--;) {
    mTrDer[mCurHyp][i].set(mCandTr[mCurHyp][i], mBz);
  }
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() double DCAFitterN<N, Args...>::calcChi2() const
{
  // calculate current chi2
  double chi2 = 0;
  for (int i = N; i--;) {
    const auto& res = mTrRes[mCurHyp][i];
    const auto& covI = mTrcEInv[mCurHyp][i];
    chi2 += res[0] * res[0] * covI.sxx + res[1] * res[1] * covI.syy + res[2] * res[2] * covI.szz + 2. * res[1] * res[2] * covI.syz;
  }
  return chi2;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() double DCAFitterN<N, Args...>::calcChi2NoErr() const
{
  // calculate current chi2 of abs. distance minimization
  double chi2 = 0;
  for (int i = N; i--;) {
    const auto& res = mTrRes[mCurHyp][i];
    chi2 += res[0] * res[0] + res[1] * res[1] + res[2] * res[2];
  }
  return chi2;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::correctTracks(const VecND& corrX)
{
  // propagate tracks to updated X
  for (int i = N; i--;) {
    const auto& trDer = mTrDer[mCurHyp][i];
    auto dx2h = 0.5 * corrX[i] * corrX[i];
    mTrPos[mCurHyp][i][0] -= corrX[i];
    mTrPos[mCurHyp][i][1] -= trDer.dydx * corrX[i] - dx2h * trDer.d2ydx2;
    mTrPos[mCurHyp][i][2] -= trDer.dzdx * corrX[i] - dx2h * trDer.d2zdx2;
  }
  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::propagateTracksToVertex(int icand)
{
  // propagate tracks to current vertex
  int ord = mOrder[icand];
  if (mTrPropDone[ord]) {
    return true;
  }

  // need to refit taking as a seed already found vertex
  if (mRefitWithMatCorr) {
    int curHypSav = mCurHyp, curCrosIDAlt = mCrossIDAlt; // save
    mCurHyp = ord;
    mCrossIDAlt = -1; // disable alternative check
    auto restore = [this, curHypSav, curCrosIDAlt]() { this->mCurHyp = curHypSav; this->mCrossIDAlt = curCrosIDAlt; };
    if (!(mUseAbsDCA ? minimizeChi2NoErr() : minimizeChi2())) { // do final propagation
      restore();
      return false;
    }
    restore();
  }

  for (int i = N; i--;) {
    if (mUseAbsDCA || mUsePropagator || mMatCorr != o2::base::Propagator::MatCorrType::USEMatCorrNONE) {
      mCandTr[ord][i] = *mOrigTrPtr[i]; // fetch the track again, as mCandTr might have been propagated w/o errors or material corrections might be wrong
    }
    auto x = mTrAux[i].c * mPCA[ord][0] + mTrAux[i].s * mPCA[ord][1]; // X of PCA in the track frame
    if (!propagateToX(mCandTr[ord][i], x)) {
      return false;
    }
  }

  mTrPropDone[ord] = true;
  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() o2::track::TrackPar DCAFitterN<N, Args...>::getTrackParamAtPCA(int i, int icand)
{
  // propagate tracks param only to current vertex (if not already done)
  int ord = mOrder[icand];
  o2::track::TrackPar trc(mCandTr[ord][i]);
  if (!mTrPropDone[ord]) {
    auto x = mTrAux[i].c * mPCA[ord][0] + mTrAux[i].s * mPCA[ord][1]; // X of PCA in the track frame
    if (!propagateParamToX(trc, x)) {
      trc.invalidate();
    }
  }
  return trc;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() double DCAFitterN<N, Args...>::getAbsMax(const VecND& v)
{
  double mx = -1;
  for (int i = N; i--;) {
    auto vai = o2::gpu::GPUCommonMath::Abs(v[i]);
    if (mx < vai) {
      mx = vai;
    }
  }
  return mx;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::minimizeChi2()
{
  // find best chi2 (weighted DCA) of N tracks in the vicinity of the seed PCA
  for (int i = N; i--;) {
    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];
    auto x = mTrAux[i].c * mPCA[mCurHyp][0] + mTrAux[i].s * mPCA[mCurHyp][1]; // X of PCA in the track frame
    if (x < mMinXSeed || !propagateToX(mCandTr[mCurHyp][i], x)) {
      return false;
    }
    setTrackPos(mTrPos[mCurHyp][i], mCandTr[mCurHyp][i]);      // prepare positions
    mTrcEInv[mCurHyp][i].set(mCandTr[mCurHyp][i], XerrFactor); // prepare inverse cov.matrices at starting point
  }

  if (mMaxDZIni > 0 && !roughDZCut()) { // apply rough cut on tracks Z difference
    return false;
  }

  if (!calcPCACoefs()) { // prepare tracks contribution matrices to the global PCA
    return false;
  }
  calcPCA();            // current PCA
  calcTrackResiduals(); // current track residuals
  float chi2Upd, chi2 = calcChi2();
  do {
    calcTrackDerivatives(); // current track derivatives (1st and 2nd)
    calcResidDerivatives(); // current residals derivatives (1st and 2nd)
    calcChi2Derivatives();  // current chi2 derivatives (1st and 2nd)

    // do Newton-Rapson iteration with corrections = - dchi2/d{x0..xN} * [ d^2chi2/d{x0..xN}^2 ]^-1
    if (!mD2Chi2Dx2.Invert()) {
#ifndef GPUCA_GPUCODE_DEVICE
      LOG(error) << "InversionFailed";
#else
      printf("InversionFailed\n");
#endif
      return false;
    }
    VecND dx = mD2Chi2Dx2 * mDChi2Dx;
    if (!correctTracks(dx)) {
      return false;
    }
    calcPCA(); // updated PCA
    if (mCrossIDAlt >= 0 && closerToAlternative()) {
      mAllowAltPreference = false;
      return false;
    }
    calcTrackResiduals(); // updated residuals
    chi2Upd = calcChi2(); // updated chi2
    if (getAbsMax(dx) < mMinParamChange || chi2Upd > chi2 * mMinRelChi2Change) {
      chi2 = chi2Upd;
      break; // converged
    }
    chi2 = chi2Upd;
  } while (++mNIters[mCurHyp] < mMaxIter);
  //
  mChi2[mCurHyp] = chi2 * NInv;
  return mChi2[mCurHyp] < mMaxChi2;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::minimizeChi2NoErr()
{
  // find best chi2 (absolute DCA) of N tracks in the vicinity of the PCA seed

  for (int i = N; i--;) {
    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];
    auto x = mTrAux[i].c * mPCA[mCurHyp][0] + mTrAux[i].s * mPCA[mCurHyp][1]; // X of PCA in the track frame
    if (x < mMinXSeed || !propagateParamToX(mCandTr[mCurHyp][i], x)) {
      return false;
    }
    setTrackPos(mTrPos[mCurHyp][i], mCandTr[mCurHyp][i]); // prepare positions
  }
  if (mMaxDZIni > 0 && !roughDZCut()) { // apply rough cut on tracks Z difference
    return false;
  }

  calcPCANoErr();       // current PCA
  calcTrackResiduals(); // current track residuals
  float chi2Upd, chi2 = calcChi2NoErr();
  do {
    calcTrackDerivatives();      // current track derivatives (1st and 2nd)
    calcResidDerivativesNoErr(); // current residals derivatives (1st and 2nd)
    calcChi2DerivativesNoErr();  // current chi2 derivatives (1st and 2nd)

    // do Newton-Rapson iteration with corrections = - dchi2/d{x0..xN} * [ d^2chi2/d{x0..xN}^2 ]^-1
    if (!mD2Chi2Dx2.Invert()) {
#ifndef GPUCA_GPUCODE_DEVICE
      LOG(error) << "InversionFailed";
#else
      printf("InversionFailed\n");
#endif
      return false;
    }
    VecND dx = mD2Chi2Dx2 * mDChi2Dx;
    if (!correctTracks(dx)) {
      return false;
    }
    calcPCANoErr(); // updated PCA
    if (mCrossIDAlt >= 0 && closerToAlternative()) {
      mAllowAltPreference = false;
      return false;
    }
    calcTrackResiduals();      // updated residuals
    chi2Upd = calcChi2NoErr(); // updated chi2
    if (getAbsMax(dx) < mMinParamChange || chi2Upd > chi2 * mMinRelChi2Change) {
      chi2 = chi2Upd;
      break; // converged
    }
    chi2 = chi2Upd;
  } while (++mNIters[mCurHyp] < mMaxIter);
  //
  mChi2[mCurHyp] = chi2 * NInv;
  return mChi2[mCurHyp] < mMaxChi2;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::roughDZCut() const
{
  // apply rough cut on DZ between the tracks in the seed point
  bool accept = true;
  for (int i = N; accept && i--;) {
    for (int j = i; j--;) {
      if (o2::gpu::GPUCommonMath::Abs(mCandTr[mCurHyp][i].getZ() - mCandTr[mCurHyp][j].getZ()) > mMaxDZIni) {
        accept = false;
        break;
      }
    }
  }
  return accept;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() bool DCAFitterN<N, Args...>::closerToAlternative() const
{
  // check if the point current PCA point is closer to the seeding XY point being tested or to alternative see (if any)
  auto dxCur = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDCur], dyCur = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDCur];
  auto dxAlt = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDAlt], dyAlt = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDAlt];
  return dxCur * dxCur + dyCur * dyCur > dxAlt * dxAlt + dyAlt * dyAlt;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() void DCAFitterN<N, Args...>::print() const
{
#ifndef GPUCA_GPUCODE_DEVICE
  LOG(info) << N << "-prong vertex fitter in " << (mUseAbsDCA ? "abs." : "weighted") << " distance minimization mode";
  LOG(info) << "Bz: " << mBz << " MaxIter: " << mMaxIter << " MaxChi2: " << mMaxChi2;
  LOG(info) << "Stopping condition: Max.param change < " << mMinParamChange << " Rel.Chi2 change > " << mMinRelChi2Change;
  LOG(info) << "Discard candidates for : Rvtx > " << getMaxR() << " DZ between tracks > " << mMaxDZIni;
#else
  if (mUseAbsDCA) {
    printf("%d-prong vertex fitter in abs. distance minimization mode\n", N);
  } else {
    printf("%d-prong vertex fitter in weighted distance minimization mode\n", N);
  }
  printf("Bz: %1.f MaxIter: %3.d  MaxChi2: %2.3f\n", mBz, mMaxIter, mMaxChi2);
  printf("Stopping condition: Max.param change < %2.3f Rel.Chi2 change > %2.3f\n", mMinParamChange, mMinRelChi2Change);
  printf("Discard candidates for : Rvtx > %2.3f DZ between tracks > %2.3f\n", getMaxR(), mMaxDZIni);
#endif
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() o2::track::TrackParCov DCAFitterN<N, Args...>::createParentTrackParCov(int cand, bool sectorAlpha) const
{
  const auto& trP = getTrack(0, cand);
  const auto& trN = getTrack(1, cand);
  o2::gpu::gpustd::array<float, 21> covV = {0.};
  o2::gpu::gpustd::array<float, 3> pvecV = {0.};
  int q = 0;
  for (int it = 0; it < N; it++) {
    const auto& trc = getTrack(it, cand);
    o2::gpu::gpustd::array<float, 3> pvecT = {0.};
    o2::gpu::gpustd::array<float, 21> covT = {0.};
    trc.getPxPyPzGlo(pvecT);
    trc.getCovXYZPxPyPzGlo(covT);
    constexpr int MomInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
    for (int i = 0; i < 6; i++) {
      covV[MomInd[i]] += covT[MomInd[i]];
    }
    for (int i = 0; i < 3; i++) {
      pvecV[i] += pvecT[i];
    }
    q += trc.getCharge();
  }
  auto covVtxV = calcPCACovMatrix(cand);
  covV[0] = covVtxV(0, 0);
  covV[1] = covVtxV(1, 0);
  covV[2] = covVtxV(1, 1);
  covV[3] = covVtxV(2, 0);
  covV[4] = covVtxV(2, 1);
  covV[5] = covVtxV(2, 2);
  return o2::track::TrackParCov(getPCACandidatePos(cand), pvecV, covV, q, sectorAlpha);
}

//___________________________________________________________________
template <int N, typename... Args>
GPUd() o2::track::TrackPar DCAFitterN<N, Args...>::createParentTrackPar(int cand, bool sectorAlpha) const
{
  const auto& trP = getTrack(0, cand);
  const auto& trN = getTrack(1, cand);
  const auto& wvtx = getPCACandidate(cand);
  o2::gpu::gpustd::array<float, 3> pvecV = {0.};
  int q = 0;
  for (int it = 0; it < N; it++) {
    const auto& trc = getTrack(it, cand);
    o2::gpu::gpustd::array<float, 3> pvecT = {0.};
    trc.getPxPyPzGlo(pvecT);
    for (int i = 0; i < 3; i++) {
      pvecV[i] += pvecT[i];
    }
    q += trc.getCharge();
  }
  const o2::gpu::gpustd::array<float, 3> vertex = {(float)wvtx[0], (float)wvtx[1], (float)wvtx[2]};
  return o2::track::TrackPar(vertex, pvecV, q, sectorAlpha);
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() bool DCAFitterN<N, Args...>::propagateParamToX(o2::track::TrackPar& t, float x)
{
  bool res = true;
  if (mUsePropagator || mMatCorr != o2::base::Propagator::MatCorrType::USEMatCorrNONE) {
#ifndef GPUCA_GPUCODE
    res = o2::base::Propagator::Instance()->PropagateToXBxByBz(t, x, mMaxSnp, mMaxStep, mMatCorr);
#endif
  } else {
    res = t.propagateParamTo(x, mBz);
  }
  if (!res) {
    mPropFailed[mCurHyp] = true;
  }
  return res;
}

//___________________________________________________________________
template <int N, typename... Args>
GPUdi() bool DCAFitterN<N, Args...>::propagateToX(o2::track::TrackParCov& t, float x)
{
  bool res = true;
  if (mUsePropagator || mMatCorr != o2::base::Propagator::MatCorrType::USEMatCorrNONE) {
#ifndef GPUCA_GPUCODE
    res = o2::base::Propagator::Instance()->PropagateToXBxByBz(t, x, mMaxSnp, mMaxStep, mMatCorr);
#endif
  } else {
    res = t.propagateTo(x, mBz);
  }
  if (!res) {
    mPropFailed[mCurHyp] = true;
  }
  return res;
}

using DCAFitter2 = DCAFitterN<2, o2::track::TrackParCov>;
using DCAFitter3 = DCAFitterN<3, o2::track::TrackParCov>;

namespace device
{
template <typename Fitter>
void print(const int nBlocks, const int nThreads, Fitter& ft);

template <typename Fitter, class... Tr>
int process(const int nBlocks, const int nThreads, Fitter&, Tr&... args);

template <class Fitter, class... Tr>
std::vector<int> processBulk(const int nBlocks, const int nThreads, std::vector<Fitter>& fitters, std::vector<Tr>&... args);
} // namespace device

} // namespace vertexing
} // namespace o2
#endif // _ALICEO2_DCA_FITTERN_
