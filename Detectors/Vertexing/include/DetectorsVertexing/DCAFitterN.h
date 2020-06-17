// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <TMath.h>
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsVertexing/HelixHelper.h"

namespace o2
{
namespace vertexing
{
///__________________________________________________________________________________
///< Inverse cov matrix (augmented by a dummy X error) of the point defined by the track
struct TrackCovI {
  float sxx, syy, syz, szz;

  TrackCovI(const o2::track::TrackParCov& trc) { set(trc); }

  TrackCovI() = default;

  void set(const o2::track::TrackParCov& trc)
  {
    // we assign Y error to X for DCA calculation
    // (otherwise for quazi-collinear tracks the X will not be constrained)
    float cyy = trc.getSigmaY2(), czz = trc.getSigmaZ2(), cyz = trc.getSigmaZY(), cxx = cyy;
    float detYZ = cyy * czz - cyz * cyz;
    if (detYZ > 0.) {
      auto detYZI = 1. / detYZ;
      sxx = 1. / cxx;
      syy = czz * detYZI;
      syz = -cyz * detYZI;
      szz = cyy * detYZI;
    } else {
      throw std::runtime_error("invalid track covariance");
    }
  }
};

///__________________________________________________________________________
///< Derivative (up to 2) of the TrackParam position over its running param X
struct TrackDeriv {
  float dydx, dzdx, d2ydx2, d2zdx2;
  TrackDeriv() = default;
  TrackDeriv(const o2::track::TrackPar& trc, float bz) { set(trc, bz); }
  void set(const o2::track::TrackPar& trc, float bz)
  {
    float snp = trc.getSnp(), csp = std::sqrt((1. - snp) * (1. + snp)), cspI = 1. / csp, crv2c = trc.getCurvature(bz) * cspI;
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

  using Track = o2::track::TrackParCov;
  using TrackAuxPar = o2::track::TrackAuxPar;
  using CrossInfo = o2::track::CrossInfo;

  using Vec3D = ROOT::Math::SVector<double, 3>;
  using VecND = ROOT::Math::SVector<double, N>;
  using MatSym3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>;
  using MatStd3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepStd<double, 3>>;
  using MatSymND = ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N>>;
  using MatStdND = ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepStd<double, N>>;
  using TrackCoefVtx = MatStd3D;
  using ArrTrack = std::array<Track, N>;         // container for prongs (tracks) at single vertex cand.
  using ArrTrackCovI = std::array<TrackCovI, N>; // container for inv.cov.matrices at single vertex cand.
  using ArrTrCoef = std::array<TrackCoefVtx, N>; // container of TrackCoefVtx coefficients at single vertex cand.
  using ArrTrDer = std::array<TrackDeriv, N>;    // container of Track 1st and 2nd derivative over their X param
  using ArrTrPos = std::array<Vec3D, N>;         // container of Track positions

 public:
  static constexpr int getNProngs() { return N; }

  DCAFitterN() = default;
  DCAFitterN(float bz, bool useAbsDCA, bool prop2DCA) : mBz(bz), mUseAbsDCA(useAbsDCA), mPropagateToPCA(prop2DCA)
  {
    static_assert(N >= NMin && N <= NMax, "N prongs outside of allowed range");
  }

  //=========================================================================
  ///< return PCA candidate, by default best on is provided (no check for the index validity)
  const Vec3D& getPCACandidate(int cand = 0) const { return mPCA[mOrder[cand]]; }

  ///< return Chi2 at PCA candidate (no check for its validity)
  float getChi2AtPCACandidate(int cand = 0) const { return mChi2[mOrder[cand]]; }

  ///< track param positions at V0 candidate (no check for the candidate validity)
  const Vec3D& getTrackPos(int i, int cand = 0) const { return mTrPos[mOrder[cand]][i]; }

  ///< prapare copies of tracks at the V0 candidate (no check for the candidate validity)
  ///  must be called before getTrack(i,cand) query
  bool propagateTracksToVertex(int cand = 0);

  ///< track X-param at V0 candidate (no check for the candidate validity)
  float getTrackX(int i, int cand = 0) const { return getTrackPos(i, cand)[0]; }

  ///< track param propagated to V0 candidate (no check for the candidate validity)
  ///  propagateTracksToVertex must be called in advance
  const Track& getTrack(int i, int cand = 0) const
  {
    if (!mTrPropDone[mOrder[cand]]) {
      throw std::runtime_error("propagateTracksToVertex was not called yet");
    }
    return mCandTr[mOrder[cand]][i];
  }

  const Track* getOrigTrackPtr(int i) const { return mOrigTrPtr[i]; }

  ///< return number of iterations during minimization (no check for its validity)
  int getNIterations(int cand = 0) const { return mNIters[mOrder[cand]]; }
  void setPropagateToPCA(bool v = true) { mPropagateToPCA = v; }
  void setMaxIter(int n = 20) { mMaxIter = n > 2 ? n : 2; }
  void setMaxR(float r = 200.) { mMaxR2 = r * r; }
  void setMaxDZIni(float d = 4.) { mMaxDZIni = d; }
  void setMaxChi2(float chi2 = 999.) { mMaxChi2 = chi2; }
  void setBz(float bz) { mBz = std::abs(bz) > o2::constants::math::Almost0 ? bz : 0.f; }
  void setMinParamChange(float x = 1e-3) { mMinParamChange = x > 1e-4 ? x : 1.e-4; }
  void setMinRelChi2Change(float r = 0.9) { mMinRelChi2Change = r > 0.1 ? r : 999.; }
  void setUseAbsDCA(bool v) { mUseAbsDCA = v; }
  void setMaxDistance2ToMerge(float v) { mMaxDist2ToMergeSeeds = v; }

  int getNCandidates() const { return mCurHyp; }
  int getMaxIter() const { return mMaxIter; }
  float getMaxR() const { return std::sqrt(mMaxR2); }
  float getMaxDZIni() const { return mMaxDZIni; }
  float getMaxChi2() const { return mMaxChi2; }
  float getMinParamChange() const { return mMinParamChange; }
  float getBz() const { return mBz; }
  float getMaxDistance2ToMerge() const { return mMaxDist2ToMergeSeeds; }
  bool getUseAbsDCA() const { return mUseAbsDCA; }
  bool getPropagateToPCA() const { return mPropagateToPCA; }

  template <class... Tr>
  int process(const Tr&... args);
  void print() const;

 protected:
  bool calcPCACoefs();
  bool calcInverseWeight();
  void calcResidDerivatives();
  void calcResidDerivativesNoErr();
  void calcRMatrices();
  void calcChi2Derivatives();
  void calcChi2DerivativesNoErr();
  void calcPCA();
  void calcPCANoErr();
  void calcTrackResiduals();
  void calcTrackDerivatives();
  double calcChi2() const;
  double calcChi2NoErr() const;
  bool correctTracks(const VecND& corrX);
  bool minimizeChi2();
  bool minimizeChi2NoErr();
  bool roughDZCut() const;
  bool closerToAlternative() const;
  static double getAbsMax(const VecND& v);

  void assign(int) {}
  template <class T, class... Tr>
  void assign(int i, const T& t, const Tr&... args)
  {
    static_assert(std::is_convertible<T, Track>(), "Wrong track type");
    mOrigTrPtr[i] = &t;
    assign(i + 1, args...);
  }

  void clear()
  {
    mCurHyp = 0;
    mAllowAltPreference = true;
  }

  static void setTrackPos(Vec3D& pnt, const Track& tr)
  {
    pnt[0] = tr.getX();
    pnt[1] = tr.getY();
    pnt[2] = tr.getZ();
  }

 private:
  // vectors of 1st derivatives of track local residuals over X parameters
  std::array<std::array<Vec3D, N>, N> mDResidDx;
  // vectors of 1nd derivatives of track local residuals over X parameters
  // (cross-derivatives DR/(dx_j*dx_k) = 0 for j!=k, therefore the hessian is diagonal)
  std::array<std::array<Vec3D, N>, N> mD2ResidDx2;
  VecND mDChi2Dx;      // 1st derivatives of chi2 over tracks X params
  MatSymND mD2Chi2Dx2; // 2nd derivatives of chi2 over tracks X params (symmetric matrix)
  MatSymND mCosDif;    // matrix with cos(alp_j-alp_i) for j<i
  MatSymND mSinDif;    // matrix with sin(alp_j-alp_i) for j<i
  std::array<const Track*, N> mOrigTrPtr;
  std::array<TrackAuxPar, N> mTrAux; // Aux track info for each track at each cand. vertex
  CrossInfo mCrossings;              // info on track crossing

  std::array<ArrTrackCovI, MAXHYP> mTrcEInv; // errors for each track at each cand. vertex
  std::array<ArrTrack, MAXHYP> mCandTr;      // tracks at each cond. vertex (Note: Errors are at seed XY point)
  std::array<ArrTrCoef, MAXHYP> mTrCFVT;     // TrackCoefVtx for each track at each cand. vertex
  std::array<ArrTrDer, MAXHYP> mTrDer;       // Track derivativse
  std::array<ArrTrPos, MAXHYP> mTrPos;       // Track positions
  std::array<ArrTrPos, MAXHYP> mTrRes;       // Track residuals
  std::array<Vec3D, MAXHYP> mPCA;            // PCA for each vertex candidate
  std::array<float, MAXHYP> mChi2 = {0};     // Chi2 at PCA candidate
  std::array<int, MAXHYP> mNIters;           // number of iterations for each seed
  std::array<bool, MAXHYP> mTrPropDone;      // Flag that the tracks are fully propagated to PCA
  MatSym3D mWeightInv;                       // inverse weight of single track, [sum{M^T E M}]^-1 in EQ.T
  std::array<int, MAXHYP> mOrder{0};
  int mCurHyp = 0;
  int mCrossIDCur = 0;
  int mCrossIDAlt = -1;
  bool mAllowAltPreference = true; // if the fit converges to alternative PCA seed, abandon the current one
  bool mUseAbsDCA = false;       // use abs. distance minimization rather than chi2
  bool mPropagateToPCA = true;   // create tracks version propagated to PCA
  int mMaxIter = 20;             // max number of iterations
  float mBz = 0;                 // bz field, to be set by user
  float mMaxR2 = 200. * 200.;    // reject PCA's above this radius
  float mMaxDZIni = 4.;          // reject (if>0) PCA candidate if tracks DZ exceeds threshold
  float mMinParamChange = 1e-3;  // stop iterations if largest change of any X is smaller than this
  float mMinRelChi2Change = 0.9; // stop iterations is chi2/chi2old > this
  float mMaxChi2 = 100;          // abs cut on chi2 or abs distance
  float mMaxDist2ToMergeSeeds = 1.; // merge 2 seeds to their average if their distance^2 is below the threshold

  ClassDefNV(DCAFitterN, 1);
};

///_________________________________________________________________________
template <int N, typename... Args>
template <class... Tr>
int DCAFitterN<N, Args...>::process(const Tr&... args)
{
  // This is a main entry point: fit PCA of N tracks
  static_assert(sizeof...(args) == N, "incorrect number of input tracks");
  assign(0, args...);
  clear();
  for (int i = 0; i < N; i++) {
    mTrAux[i].set(*mOrigTrPtr[i], mBz);
  }
  if (!mCrossings.set(mTrAux[0], *mOrigTrPtr[0], mTrAux[1], *mOrigTrPtr[1])) { // even for N>2 it should be enough to test just 1 loop
    return 0;                                  // no crossing
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
        return false;
      }
      mCurHyp++;
    }
  }

  for (int i = mCurHyp; i--;) { // order in quality
    for (int j = i; j--;) {
      if (mChi2[mOrder[i]] < mChi2[mOrder[j]]) {
        std::swap(mOrder[i], mOrder[j]);
      }
    }
  }
  return mCurHyp;
}

//__________________________________________________________________________
template <int N, typename... Args>
bool DCAFitterN<N, Args...>::calcPCACoefs()
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
    //miei[2][0] = 0;
    miei[2][1] = tcov.syz;
    miei[2][2] = tcov.szz;
    mTrCFVT[mCurHyp][i] = mWeightInv * miei;
  }
  return true;
}

//__________________________________________________________________________
template <int N, typename... Args>
bool DCAFitterN<N, Args...>::calcInverseWeight()
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
void DCAFitterN<N, Args...>::calcResidDerivatives()
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
  }   // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
void DCAFitterN<N, Args...>::calcResidDerivativesNoErr()
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
  }   // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
void DCAFitterN<N, Args...>::calcRMatrices()
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
void DCAFitterN<N, Args...>::calcChi2Derivatives()
{
  //< calculate 1st and 2nd derivatives of wighted DCA (chi2) over track parameters X, see EQ.Chi2 in the ref
  std::array<std::array<Vec3D, N>, N> covIDrDx; // tempory vectors of covI_j * dres_j/dx_i

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
      dchi1 += ROOT::Math::Dot(res, cidr);
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
        dchi2 += ROOT::Math::Dot(dr1j, cidrkj);
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
void DCAFitterN<N, Args...>::calcChi2DerivativesNoErr()
{
  //< calculate 1st and 2nd derivatives of abs DCA (chi2) over track parameters X, see (6) in the ref
  for (int i = N; i--;) {
    auto& dchi1 = mDChi2Dx[i]; // DChi2/Dx_i = sum_j { res_j * Dres_j/Dx_i }
    dchi1 = 0;                 // chi2 1st derivative
    for (int j = N; j--;) {
      const auto& res = mTrRes[mCurHyp][j]; // vector of residuals of track j
      const auto& dr1 = mDResidDx[j][i];    // vector of j-th residuals 1st derivative over X param of track i
      dchi1 += ROOT::Math::Dot(res, dr1);
      if (i >= j) { // symmetrix matrix
        // chi2 2nd derivative
        auto& dchi2 = mD2Chi2Dx2[i][j]; // D2Chi2/Dx_i/Dx_j = sum_k { Dres_k/Dx_j * covI_k * Dres_k/Dx_i + res_k * covI_k * D2res_k/Dx_i/Dx_j }
        dchi2 = ROOT::Math::Dot(mTrRes[mCurHyp][i], mD2ResidDx2[i][j]);
        for (int k = N; k--;) {
          dchi2 += ROOT::Math::Dot(mDResidDx[k][i], mDResidDx[k][j]);
        }
      }
    }
  }
}

//___________________________________________________________________
template <int N, typename... Args>
void DCAFitterN<N, Args...>::calcPCA()
{
  // calculate point of closest approach for N prongs
  mPCA[mCurHyp] = mTrCFVT[mCurHyp][N - 1] * mTrPos[mCurHyp][N - 1];
  for (int i = N - 1; i--;) {
    mPCA[mCurHyp] += mTrCFVT[mCurHyp][i] * mTrPos[mCurHyp][i];
  }
}

//___________________________________________________________________
template <int N, typename... Args>
void DCAFitterN<N, Args...>::calcPCANoErr()
{
  // calculate point of closest approach for N prongs w/o errors
  auto& pca = mPCA[mCurHyp];
  o2::utils::rotateZ(mTrPos[mCurHyp][N - 1][0], mTrPos[mCurHyp][N - 1][1], pca[0], pca[1], mTrAux[N - 1].s, mTrAux[N - 1].c);
  //RRRR    mTrAux[N-1].loc2glo(mTrPos[mCurHyp][N-1][0], mTrPos[mCurHyp][N-1][1], pca[0], pca[1] );
  pca[2] = mTrPos[mCurHyp][N - 1][2];
  for (int i = N - 1; i--;) {
    double x, y;
    o2::utils::rotateZ(mTrPos[mCurHyp][i][0], mTrPos[mCurHyp][i][1], x, y, mTrAux[i].s, mTrAux[i].c);
    //RRRR mTrAux[i].loc2glo(mTrPos[mCurHyp][i][0], mTrPos[mCurHyp][i][1], x, y );
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
void DCAFitterN<N, Args...>::calcTrackResiduals()
{
  // calculate residuals
  Vec3D vtxLoc;
  for (int i = N; i--;) {
    mTrRes[mCurHyp][i] = mTrPos[mCurHyp][i];
    vtxLoc = mPCA[mCurHyp];
    o2::utils::rotateZInv(vtxLoc[0], vtxLoc[1], vtxLoc[0], vtxLoc[1], mTrAux[i].s, mTrAux[i].c); // glo->loc
    mTrRes[mCurHyp][i] -= vtxLoc;
  }
}

//___________________________________________________________________
template <int N, typename... Args>
inline void DCAFitterN<N, Args...>::calcTrackDerivatives()
{
  // calculate track derivatives over X param
  for (int i = N; i--;) {
    mTrDer[mCurHyp][i].set(mCandTr[mCurHyp][i], mBz);
  }
}

//___________________________________________________________________
template <int N, typename... Args>
inline double DCAFitterN<N, Args...>::calcChi2() const
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
inline double DCAFitterN<N, Args...>::calcChi2NoErr() const
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
bool DCAFitterN<N, Args...>::correctTracks(const VecND& corrX)
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
bool DCAFitterN<N, Args...>::propagateTracksToVertex(int icand)
{
  // propagate tracks to current vertex
  int ord = mOrder[icand];
  if (mTrPropDone[ord]) {
    return true;
  }
  const Vec3D& pca = mPCA[ord];
  for (int i = N; i--;) {
    if (mUseAbsDCA) {
      mCandTr[ord][i] = *mOrigTrPtr[i]; // fetch the track again, as mCandTr might have been propagated w/o errors
    }
    auto& trc = mCandTr[ord][i];
    auto x = mTrAux[i].c * pca[0] + mTrAux[i].s * pca[1]; // X of PCA in the track frame
    if (!trc.propagateTo(x, mBz)) {
      return false;
    }
  }
  mTrPropDone[ord] = true;
  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
inline double DCAFitterN<N, Args...>::getAbsMax(const VecND& v)
{
  double mx = -1;
  for (int i = N; i--;) {
    auto vai = std::abs(v[i]);
    if (mx < vai) {
      mx = vai;
    }
  }
  return mx;
}

//___________________________________________________________________
template <int N, typename... Args>
bool DCAFitterN<N, Args...>::minimizeChi2()
{
  // find best chi2 (weighted DCA) of N tracks in the vicinity of the seed PCA
  for (int i = N; i--;) {
    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];
    auto x = mTrAux[i].c * mPCA[mCurHyp][0] + mTrAux[i].s * mPCA[mCurHyp][1]; // X of PCA in the track frame
    if (!mCandTr[mCurHyp][i].propagateTo(x, mBz)) {
      return false;
    }
    setTrackPos(mTrPos[mCurHyp][i], mCandTr[mCurHyp][i]); // prepare positions
    mTrcEInv[mCurHyp][i].set(mCandTr[mCurHyp][i]);        // prepare inverse cov.matrices at starting point
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
      LOG(ERROR) << "InversionFailed";
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
bool DCAFitterN<N, Args...>::minimizeChi2NoErr()
{
  // find best chi2 (absolute DCA) of N tracks in the vicinity of the PCA seed

  for (int i = N; i--;) {
    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];
    auto x = mTrAux[i].c * mPCA[mCurHyp][0] + mTrAux[i].s * mPCA[mCurHyp][1]; // X of PCA in the track frame
    if (!mCandTr[mCurHyp][i].propagateParamTo(x, mBz)) {
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
      LOG(ERROR) << "InversionFailed";
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
bool DCAFitterN<N, Args...>::roughDZCut() const
{
  // apply rough cut on DZ between the tracks in the seed point
  bool accept = true;
  for (int i = N; accept && i--;) {
    for (int j = i; j--;) {
      if (std::abs(mCandTr[mCurHyp][i].getZ() - mCandTr[mCurHyp][j].getZ()) > mMaxDZIni) {
        accept = false;
        break;
      }
    }
  }
  return accept;
}

//___________________________________________________________________
template <int N, typename... Args>
bool DCAFitterN<N, Args...>::closerToAlternative() const
{
  // check if the point current PCA point is closer to the seeding XY point being tested or to alternative see (if any)
  auto dxCur = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDCur], dyCur = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDCur];
  auto dxAlt = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDAlt], dyAlt = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDAlt];
  return dxCur * dxCur + dyCur * dyCur > dxAlt * dxAlt + dyAlt * dyAlt;
}

//___________________________________________________________________
template <int N, typename... Args>
void DCAFitterN<N, Args...>::print() const
{
  LOG(INFO) << N << "-prong vertex fitter in " << (mUseAbsDCA ? "abs." : "weighted") << " distance minimization mode";
  LOG(INFO) << "Bz: " << mBz << " MaxIter: " << mMaxIter << " MaxChi2: " << mMaxChi2;
  LOG(INFO) << "Stopping condition: Max.param change < " << mMinParamChange << " Rel.Chi2 change > " << mMinRelChi2Change;
  LOG(INFO) << "Discard candidates for : Rvtx > " << getMaxR() << " DZ between tracks > " << mMaxDZIni;
}

using DCAFitter2 = DCAFitterN<2, o2::track::TrackParCov>;
using DCAFitter3 = DCAFitterN<3, o2::track::TrackParCov>;

} // namespace vertexing
} // namespace o2
#endif // _ALICEO2_DCA_FITTERN_
