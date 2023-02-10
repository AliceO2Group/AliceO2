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

/// \file FwdDCAFitterN.h
/// \brief Defintions for N-prongs secondary vertex fit
/// \author ruben.shahoyan@cern.ch, adapted from central barrel to fwd rapidities by Rita Sadek, rita.sadek@cern.ch
/// For the formulae derivation see /afs/cern.ch/user/s/shahoian/public/O2/DCAFitter/DCAFitterN.pdf

#ifndef _ALICEO2_DCA_FWDFITTERN_
#define _ALICEO2_DCA_FWDFITTERN_
#include <TMath.h>
#include "MathUtils/Cartesian.h"
#include "ReconstructionDataFormats/TrackFwd.h"
#include "ReconstructionDataFormats/Track.h"
#include "DCAFitter/HelixHelper.h"
#include <TRandom.h>

namespace o2
{
namespace vertexing
{

///__________________________________________________________________________________
///< Fwd Inverse cov matrix (augmented by a dummy Z error) of the point defined by the track
struct FwdTrackCovI {
  float sxx, syy, sxy, szz;

  FwdTrackCovI(const o2::track::TrackParCovFwd& trc, float zerrFactor = 1.) { set(trc, zerrFactor); }
  FwdTrackCovI() = default;
  void set(const o2::track::TrackParCovFwd& trc, float zerrFactor = 1)
  {
    float cxx = trc.getSigma2X(), cyy = trc.getSigma2Y(), cxy = trc.getSigmaXY(), czz = cyy * zerrFactor;
    float detXY = cxx * cyy - cxy * cxy;
    if (detXY > 0.) {
      auto detXYI = 1. / detXY;
      sxx = cyy * detXYI;
      syy = cxx * detXYI;
      sxy = -cxy * detXYI;
      szz = 1. / czz;
    } else {
      throw std::runtime_error("invalid track covariance");
    }
  }
};

///__________________________________________________________________________
///< Fwd derivative (up to 2) of the TrackParam position over its running param Z
struct FwdTrackDeriv {
  float dxdz, dydz, d2xdz2, d2ydz2;
  FwdTrackDeriv() = default;
  FwdTrackDeriv(const o2::track::TrackParFwd& trc, float bz) { set(trc, bz); }
  void set(const o2::track::TrackParFwd& trc, float bz)
  {
    float snp = trc.getSnp(), csp = std::sqrt((1. - snp) * (1. + snp)), cspI = 1. / csp, crv2c = trc.getCurvature(bz), tgl = trc.getTanl(), tglI = 1. / tgl;
    if (crv2c == 0.) {
      crv2c = (trc.getCharge()) * 0.3 * bz * (-1e-3);
    }

    dxdz = csp * tglI;
    dydz = snp * tglI;
    d2xdz2 = crv2c * snp * tglI * tglI;
    d2ydz2 = -crv2c * csp * tglI * tglI;
  }
};

template <int N, typename... Args>
class FwdDCAFitterN
{
  static constexpr double NMin = 2;
  static constexpr double NMax = 4;
  static constexpr double NInv = 1. / N;
  static constexpr int MAXHYP = 2;
  static constexpr float ZerrFactor = 5.; // factor for conversion of track covXX to dummy covZZ
  using Track = o2::track::TrackParCovFwd;
  using TrackAuxPar = o2::track::TrackAuxPar;
  using CrossInfo = o2::track::CrossInfo;
  using Vec3D = ROOT::Math::SVector<double, 3>;
  using VecND = ROOT::Math::SVector<double, N>;
  using MatSym3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>;
  using MatStd3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepStd<double, 3>>;
  using MatSymND = ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N>>;
  using MatStdND = ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepStd<double, N>>;
  using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
  using TrackCoefVtx = MatStd3D;
  using ArrTrack = std::array<Track, N>;            // container for prongs (tracks) at single vertex cand.
  using ArrTrackCovI = std::array<FwdTrackCovI, N>; // container for inv.cov.matrices at single vertex cand.
  using ArrTrCoef = std::array<TrackCoefVtx, N>;    // container of TrackCoefVtx coefficients at single vertex cand.
  using ArrTrDer = std::array<FwdTrackDeriv, N>;    // container of Track 1st and 2nd derivative over their Z param
  using ArrTrPos = std::array<Vec3D, N>;            // container of Track positions

 public:
  static constexpr int getNProngs() { return N; }

  FwdDCAFitterN() = default;
  FwdDCAFitterN(float bz, bool useAbsDCA, bool prop2DCA) : mBz(bz), mUseAbsDCA(useAbsDCA), mPropagateToPCA(prop2DCA)
  {
    static_assert(N >= NMin && N <= NMax, "N prongs outside of allowed range");
  }

  //=========================================================================
  ///< return PCA candidate, by default best on is provided (no check for the index validity)
  const Vec3D& getPCACandidate(int cand = 0) const { return mPCA[mOrder[cand]]; }
  const auto getPCACandidatePos(int cand = 0) const
  {
    const auto& vd = mPCA[mOrder[cand]];
    return std::array<float, 3>{float(vd[0]), float(vd[1]), float(vd[2])};
  }

  ///< return Chi2 at PCA candidate (no check for its validity)
  float getChi2AtPCACandidate(int cand = 0) const { return mChi2[mOrder[cand]]; }

  ///< prepare copies of tracks at the V0 candidate (no check for the candidate validity)
  ///  must be called before getTrack(i,cand) query
  bool FwdpropagateTracksToVertex(int cand = 0);

  ///< check if propagation of tracks to candidate vertex was done
  bool isPropagateTracksToVertexDone(int cand = 0) const { return mTrPropDone[mOrder[cand]]; }

  ///< track param propagated to V0 candidate (no check for the candidate validity)
  ///  propagateTracksToVertex must be called in advance
  Track& getTrack(int i, int cand = 0)
  {
    if (!mTrPropDone[mOrder[cand]]) {
      throw std::runtime_error("propagateTracksToVertex was not called yet");
    }
    return mCandTr[mOrder[cand]][i];
  }

  ///< calculate on the fly track param (no cov mat) at candidate
  o2::track::TrackParFwd FwdgetTrackParamAtPCA(int i, int cand = 0) const;

  MatSym3D calcPCACovMatrix(int cand = 0) const;

  std::array<float, 6> calcPCACovMatrixFlat(int cand = 0) const
  {
    auto m = calcPCACovMatrix(cand);
    return {float(m(0, 0)), float(m(1, 0)), float(m(1, 1)), float(m(2, 0)), float(m(2, 1)), float(m(2, 2))};
  }

  const Track* getOrigTrackPtr(int i) const { return mOrigTrPtr[i]; }

  ///< return number of iterations during minimization (no check for its validity)
  int getNIterations(int cand = 0) const { return mNIters[mOrder[cand]]; }
  void setPropagateToPCA(bool v = true) { mPropagateToPCA = v; }
  void setMaxIter(int n = 60) { mMaxIter = n > 2 ? n : 2; }
  void setMaxR(float r = 200.) { mMaxR2 = r * r; }
  void setMaxDXIni(float d = 4.) { mMaxDXIni = d; }
  void setMaxChi2(float chi2 = 999.) { mMaxChi2 = chi2; }
  void setBz(float bz) { mBz = std::abs(bz) > o2::constants::math::Almost0 ? bz : 0.f; }
  void setMinParamChange(float x = 1e-3) { mMinParamChange = x > 1e-4 ? x : 1.e-4; }
  void setMinRelChi2Change(float r = 0.9) { mMinRelChi2Change = r > 0.1 ? r : 999.; }
  void setUseAbsDCA(bool v) { mUseAbsDCA = v; }
  void setMaxDistance2ToMerge(float v) { mMaxDist2ToMergeSeeds = v; }

  int getNCandidates() const { return mCurHyp; }
  int getMaxIter() const { return mMaxIter; }
  float getMaxR() const { return std::sqrt(mMaxR2); }
  float getMaxDXIni() const { return mMaxDXIni; }
  float getMaxChi2() const { return mMaxChi2; }
  float getMinParamChange() const { return mMinParamChange; }
  float getBz() const { return mBz; }
  double getK(double b) const { return std::abs(o2::constants::math::B2C * b); }
  double getHz(double b) const { return std::copysign(1, b); }

  float getMaxDistance2ToMerge() const { return mMaxDist2ToMergeSeeds; }
  bool getUseAbsDCA() const { return mUseAbsDCA; }
  bool getPropagateToPCA() const { return mPropagateToPCA; }

  template <class... Tr>
  int process(const Tr&... args);
  void print() const;

 protected:
  bool FwdcalcPCACoefs();
  bool FwdcalcInverseWeight();
  void FwdcalcResidDerivatives();
  void FwdcalcResidDerivativesNoErr();
  void FwdcalcChi2Derivatives();
  void FwdcalcChi2DerivativesNoErr();
  void FwdcalcPCA();
  void FwdcalcPCANoErr();
  void FwdcalcTrackResiduals();
  void calcTrackDerivatives();
  float findZatXY(int cand = 0);
  void findZatXY_mid(int cand = 0);
  void findZatXY_lineApprox(int cand = 0);
  void findZatXY_quad(int cand = 0);
  void findZatXY_linear(int cand = 0);
  double FwdcalcChi2() const;
  double FwdcalcChi2NoErr() const;
  bool FwdcorrectTracks(const VecND& corrZ);
  bool minimizeChi2();
  bool minimizeChi2NoErr();
  bool roughDXCut() const;
  bool closerToAlternative() const;
  static double getAbsMax(const VecND& v);

  ///< track param positions at V0 candidate (no check for the candidate validity)
  const Vec3D& getTrackPos(int i, int cand = 0) const { return mTrPos[mOrder[cand]][i]; }

  ///< track Z-param at V0 candidate (no check for the candidate validity)
  float getTrackZ(int i, int cand = 0) const { return getTrackPos(i, cand)[2]; }

  MatStd3D getTrackRotMatrix(int i) const // generate 3D matrix for track rotation to global frame
  // no rotation for fwd: mat=I
  {
    MatStd3D mat;
    mat(0, 0) = 1;
    mat(1, 1) = 1;
    mat(2, 2) = 1;
    return mat;
  }

  MatSym3D getTrackCovMatrix(int i, int cand = 0) const // generate covariance matrix of track position, adding fake Z error
  {
    const auto& trc = mCandTr[mOrder[cand]][i];
    MatSym3D mat;
    mat(0, 0) = trc.getSigma2X();
    mat(1, 1) = trc.getSigma2Y();
    mat(1, 0) = trc.getSigmaXY();
    mat(2, 2) = trc.getSigma2Y() * ZerrFactor;
    return mat;
  }

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
  // vectors of 1st derivatives of track local residuals over Z parameters
  std::array<std::array<Vec3D, N>, N> mDResidDz;
  // vectors of 1nd derivatives of track local residuals over Z parameters
  std::array<std::array<Vec3D, N>, N> mD2ResidDz2;
  VecND mDChi2Dz;      // 1st derivatives of chi2 over tracks Z params
  MatSymND mD2Chi2Dz2; // 2nd derivatives of chi2 over tracks Z params (symmetric matrix)

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
  bool mAllowAltPreference = true;  // if the fit converges to alternative PCA seed, abandon the current one
  bool mUseAbsDCA = false;          // use abs. distance minimization rather than chi2
  bool mPropagateToPCA = true;      // create tracks version propagated to PCA
  int mMaxIter = 60;                // max number of iterations
  float mBz = 0;                    // bz field, to be set by user
  float mMaxR2 = 200. * 200.;       // reject PCA's above this radius
  float mMaxDXIni = 4.;             // reject (if>0) PCA candidate if tracks DZ exceeds threshold
  float mMinParamChange = 1e-5;     // stop iterations if largest change of any X is smaller than this
  float mMinRelChi2Change = 0.98;   // stop iterations is chi2/chi2old > this
  float mMaxChi2 = 100;             // abs cut on chi2 or abs distance
  float mMaxDist2ToMergeSeeds = 1.; // merge 2 seeds to their average if their distance^2 is below the threshold

  ClassDefNV(FwdDCAFitterN, 1);
};

///_________________________________________________________________________
template <int N, typename... Args>
template <class... Tr>
int FwdDCAFitterN<N, Args...>::process(const Tr&... args)
{

  static_assert(sizeof...(args) == N, "incorrect number of input tracks");
  assign(0, args...);
  clear();

  for (int i = 0; i < N; i++) {
    mTrAux[i].set(*mOrigTrPtr[i], mBz);
  }

  if (!mCrossings.set(mTrAux[0], *mOrigTrPtr[0], mTrAux[1], *mOrigTrPtr[1])) { // even for N>2 it should be enough to test just 1 loop
    return 0;                                                                  // no crossing
  }

  if (mCrossings.nDCA == MAXHYP) { // if there are 2 candidates
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

    findZatXY_mid(mCurHyp);

    if (mUseAbsDCA ? minimizeChi2NoErr() : minimizeChi2()) {
      mOrder[mCurHyp] = mCurHyp;
      if (mPropagateToPCA && !FwdpropagateTracksToVertex(mCurHyp)) {
        continue;
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
bool FwdDCAFitterN<N, Args...>::FwdcalcPCACoefs()
{
  //< calculate Ti matrices for global vertex decomposition to V = sum_{0<i<N} Ti pi, see EQ.T in the ref
  if (!FwdcalcInverseWeight()) {
    return false;
  }
  for (int i = N; i--;) { // build Mi*Ei matrix, with Mi = I
    const auto& tcov = mTrcEInv[mCurHyp][i];
    MatStd3D miei;

    miei[0][0] = tcov.sxx;
    miei[0][1] = tcov.sxy;
    miei[1][0] = tcov.sxy;
    miei[1][1] = tcov.syy;
    miei[2][2] = tcov.szz;

    mTrCFVT[mCurHyp][i] = mWeightInv * miei;
  }
  return true;
}

//__________________________________________________________________________
template <int N, typename... Args>
bool FwdDCAFitterN<N, Args...>::FwdcalcInverseWeight()
{
  //< calculate [sum_{0<j<N} M_j*E_j*M_j^T]^-1 used for Ti matrices, see EQ.T, with M_i = I
  auto* arrmat = mWeightInv.Array();
  memset(arrmat, 0, sizeof(mWeightInv));
  enum { XX,
         XY,
         YY,
         XZ,
         YZ,
         ZZ };
  for (int i = N; i--;) {
    const auto& tcov = mTrcEInv[mCurHyp][i];
    arrmat[XX] += tcov.sxx;
    arrmat[XY] += tcov.sxy;
    arrmat[XZ] += 0;
    arrmat[YY] += tcov.syy;
    arrmat[YZ] += 0;
    arrmat[ZZ] += tcov.szz;
  }

  // invert 3x3 symmetrix matrix
  return mWeightInv.Invert();
}

//__________________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcResidDerivatives()
{
  //< calculate matrix of derivatives for weighted chi2: residual i vs parameter Z of track j
  MatStd3D matMT;
  for (int i = N; i--;) { // residual being differentiated
    // const auto& taux = mTrAux[i];
    for (int j = N; j--;) {                   // track over which we differentiate
      const auto& matT = mTrCFVT[mCurHyp][j]; // coefficient matrix for track J
      const auto& trDz = mTrDer[mCurHyp][j];  // track point derivs over track Z param
      auto& dr1 = mDResidDz[i][j];
      auto& dr2 = mD2ResidDz2[i][j];
      // calculate M_i^transverse * T_j , M_i^transverse=I -> MT=T
      matMT[0][0] = matT[0][0];
      matMT[0][1] = matT[0][1];
      matMT[0][2] = matT[0][2];
      matMT[1][0] = matT[1][0];
      matMT[1][1] = matT[1][1];
      matMT[1][2] = matT[1][2];
      matMT[2][0] = matT[2][0];
      matMT[2][1] = matT[2][1];
      matMT[2][2] = matT[2][2];

      // calculate DResid_i/Dz_j = (delta_ij - M_i^tr * T_j) * DTrack_k/Dz_k
      dr1[0] = -(matMT[0][0] * trDz.dxdz + matMT[0][1] * trDz.dydz + matMT[0][2]);
      dr1[1] = -(matMT[1][0] * trDz.dxdz + matMT[1][1] * trDz.dydz + matMT[1][2]);
      dr1[2] = -(matMT[2][0] * trDz.dxdz + matMT[2][1] * trDz.dydz + matMT[2][2]);

      // calculate D2Resid_I/(Dz_J Dz_K) = (delta_ijk - M_i^tr * T_j * delta_jk) * D2Track_k/dz_k^2
      dr2[0] = -(matMT[0][1] * trDz.d2ydz2 + matMT[0][0] * trDz.d2xdz2);
      dr2[1] = -(matMT[1][1] * trDz.d2ydz2 + matMT[1][0] * trDz.d2xdz2);
      dr2[2] = -(matMT[2][1] * trDz.d2ydz2 + matMT[2][0] * trDz.d2xdz2);

      if (i == j) {
        dr1[0] += trDz.dxdz;
        dr1[1] += trDz.dydz;
        dr1[2] += 1.;

        dr2[0] += trDz.d2xdz2;
        dr2[1] += trDz.d2ydz2;
      }
    } // track over which we differentiate
  }   // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcResidDerivativesNoErr()
{
  //< calculate matrix of derivatives for absolute distance chi2: residual i vs parameter Z of track j
  constexpr double NInv1 = 1. - NInv;       // profit from Rii = I/Ninv
  for (int i = N; i--;) {                   // residual being differentiated
    const auto& trDzi = mTrDer[mCurHyp][i]; // track point derivs over track Z param
    auto& dr1ii = mDResidDz[i][i];
    auto& dr2ii = mD2ResidDz2[i][i];

    dr1ii[0] = NInv1 * trDzi.dxdz;
    dr1ii[1] = NInv1 * trDzi.dydz;
    dr1ii[2] = NInv1;

    dr2ii[0] = NInv1 * trDzi.d2xdz2;
    dr2ii[1] = NInv1 * trDzi.d2ydz2;
    dr2ii[2] = 0;

    for (int j = i; j--;) { // track over which we differentiate
      auto& dr1ij = mDResidDz[i][j];
      auto& dr1ji = mDResidDz[j][i];
      const auto& trDzj = mTrDer[mCurHyp][j]; // track point derivs over track Z param

      // calculate DResid_i/Dz_j = (delta_ij - R_ij) * DTrack_j/Dz_j  for j<i
      dr1ij[0] = -trDzj.dxdz * NInv;
      dr1ij[1] = -trDzj.dydz * NInv;
      dr1ij[2] = -1 * NInv;

      // calculate DResid_j/Dz_i = (delta_ij - R_ji) * DTrack_i/Dz_i  for j<i
      dr1ji[0] = -trDzi.dxdz * NInv;
      dr1ji[1] = -trDzi.dydz * NInv;
      dr1ji[2] = -1 * NInv;

      auto& dr2ij = mD2ResidDz2[i][j];
      auto& dr2ji = mD2ResidDz2[j][i];

      // calculate D2Resid_I/(Dz_J Dz_K) = (delta_ij - Rij) * D2Track_j/dz_j^2 * delta_jk for j<i
      dr2ij[0] = -trDzj.d2xdz2 * NInv;
      dr2ij[1] = -trDzj.d2ydz2 * NInv;
      dr2ij[2] = 0;

      // calculate D2Resid_j/(Dz_i Dz_k) = (delta_ij - Rji) * D2Track_i/dz_i^2 * delta_ik for j<i
      dr2ji[0] = -trDzi.d2xdz2 * NInv;
      dr2ji[1] = -trDzi.d2ydz2 * NInv;
      dr2ji[2] = 0;

    } // track over which we differentiate
  }   // residual being differentiated
}

//__________________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcChi2Derivatives()
{
  //< calculate 1st and 2nd derivatives of wighted DCA (chi2) over track parameters Z
  std::array<std::array<Vec3D, N>, N> covIDrDz; // tempory vectors of covI_j * dres_j/dz_i

  // chi2 1st derivative
  for (int i = N; i--;) {
    auto& dchi1 = mDChi2Dz[i]; // DChi2/Dz_i = sum_j { res_j * covI_j * Dres_j/Dz_i }
    dchi1 = 0;
    for (int j = N; j--;) {
      const auto& res = mTrRes[mCurHyp][j];    // vector of residuals of track j
      const auto& covI = mTrcEInv[mCurHyp][j]; // inverse cov matrix of track j
      const auto& dr1 = mDResidDz[j][i];       // vector of j-th residuals 1st derivative over Z param of track i
      auto& cidr = covIDrDz[i][j];             // vector covI_j * dres_j/dz_i, save for 2nd derivative calculation
      cidr[0] = covI.sxx * dr1[0] + covI.sxy * dr1[1];
      cidr[1] = covI.sxy * dr1[0] + covI.syy * dr1[1];
      cidr[2] = covI.szz * dr1[2];

      dchi1 += ROOT::Math::Dot(res, cidr);
    }
  }

  // chi2 2nd derivative
  for (int i = N; i--;) {
    for (int j = i + 1; j--;) {       // symmetric matrix
      auto& dchi2 = mD2Chi2Dz2[i][j]; // D2Chi2/Dz_i/Dz_j = sum_k { Dres_k/Dz_j * covI_k * Dres_k/Dz_i + res_k * covI_k * D2res_k/Dz_i/Dz_j }
      dchi2 = 0;
      for (int k = N; k--;) {
        const auto& dr1j = mDResidDz[k][j];  // vector of k-th residuals 1st derivative over Z param of track j
        const auto& cidrkj = covIDrDz[i][k]; // vector covI_k * dres_k/dz_i
        dchi2 += ROOT::Math::Dot(dr1j, cidrkj);
        if (k == j) {
          const auto& res = mTrRes[mCurHyp][k];    // vector of residuals of track k
          const auto& covI = mTrcEInv[mCurHyp][k]; // inverse cov matrix of track k
          const auto& dr2ij = mD2ResidDz2[k][j];   // vector of k-th residuals 2nd derivative over Z params of track j
          dchi2 += res[0] * (covI.sxx * dr2ij[0] + covI.sxy * dr2ij[1]) + res[1] * (covI.sxy * dr2ij[0] + covI.syy * dr2ij[1]) + res[2] * covI.szz * dr2ij[2];
        }
      }
    }
  }
}

//__________________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcChi2DerivativesNoErr()
{
  //< calculate 1st and 2nd derivatives of abs DCA (chi2) over track parameters Z
  for (int i = N; i--;) {
    auto& dchi1 = mDChi2Dz[i]; // DChi2/Dz_i = sum_j { res_j * Dres_j/Dz_i }
    dchi1 = 0;                 // chi2 1st derivative
    for (int j = N; j--;) {
      const auto& res = mTrRes[mCurHyp][j]; // vector of residuals of track j
      const auto& dr1 = mDResidDz[j][i];    // vector of j-th residuals 1st derivative over Z param of track i
      dchi1 += ROOT::Math::Dot(res, dr1);
      if (i >= j) { // symmetrix matrix
        // chi2 2nd derivative
        auto& dchi2 = mD2Chi2Dz2[i][j]; // D2Chi2/Dz_i/Dz_j = sum_k { Dres_k/Dz_j * covI_k * Dres_k/Dz_i + res_k * covI_k * D2res_k/Dz_i/Dz_j }
        dchi2 = ROOT::Math::Dot(mTrRes[mCurHyp][i], mD2ResidDz2[i][j]);
        for (int k = N; k--;) {
          dchi2 += ROOT::Math::Dot(mDResidDz[k][i], mDResidDz[k][j]);
        }
      }
    }
  }
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcPCA()
{
  // calculate point of closest approach for N prongs
  // calculating V = sum (Ti*Pi)
  mPCA[mCurHyp] = mTrCFVT[mCurHyp][N - 1] * mTrPos[mCurHyp][N - 1];
  for (int i = N - 1; i--;) {
    mPCA[mCurHyp] += mTrCFVT[mCurHyp][i] * mTrPos[mCurHyp][i];
  }
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcPCANoErr()
{
  // calculate point of closest approach for N prongs w/o errors
  auto& pca = mPCA[mCurHyp];

  pca[0] = mTrPos[mCurHyp][N - 1][0];
  pca[1] = mTrPos[mCurHyp][N - 1][1];
  pca[2] = mTrPos[mCurHyp][N - 1][2];

  for (int i = N - 1; i--;) {
    pca[0] += mTrPos[mCurHyp][i][0];
    pca[1] += mTrPos[mCurHyp][i][1];
    pca[2] += mTrPos[mCurHyp][i][2];
  }
  pca[0] *= NInv;
  pca[1] *= NInv;
  pca[2] *= NInv;
}

//___________________________________________________________________
template <int N, typename... Args>
ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> FwdDCAFitterN<N, Args...>::calcPCACovMatrix(int cand) const
{
  // calculate covariance matrix for the point of closest approach
  MatSym3D covm;
  for (int i = N; i--;) {
    covm += ROOT::Math::Similarity(mUseAbsDCA ? getTrackRotMatrix(i) : mTrCFVT[mOrder[cand]][i], getTrackCovMatrix(i, cand));
  }
  return covm;
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::FwdcalcTrackResiduals()
{
  // calculate residuals, res = Pi - V
  Vec3D vtxLoc;
  for (int i = N; i--;) {
    mTrRes[mCurHyp][i] = mTrPos[mCurHyp][i];
    vtxLoc = mPCA[mCurHyp];
    mTrRes[mCurHyp][i] -= vtxLoc;
  }
}

//___________________________________________________________________
template <int N, typename... Args>
inline void FwdDCAFitterN<N, Args...>::calcTrackDerivatives()
{
  // calculate track derivatives over Z param
  for (int i = N; i--;) {
    mTrDer[mCurHyp][i].set(mCandTr[mCurHyp][i], mBz);
  }
}

//___________________________________________________________________
template <int N, typename... Args>
inline double FwdDCAFitterN<N, Args...>::FwdcalcChi2() const
{
  // calculate current chi2
  double chi2 = 0;
  for (int i = N; i--;) {
    const auto& res = mTrRes[mCurHyp][i];
    const auto& covI = mTrcEInv[mCurHyp][i];
    chi2 += res[0] * res[0] * covI.sxx + res[1] * res[1] * covI.syy + res[2] * res[2] * covI.szz + 2. * res[0] * res[1] * covI.sxy;
  }
  return chi2;
}

//___________________________________________________________________
template <int N, typename... Args>
inline double FwdDCAFitterN<N, Args...>::FwdcalcChi2NoErr() const
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
bool FwdDCAFitterN<N, Args...>::FwdcorrectTracks(const VecND& corrZ)
{
  // propagate tracks to updated Z
  for (int i = N; i--;) {
    const auto& trDer = mTrDer[mCurHyp][i];
    auto dz2h = 0.5 * corrZ[i] * corrZ[i];
    mTrPos[mCurHyp][i][0] -= trDer.dxdz * corrZ[i] - dz2h * trDer.d2xdz2;
    mTrPos[mCurHyp][i][1] -= trDer.dydz * corrZ[i] - dz2h * trDer.d2ydz2;
    mTrPos[mCurHyp][i][2] -= corrZ[i];
  }

  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
bool FwdDCAFitterN<N, Args...>::FwdpropagateTracksToVertex(int icand)
{
  // propagate on z axis to vertex
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
    auto z = pca[2];

    trc.propagateToZquadratic(z, mBz);
  }

  mTrPropDone[ord] = true;
  return true;
}

//___________________________________________________________________
template <int N, typename... Args>
float FwdDCAFitterN<N, Args...>::findZatXY(int mCurHyp) // Between 2 tracks
{

  double step = 0.001;     // initial step
  double startPoint = 20.; // first MFT disk

  double z[2] = {startPoint, startPoint};
  double newX[2], newY[2];

  double X = mPCA[mCurHyp][0]; // X seed
  double Y = mPCA[mCurHyp][1]; // Y seed

  mCandTr[mCurHyp][0] = *mOrigTrPtr[0];
  mCandTr[mCurHyp][1] = *mOrigTrPtr[1];

  double dstXY[2][3] = {{999., 999., 999.}, {999., 999., 999.}};

  double Z[2];
  double finalZ[2];

  double newDstXY;

  for (int i = 0; i < 2; i++) {

    while (z[i] > -10) {

      mCandTr[mCurHyp][i].propagateParamToZquadratic(z[i], mBz);
      newX[i] = mCandTr[mCurHyp][i].getX();
      newY[i] = mCandTr[mCurHyp][i].getY();

      newDstXY = std::sqrt((newX[i] - X) * (newX[i] - X) +
                           (newY[i] - Y) * (newY[i] - Y));

      // Update points
      dstXY[i][0] = dstXY[i][1];
      dstXY[i][1] = dstXY[i][2];
      dstXY[i][2] = newDstXY;

      if (dstXY[i][2] > dstXY[i][1] && dstXY[i][1] < dstXY[i][0]) {
        finalZ[i] = z[i] + step;
        break;
      }

      z[i] -= step;
    }
  }

  float rez = 0.5 * (finalZ[0] + finalZ[1]);
  return rez;
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::findZatXY_mid(int mCurHyp)
{
  // look into dXY of T0 - T1 between 2 points(0,40cm); the one with the highest dXY is moved to mid

  double startPoint = -40.;
  double endPoint = 50.;
  double midPoint = 0.5 * (startPoint + endPoint);

  double z[2][2] = {{startPoint, endPoint}, {startPoint, endPoint}}; // z for tracks 0/1 on starting poing and endpoint

  double DeltaZ = std::abs(endPoint - startPoint);

  double newX[2][2];
  double newY[2][2];

  double epsilon = 0.0001;

  double X = mPCA[mCurHyp][0]; // X seed
  double Y = mPCA[mCurHyp][1]; // Y seed

  mCandTr[mCurHyp][0] = *mOrigTrPtr[0];
  mCandTr[mCurHyp][1] = *mOrigTrPtr[1];

  double finalZ;

  double dstXY[2]; // 0 -> distance btwn both tracks at startPoint

  while (DeltaZ > epsilon) {

    midPoint = 0.5 * (startPoint + endPoint);

    for (int i = 0; i < 2; i++) {
      mCandTr[mCurHyp][i].propagateParamToZquadratic(startPoint, mBz);
      newX[i][0] = mCandTr[mCurHyp][i].getX();
      newY[i][0] = mCandTr[mCurHyp][i].getY();

      mCandTr[mCurHyp][i].propagateParamToZquadratic(endPoint, mBz);
      newX[i][1] = mCandTr[mCurHyp][i].getX();
      newY[i][1] = mCandTr[mCurHyp][i].getY();
    }

    dstXY[0] = (newX[0][0] - newX[1][0]) * (newX[0][0] - newX[1][0]) +
               (newY[0][0] - newY[1][0]) * (newY[0][0] - newY[1][0]);

    dstXY[1] = (newX[0][1] - newX[1][1]) * (newX[0][1] - newX[1][1]) +
               (newY[0][1] - newY[1][1]) * (newY[0][1] - newY[1][1]);

    DeltaZ = std::abs(endPoint - startPoint);

    if (DeltaZ < epsilon) {
      finalZ = 0.5 * (startPoint + endPoint);
      break;
    }

    // chose new start and end Point according to the smallest D_XY
    if (dstXY[1] > dstXY[0]) {
      endPoint = midPoint;
    } else {
      startPoint = midPoint;
    }
  }

  mPCA[mCurHyp][2] = finalZ;
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::findZatXY_lineApprox(int mCurHyp)
{
  // approx method: z=(b-b')/(a'-a) -> tracks to lines with y0,1=az0,1+b for each track (in YZ and XZ plane)

  double startPoint = 1.;
  double endPoint = 50.; // first disk

  double X = mPCA[mCurHyp][0]; // X seed
  double Y = mPCA[mCurHyp][1]; // Y seed

  mCandTr[mCurHyp][0] = *mOrigTrPtr[0];
  mCandTr[mCurHyp][1] = *mOrigTrPtr[1];

  double y[2][2]; // Y00: y track 0 at point 0; Y01: y track 0 at point 1
  double z[2][2];
  double x[2][2];

  double aYZ[2];
  double bYZ[2];

  double aXZ[2];
  double bXZ[2];

  double finalZ;

  // find points of the tracks = 2 straight lines
  for (int i = 0; i < 2; i++) {

    mCandTr[mCurHyp][i].propagateToZquadratic(startPoint, mBz);
    //  mCandTr[mCurHyp][i].propagateToZlinear(startPoint);
    z[i][0] = startPoint;
    y[i][0] = mCandTr[mCurHyp][i].getY();
    x[i][0] = mCandTr[mCurHyp][i].getX();

    mCandTr[mCurHyp][i].propagateToZquadratic(endPoint, mBz);
    //  mCandTr[mCurHyp][i].propagateToZlinear(endPoint);
    z[i][1] = endPoint;
    y[i][1] = mCandTr[mCurHyp][i].getY();
    x[i][1] = mCandTr[mCurHyp][i].getX();

    bYZ[i] = (y[i][1] - y[i][0] * z[i][1] / z[i][0]) / (1 - z[i][1] / z[i][0]);
    aYZ[i] = (y[i][0] - bYZ[i]) / z[i][0];

    bXZ[i] = (x[i][1] - x[i][0] * z[i][1] / z[i][0]) / (1 - z[i][1] / z[i][0]);
    aXZ[i] = (x[i][0] - bXZ[i]) / z[i][0];
  }

  // z seed: equ. for intersection of these lines
  finalZ = 0.5 * ((bYZ[0] - bYZ[1]) / (aYZ[1] - aYZ[0]) + (bXZ[0] - bXZ[1]) / (aXZ[1] - aXZ[0]));

  mPCA[mCurHyp][2] = finalZ;
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::findZatXY_quad(int mCurHyp)
{
  double startPoint = 0.;
  double endPoint = 40.; // first disk

  double X = mPCA[mCurHyp][0]; // X seed
  double Y = mPCA[mCurHyp][1]; // Y seed

  mCandTr[mCurHyp][0] = *mOrigTrPtr[0];
  mCandTr[mCurHyp][1] = *mOrigTrPtr[1];

  double x[2];
  double y[2];
  double sinPhi0[2];
  double cosPhi0[2];
  double tanL0[2];
  double qpt0[2];

  double k[2];  // B2C *abs(mBz)
  double Hz[2]; // mBz/abs(mBz)

  double Ax[2], Bx[2], Cx[2];
  double Ay[2], By[2], Cy[2];

  double deltaX[2], deltaY[2];

  bool posX[2], nulX[2], negX[2];
  double z1X[2], z2X[2], z12X[2];

  bool posY[2], nulY[2], negY[2];
  double z1Y[2], z2Y[2], z12Y[2];

  double finalZ[2];

  // find all variables for 2 tracks at z0 = startPoint
  // set A, B, C variables for x/y equation for 2 tracks
  // calculate Deltax/y for both and roots

  for (int i = 0; i < 2; i++) {
    mCandTr[mCurHyp][i].propagateToZquadratic(startPoint, mBz);
    x[i] = mCandTr[mCurHyp][i].getX();
    y[i] = mCandTr[mCurHyp][i].getY();
    sinPhi0[i] = mCandTr[mCurHyp][i].getSnp();
    cosPhi0[i] = std::sqrt((1. - sinPhi0[i]) * (1. + sinPhi0[i]));
    tanL0[i] = mCandTr[mCurHyp][i].getTanl();
    qpt0[i] = mCandTr[mCurHyp][i].getInvQPt();
    k[i] = getK(mBz);
    Hz[i] = getHz(mBz);

    Ax[i] = qpt0[i] * Hz[i] * k[i] * sinPhi0[i] / (2 * tanL0[i] * tanL0[i]);
    Bx[i] = cosPhi0[i] / tanL0[i];
    Cx[i] = x[i] - X;

    Ay[i] = -qpt0[i] * Hz[i] * k[i] * cosPhi0[i] / (2 * tanL0[i] * tanL0[i]);
    By[i] = sinPhi0[i] / tanL0[i];
    Cy[i] = y[i] - Y; //

    deltaX[i] = Bx[i] * Bx[i] - 4 * Ax[i] * Cx[i];
    deltaY[i] = By[i] * By[i] - 4 * Ay[i] * Cy[i];

    if (deltaX[i] > 0) {
      posX[i] = true;
      z1X[i] = (-Bx[i] - std::sqrt(deltaX[i])) / (2 * Ax[i]);
      z2X[i] = (-Bx[i] + std::sqrt(deltaX[i])) / (2 * Ax[i]);
    } else if (deltaX[i] == 0) {
      nulX[i] = true;
      z12X[i] = -Bx[i] / (2 * Ax[i]);
    } else {
      negX[i] = true;
      z12X[i] = 0;
    } // discard

    if (deltaY[i] > 0) {
      posY[i] = true;
      z1Y[i] = (-By[i] - std::sqrt(deltaY[i])) / (2 * Ay[i]);
      z2Y[i] = (-By[i] + std::sqrt(deltaY[i])) / (2 * Ay[i]);
    } else if (deltaX[i] == 0) {
      nulY[i] = true;
      z12Y[i] = -By[i] / (2 * Ay[i]);
    } else {
      negY[i] = true;
      z12Y[i] = 0;
    }

    // find the z located in an acceptable interval
    if (posX[i]) {
      if (z1X[i] < endPoint && z1X[i] > startPoint) {
        z12X[i] = z1X[i];
      } else {
        z12X[i] = z2X[i];
      }
    }

    if (posY[i]) {
      if (z1Y[i] < endPoint && z1Y[i] > startPoint) {
        z12Y[i] = z1Y[i];
      } else {
        z12Y[i] = z2Y[i];
      }
    }

    finalZ[i] = 0.5 * (z12X[i] + z12Y[i]);
  }

  mPCA[mCurHyp][2] = 0.5 * (finalZ[0] + finalZ[1]);
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::findZatXY_linear(int mCurHyp)
{

  double startPoint = 0.;

  double X = mPCA[mCurHyp][0]; // X seed
  double Y = mPCA[mCurHyp][1]; // Y seed

  mCandTr[mCurHyp][0] = *mOrigTrPtr[0];
  mCandTr[mCurHyp][1] = *mOrigTrPtr[1];

  double x[2];
  double y[2];
  double sinPhi0[2];
  double cosPhi0[2];
  double tanL0[2];

  double Ax[2], Bx[2];
  double Ay[2], By[2];

  double z12X[2];
  double z12Y[2];

  double finalZ[2];

  // find all variables for 2 tracks at z0 = startPoint
  // set A, B variables for x/y equation for 2 tracks
  // calculate root

  for (int i = 0; i < 2; i++) {
    mCandTr[mCurHyp][i].propagateToZlinear(startPoint);
    x[i] = mCandTr[mCurHyp][i].getX();
    y[i] = mCandTr[mCurHyp][i].getY();
    sinPhi0[i] = mCandTr[mCurHyp][i].getSnp();
    cosPhi0[i] = std::sqrt((1. - sinPhi0[i]) * (1. + sinPhi0[i]));
    tanL0[i] = mCandTr[mCurHyp][i].getTanl();

    Ax[i] = cosPhi0[i] / tanL0[i];
    Bx[i] = x[i] - X;

    Ay[i] = sinPhi0[i] / tanL0[i];
    By[i] = y[i] - Y;

    z12X[i] = -Bx[i] / Ax[i];
    z12Y[i] = -By[i] / Ay[i];

    finalZ[i] = 0.5 * (z12X[i] + z12Y[i]);
  }

  mPCA[mCurHyp][2] = 0.5 * (finalZ[0] + finalZ[1]);
}

//___________________________________________________________________
template <int N, typename... Args>
inline o2::track::TrackParFwd FwdDCAFitterN<N, Args...>::FwdgetTrackParamAtPCA(int i, int icand) const
{
  // propagate tracks param only to current vertex (if not already done)
  int ord = mOrder[icand];
  o2::track::TrackParFwd trc(mCandTr[ord][i]);
  if (!mTrPropDone[ord]) {
    auto z = mPCA[ord][2];
    trc.propagateParamToZquadratic(z, mBz);
  }

  return {trc};
}

//___________________________________________________________________
template <int N, typename... Args>
inline double FwdDCAFitterN<N, Args...>::getAbsMax(const VecND& v)
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
bool FwdDCAFitterN<N, Args...>::minimizeChi2()
{
  // find best chi2 (weighted DCA) of N tracks in the vicinity of the seed PCA
  double x[2], y[2];
  double sumX = 0.;
  double sumY = 0.;

  for (int i = N; i--;) {
    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];
    auto z = mPCA[mCurHyp][2];

    mCandTr[mCurHyp][i].propagateToZquadratic(z, mBz);

    x[i] = mCandTr[mCurHyp][i].getX();
    y[i] = mCandTr[mCurHyp][i].getY();

    setTrackPos(mTrPos[mCurHyp][i], mCandTr[mCurHyp][i]);      // prepare positions
    mTrcEInv[mCurHyp][i].set(mCandTr[mCurHyp][i], ZerrFactor); // prepare inverse cov.matrices at starting point

    sumX = sumX + x[i];
    sumY = sumY + y[i];
  }

  mPCA[mCurHyp][0] = sumX / N;
  mPCA[mCurHyp][1] = sumY / N;

  if (mMaxDXIni > 0 && !roughDXCut()) { // apply rough cut on tracks X difference
    return false;
  }

  if (!FwdcalcPCACoefs()) { // prepare tracks contribution matrices to the global PCA
    return false;
  }
  FwdcalcPCA();            // current PCA
  FwdcalcTrackResiduals(); // current track residuals
  float chi2Upd, chi2 = FwdcalcChi2();
  do {
    calcTrackDerivatives();    // current track derivatives (1st and 2nd)
    FwdcalcResidDerivatives(); // current residals derivatives (1st and 2nd)
    FwdcalcChi2Derivatives();  // current chi2 derivatives (1st and 2nd) to proceed for dz calculation

    // do Newton-Rapson iteration with corrections = - dchi2/d{x0..xN} * [ d^2chi2/d{x0..xN}^2 ]^-1
    if (!mD2Chi2Dz2.Invert()) {
      return false;
    }

    VecND dz = mD2Chi2Dz2 * mDChi2Dz;

    if (!FwdcorrectTracks(dz)) { // calculate new Pi (mTrPos) following Newton-Rapson iteration
      return false;
    }

    FwdcalcPCA(); // updated mPCA (new V coordinates with new mTrPos (Pi))
    if (mCrossIDAlt >= 0 && closerToAlternative()) {
      mAllowAltPreference = false;
      return false;
    }

    FwdcalcTrackResiduals(); // updated residuals
    chi2Upd = FwdcalcChi2(); // updated chi2

    if (getAbsMax(dz) < mMinParamChange || chi2Upd > chi2 * mMinRelChi2Change) {
      chi2 = chi2Upd;
      break; // converged
    }

    chi2 = chi2Upd;
  } while (++mNIters[mCurHyp] < mMaxIter);

  mChi2[mCurHyp] = chi2 * NInv;
  return mChi2[mCurHyp] < mMaxChi2;
}

//___________________________________________________________________
template <int N, typename... Args>
bool FwdDCAFitterN<N, Args...>::minimizeChi2NoErr()
{
  // find best chi2 (absolute DCA) of N tracks in the vicinity of the PCA seed
  double x[2], y[2];
  double sumX = 0.;
  double sumY = 0.;

  for (int i = N; i--;) {

    mCandTr[mCurHyp][i] = *mOrigTrPtr[i];

    auto z = mPCA[mCurHyp][2];
    mCandTr[mCurHyp][i].propagateParamToZquadratic(z, mBz);

    x[i] = mCandTr[mCurHyp][i].getX();
    y[i] = mCandTr[mCurHyp][i].getY();

    mPCA[mCurHyp][2] = z;

    setTrackPos(mTrPos[mCurHyp][i], mCandTr[mCurHyp][i]); // prepare positions

    sumX = sumX + x[i];
    sumY = sumY + y[i];
  }

  mPCA[mCurHyp][0] = sumX / N;
  mPCA[mCurHyp][1] = sumY / N;

  if (mMaxDXIni > 0 && !roughDXCut()) { // apply rough cut on tracks Z difference
    return false;
  }

  FwdcalcPCANoErr();       // current PCA
  FwdcalcTrackResiduals(); // current track residuals
  float chi2Upd, chi2 = FwdcalcChi2NoErr();
  do {
    calcTrackDerivatives();         // current track derivatives (1st and 2nd)
    FwdcalcResidDerivativesNoErr(); // current residals derivatives (1st and 2nd)
    FwdcalcChi2DerivativesNoErr();  // current chi2 derivatives (1st and 2nd)

    // do Newton-Rapson iteration with corrections = - dchi2/d{x0..xN} * [ d^2chi2/d{x0..xN}^2 ]^-1
    if (!mD2Chi2Dz2.Invert()) {
      return false;
    }
    VecND dz = mD2Chi2Dz2 * mDChi2Dz;

    if (!FwdcorrectTracks(dz)) {
      return false;
    }
    FwdcalcPCANoErr(); // updated PCA
    if (mCrossIDAlt >= 0 && closerToAlternative()) {
      mAllowAltPreference = false;
      return false;
    }
    FwdcalcTrackResiduals();      // updated residuals
    chi2Upd = FwdcalcChi2NoErr(); // updated chi2
    if (getAbsMax(dz) < mMinParamChange || chi2Upd > chi2 * mMinRelChi2Change) {
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
bool FwdDCAFitterN<N, Args...>::roughDXCut() const
{
  // apply rough cut on DX between the tracks in the seed point

  bool accept = true;
  for (int i = N; accept && i--;) {
    for (int j = i; j--;) {
      if (std::abs(mCandTr[mCurHyp][i].getX() - mCandTr[mCurHyp][j].getX()) > mMaxDXIni) {
        accept = false;
        break;
      }
    }
  }
  return accept;
}

//___________________________________________________________________
template <int N, typename... Args>
bool FwdDCAFitterN<N, Args...>::closerToAlternative() const
{
  // check if the point current PCA point is closer to the seeding XY point being tested or to alternative see (if any)
  auto dxCur = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDCur], dyCur = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDCur];
  auto dxAlt = mPCA[mCurHyp][0] - mCrossings.xDCA[mCrossIDAlt], dyAlt = mPCA[mCurHyp][1] - mCrossings.yDCA[mCrossIDAlt];
  return dxCur * dxCur + dyCur * dyCur > dxAlt * dxAlt + dyAlt * dyAlt;
}

//___________________________________________________________________
template <int N, typename... Args>
void FwdDCAFitterN<N, Args...>::print() const
{
  LOG(info) << N << "-prong vertex fitter in " << (mUseAbsDCA ? "abs." : "weighted") << " distance minimization mode";
  LOG(info) << "Bz: " << mBz << " MaxIter: " << mMaxIter << " MaxChi2: " << mMaxChi2;
  LOG(info) << "Stopping condition: Max.param change < " << mMinParamChange << " Rel.Chi2 change > " << mMinRelChi2Change;
  LOG(info) << "Discard candidates for : Rvtx > " << getMaxR() << " DZ between tracks > " << mMaxDXIni;
}

using FwdDCAFitter2 = FwdDCAFitterN<2, o2::track::TrackParCovFwd>;
using FwdDCAFitter3 = FwdDCAFitterN<3, o2::track::TrackParCovFwd>;

} // namespace vertexing
} // namespace o2
#endif // _ALICEO2_DCA_FWDFITTERN_
