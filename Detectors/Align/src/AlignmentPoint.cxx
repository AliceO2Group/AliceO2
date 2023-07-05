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

/// @file   AlignmentPoint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Meausered point in the sensor.

#include <cstdio>
#include <TMath.h>
#include <TString.h>
#include "Align/AlignmentPoint.h"

using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//_____________________________________
void AlignmentPoint::init()
{
  // compute aux info
  const double kCorrToler = 1e-6;
  const double kDiagToler = 1e-14;
  //
  // compute parameters of tranformation to diagonal error matrix
  if (!isZeroPos(mErrYZTracking[0] + mErrYZTracking[2])) {
    //
    // is there a correlation?
    if (math_utils::detail::abs(mErrYZTracking[1] * mErrYZTracking[1] / (mErrYZTracking[0] * mErrYZTracking[2])) < kCorrToler) {
      mCosDiagErr = 1.;
      mSinDiagErr = 0.;
      mErrDiag[0] = mErrYZTracking[0];
      mErrDiag[1] = mErrYZTracking[2];
    } else {
      double dfd = 0.5 * (mErrYZTracking[2] - mErrYZTracking[0]);
      double phi = 0;
      // special treatment if errors are equal
      if (Abs(dfd) < kDiagToler) {
        phi = mErrYZTracking[1] > 0 ? (Pi() * 0.25) : (Pi() * 0.75);
      } else {
        phi = 0.5 * ATan2(mErrYZTracking[1], dfd);
      }
      //
      mCosDiagErr = Cos(phi);
      mSinDiagErr = Sin(phi);
      //
      //      double det = dfd*dfd + mErrYZTracking[1]*mErrYZTracking[1];
      //      det = det>0 ? Sqrt(det) : 0;
      //      double smd = 0.5*(mErrYZTracking[0] + mErrYZTracking[2]);
      //      mErrDiag[0] = smd + det;
      //      mErrDiag[1] = smd - det;
      double xterm = 2 * mCosDiagErr * mSinDiagErr * mErrYZTracking[1];
      double cc = mCosDiagErr * mCosDiagErr;
      double ss = mSinDiagErr * mSinDiagErr;
      mErrDiag[0] = mErrYZTracking[0] * cc + mErrYZTracking[2] * ss - xterm;
      mErrDiag[1] = mErrYZTracking[0] * ss + mErrYZTracking[2] * cc + xterm;
    }
  }
  //
}

//_____________________________________
void AlignmentPoint::updatePointByTrackInfo(const trackParam_t* t)
{
  //  // recalculate point errors using info about the track in the sensor tracking frame
  mSensor->updatePointByTrackInfo(this, t);
}

//_____________________________________
void AlignmentPoint::print(uint16_t opt) const
{
  // print
  printf("%cDet%d SID:%4d Alp:%+.3f X:%+9.4f Meas:%s Mat: ", isInvDir() ? '*' : ' ',
         getDetID(), getSID(), getAlphaSens(), getXSens(), containsMeasurement() ? "ON" : "OFF");
  if (!containsMaterial()) {
    printf("OFF\n");
  } else {
    printf("x2X0: %.4f x*rho: %.4f | pars:[%3d:%3d)\n", getX2X0(), getXTimesRho(), getMinLocVarID(), getMaxLocVarID());
  }
  //
  if ((opt & kMeasurementBit) && containsMeasurement()) {
    printf("  MeasPnt: Xtr: %+9.4f Ytr: %+8.4f Ztr: %+9.4f | ErrYZ: %+e %+e %+e | %d DOFglo\n",
           getXTracking(), getYTracking(), getZTracking(),
           mErrYZTracking[0], mErrYZTracking[1], mErrYZTracking[2], getNGloDOFs());
    printf("  DiagErr: %+e %+e\n", mErrDiag[0], mErrDiag[1]);
  }
  //
  if ((opt & kMaterialBit) && containsMaterial()) {
    printf("  MatCorr Exp(ELOSS): %+.4e %+.4e %+.4e %+.4e %+.4e\n",
           mMatCorrExp[0], mMatCorrExp[1], mMatCorrExp[2], mMatCorrExp[3], mMatCorrExp[4]);
    printf("  MatCorr Cov (diag): %+.4e %+.4e %+.4e %+.4e %+.4e\n",
           mMatCorrCov[0], mMatCorrCov[1], mMatCorrCov[2], mMatCorrCov[3], mMatCorrCov[4]);
    //
    if (opt & kOptUMAT) {
      float covUndiag[15];
      memset(covUndiag, 0, 15 * sizeof(float));
      int np = getNMatPar();
      for (int i = 0; i < np; i++) {
        for (int j = 0; j <= i; j++) {
          double val = 0;
          for (int k = np; k--;) {
            val += mMatDiag[i][k] * mMatDiag[j][k] * mMatCorrCov[k];
          }
          int ij = (i * (i + 1) / 2) + j;
          covUndiag[ij] = val;
        }
      }
      if (np < kNMatDOFs) {
        covUndiag[14] = mMatCorrCov[4];
      } // eloss was fixed
      printf("  MatCorr Cov in normal form:\n");
      printf("  %+e\n", covUndiag[0]);
      printf("  %+e %+e\n", covUndiag[1], covUndiag[2]);
      printf("  %+e %+e %+e\n", covUndiag[3], covUndiag[4], covUndiag[5]);
      printf("  %+e %+e %+e %+e\n", covUndiag[6], covUndiag[7], covUndiag[8], covUndiag[9]);
      printf("  %+e %+e %+e %+e +%e\n", covUndiag[10], covUndiag[11], covUndiag[12], covUndiag[13], covUndiag[14]);
    }
  }
  //
  if ((opt & kOptDiag) && containsMaterial()) {
    printf("  Matrix for Mat.corr.errors diagonalization:\n");
    int npar = getNMatPar();
    for (int i = 0; i < npar; i++) {
      for (int j = 0; j < npar; j++) {
        printf("%+.4e ", mMatDiag[i][j]);
      }
      printf("\n");
    }
  }
  //
  if (opt & kOptWSA) { // printf track state at this point stored during residuals calculation
    printf("  Local Track (A): ");
    for (int i = 0; i < 5; i++) {
      printf("%+.3e ", mTrParamWSA[i]);
    }
    printf("\n");
  }
  if (opt & kOptWSB) { // printf track state at this point stored during residuals calculation
    printf("  Local Track (B): ");
    for (int i = 0; i < 5; i++) {
      printf("%+.3e ", mTrParamWSB[i]);
    }
    printf("\n");
  }
  //
}

//_____________________________________
void AlignmentPoint::dumpCoordinates() const
{
  // dump various corrdinates for inspection
  // global xyz
  dim3_t xyz;
  getXYZGlo(xyz.data());

  auto print3d = [](dim3_t& xyz) {
    for (auto i : xyz) {
      printf("%+.4e ", i);
    }
  };

  print3d(xyz);
  trackParam_t wsb;
  trackParam_t wsa;
  getTrWSB(wsb);
  getTrWSA(wsa);

  wsb.getXYZGlo(xyz);
  print3d(xyz); // track before mat corr

  wsa.getXYZGlo(xyz);
  print3d(xyz); // track after mat corr

  printf("%+.4f ", mAlphaSens);
  printf("%+.4e ", getXTracking());
  printf("%+.4e ", getYTracking());
  printf("%+.4e ", getZTracking());
  //
  printf("%+.4e %.4e ", wsb.getY(), wsb.getZ());
  printf("%+.4e %.4e ", wsa.getY(), wsa.getZ());
  //
  printf("%4e %4e", Sqrt(mErrYZTracking[0]), Sqrt(mErrYZTracking[2]));
  printf("\n");
}

//_____________________________________
void AlignmentPoint::clear()
{
  // reset the point
  mBits = 0;
  mMaxLocVarID = -1;
  mDetID = -1;
  mSID = -1;
  mNGloDOFs = 0;
  mDGloOffs = 0;
  mSensor = nullptr;
  setXYZTracking(0., 0., 0.);
}

//__________________________________________________________________
bool AlignmentPoint::isAfter(const AlignmentPoint& pnt) const
{
  // sort points in direction opposite to track propagation, i.e.
  // 1) for tracks from collision: range in decreasing tracking X
  // 2) for cosmic tracks: upper leg (pnt->isInvDir()==true) ranged in increasing X
  //                       lower leg - in decreasing X
  double x = getXTracking();
  double xp = pnt.getXTracking();
  if (!isInvDir()) {        // track propagates from low to large X via this point
    if (!pnt.isInvDir()) {  // via this one also
      return x > xp ? -1 : 1;
    } else {
      return true;         // any point on the lower leg has higher priority than on the upper leg
    }                      // range points of lower leg 1st
  } else {                 // this point is from upper cosmic leg: track propagates from large to low X
    if (pnt.isInvDir()) {  // this one also
      return x > xp ? 1 : -1;
    } else {
      return 1;
    } // other point is from lower leg
  }
  //
}

//__________________________________________________________________
void AlignmentPoint::getXYZGlo(double r[3]) const
{
  // position in lab frame
  double cs = TMath::Cos(mAlphaSens);
  double sn = TMath::Sin(mAlphaSens);
  double x = getXTracking();
  r[0] = x * cs - getYTracking() * sn;
  r[1] = x * sn + getYTracking() * cs;
  r[2] = getZTracking();
  //
}

//__________________________________________________________________
double AlignmentPoint::getPhiGlo() const
{
  // phi angle (-pi:pi) in global frame
  double xyz[3];
  getXYZGlo(xyz);
  return ATan2(xyz[1], xyz[0]);
}

//__________________________________________________________________
int AlignmentPoint::getAliceSector() const
{
  // get global sector ID corresponding to this point phi
  return math_utils::detail::angle2Sector(getPhiGlo());
}

//__________________________________________________________________
void AlignmentPoint::setMatCovDiagonalizationMatrix(const TMatrixD& d)
{
  // save non-sym matrix for material corrections cov.matrix diagonalization
  // (actually, the eigenvectors are stored)
  int sz = d.GetNrows();
  for (int i = sz; i--;) {
    for (int j = sz; j--;) {
      mMatDiag[i][j] = d(i, j);
    }
  }
}

//__________________________________________________________________
void AlignmentPoint::setMatCovDiag(const TVectorD& v)
{
  // save material correction diagonalized matrix
  // (actually, the eigenvalues are stored w/o reordering them to correspond to the
  // AliExternalTrackParam variables)
  for (int i = v.GetNrows(); i--;) {
    mMatCorrCov[i] = v(i);
  }
}

//__________________________________________________________________
void AlignmentPoint::unDiagMatCorr(const double* diag, double* nodiag) const
{
  // transform material corrections from the frame diagonalizing the errors to point frame
  // nodiag = mMatDiag * diag
  int np = getNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;) {
      v += mMatDiag[ip][jp] * diag[jp];
    }
    nodiag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AlignmentPoint::unDiagMatCorr(const float* diag, float* nodiag) const
{
  // transform material corrections from the frame diagonalizing the errors to point frame
  // nodiag = mMatDiag * diag
  int np = getNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;) {
      v += double(mMatDiag[ip][jp]) * diag[jp];
    }
    nodiag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AlignmentPoint::diagMatCorr(const double* nodiag, double* diag) const
{
  // transform material corrections from the AliExternalTrackParam frame to
  // the frame diagonalizing the errors
  // diag = mMatDiag^T * nodiag
  int np = getNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;) {
      v += mMatDiag[jp][ip] * nodiag[jp];
    }
    diag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AlignmentPoint::diagMatCorr(const float* nodiag, float* diag) const
{
  // transform material corrections from the AliExternalTrackParam frame to
  // the frame diagonalizing the errors
  // diag = mMatDiag^T * nodiag
  int np = getNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;) {
      v += double(mMatDiag[jp][ip]) * nodiag[jp];
    }
    diag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AlignmentPoint::getTrWSA(trackParam_t& etp) const
{
  // assign WSA (after material corrections) parameters to supplied track
  const trackParam_t::covMat_t covDum{
    1.e-4,
    0, 1.e-4,
    0, 0, 1.e-4,
    0, 0, 0, 1.e-4,
    0, 0, 0, 0, 1e-4};
  params_t tmp;
  std::copy(std::begin(mTrParamWSA), std::end(mTrParamWSA), std::begin(tmp));

  etp.set(getXTracking(), getAlphaSens(), tmp, covDum);
}

//__________________________________________________________________
void AlignmentPoint::getTrWSB(trackParam_t& etp) const
{
  // assign WSB parameters (before material corrections) to supplied track
  const trackParam_t::covMat_t covDum{
    1.e-4,
    0, 1.e-4,
    0, 0, 1.e-4,
    0, 0, 0, 1.e-4,
    0, 0, 0, 0, 1e-4};
  params_t tmp;
  std::copy(std::begin(mTrParamWSB), std::end(mTrParamWSB), std::begin(tmp));

  etp.set(getXTracking(), getAlphaSens(), tmp, covDum);
}

} // namespace align
} // namespace o2
