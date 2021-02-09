// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgPoint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Meausered point in the sensor.

#include <stdio.h>
#include <TMath.h>
#include <TString.h>
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgAux.h"
//#include "AliExternalTrackParam.h"

using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

//_____________________________________
AliAlgPoint::AliAlgPoint()
  : fMinLocVarID(0), fMaxLocVarID(0), fDetID(-1), fSID(-1), fAlphaSens(0), fXSens(0), fCosDiagErr(0), fSinDiagErr(0), fX2X0(0), fXTimesRho(0), fNGloDOFs(0), fDGloOffs(0), fSensor(0)
{
  // def c-tor
  for (int i = 3; i--;) {
    fXYZTracking[i] = 0;
    fErrYZTracking[i] = 0;
  }
  memset(fMatCorrExp, 0, 5 * sizeof(float));
  memset(fMatCorrCov, 0, 5 * sizeof(float));
  memset(fMatDiag, 0, 5 * 5 * sizeof(float));
  //
  memset(fTrParamWSA, 0, 5 * sizeof(double));
  memset(fTrParamWSB, 0, 5 * sizeof(double));
  //
}

//_____________________________________
void AliAlgPoint::Init()
{
  // compute aux info
  const double kCorrToler = 1e-6;
  const double kDiagToler = 1e-14;
  //
  // compute parameters of tranformation to diagonal error matrix
  if (!IsZeroPos(fErrYZTracking[0] + fErrYZTracking[2])) {
    //
    // is there a correlation?
    if (SmallerAbs(fErrYZTracking[1] * fErrYZTracking[1] / (fErrYZTracking[0] * fErrYZTracking[2]), kCorrToler)) {
      fCosDiagErr = 1.;
      fSinDiagErr = 0.;
      fErrDiag[0] = fErrYZTracking[0];
      fErrDiag[1] = fErrYZTracking[2];
    } else {
      double dfd = 0.5 * (fErrYZTracking[2] - fErrYZTracking[0]);
      double phi = 0;
      // special treatment if errors are equal
      if (Abs(dfd) < kDiagToler)
        phi = fErrYZTracking[1] > 0 ? (Pi() * 0.25) : (Pi() * 0.75);
      else
        phi = 0.5 * ATan2(fErrYZTracking[1], dfd);
      //
      fCosDiagErr = Cos(phi);
      fSinDiagErr = Sin(phi);
      //
      //      double det = dfd*dfd + fErrYZTracking[1]*fErrYZTracking[1];
      //      det = det>0 ? Sqrt(det) : 0;
      //      double smd = 0.5*(fErrYZTracking[0] + fErrYZTracking[2]);
      //      fErrDiag[0] = smd + det;
      //      fErrDiag[1] = smd - det;
      double xterm = 2 * fCosDiagErr * fSinDiagErr * fErrYZTracking[1];
      double cc = fCosDiagErr * fCosDiagErr;
      double ss = fSinDiagErr * fSinDiagErr;
      fErrDiag[0] = fErrYZTracking[0] * cc + fErrYZTracking[2] * ss - xterm;
      fErrDiag[1] = fErrYZTracking[0] * ss + fErrYZTracking[2] * cc + xterm;
    }
  }
  //
}

//_____________________________________
void AliAlgPoint::UpdatePointByTrackInfo(const AliExternalTrackParam* t)
{
  // recalculate point errors using info about the track in the sensor tracking frame
  fSensor->UpdatePointByTrackInfo(this, t);
}

//_____________________________________
void AliAlgPoint::Print(Option_t* opt) const
{
  // print
  TString opts = opt;
  opts.ToLower();
  printf("%cDet%d SID:%4d Alp:%+.3f X:%+9.4f Meas:%s Mat: ", IsInvDir() ? '*' : ' ',
         GetDetID(), GetSID(), GetAlphaSens(), GetXSens(), ContainsMeasurement() ? "ON" : "OFF");
  if (!ContainsMaterial())
    printf("OFF\n");
  else
    printf("x2X0: %.4f x*rho: %.4f | pars:[%3d:%3d)\n", GetX2X0(), GetXTimesRho(), GetMinLocVarID(), GetMaxLocVarID());
  //
  if (opts.Contains("meas") && ContainsMeasurement()) {
    printf("  MeasPnt: Xtr: %+9.4f Ytr: %+8.4f Ztr: %+9.4f | ErrYZ: %+e %+e %+e | %d DOFglo\n",
           GetXTracking(), GetYTracking(), GetZTracking(),
           fErrYZTracking[0], fErrYZTracking[1], fErrYZTracking[2], GetNGloDOFs());
    printf("  DiagErr: %+e %+e\n", fErrDiag[0], fErrDiag[1]);
  }
  //
  if (opts.Contains("mat") && ContainsMaterial()) {
    printf("  MatCorr Exp(ELOSS): %+.4e %+.4e %+.4e %+.4e %+.4e\n",
           fMatCorrExp[0], fMatCorrExp[1], fMatCorrExp[2], fMatCorrExp[3], fMatCorrExp[4]);
    printf("  MatCorr Cov (diag): %+.4e %+.4e %+.4e %+.4e %+.4e\n",
           fMatCorrCov[0], fMatCorrCov[1], fMatCorrCov[2], fMatCorrCov[3], fMatCorrCov[4]);
    //
    if (opts.Contains("umat")) {
      float covUndiag[15];
      memset(covUndiag, 0, 15 * sizeof(float));
      int np = GetNMatPar();
      for (int i = 0; i < np; i++) {
        for (int j = 0; j <= i; j++) {
          double val = 0;
          for (int k = np; k--;)
            val += fMatDiag[i][k] * fMatDiag[j][k] * fMatCorrCov[k];
          int ij = (i * (i + 1) / 2) + j;
          covUndiag[ij] = val;
        }
      }
      if (np < kNMatDOFs)
        covUndiag[14] = fMatCorrCov[4]; // eloss was fixed
      printf("  MatCorr Cov in normal form:\n");
      printf("  %+e\n", covUndiag[0]);
      printf("  %+e %+e\n", covUndiag[1], covUndiag[2]);
      printf("  %+e %+e %+e\n", covUndiag[3], covUndiag[4], covUndiag[5]);
      printf("  %+e %+e %+e %+e\n", covUndiag[6], covUndiag[7], covUndiag[8], covUndiag[9]);
      printf("  %+e %+e %+e %+e +%e\n", covUndiag[10], covUndiag[11], covUndiag[12], covUndiag[13], covUndiag[14]);
    }
  }
  //
  if (opts.Contains("diag") && ContainsMaterial()) {
    printf("  Matrix for Mat.corr.errors diagonalization:\n");
    int npar = GetNMatPar();
    for (int i = 0; i < npar; i++) {
      for (int j = 0; j < npar; j++)
        printf("%+.4e ", fMatDiag[i][j]);
      printf("\n");
    }
  }
  //
  if (opts.Contains("wsa")) { // printf track state at this point stored during residuals calculation
    printf("  Local Track (A): ");
    for (int i = 0; i < 5; i++)
      printf("%+.3e ", fTrParamWSA[i]);
    printf("\n");
  }
  if (opts.Contains("wsb")) { // printf track state at this point stored during residuals calculation
    printf("  Local Track (B): ");
    for (int i = 0; i < 5; i++)
      printf("%+.3e ", fTrParamWSB[i]);
    printf("\n");
  }
  //
}

//_____________________________________
void AliAlgPoint::DumpCoordinates() const
{
  // dump various corrdinates for inspection
  // global xyz
  double xyz[3];
  GetXYZGlo(xyz);
  for (int i = 0; i < 3; i++)
    printf("%+.4e ", xyz[i]);
  //
  AliExternalTrackParam wsb;
  AliExternalTrackParam wsa;
  GetTrWSB(wsb);
  GetTrWSA(wsa);
  wsb.GetXYZ(xyz);
  for (int i = 0; i < 3; i++)
    printf("%+.4e ", xyz[i]); // track before mat corr
  wsa.GetXYZ(xyz);
  for (int i = 0; i < 3; i++)
    printf("%+.4e ", xyz[i]); // track after mat corr
  //
  printf("%+.4f ", fAlphaSens);
  printf("%+.4e ", GetXPoint());
  printf("%+.4e ", GetYTracking());
  printf("%+.4e ", GetZTracking());
  //
  printf("%+.4e %.4e ", wsb.GetY(), wsb.GetZ());
  printf("%+.4e %.4e ", wsa.GetY(), wsa.GetZ());
  //
  printf("%4e %4e", Sqrt(fErrYZTracking[0]), Sqrt(fErrYZTracking[2]));
  printf("\n");
}

//_____________________________________
void AliAlgPoint::Clear(Option_t*)
{
  // reset the point
  ResetBit(0xfffffff);
  fMaxLocVarID = -1;
  fDetID = -1;
  fSID = -1;
  fNGloDOFs = 0;
  fDGloOffs = 0;
  //
  fSensor = 0;
}

//__________________________________________________________________
Int_t AliAlgPoint::Compare(const TObject* b) const
{
  // sort points in direction opposite to track propagation, i.e.
  // 1) for tracks from collision: range in decreasing tracking X
  // 2) for cosmic tracks: upper leg (pnt->IsInvDir()==kTRUE) ranged in increasing X
  //                       lower leg - in decreasing X
  AliAlgPoint* pnt = (AliAlgPoint*)b;
  double x = GetXPoint();
  double xp = pnt->GetXPoint();
  if (!IsInvDir()) {        // track propagates from low to large X via this point
    if (!pnt->IsInvDir()) { // via this one also
      return x > xp ? -1 : 1;
    } else
      return -1;           // range points of lower leg 1st
  } else {                 // this point is from upper cosmic leg: track propagates from large to low X
    if (pnt->IsInvDir()) { // this one also
      return x > xp ? 1 : -1;
    } else
      return 1; // other point is from lower leg
  }
  //
}

//__________________________________________________________________
void AliAlgPoint::GetXYZGlo(Double_t r[3]) const
{
  // position in lab frame
  double cs = TMath::Cos(fAlphaSens);
  double sn = TMath::Sin(fAlphaSens);
  double x = GetXPoint();
  r[0] = x * cs - GetYTracking() * sn;
  r[1] = x * sn + GetYTracking() * cs;
  r[2] = GetZTracking();
  //
}

//__________________________________________________________________
Double_t AliAlgPoint::GetPhiGlo() const
{
  // phi angle (-pi:pi) in global frame
  double xyz[3];
  GetXYZGlo(xyz);
  return ATan2(xyz[1], xyz[0]);
}

//__________________________________________________________________
Int_t AliAlgPoint::GetAliceSector() const
{
  // get global sector ID corresponding to this point phi
  return Phi2Sector(GetPhiGlo());
}

//__________________________________________________________________
void AliAlgPoint::SetMatCovDiagonalizationMatrix(const TMatrixD& d)
{
  // save non-sym matrix for material corrections cov.matrix diagonalization
  // (actually, the eigenvectors are stored)
  int sz = d.GetNrows();
  for (int i = sz; i--;)
    for (int j = sz; j--;)
      fMatDiag[i][j] = d(i, j);
}

//__________________________________________________________________
void AliAlgPoint::SetMatCovDiag(const TVectorD& v)
{
  // save material correction diagonalized matrix
  // (actually, the eigenvalues are stored w/o reordering them to correspond to the
  // AliExternalTrackParam variables)
  for (int i = v.GetNrows(); i--;)
    fMatCorrCov[i] = v(i);
}

//__________________________________________________________________
void AliAlgPoint::UnDiagMatCorr(const double* diag, double* nodiag) const
{
  // transform material corrections from the frame diagonalizing the errors to point frame
  // nodiag = fMatDiag * diag
  int np = GetNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;)
      v += fMatDiag[ip][jp] * diag[jp];
    nodiag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AliAlgPoint::UnDiagMatCorr(const float* diag, float* nodiag) const
{
  // transform material corrections from the frame diagonalizing the errors to point frame
  // nodiag = fMatDiag * diag
  int np = GetNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;)
      v += double(fMatDiag[ip][jp]) * diag[jp];
    nodiag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AliAlgPoint::DiagMatCorr(const double* nodiag, double* diag) const
{
  // transform material corrections from the AliExternalTrackParam frame to
  // the frame diagonalizing the errors
  // diag = fMatDiag^T * nodiag
  int np = GetNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;)
      v += fMatDiag[jp][ip] * nodiag[jp];
    diag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AliAlgPoint::DiagMatCorr(const float* nodiag, float* diag) const
{
  // transform material corrections from the AliExternalTrackParam frame to
  // the frame diagonalizing the errors
  // diag = fMatDiag^T * nodiag
  int np = GetNMatPar();
  for (int ip = np; ip--;) {
    double v = 0;
    for (int jp = np; jp--;)
      v += double(fMatDiag[jp][ip]) * nodiag[jp];
    diag[ip] = v;
  }
  //
}

//__________________________________________________________________
void AliAlgPoint::GetTrWSA(AliExternalTrackParam& etp) const
{
  // assign WSA (after material corrections) parameters to supplied track
  double covDum[15] = {
    1.e-4,
    0, 1.e-4,
    0, 0, 1.e-4,
    0, 0, 0, 1.e-4,
    0, 0, 0, 0, 1e-4};
  etp.Set(GetXPoint(), GetAlphaSens(), fTrParamWSA, covDum);
}

//__________________________________________________________________
void AliAlgPoint::GetTrWSB(AliExternalTrackParam& etp) const
{
  // assign WSB parameters (before material corrections) to supplied track
  double covDum[15] = {
    1.e-4,
    0, 1.e-4,
    0, 0, 1.e-4,
    0, 0, 0, 1.e-4,
    0, 0, 0, 0, 1e-4};
  etp.Set(GetXPoint(), GetAlphaSens(), fTrParamWSB, covDum);
}

} // namespace align
} // namespace o2
