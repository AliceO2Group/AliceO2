// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgTrack.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Track model for the alignment

#include <stdio.h>
#include "Align/AliAlgTrack.h"
//#include "AliTrackerBase.h"
#include "Framework/Logger.h"
#include "Align/AliAlgSens.h"
#include "Align/AliAlgVol.h"
#include "Align/AliAlgDet.h"
#include "Align/AliAlgAux.h"
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TMatrixDSymEigen.h>

using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

// RS: this is not good: we define constants outside the class, but it is to
// bypass the CINT limitations on static arrays initializations
const Int_t kRichardsonOrd = 1;                // Order of Richardson extrapolation for derivative (min=1)
const Int_t kRichardsonN = kRichardsonOrd + 1; // N of 2-point symmetric derivatives needed for requested order
const Int_t kNRDClones = kRichardsonN * 2;     // number of variations for derivative of requested order

//____________________________________________________________________________
AliAlgTrack::AliAlgTrack() : TrackParametrizationWithError<double>(), TObject(), fNLocPar(0), fNLocExtPar(0), fNGloPar(0), fNDF(0), fInnerPointID(0)
                             //  ,fMinX2X0Pt2Account(5/1.0)
                             ,
                             fMinX2X0Pt2Account(0.5e-3 / 1.0),
                             fMass(0.14),
                             fChi2(0),
                             fChi2CosmUp(0),
                             fChi2CosmDn(0),
                             fChi2Ini(0),
                             fPoints(0),
                             fLocPar(),
                             fGloParID(0),
                             fGloParIDA(0),
                             fLocParA(0)
{
  // def c-tor
  for (int i = 0; i < 2; i++) {
    // we start with 0 size buffers for derivatives, they will be expanded automatically
    fResid[i].Set(0);
    fDResDGlo[i].Set(0);
    fDResDLoc[i].Set(0);
    //
    fResidA[i] = 0;
    fDResDLocA[i] = 0;
    fDResDGloA[i] = 0;
  }
  fNeedInv[0] = fNeedInv[1] = kFALSE;
  //
}

//____________________________________________________________________________
AliAlgTrack::~AliAlgTrack()
{
  // d-tor
}

//____________________________________________________________________________
void AliAlgTrack::Clear(Option_t*)
{
  // reset the track
  TObject::Clear();
  ResetBit(0xffffffff);
  fPoints.Clear();
  fChi2 = fChi2CosmUp = fChi2CosmDn = fChi2Ini = 0;
  fNDF = 0;
  fInnerPointID = -1;
  fNeedInv[0] = fNeedInv[1] = kFALSE;
  fNLocPar = fNLocExtPar = fNGloPar = 0;
  //
}

//____________________________________________________________________________
void AliAlgTrack::DefineDOFs()
{
  // define varied DOF's (local parameters) for the track:
  // 1) kinematic params (5 or 4 depending on Bfield)
  // 2) mult. scattering angles (2)
  // 3) if requested by point: energy loss
  //
  fNLocPar = fNLocExtPar = GetFieldON() ? kNKinParBON : kNKinParBOFF;
  int np = GetNPoints();
  //
  // the points are sorted in order opposite to track direction -> outer points come 1st,
  // but for the 2-leg cosmic track the innermost points are in the middle (1st lower leg, then upper one)
  //
  // start along track direction, i.e. last point in the ordered array
  int minPar = fNLocPar;
  for (int ip = GetInnerPointID() + 1; ip--;) { // collision track or cosmic lower leg
    AliAlgPoint* pnt = GetPoint(ip);
    pnt->SetMinLocVarID(minPar);
    if (pnt->ContainsMaterial())
      fNLocPar += pnt->GetNMatPar();
    pnt->SetMaxLocVarID(fNLocPar); // flag up to which parameted ID this points depends on
  }
  //
  if (IsCosmic()) {
    minPar = fNLocPar;
    for (int ip = GetInnerPointID() + 1; ip < np; ip++) { // collision track or cosmic lower leg
      AliAlgPoint* pnt = GetPoint(ip);
      pnt->SetMinLocVarID(minPar);
      if (pnt->ContainsMaterial())
        fNLocPar += pnt->GetNMatPar();
      pnt->SetMaxLocVarID(fNLocPar); // flag up to which parameted ID this points depends on
    }
  }
  //
  if (fLocPar.GetSize() < fNLocPar)
    fLocPar.Set(fNLocPar);
  fLocPar.Reset();
  fLocParA = fLocPar.GetArray();
  //
  if (fResid[0].GetSize() < np) {
    fResid[0].Set(np);
    fResid[1].Set(np);
  }
  if (fDResDLoc[0].GetSize() < fNLocPar * np) {
    fDResDLoc[0].Set(fNLocPar * np);
    fDResDLoc[1].Set(fNLocPar * np);
  }
  for (int i = 2; i--;) {
    fResid[i].Reset();
    fDResDLoc[i].Reset();
    fResidA[i] = fResid[i].GetArray();
    fDResDLocA[i] = fDResDLoc[i].GetArray();
  }
  //
  //  memcpy(fLocParA,GetParameter(),fNLocExtPar*sizeof(Double_t));
  memset(fLocParA, 0, fNLocExtPar * sizeof(Double_t));
}

//______________________________________________________
Bool_t AliAlgTrack::CalcResidDeriv(Double_t* params)
{
  // Propagate for given local params and calculate residuals and their derivatives.
  // The 1st 4 or 5 elements of params vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the ApplyMatCorr routine.
  //
  // If params are not provided, use internal params array
  //
  if (!params)
    params = fLocParA;
  //
  if (!GetResidDone())
    CalcResiduals(params);
  //
  int np = GetNPoints();
  //
  // collision track or cosmic lower leg
  if (!CalcResidDeriv(params, fNeedInv[0], GetInnerPointID(), 0)) {
#if DEBUG > 3
    LOG(warn) << "Failed on derivatives calculation 0";
#endif
    return kFALSE;
  }
  //
  if (IsCosmic()) { // cosmic upper leg
    if (!CalcResidDeriv(params, fNeedInv[1], GetInnerPointID() + 1, np - 1)) {
#if DEBUG > 3
      LOG(warn) << "Failed on derivatives calculation 0";
#endif
    }
  }
  //
  SetDerivDone();
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::CalcResidDeriv(double* extendedParams, Bool_t invert, int pFrom, int pTo)
{
  // Calculate derivatives of residuals vs params for points pFrom to pT. For cosmic upper leg
  // track parameter may require inversion.
  // The 1st 4 or 5 elements of params vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the ApplyMatCorr routine.
  //
  // The derivatives are calculated using Richardson extrapolation
  // (like http://root.cern.ch/root/html/ROOT__Math__RichardsonDerivator.html)
  //
  trackParam_t probD[kNRDClones]; // use this to vary supplied param for derivative calculation
  double varDelta[kRichardsonN];
  const int kInvElem[kNKinParBON] = {-1, 1, 1, -1, -1};
  //
  const double kDelta[kNKinParBON] = {0.02, 0.02, 0.001, 0.001, 0.01}; // variations for ExtTrackParam and material effects
  //
  double delta[kNKinParBON]; // variations of curvature term are relative
  for (int i = kNKinParBOFF; i--;)
    delta[i] = kDelta[i];
  if (GetFieldON())
    delta[kParQ2Pt] = kDelta[kParQ2Pt] * Abs(getQ2Pt());
  //
  int pinc;
  if (pTo > pFrom) { // fit in points decreasing order: cosmics upper leg
    pTo++;
    pinc = 1;
  } else { // fit in points increasing order: collision track or cosmics lower leg
    pTo--;
    pinc = -1;
  }

  // 1) derivative wrt trackParam_t parameters
  for (int ipar = fNLocExtPar; ipar--;) {

    SetParams(probD, kNRDClones, getX(), getAlpha(), extendedParams, kTRUE);
    if (invert)
      for (int ic = kNRDClones; ic--;)
        probD[ic].invert();
    double del = delta[ipar];
    //
    for (int icl = 0; icl < kRichardsonN; icl++) { // calculate kRichardsonN variations with del, del/2, del/4...
      varDelta[icl] = del;
      ModParam(probD[(icl << 1) + 0], ipar, del);
      ModParam(probD[(icl << 1) + 1], ipar, -del);
      del *= 0.5;
    }
    // propagate varied tracks to each point
    for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
      AliAlgPoint* pnt = GetPoint(ip);
      if (!PropagateParamToPoint(probD, kNRDClones, pnt))
        return kFALSE;
      //      if (pnt->ContainsMaterial()) { // apply material corrections
      if (!ApplyMatCorr(probD, kNRDClones, extendedParams, pnt))
        return kFALSE;
      //      }
      //
      if (pnt->ContainsMeasurement()) {
        int offsDer = ip * fNLocPar + ipar;
        RichardsonDeriv(probD, varDelta, pnt, fDResDLocA[0][offsDer], fDResDLocA[1][offsDer]); // calculate derivatives
        if (invert && kInvElem[ipar] < 0) {
          fDResDLocA[0][offsDer] = -fDResDLocA[0][offsDer];
          fDResDLocA[1][offsDer] = -fDResDLocA[1][offsDer];
        }
      }
    } // loop over points
  }   // loop over ExtTrackParam parameters
  //
  // 2) now vary material effect related parameters: MS and eventually ELoss
  //
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
    AliAlgPoint* pnt = GetPoint(ip);
    //
    // global derivatives at this point
    if (pnt->ContainsMeasurement() && !CalcResidDerivGlo(pnt)) {
#if DEBUG > 3
      AliWarningF("Failed on global derivatives calculation at point %d", ip);
      pnt->print("meas");
#endif
      return kFALSE;
    }
    //
    if (!pnt->ContainsMaterial())
      continue;
    //
    int nParFreeI = pnt->GetNMatPar();
    //
    // array delta gives desired variation of parameters in trackParam_t definition,
    // while the variation should be done for parameters in the frame where the vector
    // of material corrections has diagonal cov. matrix -> rotate the delta to this frame
    double deltaMatD[kNKinParBON];
    pnt->DiagMatCorr(delta, deltaMatD);
    //
    //    printf("Vary %d [%+.3e %+.3e %+.3e %+.3e] ",ip,deltaMatD[0],deltaMatD[1],deltaMatD[2],deltaMatD[3]); pnt->print();

    int offsI = pnt->GetMaxLocVarID() - nParFreeI; // the parameters for this point start with this offset
                                                   // they are irrelevant for the points upstream
    for (int ipar = 0; ipar < nParFreeI; ipar++) { // loop over DOFs related to MS and ELoss are point ip
      double del = deltaMatD[ipar];
      //
      // We will vary the tracks starting from the original parameters propagated to given point
      // and stored there (before applying material corrections for this point)
      //
      SetParams(probD, kNRDClones, pnt->GetXPoint(), pnt->GetAlphaSens(), pnt->GetTrParamWSB(), kFALSE);
      // no need for eventual track inversion here: if needed, this is already done in ParamWSB
      //
      int offsIP = offsI + ipar; // parameter entry in the extendedParams array
      //      printf("  Var:%d (%d)  %e\n",ipar,offsIP, del);

      for (int icl = 0; icl < kRichardsonN; icl++) { // calculate kRichardsonN variations with del, del/2, del/4...
        varDelta[icl] = del;
        double parOrig = extendedParams[offsIP];
        extendedParams[offsIP] += del;
        //
        // apply varied material effects : incremented by delta
        if (!ApplyMatCorr(probD[(icl << 1) + 0], extendedParams, pnt))
          return kFALSE;
        //
        // apply varied material effects : decremented by delta
        extendedParams[offsIP] = parOrig - del;
        if (!ApplyMatCorr(probD[(icl << 1) + 1], extendedParams, pnt))
          return kFALSE;
        //
        extendedParams[offsIP] = parOrig;
        del *= 0.5;
      }
      if (pnt->ContainsMeasurement()) { // calculate derivatives at the scattering point itself
        int offsDerIP = ip * fNLocPar + offsIP;
        RichardsonDeriv(probD, varDelta, pnt, fDResDLocA[0][offsDerIP], fDResDLocA[1][offsDerIP]); // calculate derivatives for ip
                                                                                                   //	printf("DR SELF: %e %e at %d (%d)\n",fDResDLocA[0][offsDerIP], fDResDLocA[1][offsDerIP],offsI, offsDerIP);
      }
      //
      // loop over points whose residuals can be affected by the material effects on point ip
      for (int jp = ip + pinc; jp != pTo; jp += pinc) {
        AliAlgPoint* pntJ = GetPoint(jp);

        //	printf("  DerFor:%d ",jp); pntJ->print();

        if (!PropagateParamToPoint(probD, kNRDClones, pntJ))
          return kFALSE;
        //
        if (pntJ->ContainsMaterial()) { // apply material corrections
          if (!ApplyMatCorr(probD, kNRDClones, extendedParams, pntJ))
            return kFALSE;
        }
        //
        if (pntJ->ContainsMeasurement()) {
          int offsDerJ = jp * fNLocPar + offsIP;
          // calculate derivatives
          RichardsonDeriv(probD, varDelta, pntJ, fDResDLocA[0][offsDerJ], fDResDLocA[1][offsDerJ]);
        }
        //
      } // << loop over points whose residuals can be affected by the material effects on point ip
    }   // << loop over DOFs related to MS and ELoss are point ip
  }     // << loop over all points of the track
  //
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::CalcResidDerivGlo(AliAlgPoint* pnt)
{
  // calculate residuals derivatives over point's sensor and its parents global params
  double deriv[AliAlgVol::kNDOFGeom * 3];
  //
  const AliAlgSens* sens = pnt->GetSensor();
  const AliAlgVol* vol = sens;
  // precalculated track parameters
  double snp = pnt->GetTrParamWSA(kParSnp), tgl = pnt->GetTrParamWSA(kParTgl);
  // precalculate track slopes to account tracking X veriation
  // these are coeffs to translate deltaX of the point to deltaY and deltaZ of track
  double cspi = 1. / Sqrt((1 - snp) * (1 + snp)), slpY = snp * cspi, slpZ = tgl * cspi;
  //
  pnt->SetDGloOffs(fNGloPar); // mark 1st entry of derivatives
  do {
    // measurement residuals
    int nfree = vol->GetNDOFFree();
    if (!nfree)
      continue; // no free parameters?
    sens->DPosTraDParGeom(pnt, deriv, vol == sens ? 0 : vol);
    //
    CheckExpandDerGloBuffer(fNGloPar + nfree); // if needed, expand derivatives buffer
    //
    for (int ip = 0; ip < AliAlgVol::kNDOFGeom; ip++) { // we need only free parameters
      if (!vol->IsFreeDOF(ip))
        continue;
      double* dXYZ = &deriv[ip * 3]; // tracking XYZ derivatives over this parameter
      // residual is defined as diagonalized track_estimate - measured Y,Z in tracking frame
      // where the track is evaluated at measured X!
      // -> take into account modified X using track parameterization at the point (paramWSA)
      // Attention: small simplifications(to be checked if it is ok!!!):
      // effect of changing X is accounted neglecting track curvature to preserve linearity
      //
      // store diagonalized residuals in track buffer
      pnt->DiagonalizeResiduals((dXYZ[AliAlgPoint::kX] * slpY - dXYZ[AliAlgPoint::kY]),
                                (dXYZ[AliAlgPoint::kX] * slpZ - dXYZ[AliAlgPoint::kZ]),
                                fDResDGloA[0][fNGloPar], fDResDGloA[1][fNGloPar]);
      // and register global ID of varied parameter
      fGloParIDA[fNGloPar] = vol->GetParGloID(ip);
      fNGloPar++;
    }
    //
  } while ((vol = vol->GetParent()));
  //
  // eventual detector calibration parameters
  const AliAlgDet* det = sens->GetDetector();
  int ndof = 0;
  if (det && (ndof = det->GetNCalibDOFs())) {
    // if needed, expand derivatives buffer
    CheckExpandDerGloBuffer(fNGloPar + det->GetNCalibDOFsFree());
    for (int idf = 0; idf < ndof; idf++) {
      if (!det->IsFreeDOF(idf))
        continue;
      sens->DPosTraDParCalib(pnt, deriv, idf, 0);
      pnt->DiagonalizeResiduals((deriv[AliAlgPoint::kX] * slpY - deriv[AliAlgPoint::kY]),
                                (deriv[AliAlgPoint::kX] * slpZ - deriv[AliAlgPoint::kZ]),
                                fDResDGloA[0][fNGloPar], fDResDGloA[1][fNGloPar]);
      // and register global ID of varied parameter
      fGloParIDA[fNGloPar] = det->GetParGloID(idf);
      fNGloPar++;
    }
  }
  //
  pnt->SetNGloDOFs(fNGloPar - pnt->GetDGloOffs()); // mark number of global derivatives filled
  //
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::CalcResiduals(const double* extendedParams)
{
  // Propagate for given local params and calculate residuals
  // The 1st 4 or 5 elements of extendedParams vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the ApplyMatCorr routine.
  //
  // If extendedParams are not provided, use internal extendedParams array
  //
  if (!extendedParams)
    extendedParams = fLocParA;
  int np = GetNPoints();
  fChi2 = 0;
  fNDF = 0;
  //
  // collision track or cosmic lower leg
  if (!CalcResiduals(extendedParams, fNeedInv[0], GetInnerPointID(), 0)) {
#if DEBUG > 3
    LOG(warn) << "Failed on residuals calculation 0";
#endif
    return kFALSE;
  }
  //
  if (IsCosmic()) { // cosmic upper leg
    if (!CalcResiduals(extendedParams, fNeedInv[1], GetInnerPointID() + 1, np - 1)) {
#if DEBUG > 3
      LOG(warn) << "Failed on residuals calculation 1";
#endif
      return kFALSE;
    }
  }
  //
  fNDF -= fNLocExtPar;
  SetResidDone();
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::CalcResiduals(const double* extendedParams, Bool_t invert, int pFrom, int pTo)
{
  // Calculate residuals for the single leg from points pFrom to pT
  // The 1st 4 or 5 elements of extendedParams vector should be corrections to
  // the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the ApplyMatCorr routine.
  //
  trackParam_t probe;
  SetParams(probe, getX(), getAlpha(), extendedParams, kTRUE);
  if (invert)
    probe.invert();
  int pinc;
  if (pTo > pFrom) { // fit in points decreasing order: cosmics upper leg
    pTo++;
    pinc = 1;
  } else { // fit in points increasing order: collision track or cosmics lower leg
    pTo--;
    pinc = -1;
  }
  //
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
    AliAlgPoint* pnt = GetPoint(ip);
    if (!PropagateParamToPoint(probe, pnt))
      return kFALSE;
    //
    // store the current track kinematics at the point BEFORE applying eventual material
    // corrections. This kinematics will be later varied around supplied parameters (in the CalcResidDeriv)
    pnt->SetTrParamWSB(probe.getParams());
    //
    // account for materials
    //    if (pnt->ContainsMaterial()) { // apply material corrections
    if (!ApplyMatCorr(probe, extendedParams, pnt))
      return kFALSE;
    //    }
    pnt->SetTrParamWSA(probe.getParams());
    //
    if (pnt->ContainsMeasurement()) { // need to calculate residuals in the frame where errors are orthogonal
      pnt->GetResidualsDiag(probe.getParams(), fResidA[0][ip], fResidA[1][ip]);
      fChi2 += fResidA[0][ip] * fResidA[0][ip] / pnt->GetErrDiag(0);
      fChi2 += fResidA[1][ip] * fResidA[1][ip] / pnt->GetErrDiag(1);
      fNDF += 2;
    }
    //
    if (pnt->ContainsMaterial()) {
      // material degrees of freedom do not contribute to NDF since they are constrained by 0 expectation
      int nCorrPar = pnt->GetNMatPar();
      const double* corrDiag = &fLocParA[pnt->GetMaxLocVarID() - nCorrPar]; // corrections in diagonalized frame
      float* corCov = pnt->GetMatCorrCov();                                 // correction diagonalized covariance
      for (int i = 0; i < nCorrPar; i++)
        fChi2 += corrDiag[i] * corrDiag[i] / corCov[i];
    }
  }
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::PropagateParamToPoint(trackParam_t* tr, int nTr, const AliAlgPoint* pnt, double maxStep)
{
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  // FIXME(milettri): needs AliTrackerBase
  //  // Propagate set of tracks to the point  (only parameters, no error matrix)
  //  // VECTORIZE this
  //  //
  //  for (int itr = nTr; itr--;) {
  //    if (!PropagateParamToPoint(tr[itr], pnt, maxStep)) {
  //#if DEBUG > 3
  //      LOG(fatal) << "Failed on clone %d propagation" << itr;
  //      tr[itr].print();
  //      pnt->print("meas mat");
  //#endif
  //      return kFALSE;
  //    }
  //  }
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::PropagateParamToPoint(trackParam_t& tr, const AliAlgPoint* pnt, double maxStep)
{
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  // FIXME(milettri): needs AliTrackerBase
  //  // propagate tracks to the point (only parameters, no error matrix)
  //  dim3_t xyz;
  //  double bxyz[3];
  //  //
  //  if (!tr.rotateParam(pnt->GetAlphaSens())) {
  //#if DEBUG > 3
  //    LOG(error) << "Failed to rotate to alpha=" << pnt->GetAlphaSens();
  //    tr.Print();
  //    pnt->Print();
  //#endif
  //    return kFALSE;
  //  }
  //  //
  //  double xTgt = pnt->GetXPoint();
  //  double xBeg = tr.getX();
  //  double dx = xTgt - xBeg;
  //  int nstep = int(Abs(dx) / maxStep) + 1;
  //  dx /= nstep;
  //  //
  //  for (int ist = nstep; ist--;) {
  //    //
  //    double xToGo = xTgt - dx * ist;
  //    tr.getXYZGlo(xyz);
  //    //
  //    if (GetFieldON()) {
  //      if (pnt->GetUseBzOnly()) {
  //        if (!tr.propagateParamTo(xToGo, AliTrackerBase::GetBz(xyz))) {
  //#if DEBUG > 3
  //          LOG(error) << "Failed to propagate(BZ) to X=%" << pnt->GetXPoint();
  //          tr.Print();
  //          pnt->Print();
  //#endif
  //          return kFALSE;
  //        }
  //      } else {
  //        AliTrackerBase::GetBxByBz(xyz, bxyz);
  //        if (!tr.propagateParamTo(xToGo, bxyz)) {
  //#if DEBUG > 3
  //          LOG(error) << "Failed to propagate(BXYZ) to X=" << pnt->GetXPoint();
  //          tr.Print();
  //          pnt->Print();
  //#endif
  //          return kFALSE;
  //        }
  //      }
  //    } else { // straigth line propagation
  //      if (!tr.propagateParamTo(xToGo, 0)) {
  //#if DEBUG > 3
  //        LOG(error) << "Failed to propagate(B=0) to X=" << pnt->GetXPoint();
  //        tr.Print();
  //        pnt->Print();
  //#endif
  //        return kFALSE;
  //      }
  //    }
  //  } // steps
  //  //
  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::PropagateToPoint(trackParam_t& tr, const AliAlgPoint* pnt,
                                     int minNSteps, double maxStep, Bool_t matCor, double* matPar)
{
  LOG(FATAL) << __PRETTY_FUNCTION__ << " is disabled";
  // FIXME(milettri): needs AliTrackerBase
  //  // propagate tracks to the point. If matCor is true, then material corrections will be applied.
  //  // if matPar pointer is provided, it will be filled by total x2x0 and signed xrho
  //  if (!tr.rotate(pnt->GetAlphaSens())) {
  //#if DEBUG > 3
  //    LOG(WARNING) << "Failed to rotate to alpha=" << pnt->GetAlphaSens();
  //    tr.print();
  //#endif
  //    return kFALSE;
  //  }
  //  //
  //  dim3_t xyz0{0};
  //  dim3_t xyz1{0};
  //  dim3_t bxyz{0};
  //  double matarr[7];
  //  double xPoint = pnt->GetXPoint(), dx = xPoint - tr.getX(), dxa = Abs(dx), step = dxa / minNSteps;
  //  if (matPar)
  //    matPar[0] = matPar[1] = 0;
  //  if (dxa < kTinyDist)
  //    return kTRUE;
  //  if (step > maxStep)
  //    step = maxStep;
  //  int nstep = int(dxa / step);
  //  step = dxa / nstep;
  //  if (dx < 0)
  //    step = -step;
  //  //
  //  //  printf("-->will go from X:%e to X:%e in %d steps of %f\n",tr.GetX(),xPoint,nstep,step);
  //
  //  // do we go along or against track direction
  //  Bool_t alongTrackDir = (dx > 0 && !pnt->IsInvDir()) || (dx < 0 && pnt->IsInvDir());
  //  Bool_t queryXYZ = matCor || GetFieldON();
  //  if (queryXYZ)
  //    tr.getXYZGlo(xyz0);
  //  //
  //  double x2X0Tot = 0, xrhoTot = 0;
  //  for (int ist = nstep; ist--;) { // single propagation step >>
  //    double xToGo = xPoint - step * ist;
  //    //
  //    if (GetFieldON()) {
  //      if (pnt->GetUseBzOnly()) {
  //        if (!tr.propagateParamTo(xToGo, AliTrackerBase::GetBz(xyz0))) {
  //#if DEBUG > 3
  //          LOG(WARNING) << "Failed to propagate(BZ) to X=" << xToGo;
  //          tr.print();
  //#endif
  //          return kFALSE;
  //        }
  //      } else {
  //        AliTrackerBase::GetBxByBz(xyz0, bxyz);
  //        if (!tr.propagateParamTo(xToGo, bxyz)) {
  //#if DEBUG > 3
  //          AliWarningF("Failed to propagate(BXYZ) to X=%f", xToGo);
  //#endif
  //          return kFALSE;
  //        }
  //      }
  //    } else { // straigth line propagation
  //      if (!tr.propagateParamTo(xToGo, 0)) {
  //#if DEBUG > 3
  //        AliWarningF("Failed to propagate(B=0) to X=%f", xToGo);
  //#endif
  //        return kFALSE;
  //      }
  //    }
  //    //
  //    if (queryXYZ) {
  //      tr.getXYZGlo(xyz1);
  //      if (matCor) {
  //        AliTrackerBase::MeanMaterialBudget(xyz0, xyz1, matarr);
  //        Double_t xrho = matarr[0] * matarr[4], xx0 = matarr[1];
  //        if (alongTrackDir)
  //          xrho = -xrho; // if we go along track direction, energy correction is negative
  //        x2X0Tot += xx0;
  //        xrhoTot += xrho;
  //        //	printf("MAT %+7.2f %+7.2f %+7.2f -> %+7.2f %+7.2f %+7.2f | %+e %+e | -> %+e %+e | %+e %+e %+e %+e %+e\n",
  //        //	       xyz0[0],xyz0[1],xyz0[g2], xyz1[0],xyz1[1],xyz1[2], tr.Phi(), tr.GetAlpha(),
  //        //	       x2X0Tot,xrhoTot, matarr[0],matarr[1],matarr[2],matarr[3],matarr[4]);
  //        if (!tr.correctForMaterial(xx0, xrho, fMass)) {
  //#if DEBUG > 3
  //          LOG(WARNING) << "Failed on CorrectForMeanMaterial(" << xx0 << "," << xrho << "," << fMass << ")";
  //          tr.print();
  //#endif
  //          return kFALSE;
  //        }
  //      }
  //      for (int l = 3; l--;)
  //        xyz0[l] = xyz1[l];
  //    }
  //  } // single propagation step <<
  //  //
  //  if (matPar) {
  //    matPar[0] = x2X0Tot;
  //    matPar[1] = xrhoTot;
  //  }
  return kTRUE;
}

/*
//______________________________________________________
Bool_t AliAlgTrack::ApplyMS(trackParam_t& trPar, double tms,double pms)
{
  //------------------------------------------------------------------------------
  // Modify track par (e.g. trackParam_t) in the tracking frame
  // (dip angle lam, az. angle phi)
  // by multiple scattering defined by polar and azumuthal scattering angles in
  // the track collinear frame (tms and pms resp).
  // The updated direction vector in the tracking frame becomes
  //
  //  | Cos[lam]*Cos[phi] Cos[phi]*Sin[lam] -Sin[phi] |   | Cos[tms]         |
  //  | Cos[lam]*Sin[phi] Sin[lam]*Sin[phi]  Cos[phi] | x | Cos[pms]*Sin[tms]|
  //  | Sin[lam]	       -Cos[lam]	0     |   | Sin[pms]*Sin[tms]|
  //
  //------------------------------------------------------------------------------
  //
  double *par = (double*) trPar.GetParameter();
  //
  if (Abs(tms)<1e-7) return kTRUE;
  //
  double snTms = Sin(tms), csTms = Cos(tms);
  double snPms = Sin(pms), csPms = Cos(pms);
  double snPhi = par[2],  csPhi = Sqrt((1.-snPhi)*(1.+snPhi));
  double csLam = 1./Sqrt(1.+par[3]*par[3]), snLam = csLam*par[3];
  //
  double  r00 = csLam*csPhi, r01 = snLam*csPhi, &r02 = snPhi;
  double  r10 = csLam*snPhi, r11 = snLam*snPhi, &r12 = csPhi;
  double &r20 = snLam      ,&r21 = csLam;
  //
  double &v0 = csTms, v1 = snTms*csPms, v2 = snTms*snPms;
  //
  double px = r00*v0 + r01*v1 - r02*v2;
  double py = r10*v0 + r11*v1 + r12*v2;
  double pz = r20*v0 - r21*v1;
  //
  double pt = Sqrt(px*px + py*py);
  par[2] = py/pt;
  par[3] = pz/pt;
  par[4]*= csLam/pt;
  //
  return kTRUE;
}
*/

//______________________________________________________
Bool_t AliAlgTrack::ApplyMatCorr(trackParam_t& trPar, const Double_t* corrPar, const AliAlgPoint* pnt)
{
  // Modify track param (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corrPar contains delta to track parameters rotated by the matrix
  // DIAGONALIZING ITS  COVARIANCE MATRIX!
  // transform parameters from the frame diagonalizing the errors to track frame
  double corr[kNKinParBON] = {0};
  if (pnt->ContainsMaterial()) { // are there free params from meterials?
    int nCorrPar = pnt->GetNMatPar();
    const double* corrDiag = &corrPar[pnt->GetMaxLocVarID() - nCorrPar]; // material corrections for this point start here
    pnt->UnDiagMatCorr(corrDiag, corr);                                  // this is to account for MS and RANDOM Eloss (if varied)
  }
  // to this we should add expected parameters modification due to the deterministic eloss
  float* detELoss = pnt->GetMatCorrExp();
  for (int i = kNKinParBON; i--;)
    corr[i] += detELoss[i];
  //corr[kParQ2Pt] += detELoss[kParQ2Pt];
  //  printf("apply corr UD %+.3e %+.3e %+.3e %+.3e %+.3e\n",corr[0],corr[1],corr[2],corr[3],corr[4]);
  //  printf("      corr  D %+.3e %+.3e %+.3e %+.3e\n",corrDiag[0],corrDiag[1],corrDiag[2],corrDiag[3]);
  //  printf("at point :"); pnt->print();
  return ApplyMatCorr(trPar, corr);
  //
}

//______________________________________________________
Bool_t AliAlgTrack::ApplyMatCorr(trackParam_t& trPar, const Double_t* corr)
{
  // Modify track param (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corr contains delta to track frame, NOT in diagonalized one
  const double kMaxSnp = 0.95;

  const double snp = trPar.getSnp() + corr[kParSnp];
  if (Abs(snp) > kMaxSnp) {
#if DEBUG > 3
    LOG(error) << "Snp is too large: " << snp;
    printf("DeltaPar: ");
    for (int i = 0; i < kNKinParBON; i++)
      printf("%+.3e ", corr[i]);
    printf("\n");
    trPar.print();
#endif
    return kFALSE;
  }

  trPar.updateParams(corr);

  return kTRUE;
}

//______________________________________________________
Bool_t AliAlgTrack::ApplyMatCorr(trackParam_t* trSet, int ntr, const Double_t* corrDiag, const AliAlgPoint* pnt)
{
  // Modify set of track params (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corrDiag contain delta to track parameters rotated by the matrix DIAGONALIZING ITS
  // COVARIANCE MATRIX
  // transform parameters from the frame diagonalizing the errors to track frame
  double corr[kNKinParBON] = {0};
  if (pnt->ContainsMaterial()) { // are there free params from meterials?
    int nCorrPar = pnt->GetNMatPar();
    const double* corrDiagP = &corrDiag[pnt->GetMaxLocVarID() - nCorrPar]; // material corrections for this point start here
    pnt->UnDiagMatCorr(corrDiagP, corr);
  }
  float* detELoss = pnt->GetMatCorrExp();
  for (int i = kNKinParBON; i--;)
    corr[i] += detELoss[i];
  //  if (!pnt->GetELossVaried()) corr[kParQ2Pt] = pnt->GetMatCorrExp()[kParQ2Pt]; // fixed eloss expected effect
  //  printf("apply corr UD %+.3e %+.3e %+.3e %+.3e\n",corr[0],corr[1],corr[2],corr[3]);
  //  printf("      corr  D %+.3e %+.3e %+.3e %+.3e\n",corrDiagP[0],corrDiagP[1],corrDiagP[2],corrDiagP[3]);
  //  printf("at point :"); pnt->print();
  //
  for (int itr = ntr; itr--;) {
    if (!ApplyMatCorr(trSet[itr], corr)) {
#if DEBUG > 3
      LOG(error) << "Failed on clone %d materials" << itr;
      trSet[itr].print();
#endif
      return kFALSE;
    }
  }
  return kTRUE;
}

//______________________________________________
Double_t AliAlgTrack::RichardsonExtrap(double* val, int ord)
{
  // Calculate Richardson extrapolation of order ord (starting from 1)
  // The array val should contain estimates ord+1 of derivatives with variations
  // d, d/2 ... d/2^ord.
  // The array val is overwritten
  //
  if (ord == 1)
    return (4. * val[1] - val[0]) * (1. / 3);
  do {
    for (int i = 0; i < ord; i++)
      val[i] = (4. * val[i + 1] - val[i]) * (1. / 3);
  } while (--ord);
  return val[0];
}

//______________________________________________
Double_t AliAlgTrack::RichardsonExtrap(const double* val, int ord)
{
  // Calculate Richardson extrapolation of order ord (starting from 1)
  // The array val should contain estimates ord+1 of derivatives with variations
  // d, d/2 ... d/2^ord.
  // The array val is not overwritten
  //
  if (ord == 1)
    return (4. * val[1] - val[0]) * (1. / 3);
  double* buff = new double[ord + 1];
  memcpy(buff, val, (ord + 1) * sizeof(double));
  do {
    for (int i = 0; i < ord; i++)
      buff[i] = (4. * buff[i + 1] - buff[i]) * (1. / 3);
  } while (--ord);
  return buff[0];
}

//______________________________________________
void AliAlgTrack::RichardsonDeriv(const trackParam_t* trSet, const double* delta, const AliAlgPoint* pnt, double& derY, double& derZ)
{
  // Calculate Richardson derivatives for diagonalized Y and Z from a set of kRichardsonN pairs
  // of tracks with same parameter of i-th pair varied by +-delta[i]
  static double derRichY[kRichardsonN], derRichZ[kRichardsonN];
  //
  for (int icl = 0; icl < kRichardsonN; icl++) { // calculate kRichardsonN variations with del, del/2, del/4...
    double resYVP = 0, resYVN = 0, resZVP = 0, resZVN = 0;
    pnt->GetResidualsDiag(trSet[(icl << 1) + 0].getParams(), resYVP, resZVP); // variation with +delta
    pnt->GetResidualsDiag(trSet[(icl << 1) + 1].getParams(), resYVN, resZVN); // variation with -delta
    derRichY[icl] = 0.5 * (resYVP - resYVN) / delta[icl];                     // 2-point symmetric derivatives
    derRichZ[icl] = 0.5 * (resZVP - resZVN) / delta[icl];
  }
  derY = RichardsonExtrap(derRichY, kRichardsonOrd); // dY/dPar
  derZ = RichardsonExtrap(derRichZ, kRichardsonOrd); // dZ/dPar
  //
}

//______________________________________________
void AliAlgTrack::Print(Option_t* opt) const
{
  // print track data
  printf("%s ", IsCosmic() ? "  Cosmic  " : "Collision ");
  trackParam_t::print();
  printf("N Free Par: %d (Kinem: %d) | Npoints: %d (Inner:%d) | M : %.3f | Chi2Ini:%.1f Chi2: %.1f/%d",
         fNLocPar, fNLocExtPar, GetNPoints(), GetInnerPointID(), fMass, fChi2Ini, fChi2, fNDF);
  if (IsCosmic()) {
    int npLow = GetInnerPointID();
    int npUp = GetNPoints() - npLow - 1;
    printf(" [Low:%.1f/%d Up:%.1f/%d]", fChi2CosmDn, npLow, fChi2CosmUp, npUp);
  }
  printf("\n");
  //
  TString optS = opt;
  optS.ToLower();
  Bool_t res = optS.Contains("r") && GetResidDone();
  Bool_t der = optS.Contains("d") && GetDerivDone();
  Bool_t par = optS.Contains("lc");   // local param corrections
  Bool_t paru = optS.Contains("lcu"); // local param corrections in track param frame
  //
  if (par) {
    printf("Ref.track corr: ");
    for (int i = 0; i < fNLocExtPar; i++)
      printf("%+.3e ", fLocParA[i]);
    printf("\n");
  }
  //
  if (optS.Contains("p") || res || der) {
    for (int ip = 0; ip < GetNPoints(); ip++) {
      printf("#%3d ", ip);
      AliAlgPoint* pnt = GetPoint(ip);
      pnt->Print(opt);
      //
      if (res && pnt->ContainsMeasurement()) {
        printf("  Residuals  : %+.3e %+.3e -> Pulls: %+7.2f %+7.2f\n",
               GetResidual(0, ip), GetResidual(1, ip),
               GetResidual(0, ip) / sqrt(pnt->GetErrDiag(0)), GetResidual(1, ip) / sqrt(pnt->GetErrDiag(1)));
      }
      if (der && pnt->ContainsMeasurement()) {
        for (int ipar = 0; ipar < fNLocPar; ipar++) {
          printf("  Dres/dp%03d : %+.3e %+.3e\n", ipar, GetDResDLoc(0, ip)[ipar], GetDResDLoc(1, ip)[ipar]);
        }
      }
      //
      if (par && pnt->ContainsMaterial()) { // material corrections
        int nCorrPar = pnt->GetNMatPar();
        const double* corrDiag = &fLocParA[pnt->GetMaxLocVarID() - nCorrPar];
        printf("  Corr.Diag:  ");
        for (int i = 0; i < nCorrPar; i++)
          printf("%+.3e ", corrDiag[i]);
        printf("\n");
        printf("  Corr.Pull:  ");
        float* corCov = pnt->GetMatCorrCov(); // correction covariance
        //float *corExp = pnt->GetMatCorrExp(); // correction expectation
        for (int i = 0; i < nCorrPar; i++)
          printf("%+.3e ", (corrDiag[i] /* - corExp[i]*/) / Sqrt(corCov[i]));
        printf("\n");
        if (paru) { // print also mat.corrections in track frame
          double corr[5] = {0};
          pnt->UnDiagMatCorr(corrDiag, corr);
          //	  if (!pnt->GetELossVaried()) corr[kParQ2Pt] = pnt->GetMatCorrExp()[kParQ2Pt]; // fixed eloss expected effect
          printf("  Corr.Track: ");
          for (int i = 0; i < kNKinParBON; i++)
            printf("%+.3e ", corr[i]);
          printf("\n");
        }
      }
    }
  } // print points
}

//______________________________________________
void AliAlgTrack::DumpCoordinates() const
{
  // print various coordinates for inspection
  printf("gpx/D:gpy/D:gpz/D:gtxb/D:gtyb/D:gtzb/D:gtxa/D:gtya/D:gtza/D:alp/D:px/D:py/D:pz/D:tyb/D:tzb/D:tya/D:tza/D:ey/D:ez/D\n");
  for (int ip = 0; ip < GetNPoints(); ip++) {
    AliAlgPoint* pnt = GetPoint(ip);
    if (!pnt->ContainsMeasurement())
      continue;
    pnt->DumpCoordinates();
  }
}

//______________________________________________
Bool_t AliAlgTrack::IniFit()
{
  // perform initial fit of the track
  //
  const int kMinNStep = 3;
  const double kMaxDefStep = 3.0;
  //
  trackParam_t trc = *this;
  //
  if (!GetFieldON()) { // for field-off data impose nominal momentum
  }
  fChi2 = fChi2CosmUp = fChi2CosmDn = 0;
  //
  // the points are ranged from outer to inner for collision tracks,
  // and from outer point of lower leg to outer point of upper leg for the cosmic track
  //
  // the fit will always start from the outgoing track in inward direction
  if (!FitLeg(trc, 0, GetInnerPointID(), fNeedInv[0])) {
#if DEBUG > 3
    LOG(warn) << "Failed FitLeg 0";
    trc.print();
#endif
    return kFALSE; // collision track or cosmic lower leg
  }
  //
  //  printf("Lower leg: %d %d\n",0,GetInnerPointID()); trc.print();
  //
  if (IsCosmic()) {
    fChi2CosmDn = fChi2;
    trackParam_t trcU = trc;
    if (!FitLeg(trcU, GetNPoints() - 1, GetInnerPointID() + 1, fNeedInv[1])) { //fit upper leg of cosmic track
#if DEBUG > 3
      LOG(warn) << "Failed FitLeg 0";
      trc.print();
#endif
      return kFALSE; // collision track or cosmic lower leg
    }
    //
    // propagate to reference point, which is the inner point of lower leg
    const AliAlgPoint* refP = GetPoint(GetInnerPointID());
    if (!PropagateToPoint(trcU, refP, kMinNStep, kMaxDefStep, kTRUE))
      return kFALSE;
    //
    fChi2CosmUp = fChi2 - fChi2CosmDn;
    //    printf("Upper leg: %d %d\n",GetInnerPointID()+1,GetNPoints()-1); trcU.print();
    //
    if (!CombineTracks(trc, trcU))
      return kFALSE;
    //printf("Combined\n"); trc.print();
  }
  CopyFrom(&trc);
  //
  fChi2Ini = fChi2;

  return kTRUE;
}

//______________________________________________
Bool_t AliAlgTrack::CombineTracks(trackParam_t& trcL, const trackParam_t& trcU)
{
  // Assign to trcL the combined tracks (Kalman update of trcL by trcU)
  // The trcL and trcU MUST be defined at same X,Alpha
  //
  // Update equations: tracks described by vectors vL and vU and coviriances CL and CU resp.
  // then the gain matrix K = CL*(CL+CU)^-1
  // Updated vector and its covariance:
  // CL' = CL - K*CL
  // vL' = vL + K(vU-vL)
  //
  if (Abs(trcL.getX() - trcU.getX()) > kTinyDist || Abs(trcL.getAlpha() - trcU.getAlpha()) > kTinyDist) {
    LOG(error) << "Tracks must be defined at same reference X and Alpha";
    trcL.print();
    trcU.print();
    return kFALSE;
  }
  //
  //  const covMat_t& covU = trcU.getCov();
  //  const covMat_t& covL = trcL.getCov();
  //
  int mtSize = GetFieldON() ? kNKinParBON : kNKinParBOFF;
  TMatrixD matCL(mtSize, mtSize), matCLplCU(mtSize, mtSize);
  TVectorD vl(mtSize), vUmnvL(mtSize);
  //
  //  trcL.print();
  //  trcU.print();
  //
  for (int i = mtSize; i--;) {
    vUmnvL[i] = trcU.getParam(i) - trcL.getParam(i); // y = residual of 2 tracks
    vl[i] = trcL.getParam(i);
    for (int j = i + 1; j--;) {
      int indIJ = ((i * (i + 1)) >> 1) + j; // position of IJ cov element in the trackParam_t covariance array
      matCL(i, j) = matCL(j, i) = trcL.getCovarElem(i, j);
      matCLplCU(i, j) = matCLplCU(j, i) = trcL.getCovarElem(i, j) + trcU.getCovarElem(i, j);
    }
  }
  matCLplCU.Invert(); // S^-1 = (Cl + Cu)^-1
  if (!matCLplCU.IsValid()) {
#if DEBUG > 3
    LOG(error) << "Failed to invert summed cov.matrix of cosmic track";
    matCLplCU.print();
#endif
    return kFALSE; // inversion failed
  }
  TMatrixD matK(matCL, TMatrixD::kMult, matCLplCU); // gain K = Cl*(Cl+Cu)^-1
  TMatrixD matKdotCL(matK, TMatrixD::kMult, matCL); // K*Cl
  TVectorD vlUp = matK * vUmnvL;                    // K*(vl - vu)
  for (int i = mtSize; i--;) {
    trcL.updateParam(vlUp[i], i); // updated param: vL' = vL + K(vU-vL)
    for (int j = i + 1; j--;)
      trcL.updateCov(-matKdotCL(i, j), i, j); // updated covariance: Cl' = Cl - K*Cl
  }
  //
  // update chi2
  double chi2 = 0;
  for (int i = mtSize; i--;)
    for (int j = mtSize; j--;)
      chi2 += matCLplCU(i, j) * vUmnvL[i] * vUmnvL[j];
  fChi2 += chi2;
  //
  //  printf("Combined: Chi2Tot:%.2f ChiUp:%.2f ChiDn:%.2f ChiCmb:%.2f\n",fChi2,fChi2CosmUp,fChi2CosmDn, chi2);

  return kTRUE;
}

//______________________________________________
Bool_t AliAlgTrack::FitLeg(trackParam_t& trc, int pFrom, int pTo, Bool_t& inv)
{
  // perform initial fit of the track
  // the fit will always start from the outgoing track in inward direction (i.e. if cosmics - bottom leg)
  const int kMinNStep = 3;
  const double kMaxDefStep = 3.0;
  const double kErrSpace = 50.;
  const double kErrAng = 0.7;
  const double kErrRelPtI = 1.;
  const covMat_t kIniErr{// initial error
                         kErrSpace * kErrSpace,
                         0, kErrSpace * kErrSpace,
                         0, 0, kErrAng * kErrAng,
                         0, 0, 0, kErrAng * kErrAng,
                         0, 0, 0, 0, kErrRelPtI * kErrRelPtI};
  //
  // prepare seed at outer point
  AliAlgPoint* p0 = GetPoint(pFrom);
  double phi = trc.getPhi(), alp = p0->GetAlphaSens();
  BringTo02Pi(phi);
  BringTo02Pi(alp);
  double dphi = DeltaPhiSmall(phi, alp); // abs delta angle
  if (dphi > Pi() / 2.) {                // need to invert the track to new frame
    inv = kTRUE;
    //    printf("Fit in %d %d Delta: %.3f -> Inverting for\n",pFrom,pTo,dphi);
    //    p0->print("meas");
    //    printf("BeforeInv "); trc.print();
    trc.invert();
    //    printf("After Inv "); trc.print();
  }
  if (!trc.rotateParam(p0->GetAlphaSens())) {
#if DEBUG > 3
    AliWarningF("Failed on rotateParam to %f", p0->GetAlphaSens());
    trc.print();
#endif
    return kFALSE;
  }
  if (!PropagateParamToPoint(trc, p0, 30)) {
    //  if (!PropagateToPoint(trc,p0,5,30,kTRUE)) {
    //trc.PropagateParamOnlyTo(p0->GetXPoint()+kOverShootX,AliTrackerBase::GetBz())) {
#if DEBUG > 3
    AliWarningF("Failed on PropagateParamOnlyTo to %f", p0->GetXPoint() + kOverShootX);
    trc.print();
#endif
    return kFALSE;
  }
  trc.setCov(kIniErr);
  trc.setCov(trc.getQ2Pt() * trc.getQ2Pt(), 4, 4); // lowest diagonal element (Q2Pt2)
  //
  int pinc;
  if (pTo > pFrom) { // fit in points increasing order: collision track or cosmics lower leg
    pTo++;
    pinc = 1;
  } else { // fit in points decreasing order: cosmics upper leg
    pTo--;
    pinc = -1;
  }
  //
  for (int ip = pFrom; ip != pTo; ip += pinc) { // inward fit from outer point
    AliAlgPoint* pnt = GetPoint(ip);
    //
    //    printf("*** FitLeg %d (%d %d)\n",ip,pFrom,pTo);
    //    printf("Before propagate: "); trc.print();
    if (!PropagateToPoint(trc, pnt, kMinNStep, kMaxDefStep, kTRUE))
      return kFALSE;
    if (pnt->ContainsMeasurement()) {
      if (pnt->GetNeedUpdateFromTrack())
        pnt->UpdatePointByTrackInfo(&trc);
      const double* yz = pnt->GetYZTracking();
      const double* errYZ = pnt->GetYZErrTracking();
      double chi = trc.getPredictedChi2(yz, errYZ);
      //printf("***>> fitleg-> Y: %+e %+e / Z: %+e %+e -> Chi2: %e | %+e %+e\n",yz[0],trc.GetY(),yz[1],trc.GetZ(),chi,
      //  trc.Phi(),trc.GetAlpha());
      //      printf("Before update at %e %e\n",yz[0],yz[1]); trc.print();
      if (!trc.update(yz, errYZ)) {
#if DEBUG > 3
        AliWarningF("Failed on Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
        trc.print();
#endif
        return kFALSE;
      }
      fChi2 += chi;
      //      printf("After update: (%f) -> %f\n",chi,fChi2); trc.print();
    }
  }
  //
  if (inv) {
    //    printf("Before inverting back "); trc.print();
    trc.invert();
  }
  //
  return kTRUE;
}

//______________________________________________
Bool_t AliAlgTrack::ResidKalman()
{
  // calculate residuals from bi-directional Kalman smoother
  // ATTENTION: this method modifies workspaces of the points!!!
  //
  Bool_t inv = kFALSE;
  const int kMinNStep = 3;
  const double kMaxDefStep = 3.0;
  const double kErrSpace = 50.;
  const double kErrAng = 0.7;
  const double kErrRelPtI = 1.;
  const covMat_t kIniErr = {// initial error
                            kErrSpace * kErrSpace,
                            0, kErrSpace * kErrSpace,
                            0, 0, kErrAng * kErrAng,
                            0, 0, 0, kErrAng * kErrAng,
                            0, 0, 0, 0, kErrRelPtI * kErrRelPtI};
  //  const Double_t kOverShootX = 5;
  //
  trackParam_t trc = *this;
  //
  int pID = 0, nPnt = GetNPoints();
  ;
  AliAlgPoint* pnt = 0;
  // get 1st measured point
  while (pID < nPnt && !(pnt = GetPoint(pID))->ContainsMeasurement())
    pID++;
  if (!pnt)
    return kFALSE;
  double phi = trc.getPhi(), alp = pnt->GetAlphaSens();
  BringTo02Pi(phi);
  BringTo02Pi(alp);
  double dphi = DeltaPhiSmall(phi, alp);
  if (dphi > Pi() / 2.) { // need to invert the track to new frame
    inv = kTRUE;
    trc.invert();
  }
  // prepare track seed at 1st valid point
  if (!trc.rotateParam(pnt->GetAlphaSens())) {
#if DEBUG > 3
    AliWarningF("Failed on rotateParam to %f", pnt->GetAlphaSens());
    trc.print();
#endif
    return kFALSE;
  }
  if (!PropagateParamToPoint(trc, pnt, 30)) {
    //if (!trc.PropagateParamOnlyTo(pnt->GetXPoint()+kOverShootX,AliTrackerBase::GetBz())) {
#if DEBUG > 3
    AliWarningF("Failed on PropagateParamOnlyTo to %f", pnt->GetXPoint() + kOverShootX);
    trc.print();
#endif
    return kFALSE;
  }
  //
  trc.setCov(kIniErr);
  const double inwardQ2Pt2 = trc.getCovarElem(4, 4) * trc.getQ2Pt() * trc.getQ2Pt();
  trc.setCov(inwardQ2Pt2, 4, 4); // lowest diagonal element (Q2Pt2)
  //
  double chifwd = 0, chibwd = 0;
  // inward fit
  for (int ip = 0; ip < nPnt; ip++) {
    pnt = GetPoint(ip);
    if (pnt->IsInvDir() != inv) { // crossing point where the track should be inverted?
      trc.invert();
      inv = !inv;
    }
    //    printf("*** ResidKalm %d (%d %d)\n",ip,0,nPnt);
    //    printf("Before propagate: "); trc.print();
    if (!PropagateToPoint(trc, pnt, kMinNStep, kMaxDefStep, kTRUE))
      return kFALSE;
    if (!pnt->ContainsMeasurement())
      continue;
    const double* yz = pnt->GetYZTracking();
    const double* errYZ = pnt->GetYZErrTracking();
    // store track position/errors before update in the point WorkSpace-A
    double* ws = (double*)pnt->GetTrParamWSA();
    ws[0] = trc.getY();
    ws[1] = trc.getZ();
    ws[2] = trc.getSigmaY2();
    ws[3] = trc.getSigmaZY();
    ws[4] = trc.getSigmaZ2();
    double chi = trc.getPredictedChi2(yz, errYZ);
    //    printf(">> INV%d (%9d): %+.2e %+.2e | %+.2e %+.2e %+.2e %+.2e %+.2e | %.2e %d \n",ip,pnt->GetSensor()->GetInternalID(),yz[0],yz[1], ws[0],ws[1],ws[2],ws[3],ws[4],chi,inv);
    //    printf(">>Bef ");trc.print();
    // printf("KLM Before update at %e %e\n",yz[0],yz[1]); trc.print();
    if (!trc.update(yz, errYZ)) {
#if DEBUG > 3
      AliWarningF("Failed on Inward Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
      trc.print();
#endif
      return kFALSE;
    }
    //    printf(">>Aft ");trc.print();
    chifwd += chi;
    //printf("KLM After update: (%f) -> %f\n",chi,chifwd);   trc.print();
  }
  //
  // outward fit
  trc.setCov(kIniErr);
  const double outwardQ2Pt2 = trc.getCovarElem(4, 4) * trc.getQ2Pt() * trc.getQ2Pt();
  trc.setCov(outwardQ2Pt2, 4, 4); // lowest diagonal element (Q2Pt2)

  for (int ip = nPnt; ip--;) {
    pnt = GetPoint(ip);
    if (pnt->IsInvDir() != inv) { // crossing point where the track should be inverted?
      trc.invert();
      inv = !inv;
    }
    if (!PropagateToPoint(trc, pnt, kMinNStep, kMaxDefStep, kTRUE))
      return kFALSE;
    if (!pnt->ContainsMeasurement())
      continue;
    const double* yz = pnt->GetYZTracking();
    const double* errYZ = pnt->GetYZErrTracking();
    // store track position/errors before update in the point WorkSpace-B
    double* ws = (double*)pnt->GetTrParamWSB();
    ws[0] = trc.getY();
    ws[1] = trc.getZ();
    ws[2] = trc.getSigmaY2();
    ws[3] = trc.getSigmaZY();
    ws[4] = trc.getSigmaZ2();
    double chi = trc.getPredictedChi2(yz, errYZ);
    //    printf("<< OUT%d (%9d): %+.2e %+.2e | %+.2e %+.2e %+.2e %+.2e %+.2e | %.2e %d \n",ip,pnt->GetSensor()->GetInternalID(),yz[0],yz[1], ws[0],ws[1],ws[2],ws[3],ws[4],chi,inv);
    //    printf("<<Bef ");    trc.print();
    if (!trc.update(yz, errYZ)) {
#if DEBUG > 3
      AliWarningF("Failed on Outward Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
      trc.print();
#endif
      return kFALSE;
    }
    chibwd += chi;
    //    printf("<<Aft ");    trc.print();
  }
  //
  // now compute smoothed prediction and residual
  for (int ip = 0; ip < nPnt; ip++) {
    pnt = GetPoint(ip);
    if (!pnt->ContainsMeasurement())
      continue;
    double* wsA = (double*)pnt->GetTrParamWSA(); // inward measurement
    double* wsB = (double*)pnt->GetTrParamWSB(); // outward measurement
    double &yA = wsA[0], &zA = wsA[1], &sgAYY = wsA[2], &sgAYZ = wsA[3], &sgAZZ = wsA[4];
    double &yB = wsB[0], &zB = wsB[1], &sgBYY = wsB[2], &sgBYZ = wsB[3], &sgBZZ = wsB[4];
    // compute weighted average
    double sgYY = sgAYY + sgBYY, sgYZ = sgAYZ + sgBYZ, sgZZ = sgAZZ + sgBZZ;
    double detI = sgYY * sgZZ - sgYZ * sgYZ;
    if (TMath::Abs(detI) < constants::math::Almost0)
      return kFALSE;
    else
      detI = 1. / detI;
    double tmp = sgYY;
    sgYY = sgZZ * detI;
    sgZZ = tmp * detI;
    sgYZ = -sgYZ * detI;
    double dy = yB - yA, dz = zB - zA;
    double k00 = sgAYY * sgYY + sgAYZ * sgYZ, k01 = sgAYY * sgYZ + sgAYZ * sgZZ;
    double k10 = sgAYZ * sgYY + sgAZZ * sgYZ, k11 = sgAYZ * sgYZ + sgAZZ * sgZZ;
    double sgAYZt = sgAYZ;
    yA += dy * k00 + dz * k01; // these are smoothed predictions, stored in WSA
    zA += dy * k10 + dz * k11; //
    sgAYY -= k00 * sgAYY + k01 * sgAYZ;
    sgAYZ -= k00 * sgAYZt + k01 * sgAZZ;
    sgAZZ -= k10 * sgAYZt + k11 * sgAZZ;
    //    printf("|| WGH%d (%9d): | %+.2e %+.2e %+.2e %.2e %.2e\n",ip,pnt->GetSensor()->GetInternalID(), wsA[0],wsA[1],wsA[2],wsA[3],wsA[4]);
  }
  //
  fChi2 = chifwd;
  SetKalmanDone(kTRUE);
  return kTRUE;
}

//______________________________________________
Bool_t AliAlgTrack::ProcessMaterials()
{
  // attach material effect info to alignment points
  trackParam_t trc = *this;

  // collision track of cosmic lower leg: move along track direction from last (middle for cosmic lower leg)
  // point (inner) to 1st one (outer)
  if (fNeedInv[0])
    trc.invert(); // track needs to be inverted ? (should be for upper leg)
  if (!ProcessMaterials(trc, GetInnerPointID(), 0)) {
#if DEBUG > 3
    LOG(error) << "Failed to process materials for leg along the track";
#endif
    return kFALSE;
  }
  if (IsCosmic()) {
    // cosmic upper leg: move againg the track direction from middle point (inner) to last one (outer)
    trc = *this;
    if (fNeedInv[1])
      trc.invert(); // track needs to be inverted ?
    if (!ProcessMaterials(trc, GetInnerPointID() + 1, GetNPoints() - 1)) {
#if DEBUG > 3
      LOG(error) << "Failed to process materials for leg against the track";
#endif
      return kFALSE;
    }
  }
  return kTRUE;
}

//______________________________________________
Bool_t AliAlgTrack::ProcessMaterials(trackParam_t& trc, int pFrom, int pTo)
{
  // attach material effect info to alignment points
  const int kMinNStep = 3;
  const double kMaxDefStep = 3.0;
  const double kErrSpcT = 1e-6;
  const double kErrAngT = 1e-6;
  const double kErrPtIT = 1e-12;
  const covMat_t kErrTiny = {// initial tiny error
                             kErrSpcT * kErrSpcT,
                             0, kErrSpcT * kErrSpcT,
                             0, 0, kErrAngT * kErrAngT,
                             0, 0, 0, kErrAngT * kErrAngT,
                             0, 0, 0, 0, kErrPtIT * kErrPtIT};
  /*
  const double kErrSpcH = 10.0;
  const double kErrAngH = 0.5;
  const double kErrPtIH = 0.5;
  const double kErrHuge[15] = { // initial tiny error
    kErrSpcH*kErrSpcH,
    0                  , kErrSpcH*kErrSpcH,
    0                  ,                   0, kErrAngH*kErrAngH,
    0                  ,                   0,               0, kErrAngH*kErrAngH,
    0                  ,                   0,               0,               0, kErrPtIH*kErrPtIH
  };
  */
  //
  // 2 copies of the track, one will be propagated accounting for materials, other - w/o
  trackParam_t tr0;
  double x2X0xRho[2] = {0, 0};
  double dpar[5] = {0};
  covMat_t dcov{0};

  //
  int pinc;
  if (pTo > pFrom) { // fit in points decreasing order: cosmics upper leg
    pTo++;
    pinc = 1;
  } else { // fit in points increasing order: collision track or cosmics lower leg
    pTo--;
    pinc = -1;
  }
  //
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
    AliAlgPoint* pnt = GetPoint(ip);
    trc.setCov(kErrTiny); // assign tiny errors to both tracks
    tr0 = trc;
    //
    //    printf("-> ProcMat %d (%d->%d)\n",ip,pFrom,pTo);
    if (!PropagateToPoint(trc, pnt, kMinNStep, kMaxDefStep, kTRUE, x2X0xRho)) { // with material corrections
#if DEBUG > 3
      LOG(error) << "Failed to take track to point" << ip << " (dir: " << pFrom << "->" pTo << ") with mat.corr.";
      trc.print();
      pnt->print("meas");
#endif
      return kFALSE;
    }
    //
    // is there enough material to consider the point as a scatterer?
    pnt->SetContainsMaterial(x2X0xRho[0] * Abs(trc.getQ2Pt()) > GetMinX2X0Pt2Account());
    //
    //    printf("-> ProcMat000 %d (%d->%d)\n",ip,pFrom,pTo);
    if (!PropagateToPoint(tr0, pnt, kMinNStep, kMaxDefStep, kFALSE, 0)) { // no material corrections
#if DEBUG > 3
      LOG(error) << "Failed to take track to point" << ip << " (dir: " << pFrom << "->" pTo << ") with mat.corr.";
      tr0.print();
      pnt->print("meas");
#endif
      return kFALSE;
    }
    // the difference between the params, covariance of tracks with and  w/o material accounting gives
    // paramets and covariance of material correction. For params ONLY ELoss effect is revealed
    const covMat_t& cov0 = tr0.getCov();
    double* par0 = (double*)tr0.getParams();
    const covMat_t& cov1 = trc.getCov();
    double* par1 = (double*)trc.getParams();
    for (int l = 15; l--;)
      dcov[l] = cov1[l] - cov0[l];
    for (int l = kNKinParBON; l--;)
      dpar[l] = par1[l] - par0[l]; // eloss affects all parameters!
    pnt->SetMatCorrExp(dpar);
    //dpar[kParQ2Pt] = par1[kParQ2Pt] - par0[kParQ2Pt]; // only e-loss expectation is non-0
    //
    if (pnt->ContainsMaterial()) {
      //
      // MP2 handles only scalar residuals hence correlated matrix of material effect need to be diagonalized
      Bool_t eLossFree = pnt->GetELossVaried();
      int nParFree = eLossFree ? kNKinParBON : kNKinParBOFF;
      TMatrixDSym matCov(nParFree);
      for (int i = nParFree; i--;)
        for (int j = i + 1; j--;)
          matCov(i, j) = matCov(j, i) = dcov[j + ((i * (i + 1)) >> 1)];
      //
      TMatrixDSymEigen matDiag(matCov); // find eigenvectors
      const TMatrixD& matEVec = matDiag.GetEigenVectors();
      if (!matEVec.IsValid()) {
#if DEBUG > 3
        LOG(error) << "Failed to diagonalize covariance of material correction";
        matCov.print();
        return kFALSE;
#endif
      }
      pnt->SetMatCovDiagonalizationMatrix(matEVec); // store diagonalization matrix
      pnt->SetMatCovDiag(matDiag.GetEigenValues()); // store E.Values: diagonalized cov.matrix
      if (!eLossFree)
        pnt->SetMatCovDiagElem(kParQ2Pt, dcov[14]);
      //
      //printf("Add mat%d %e %e\n",ip, x2X0xRho[0],x2X0xRho[1]);
      pnt->SetX2X0(x2X0xRho[0]);
      pnt->SetXTimesRho(x2X0xRho[1]);
      //
    }
    if (pnt->ContainsMeasurement()) { // update track to have best possible kinematics
      const double* yz = pnt->GetYZTracking();
      const double* errYZ = pnt->GetYZErrTracking();
      if (!trc.update(yz, errYZ)) {
#if DEBUG > 3
        AliWarningF("Failed on Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
        trc.print();
#endif
        return kFALSE;
      }
      //
    }
    //
  }
  //
  return kTRUE;
}

//______________________________________________
void AliAlgTrack::SortPoints()
{
  // sort points in order against track direction: innermost point is last
  // for collision tracks.
  // For 2-leg cosmic tracks: 1st points of outgoing (lower) leg are added from large to
  // small radii, then the points of incomint (upper) leg are added in increasing R direction
  //
  // The fInnerPointID will mark the id of the innermost point, i.e. the last one for collision-like
  // tracks and in case of cosmics - the point of lower leg with smallest R
  //
  fPoints.Sort();
  int np = GetNPoints();
  fInnerPointID = np - 1;
  if (IsCosmic()) {
    for (int ip = np; ip--;) {
      AliAlgPoint* pnt = GetPoint(ip);
      if (pnt->IsInvDir())
        continue; // this is a point of upper leg
      fInnerPointID = ip;
      break;
    }
  }
  //
}

//______________________________________________
void AliAlgTrack::SetLocPars(const double* pars)
{
  // store loc par corrections
  memcpy(fLocParA, pars, fNLocPar * sizeof(double));
}

//______________________________________________
void AliAlgTrack::CheckExpandDerGloBuffer(int minSize)
{
  // if needed, expand global derivatives buffer
  if (fGloParID.GetSize() < minSize) {
    fGloParID.Set(minSize + 100);
    fDResDGlo[0].Set(minSize + 100);
    fDResDGlo[1].Set(minSize + 100);
    //
    // reassign fast access arrays
    fGloParIDA = fGloParID.GetArray();
    fDResDGloA[0] = fDResDGlo[0].GetArray();
    fDResDGloA[1] = fDResDGlo[1].GetArray();
  }
}

} // namespace align
} // namespace o2
