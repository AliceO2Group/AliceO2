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

/// @file   AlignmentTrack.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Track model for the alignment

#include <cstdio>
#include "Align/AlignmentTrack.h"
#include "Framework/Logger.h"
#include "Align/AlignableSensor.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableDetector.h"
#include "Align/AlignConfig.h"
#include "Align/utils.h"
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TMatrixDSymEigen.h>
#include "MathUtils/SymMatrixSolver.h"
#include "MathUtils/Utils.h"

#define DEBUG 4
using namespace o2::align::utils;
using namespace o2::base;
using namespace TMath;

namespace o2
{
namespace align
{

// RS: this is not good: we define constants outside the class, but it is to
// bypass the CINT limitations on static arrays initializations
const int kRichardsonOrd = 1;                // Order of Richardson extrapolation for derivative (min=1)
const int kRichardsonN = kRichardsonOrd + 1; // N of 2-point symmetric derivatives needed for requested order
const int kNRDClones = kRichardsonN * 2;     // number of variations for derivative of requested order

//____________________________________________________________________________
void AlignmentTrack::Clear(Option_t*)
{
  // reset the track
  TObject::Clear();
  ResetBit(0xffffffff);
  mPoints.clear();
  mDetPoints.clear();
  mChi2 = mChi2CosmUp = mChi2CosmDn = mChi2Ini = 0;
  mNDF = 0;
  mInnerPointID = -1;
  mNeedInv[0] = mNeedInv[1] = false;
  mNLocPar = mNLocExtPar = mNGloPar = 0;
  //
}

//____________________________________________________________________________
void AlignmentTrack::defineDOFs()
{
  // define varied DOF's (local parameters) for the track:
  // 1) kinematic params (5 or 4 depending on Bfield)
  // 2) mult. scattering angles (2)
  // 3) if requested by point: energy loss
  //
  mNLocPar = mNLocExtPar = getFieldON() ? kNKinParBON : kNKinParBOFF;
  int np = getNPoints();
  //
  // the points are sorted in order opposite to track direction -> outer points come 1st,
  // but for the 2-leg cosmic track the innermost points are in the middle (1st lower leg, then upper one)
  //
  // start along track direction, i.e. last point in the ordered array
  int minPar = mNLocPar;
  for (int ip = getInnerPointID() + 1; ip--;) { // collision track or cosmic lower leg
    AlignmentPoint* pnt = getPoint(ip);
    pnt->setMinLocVarID(minPar);
    if (pnt->containsMaterial()) {
      mNLocPar += pnt->getNMatPar();
    }
    pnt->setMaxLocVarID(mNLocPar); // flag up to which parameted ID this points depends on
  }
  //
  if (isCosmic()) {
    minPar = mNLocPar;
    for (int ip = getInnerPointID() + 1; ip < np; ip++) { // collision track or cosmic lower leg
      AlignmentPoint* pnt = getPoint(ip);
      pnt->setMinLocVarID(minPar);
      if (pnt->containsMaterial()) {
        mNLocPar += pnt->getNMatPar();
      }
      pnt->setMaxLocVarID(mNLocPar); // flag up to which parameted ID this points depends on
    }
  }
  mLocPar.clear();
  mLocPar.resize(mNLocPar);
  for (int i = 0; i < 2; i++) {
    mResid[i].clear();
    mDResDLoc[i].clear();
    mResid[i].resize(np);
    mDResDLoc[i].resize(mNLocPar * np);
  }
}

//______________________________________________________
bool AlignmentTrack::calcResidDeriv(double* params)
{
  // Propagate for given local params and calculate residuals and their derivatives.
  // The 1st 4 or 5 elements of params vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the applyMatCorr routine.
  //
  // If params are not provided, use internal params array
  //
  if (!params) {
    params = mLocPar.data();
  }
  //
  if (!getResidDone()) {
    calcResiduals(params);
  }
  //
  int np = getNPoints();
  //
  const auto& algConf = AlignConfig::Instance();
  // collision track or cosmic lower leg
  if (!calcResidDeriv(params, mNeedInv[0], getInnerPointID(), 0)) {
    if (algConf.verbose > 2) {
      LOG(warn) << "Failed on derivatives calculation 0";
    }
    return false;
  }
  //
  if (isCosmic()) { // cosmic upper leg
    if (!calcResidDeriv(params, mNeedInv[1], getInnerPointID() + 1, np - 1)) {
      if (algConf.verbose > 2) {
        LOG(warn) << "Failed on derivatives calculation 0";
      }
    }
  }
  //
  setDerivDone();
  return true;
}

//______________________________________________________
bool AlignmentTrack::calcResidDeriv(double* extendedParams, bool invert, int pFrom, int pTo)
{
  // Calculate derivatives of residuals vs params for points pFrom to pT. For cosmic upper leg
  // track parameter may require inversion.
  // The 1st 4 or 5 elements of params vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the applyMatCorr routine.
  //
  // The derivatives are calculated using Richardson extrapolation
  // (like http://root.cern.ch/root/html/ROOT__Math__RichardsonDerivator.html)
  //
  const auto& algConf = AlignConfig::Instance();
  trackParam_t probD[kNRDClones]; // use this to vary supplied param for derivative calculation
  double varDelta[kRichardsonN];
  const int kInvElem[kNKinParBON] = {-1, 1, 1, -1, -1};
  //
  const double kDelta[kNKinParBON] = {0.02, 0.02, 0.001, 0.001, 0.01}; // variations for ExtTrackParam and material effects
  //
  double delta[kNKinParBON]; // variations of curvature term are relative
  for (int i = kNKinParBOFF; i--;) {
    delta[i] = kDelta[i];
  }
  if (getFieldON()) {
    delta[kParQ2Pt] = kDelta[kParQ2Pt] * Abs(getQ2Pt());
  }
  //
  int pinc, signELoss = 0; // RS Validate for cosmic. Propagation is done from inner point to outer one:
  // energy loss is applied for collision tracks and cosmic lower leg, compensated for cosmic upper leg
  if (pTo > pFrom) { // fit in points decreasing order: cosmics upper leg
    pTo++;
    pinc = 1;
    signELoss = 1; // eloss is corrected
  } else { // fit in points increasing order: collision track or cosmics lower leg
    pTo--;
    pinc = -1;
    signELoss = -1; // eloss is applied
  }

  // 1) derivative wrt trackParam_t parameters
  for (int ipar = mNLocExtPar; ipar--;) {
    setParams(probD, kNRDClones, getX(), getAlpha(), extendedParams, true);
    if (invert) {
      for (int ic = kNRDClones; ic--;) {
        probD[ic].invert();
      }
    }
    double del = delta[ipar];
    //
    for (int icl = 0; icl < kRichardsonN; icl++) { // calculate kRichardsonN variations with del, del/2, del/4...
      varDelta[icl] = del;
      modParam(probD[(icl << 1) + 0], ipar, del);
      modParam(probD[(icl << 1) + 1], ipar, -del);
      del *= 0.5;
    }
    // propagate varied tracks to each point
    for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
      AlignmentPoint* pnt = getPoint(ip);
      // we propagate w/o mat. corrections, they will be accounted in applyMatCorr
      if (!propagateParamToPoint(probD, kNRDClones, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType::USEMatCorrNONE, signELoss)) {
        return false;
      }
      if (!applyMatCorr(probD, kNRDClones, extendedParams, pnt)) {
        return false;
      }
      if (pnt->containsMeasurement()) {
        int offsDer = ip * mNLocPar + ipar;
        richardsonDeriv(probD, varDelta, pnt, mDResDLoc[0][offsDer], mDResDLoc[1][offsDer]); // calculate derivatives
        if (invert && kInvElem[ipar] < 0) {
          mDResDLoc[0][offsDer] = -mDResDLoc[0][offsDer];
          mDResDLoc[1][offsDer] = -mDResDLoc[1][offsDer];
        }
      }
    } // loop over points
  }   // loop over ExtTrackParam parameters
  // 2) now vary material effect related parameters: MS and eventually ELoss
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
    AlignmentPoint* pnt = getPoint(ip);
    // global derivatives at this point
    if (pnt->containsMeasurement() && !calcResidDerivGlo(pnt)) {
      if (algConf.verbose > 2) {
        LOGF(warn, "Failed on global derivatives calculation at point %d", ip);
        pnt->print(AlignmentPoint::kMeasurementBit);
      }
      return false;
    }
    //
    if (!pnt->containsMaterial()) {
      continue;
    }
    //
    int nParFreeI = pnt->getNMatPar();
    //
    // array delta gives desired variation of parameters in trackParam_t definition,
    // while the variation should be done for parameters in the frame where the vector
    // of material corrections has diagonal cov. matrix -> rotate the delta to this frame
    double deltaMatD[kNKinParBON];
    //    pnt->diagMatCorr(delta, deltaMatD);
    for (int ipar = 0; ipar < nParFreeI; ipar++) {
      double d = pnt->getMatCorrCov()[ipar];
      deltaMatD[ipar] = d > 0. ? Sqrt(d) * 5 : delta[ipar];
    }
    //
    //    printf("Vary %d [%+.3e %+.3e %+.3e %+.3e] ",ip,deltaMatD[0],deltaMatD[1],deltaMatD[2],deltaMatD[3]); pnt->print();

    int offsI = pnt->getMaxLocVarID() - nParFreeI; // the parameters for this point start with this offset
                                                   // they are irrelevant for the points upstream
    for (int ipar = 0; ipar < nParFreeI; ipar++) { // loop over DOFs related to MS and ELoss are point ip
      double del = deltaMatD[ipar];
      //
      // We will vary the tracks starting from the original parameters propagated to given point
      // and stored there (before applying material corrections for this point)
      //
      setParams(probD, kNRDClones, pnt->getXTracking(), pnt->getAlphaSens(), pnt->getTrParamWSB(), false);
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
        if (!applyMatCorr(probD[(icl << 1) + 0], extendedParams, pnt)) {
          return false;
        }
        //
        // apply varied material effects : decremented by delta
        extendedParams[offsIP] = parOrig - del;
        if (!applyMatCorr(probD[(icl << 1) + 1], extendedParams, pnt)) {
          return false;
        }
        //
        extendedParams[offsIP] = parOrig;
        del *= 0.5;
      }
      if (pnt->containsMeasurement()) { // calculate derivatives at the scattering point itself
        int offsDerIP = ip * mNLocPar + offsIP;
        richardsonDeriv(probD, varDelta, pnt, mDResDLoc[0][offsDerIP], mDResDLoc[1][offsDerIP]); // calculate derivatives for ip
                                                                                                 // printf("DR SELF: %e %e at %d (%d)\n",mDResDLoc[0][offsDerIP], mDResDLoc[1][offsDerIP],offsI, offsDerIP);
      }
      //
      // loop over points whose residuals can be affected by the material effects on point ip
      for (int jp = ip + pinc; jp != pTo; jp += pinc) {
        AlignmentPoint* pntJ = getPoint(jp);

        // printf("  DerFor:%d ",jp); pntJ->print();

        if (!propagateParamToPoint(probD, kNRDClones, pntJ, algConf.maxStep, algConf.maxSnp, MatCorrType::USEMatCorrNONE, signELoss)) {
          return false;
        }
        //
        if (pntJ->containsMaterial()) { // apply material corrections
          if (!applyMatCorr(probD, kNRDClones, extendedParams, pntJ)) {
            return false;
          }
        }
        //
        if (pntJ->containsMeasurement()) {
          int offsDerJ = jp * mNLocPar + offsIP;
          // calculate derivatives
          richardsonDeriv(probD, varDelta, pntJ, mDResDLoc[0][offsDerJ], mDResDLoc[1][offsDerJ]);
        }
        //
      } // << loop over points whose residuals can be affected by the material effects on point ip
    }   // << loop over DOFs related to MS and ELoss are point ip
  }     // << loop over all points of the track
  //
  return true;
}

//______________________________________________________
bool AlignmentTrack::calcResidDerivGlo(AlignmentPoint* pnt)
{
  // calculate residuals derivatives over point's sensor and its parents global params
  double deriv[AlignableVolume::kNDOFGeom * 3];
  //
  const AlignableSensor* sens = pnt->getSensor();
  const AlignableVolume* vol = sens;
  // precalculated track parameters
  double snp = pnt->getTrParamWSA(kParSnp), tgl = pnt->getTrParamWSA(kParTgl);
  // precalculate track slopes to account tracking X variation
  // these are coeffs to translate deltaX of the point to deltaY and deltaZ of track
  double cspi = 1. / Sqrt((1 - snp) * (1 + snp)), slpY = snp * cspi, slpZ = tgl * cspi;
  //
  pnt->setDGloOffs(mNGloPar); // mark 1st entry of derivatives
  do {
    // measurement residuals
    int nfree = vol->getNDOFsFree();
    if (!nfree) {
      continue;
    } // no free parameters?
    sens->dPosTraDParGeom(pnt, deriv, vol == sens ? nullptr : vol);
    //
    checkExpandDerGloBuffer(mNGloPar + nfree); // if needed, expand derivatives buffer
    //
    for (int ip = 0; ip < AlignableVolume::kNDOFGeom; ip++) { // we need only free parameters
      if (!vol->isFreeDOF(ip)) {
        continue;
      }
      double* dXYZ = &deriv[ip * 3]; // tracking XYZ derivatives over this parameter
      // residual is defined as diagonalized track_estimate - measured Y,Z in tracking frame
      // where the track is evaluated at measured X!
      // -> take into account modified X using track parameterization at the point (paramWSA)
      // Attention: small simplifications(to be checked if it is ok!!!):
      // effect of changing X is accounted neglecting track curvature to preserve linearity
      //
      // store diagonalized residuals in track buffer
      pnt->diagonalizeResiduals((dXYZ[AlignmentPoint::kX] * slpY - dXYZ[AlignmentPoint::kY]),
                                (dXYZ[AlignmentPoint::kX] * slpZ - dXYZ[AlignmentPoint::kZ]),
                                mDResDGlo[0][mNGloPar], mDResDGlo[1][mNGloPar]);
      // and register global ID of varied parameter
      mGloParID[mNGloPar] = vol->getParGloID(ip);
      mNGloPar++;
    }
    //
  } while ((vol = vol->getParent()));
  //
  // eventual detector calibration parameters
  const AlignableDetector* det = sens->getDetector();
  int ndof = 0;
  if (det && (ndof = det->getNCalibDOFs())) {
    // if needed, expand derivatives buffer
    checkExpandDerGloBuffer(mNGloPar + det->getNCalibDOFsFree());
    for (int idf = 0; idf < ndof; idf++) {
      if (!det->isFreeDOF(idf)) {
        continue;
      }
      sens->dPosTraDParCalib(pnt, deriv, idf, nullptr);
      pnt->diagonalizeResiduals((deriv[AlignmentPoint::kX] * slpY - deriv[AlignmentPoint::kY]),
                                (deriv[AlignmentPoint::kX] * slpZ - deriv[AlignmentPoint::kZ]),
                                mDResDGlo[0][mNGloPar], mDResDGlo[1][mNGloPar]);
      // and register global ID of varied parameter
      mGloParID[mNGloPar] = det->getParGloID(idf);
      mNGloPar++;
    }
  }
  //
  pnt->setNGloDOFs(mNGloPar - pnt->getDGloOffs()); // mark number of global derivatives filled
  //
  return true;
}

//______________________________________________________
bool AlignmentTrack::calcResiduals(const double* extendedParams)
{
  // Propagate for given local params and calculate residuals
  // The 1st 4 or 5 elements of extendedParams vector should be the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the applyMatCorr routine.
  //
  // If extendedParams are not provided, use internal extendedParams array
  //
  if (!extendedParams) {
    extendedParams = mLocPar.data();
  }
  int np = getNPoints();
  mChi2 = 0;
  mNDF = 0;
  //
  const auto& algConf = AlignConfig::Instance();
  // collision track or cosmic lower leg
  if (!calcResiduals(extendedParams, mNeedInv[0], getInnerPointID(), 0)) {
    if (algConf.verbose > 2) {
      LOG(warn) << "Failed on residuals calculation 0";
    }
    return false;
  }
  //
  if (isCosmic()) { // cosmic upper leg
    if (!calcResiduals(extendedParams, mNeedInv[1], getInnerPointID() + 1, np - 1)) {
      if (algConf.verbose > 2) {
        LOG(warn) << "Failed on residuals calculation 1";
      }
      return false;
    }
  }
  //
  mNDF -= mNLocExtPar;
  setResidDone();
  return true;
}

//______________________________________________________
bool AlignmentTrack::calcResiduals(const double* extendedParams, bool invert, int pFrom, int pTo)
{
  // Calculate residuals for the single leg from points pFrom to pT
  // The 1st 4 or 5 elements of extendedParams vector should be corrections to
  // the reference trackParam_t
  // Then parameters of material corrections for each point
  // marked as having materials should come (4 or 5 dependending if ELoss is varied or fixed).
  // They correspond to kink parameters
  // (trackParam_t_after_material - trackParam_t_before_material)
  // rotated to frame where they error matrix is diagonal. Their conversion to trackParam_t
  // increment will be done locally in the applyMatCorr routine.
  //
  trackParam_t probe;
  setParams(probe, getX(), getAlpha(), extendedParams, true);
  if (invert) {
    probe.invert();
  }
  int pinc, signELoss = 0; // RS Validate for cosmic. Propagation is done from inner point to outer one:
  // energy loss is applied for collision tracks and cosmic lower leg, compensated for cosmic upper leg
  if (pTo > pFrom) { // fit in points decreasing order: cosmics upper leg
    pTo++;
    pinc = 1;
    signELoss = 1; // eloss is corrected
  } else { // fit in points increasing order: collision track or cosmics lower leg
    pTo--;
    pinc = -1;
    signELoss = -1; // eloss is applied
  }
  //
  const auto& algConf = AlignConfig::Instance();
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction:
    AlignmentPoint* pnt = getPoint(ip);
    if (!propagateParamToPoint(probe, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType::USEMatCorrNONE, signELoss)) {
      return false;
    }
    //
    // store the current track kinematics at the point BEFORE applying eventual material
    // corrections. This kinematics will be later varied around supplied parameters (in the calcResidDeriv)
    pnt->setTrParamWSB(probe.getParams());
    // account for materials
    if (!applyMatCorr(probe, extendedParams, pnt)) {
      return false;
    }
    pnt->setTrParamWSA(probe.getParams());
    //
    if (pnt->containsMeasurement()) { // need to calculate residuals in the frame where errors are orthogonal
      pnt->getResidualsDiag(probe.getParams(), mResid[0][ip], mResid[1][ip]);
      mChi2 += mResid[0][ip] * mResid[0][ip] / pnt->getErrDiag(0);
      mChi2 += mResid[1][ip] * mResid[1][ip] / pnt->getErrDiag(1);
      mNDF += 2;
    }
    //
    if (pnt->containsMaterial()) {
      // material degrees of freedom do not contribute to NDF since they are constrained by 0 expectation
      int nCorrPar = pnt->getNMatPar();
      const float* corCov = pnt->getMatCorrCov(); // correction diagonalized covariance
      auto offs = pnt->getMaxLocVarID() - nCorrPar;
      for (int i = 0; i < nCorrPar; i++) {
        mChi2 += mLocPar[offs + i] * mLocPar[offs + i] / corCov[i];
      }
    }
  }
  return true;
}

//______________________________________________________
bool AlignmentTrack::propagateParamToPoint(trackParam_t* tr, int nTr, const AlignmentPoint* pnt, double maxStep, double maxSnp, MatCorrType mt, int signCorr)
{
  // Propagate set of tracks to the point  (only parameters, no error matrix)
  // VECTORIZE this
  //
  const auto& algConf = AlignConfig::Instance();
  for (int itr = nTr; itr--;) {
    if (!propagateParamToPoint(tr[itr], pnt, maxStep, maxSnp, mt, signCorr)) {
      if (algConf.verbose > 2) {
        LOG(error) << "Failed on clone " << itr << " propagation ";
        tr[itr].print();
        pnt->print(AlignmentPoint::kMeasurementBit | AlignmentPoint::kMaterialBit);
      }
      return false;
    }
  }
  return true;
}

//______________________________________________________
bool AlignmentTrack::propagateParamToPoint(trackParam_t& tr, const AlignmentPoint* pnt, double maxStep, double maxSnp, MatCorrType mt, int signCorr)
{
  // propagate tracks to the point (only parameters, no error matrix)
  return propagate(tr, pnt, maxStep, maxSnp, mt, nullptr, signCorr);
}

//______________________________________________________
bool AlignmentTrack::propagateToPoint(trackParam_t& tr, const AlignmentPoint* pnt, double maxStep, double maxSnp, MatCorrType mt, track::TrackLTIntegral* tLT, int signCorr)
{
  // propagate tracks to the point. If matCor is true, then material corrections will be applied.
  // if matPar pointer is provided, it will be filled by total x2x0 and signed xrho
  return propagate(tr, pnt, maxStep, maxSnp, mt, tLT, signCorr);
}

bool AlignmentTrack::propagate(trackParam_t& track, const AlignmentPoint* pnt, double maxStep, double maxSnp, MatCorrType mt, track::TrackLTIntegral* tLT, int signCorr)
{
  if (signCorr == 0) { // auto
    // calculate the sign of the energy loss correction and ensure the upper leg of cosmics is calculated correctly.
    double dx = pnt->getXTracking() - track.getX();
    int dir = dx > 0.f ? 1 : -1;
    signCorr = pnt->isInvDir() ? dir : -dir; // propagation along the track direction should have signCorr=-1
  }
  // do propagation in at least 2 step to reveal eventual effect of MS on the position
  return PropagatorD::Instance()->propagateToAlphaX(track, pnt->getAlphaSens(), pnt->getXTracking(), pnt->getUseBzOnly(), maxSnp, maxStep, 2, mt, tLT, signCorr);
}

/*
//______________________________________________________
bool AlignmentTrack::ApplyMS(trackParam_t& trPar, double tms,double pms)
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
  //  | Sin[lam]        -Cos[lam] 0     |   | Sin[pms]*Sin[tms]|
  //
  //------------------------------------------------------------------------------
  //
  double *par = (double*) trPar.GetParameter();
  //
  if (Abs(tms)<1e-7) return true;
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
  return true;
}
*/

//______________________________________________________
bool AlignmentTrack::applyMatCorr(trackParam_t& trPar, const double* corrPar, const AlignmentPoint* pnt)
{
  // Modify track param (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corrPar contains delta to track parameters rotated by the matrix
  // DIAGONALIZING ITS  COVARIANCE MATRIX!
  // transform parameters from the frame diagonalizing the errors to track frame
  double corr[kNKinParBON] = {0};
  if (pnt->containsMaterial()) { // are there free params from materials?
    int nCorrPar = pnt->getNMatPar();
    const double* corrDiag = &corrPar[pnt->getMaxLocVarID() - nCorrPar]; // material corrections for this point start here
    pnt->unDiagMatCorr(corrDiag, corr);                                  // this is to account for MS and RANDOM Eloss (if varied)
  }
  // to this we should add expected parameters modification due to the deterministic eloss
  float* detELoss = pnt->getMatCorrExp();
  for (int i = kNKinParBON; i--;) {
    corr[i] += detELoss[i];
  }
  //corr[kParQ2Pt] += detELoss[kParQ2Pt];
  //  printf("apply corr UD %+.3e %+.3e %+.3e %+.3e %+.3e\n",corr[0],corr[1],corr[2],corr[3],corr[4]);
  //  printf("      corr  D %+.3e %+.3e %+.3e %+.3e\n",corrDiag[0],corrDiag[1],corrDiag[2],corrDiag[3]);
  //  printf("at point :"); pnt->print();
  return applyMatCorr(trPar, corr);
  //
}

//______________________________________________________
bool AlignmentTrack::applyMatCorr(trackParam_t& trPar, const double* corr)
{
  // Modify track param (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corr contains delta to track frame, NOT in diagonalized one
  const double snp = trPar.getSnp() + corr[kParSnp];
  const auto& algConf = AlignConfig::Instance();
  if (Abs(snp) > algConf.maxSnp) {
    if (algConf.verbose > 2) {
      LOG(error) << "Snp is too large: " << snp;
      printf("DeltaPar: ");
      for (int i = 0; i < kNKinParBON; i++) {
        printf("%+.3e ", corr[i]);
      }
      printf("\n");
      trPar.print();
    }
    return false;
  }

  trPar.updateParams(corr);

  return true;
}

//______________________________________________________
bool AlignmentTrack::applyMatCorr(trackParam_t* trSet, int ntr, const double* corrDiag, const AlignmentPoint* pnt)
{
  // Modify set of track params (e.g. trackParam_t) in the tracking frame
  // by delta accounting for material effects
  // Note: corrDiag contain delta to track parameters rotated by the matrix DIAGONALIZING ITS
  // COVARIANCE MATRIX
  // transform parameters from the frame diagonalizing the errors to track frame
  const auto& algConf = AlignConfig::Instance();
  double corr[kNKinParBON] = {0};
  if (pnt->containsMaterial()) { // are there free params from materials?
    int nCorrPar = pnt->getNMatPar();
    const double* corrDiagP = &corrDiag[pnt->getMaxLocVarID() - nCorrPar]; // material corrections for this point start here
    pnt->unDiagMatCorr(corrDiagP, corr);
  }
  float* detELoss = pnt->getMatCorrExp();
  for (int i = kNKinParBON; i--;) {
    corr[i] += detELoss[i];
  }
  //  if (!pnt->getELossVaried()) corr[kParQ2Pt] = pnt->getMatCorrExp()[kParQ2Pt]; // fixed eloss expected effect
  //  printf("apply corr UD %+.3e %+.3e %+.3e %+.3e\n",corr[0],corr[1],corr[2],corr[3]);
  //  printf("      corr  D %+.3e %+.3e %+.3e %+.3e\n",corrDiagP[0],corrDiagP[1],corrDiagP[2],corrDiagP[3]);
  //  printf("at point :"); pnt->print();
  //
  for (int itr = ntr; itr--;) {
    if (!applyMatCorr(trSet[itr], corr)) {
      if (algConf.verbose > 2) {
        LOGP(error, "Failed on clone {} materials", itr);
        trSet[itr].print();
      }
      return false;
    }
  }
  return true;
}

//______________________________________________
double AlignmentTrack::richardsonExtrap(double* val, int ord)
{
  // Calculate Richardson extrapolation of order ord (starting from 1)
  // The array val should contain estimates ord+1 of derivatives with variations
  // d, d/2 ... d/2^ord.
  // The array val is overwritten
  //
  if (ord == 1) {
    return (4. * val[1] - val[0]) * (1. / 3);
  }
  do {
    for (int i = 0; i < ord; i++) {
      val[i] = (4. * val[i + 1] - val[i]) * (1. / 3);
    }
  } while (--ord);
  return val[0];
}

//______________________________________________
double AlignmentTrack::richardsonExtrap(const double* val, int ord)
{
  // Calculate Richardson extrapolation of order ord (starting from 1)
  // The array val should contain estimates ord+1 of derivatives with variations
  // d, d/2 ... d/2^ord.
  // The array val is not overwritten
  //
  if (ord == 1) {
    return (4. * val[1] - val[0]) * (1. / 3);
  }
  double* buff = new double[ord + 1];
  memcpy(buff, val, (ord + 1) * sizeof(double));
  do {
    for (int i = 0; i < ord; i++) {
      buff[i] = (4. * buff[i + 1] - buff[i]) * (1. / 3);
    }
  } while (--ord);
  return buff[0];
}

//______________________________________________
void AlignmentTrack::richardsonDeriv(const trackParam_t* trSet, const double* delta, const AlignmentPoint* pnt, double& derY, double& derZ)
{
  // Calculate Richardson derivatives for diagonalized Y and Z from a set of kRichardsonN pairs
  // of tracks with same parameter of i-th pair varied by +-delta[i]
  static double derRichY[kRichardsonN], derRichZ[kRichardsonN];
  //
  for (int icl = 0; icl < kRichardsonN; icl++) { // calculate kRichardsonN variations with del, del/2, del/4...
    double resYVP = 0, resYVN = 0, resZVP = 0, resZVN = 0;
    pnt->getResidualsDiag(trSet[(icl << 1) + 0].getParams(), resYVP, resZVP); // variation with +delta
    pnt->getResidualsDiag(trSet[(icl << 1) + 1].getParams(), resYVN, resZVN); // variation with -delta
    derRichY[icl] = 0.5 * (resYVP - resYVN) / delta[icl];                     // 2-point symmetric derivatives
    derRichZ[icl] = 0.5 * (resZVP - resZVN) / delta[icl];
  }
  derY = richardsonExtrap(derRichY, kRichardsonOrd); // dY/dPar
  derZ = richardsonExtrap(derRichZ, kRichardsonOrd); // dZ/dPar
  if (TMath::IsNaN(derY) || TMath::IsNaN(derZ)) {
    LOGP(error, "NAN encounterd: DerY {}  : DerZ {}", derY, derZ);
  }
  //
}

//______________________________________________
void AlignmentTrack::Print(Option_t* opt) const
{
  // print track data
  printf("%s ", isCosmic() ? "  Cosmic  " : "Collision ");
  trackParam_t::print();
  printf("N Free Par: %d (Kinem: %d) | Npoints: %d (Inner:%d) | Chi2Ini:%.1f Chi2: %.1f/%d",
         mNLocPar, mNLocExtPar, getNPoints(), getInnerPointID(), mChi2Ini, mChi2, mNDF);
  if (isCosmic()) {
    int npLow = getInnerPointID();
    int npUp = getNPoints() - npLow - 1;
    printf(" [Low:%.1f/%d Up:%.1f/%d]", mChi2CosmDn, npLow, mChi2CosmUp, npUp);
  }
  printf("\n");
  //
  TString optS = opt;
  optS.ToLower();
  bool res = optS.Contains("r") && getResidDone();
  bool der = optS.Contains("d") && getDerivDone();
  bool par = optS.Contains("lc");   // local param corrections
  bool paru = optS.Contains("lcu"); // local param corrections in track param frame
  //
  if (par) {
    printf("Ref.track corr: ");
    for (int i = 0; i < mNLocExtPar; i++) {
      printf("%+.3e ", mLocPar[i]);
    }
    printf("\n");
  }
  //
  if (optS.Contains("p") || res || der) {
    for (int ip = 0; ip < getNPoints(); ip++) {
      printf("#%3d ", ip);
      auto* pnt = getPoint(ip);
      pnt->print(0); // RS FIXME OPT
      //
      if (res && pnt->containsMeasurement()) {
        printf("  Residuals  : %+.3e %+.3e -> Pulls: %+7.2f %+7.2f\n",
               getResidual(0, ip), getResidual(1, ip),
               getResidual(0, ip) / sqrt(pnt->getErrDiag(0)), getResidual(1, ip) / sqrt(pnt->getErrDiag(1)));
      }
      if (der && pnt->containsMeasurement()) {
        for (int ipar = 0; ipar < mNLocPar; ipar++) {
          printf("  Dres/dp%03d : %+.3e %+.3e\n", ipar, getDResDLoc(0, ip)[ipar], getDResDLoc(1, ip)[ipar]);
        }
      }
      //
      if (par && pnt->containsMaterial()) { // material corrections
        int nCorrPar = pnt->getNMatPar();
        printf("  Corr.Diag:  ");
        for (int i = 0; i < nCorrPar; i++) {
          printf("%+.3e ", mLocPar[i + pnt->getMaxLocVarID() - nCorrPar]);
        }
        printf("\n");
        printf("  Corr.Pull:  ");
        const float* corCov = pnt->getMatCorrCov(); // correction covariance
        // const float *corExp = pnt->getMatCorrExp(); // correction expectation
        for (int i = 0; i < nCorrPar; i++) {
          printf("%+.3e ", (mLocPar[i + pnt->getMaxLocVarID() - nCorrPar] /* - corExp[i]*/) / Sqrt(corCov[i]));
        }
        printf("\n");
        if (paru) { // print also mat.corrections in track frame
          double corr[5] = {0};
          pnt->unDiagMatCorr(&mLocPar[pnt->getMaxLocVarID() - nCorrPar], corr);
          //   if (!pnt->getELossVaried()) corr[kParQ2Pt] = pnt->getMatCorrExp()[kParQ2Pt]; // fixed eloss expected effect
          printf("  Corr.Track: ");
          for (int i = 0; i < kNKinParBON; i++) {
            printf("%+.3e ", corr[i]);
          }
          printf("\n");
        }
      }
    }
  } // print points
}

//______________________________________________
void AlignmentTrack::dumpCoordinates() const
{
  // print various coordinates for inspection
  printf("gpx/D:gpy/D:gpz/D:gtxb/D:gtyb/D:gtzb/D:gtxa/D:gtya/D:gtza/D:alp/D:px/D:py/D:pz/D:tyb/D:tzb/D:tya/D:tza/D:ey/D:ez/D\n");
  for (int ip = 0; ip < getNPoints(); ip++) {
    auto* pnt = getPoint(ip);
    if (!pnt->containsMeasurement()) {
      continue;
    }
    pnt->dumpCoordinates();
  }
}

//______________________________________________
bool AlignmentTrack::iniFit()
{
  // perform initial fit of the track
  //
  trackParam_t trc(*(trackParam_t*)this);
  //
  if (!getFieldON()) { // for field-off data impose nominal momentum
  }
  mChi2 = mChi2CosmUp = mChi2CosmDn = 0;
  //
  // the points are ranged from outer to inner for collision tracks,
  // and from outer point of lower leg to outer point of upper leg for the cosmic track
  //
  // the fit will always start from the outgoing track in inward direction
  const auto& algConf = AlignConfig::Instance();
  if (!fitLeg(trc, 0, getInnerPointID(), mNeedInv[0])) {
    if (algConf.verbose > 2) {
      LOG(warn) << "Failed fitLeg 0";
      trc.print();
    }
    return false; // collision track or cosmic lower leg
  }
  //
  //  printf("Lower leg: %d %d\n",0,getInnerPointID()); trc.print();
  //
  if (isCosmic()) {
    mChi2CosmDn = mChi2;
    trackParam_t trcU = trc;
    if (!fitLeg(trcU, getNPoints() - 1, getInnerPointID() + 1, mNeedInv[1])) { //fit upper leg of cosmic track
      if (algConf.verbose > 2) {
        LOG(warn) << "Failed fitLeg 0";
        trc.print();
      }
      return false; // collision track or cosmic lower leg
    }
    //
    // propagate to reference point, which is the inner point of lower leg
    const AlignmentPoint* refP = getPoint(getInnerPointID());
    if (!propagateToPoint(trcU, refP, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), nullptr, -1)) { // moving along the track: energy is lost
      return false;
    }
    //
    mChi2CosmUp = mChi2 - mChi2CosmDn;
    //    printf("Upper leg: %d %d\n",getInnerPointID()+1,getNPoints()-1); trcU.print();
    //
    if (!combineTracks(trc, trcU)) {
      return false;
    }
    //printf("Combined\n"); trc.print();
  }
  copyFrom(&trc);
  //
  mChi2Ini = mChi2;
  return true;
}

//______________________________________________
bool AlignmentTrack::combineTracks(trackParam_t& trcL, const trackParam_t& trcU)
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
  const auto& algConf = AlignConfig::Instance();
  if (Abs(trcL.getX() - trcU.getX()) > TinyDist || Abs(trcL.getAlpha() - trcU.getAlpha()) > TinyDist) {
    if (algConf.verbose > 2) {
      LOG(error) << "Tracks must be defined at same reference X and Alpha";
      trcL.print();
      trcU.print();
    }
    return false;
  }
  //
  //  const covMat_t& covU = trcU.getCov();
  //  const covMat_t& covL = trcL.getCov();
  //
  int mtSize = getFieldON() ? kNKinParBON : kNKinParBOFF;
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
    if (algConf.verbose > 2) {
      LOG(error) << "Failed to invert summed cov.matrix of cosmic track";
      matCLplCU.Print();
    }
    return false; // inversion failed
  }
  TMatrixD matK(matCL, TMatrixD::kMult, matCLplCU); // gain K = Cl*(Cl+Cu)^-1
  TMatrixD matKdotCL(matK, TMatrixD::kMult, matCL); // K*Cl
  TVectorD vlUp = matK * vUmnvL;                    // K*(vl - vu)
  for (int i = mtSize; i--;) {
    trcL.updateParam(vlUp[i], i); // updated param: vL' = vL + K(vU-vL)
    for (int j = i + 1; j--;) {
      trcL.updateCov(-matKdotCL(i, j), i, j);
    } // updated covariance: Cl' = Cl - K*Cl
  }
  //
  // update chi2
  double chi2 = 0;
  for (int i = mtSize; i--;) {
    for (int j = mtSize; j--;) {
      chi2 += matCLplCU(i, j) * vUmnvL[i] * vUmnvL[j];
    }
  }
  mChi2 += chi2;
  //
  //  printf("Combined: Chi2Tot:%.2f ChiUp:%.2f ChiDn:%.2f ChiCmb:%.2f\n",mChi2,mChi2CosmUp,mChi2CosmDn, chi2);

  return true;
}

//______________________________________________
bool AlignmentTrack::fitLeg(trackParam_t& trc, int pFrom, int pTo, bool& inv)
{
  // perform initial fit of the track
  // the fit will always start from the outgoing track in inward direction (i.e. if cosmics - bottom leg)
  const int kMinNStep = 3;
  const double kErrSpace = 50.;
  const double kErrAng = 0.8;
  const double kErrRelPtI = 1.;
  const covMat_t kIniErr{// initial error
                         kErrSpace * kErrSpace,
                         0, kErrSpace * kErrSpace,
                         0, 0, kErrAng * kErrAng,
                         0, 0, 0, kErrAng * kErrAng,
                         0, 0, 0, 0, kErrRelPtI * kErrRelPtI};
  //
  static int count = 0;
  const auto& algConf = AlignConfig::Instance();
  if (algConf.verbose > 2) {
    LOGP(info, "FIT COUNT {}", count++);
  }
  // prepare seed at outer point
  AlignmentPoint* p0 = getPoint(pFrom);
  double phi = trc.getPhi(), alp = p0->getAlphaSens();
  math_utils::detail::bringTo02Pi(phi);
  math_utils::detail::bringTo02Pi(alp);
  double dphi = math_utils::detail::deltaPhiSmall(phi, alp); // abs delta angle
  if (dphi > Pi() / 2.) {                // need to invert the track to new frame
    inv = true;
    trc.invert();
  }
  // Fit is done from outward to inward: normally against the track direction (hence e.loss is compensated) inless inversion is requested (cosmic upper leg)
  if (!propagateParamToPoint(trc, p0, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), -1)) {
    if (algConf.verbose > 2) {
      LOGF(warn, "Failed on PropagateParamOnlyTo to %f", p0->getXTracking());
      trc.print();
    }
    return false;
  }
  trc.setCov(kIniErr);
  trc.setCov(16 * trc.getQ2Pt() * trc.getQ2Pt(), 4, 4); // lowest diagonal element (Q2Pt2)
  //
  int pinc, signELoss = 0; // RS Validate for cosmic. Propagation is done from inner point to outer one:
  // energy loss is applied for collision tracks and cosmic lower leg, compensated for cosmic upper leg
  if (pTo > pFrom) { // fit in points increasing order: collision track or cosmics lower leg
    pTo++;
    pinc = 1;
    signELoss = 1; // e.loss is compensated
  } else { // fit in points decreasing order: cosmics upper leg
    pTo--;
    pinc = -1;
    signELoss = -1; // e.loass is applied
  }
  //
  int pntCnt = 0;
  for (int ip = pFrom; ip != pTo; ip += pinc) { // inward fit from outer point
    AlignmentPoint* pnt = getPoint(ip);
    if (!propagateToPoint(trc, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), nullptr, signELoss)) { // against track direction : e.loss is compensated
      if (algConf.verbose > 2) {
        LOGF(warn, "Failed on propagateToPoint %d (%d : %d) %f", ip, pFrom, pTo, pnt->getXTracking());
        trc.print();
      }
      return false;
    }
    if (pnt->containsMeasurement()) {
      if (pnt->getNeedUpdateFromTrack()) {
        pnt->updatePointByTrackInfo(&trc);
      }
      const double* yz = pnt->getYZTracking();
      const double* errYZ = pnt->getYZErrTracking();
      double chi = trc.getPredictedChi2(yz, errYZ);
      if (!trc.update(yz, errYZ)) {
        if (algConf.verbose > 2) {
          LOGF(warn, "Failed on Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
          trc.print();
        }
        return false;
      }
      mChi2 += chi;
      pntCnt++;
    }
  }
  //
  if (inv) {
    trc.invert();
  }
  if (algConf.verbose > 2) {
    LOGP(info, "Fitted leg with {} points, chi2: {}", pntCnt, mChi2);
  }
  //
  return true;
}

//______________________________________________
bool AlignmentTrack::residKalman()
{
  // calculate residuals from bi-directional Kalman smoother
  // ATTENTION: this method modifies workspaces of the points!!!
  //
  bool inv = false;
  const int kMinNStep = 3;
  const double kErrSpace = 100.;
  const double kErrAng = 0.7;
  const double kErrRelPtI = 1.;
  const covMat_t kIniErr = {// initial error
                            kErrSpace * kErrSpace,
                            0, kErrSpace * kErrSpace,
                            0, 0, kErrAng * kErrAng,
                            0, 0, 0, kErrAng * kErrAng,
                            0, 0, 0, 0, kErrRelPtI * kErrRelPtI};
  //  const double kOverShootX = 5;
  //
  trackParam_t trc = *this;
  const auto& algConf = AlignConfig::Instance();
  int pID = 0, nPnt = getNPoints();
  AlignmentPoint* pnt = nullptr;
  // get 1st measured point
  while (pID < nPnt && !(pnt = getPoint(pID))->containsMeasurement()) { // points are ordered from outward to inward
    pID++;
  }
  if (!pnt) {
    return false;
  }
  double phi = trc.getPhi(), alp = pnt->getAlphaSens();
  math_utils::detail::bringTo02Pi(phi);
  math_utils::detail::bringTo02Pi(alp);
  double dphi = math_utils::detail::deltaPhiSmall(phi, alp);
  if (dphi > Pi() / 2.) { // need to invert the track to new frame
    inv = true;
    trc.invert();
  }
  // prepare track seed at 1st outermost valid point
  if (!propagateParamToPoint(trc, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType::USEMatCorrNONE, 1)) {
    if (algConf.verbose > 2) {
      LOG(warn) << "Failed on propagateParamToPoint";
      trc.print();
      pnt->print(AlignmentPoint::kMeasurementBit);
    }
    return false;
  }
  //
  trc.setCov(kIniErr);
  const double inwardQ2Pt2 = 4 * trc.getCovarElem(4, 4) * trc.getQ2Pt() * trc.getQ2Pt();
  trc.setCov(inwardQ2Pt2, 4, 4); // lowest diagonal element (Q2Pt2)
  //
  double chifwd = 0, chibwd = 0;
  // inward fit
  int signELoss = 1; // normally inward fit goes against track direction: eloss is compensated. // RS Check for cosmic
  for (int ip = pID; ip < nPnt; ip++) {
    pnt = getPoint(ip);
    if (pnt->isInvDir() != inv) { // crossing point where the track should be inverted?
      trc.invert();
      inv = !inv;
    }
    if (!propagateToPoint(trc, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), nullptr, signELoss)) {
      return false;
    }
    if (!pnt->containsMeasurement()) {
      continue;
    }
    const double* yz = pnt->getYZTracking();
    const double* errYZ = pnt->getYZErrTracking();
    // store track position/errors before update in the point WorkSpace-A
    double* ws = (double*)pnt->getTrParamWSA();
    ws[0] = trc.getY();
    ws[1] = trc.getZ();
    ws[2] = trc.getSigmaY2();
    ws[3] = trc.getSigmaZY();
    ws[4] = trc.getSigmaZ2();
    double chi = trc.getPredictedChi2(yz, errYZ);
    if (!trc.update(yz, errYZ)) {
      if (algConf.verbose > 2) {
        LOGF(warn, "Failed on Inward Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
        trc.print();
      }
      return false;
    }
    //    printf(">>Aft ");trc.print();
    chifwd += chi;
    //    printf("KLM After update: (%f) -> %f\n",chi,chifwd);   trc.print();
  }
  //
  // outward fit
  trc.setCov(kIniErr);
  const double outwardQ2Pt2 = 4 * trc.getCovarElem(4, 4) * trc.getQ2Pt() * trc.getQ2Pt();
  trc.setCov(outwardQ2Pt2, 4, 4); // lowest diagonal element (Q2Pt2)
  signELoss = -1;                 // normally outward fit goes along track direction: eloss is applied. // RS Check for cosmic
  for (int ip = nPnt; ip--;) {
    pnt = getPoint(ip);
    if (pnt->isInvDir() != inv) { // crossing point where the track should be inverted?
      trc.invert();
      inv = !inv;
    }
    if (!propagateToPoint(trc, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), nullptr, signELoss)) { // we are going along track direction, e.loss is applied
      return false;
    }
    if (!pnt->containsMeasurement()) {
      continue;
    }
    const double* yz = pnt->getYZTracking();
    const double* errYZ = pnt->getYZErrTracking();
    // store track position/errors before update in the point WorkSpace-B
    double* ws = (double*)pnt->getTrParamWSB();
    ws[0] = trc.getY();
    ws[1] = trc.getZ();
    ws[2] = trc.getSigmaY2();
    ws[3] = trc.getSigmaZY();
    ws[4] = trc.getSigmaZ2();
    double chi = trc.getPredictedChi2(yz, errYZ);
    //    printf("<< OUT%d (%9d): %+.2e %+.2e | %+.2e %+.2e %+.2e %+.2e %+.2e | %.2e %d \n",ip,pnt->getSensor()->getInternalID(),yz[0],yz[1], ws[0],ws[1],ws[2],ws[3],ws[4],chi,inv);
    //    printf("<<Bef ");    trc.print();
    if (!trc.update(yz, errYZ)) {
      if (algConf.verbose > 2) {
        LOGF(warn, "Failed on Outward Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
        trc.print();
      }
      return false;
    }
    chibwd += chi;
    //    printf("<<Aft ");    trc.print();
  }
  //
  // now compute smoothed prediction and residual
  for (int ip = 0; ip < nPnt; ip++) {
    pnt = getPoint(ip);
    if (!pnt->containsMeasurement()) {
      continue;
    }
    double* wsA = (double*)pnt->getTrParamWSA(); // inward measurement
    double* wsB = (double*)pnt->getTrParamWSB(); // outward measurement
    double &yA = wsA[0], &zA = wsA[1], &sgAYY = wsA[2], &sgAYZ = wsA[3], &sgAZZ = wsA[4];
    double &yB = wsB[0], &zB = wsB[1], &sgBYY = wsB[2], &sgBYZ = wsB[3], &sgBZZ = wsB[4];
    // compute weighted average
    double sgYY = sgAYY + sgBYY, sgYZ = sgAYZ + sgBYZ, sgZZ = sgAZZ + sgBZZ;
    double detI = sgYY * sgZZ - sgYZ * sgYZ;
    if (TMath::Abs(detI) < constants::math::Almost0) {
      return false;
    } else {
      detI = 1. / detI;
    }
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
    //    printf("|| WGH%d (%9d): | %+.2e %+.2e %+.2e %.2e %.2e\n",ip,pnt->getSensor()->getInternalID(), wsA[0],wsA[1],wsA[2],wsA[3],wsA[4]);
  }
  //
  mChi2 = chifwd;
  setKalmanDone(true);
  return true;
}

//______________________________________________
bool AlignmentTrack::processMaterials()
{
  // attach material effect info to alignment points
  trackParam_t trc = *this;
  const auto& algConf = AlignConfig::Instance();
  // collision track or cosmic lower leg: move along track direction from last (middle for cosmic lower leg) point (inner) to 1st one (outer)
  if (mNeedInv[0]) {
    trc.invert();
  } // track needs to be inverted ? (should be for upper leg)
  if (!processMaterials(trc, getInnerPointID(), 0)) {
    if (algConf.verbose > 2) {
      LOG(error) << "Failed to process materials for leg along the track";
    }
    return false;
  }
  if (isCosmic()) {
    // cosmic upper leg: move againg the track direction from middle point (inner) to last one (outer)
    trc = *this;
    if (mNeedInv[1]) {
      trc.invert();
    } // track needs to be inverted ?
    if (!processMaterials(trc, getInnerPointID() + 1, getNPoints() - 1)) {
#if DEBUG > 3
      LOG(error) << "Failed to process materials for leg against the track";
#endif
      return false;
    }
  }
  return true;
}

//______________________________________________
bool AlignmentTrack::processMaterials(trackParam_t& trc, int pFrom, int pTo)
{
  // attach material effect info to alignment points
  const int kMinNStep = 3;
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
  track::TrackLTIntegral matTL;
  double dpar[5] = {0};
  covMat_t dcov{0};
  matTL.setTimeNotNeeded();
  const auto& algConf = AlignConfig::Instance();
  // calculate min X/X0 to treat the point as having materials
  double minX2X0 = algConf.minScatteringAngleToAccount / 0.014 * trc.getPt();
  minX2X0 *= minX2X0;

  //
  int pinc, signELoss = 0;
  if (pTo > pFrom) { // fit inward: cosmics upper leg
    pTo++;
    pinc = 1;
    signELoss = 1; // against the track direction, eloss is applied (RS Check for cosmic)
  } else {         // fit outward collision track or cosmics lower leg: along the track direction:
    pTo--;
    pinc = -1;
    signELoss = -1; // eloss is applied
  }
  //
  for (int ip = pFrom; ip != pTo; ip += pinc) { // points are ordered against track direction
    AlignmentPoint* pnt = getPoint(ip);
    trc.setCov(kErrTiny); // assign tiny errors to both tracks
    tr0 = trc;
    //
    matTL.clearFast();
    //    printf("-> ProcMat %d (%d->%d)\n",ip,pFrom,pTo);
    if (!propagateToPoint(trc, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType(algConf.matCorType), &matTL, signELoss)) { // with material corrections
      if (algConf.verbose > 2) {
        LOG(error) << "Failed to take track to point" << ip << " (dir: " << pFrom << "->" << pTo << ") with mat.corr.";
        trc.print();
        pnt->print(AlignmentPoint::kMeasurementBit);
      }
      return false;
    }
    //
    // is there enough material to consider the point as a scatterer?
    bool hasMaterial = matTL.getX2X0() > minX2X0;
    if (!propagateToPoint(tr0, pnt, algConf.maxStep, algConf.maxSnp, MatCorrType::USEMatCorrNONE, nullptr, signELoss)) { // no material corrections
      if (algConf.verbose > 2) {
        LOG(error) << "Failed to take track to point" << ip << " (dir: " << pFrom << "->" << pTo << ") with mat.corr.";
        tr0.print();
        pnt->print(AlignmentPoint::kMeasurementBit);
      }
      return false;
    }
    // the difference between the params, covariance of tracks with and  w/o material accounting gives
    // parameters and covariance of material correction. For params ONLY ELoss effect is relevant
    double *par0 = (double*)tr0.getParams(), *par1 = (double*)trc.getParams();
    const covMat_t &cov0 = tr0.getCov(), &cov1 = trc.getCov();
    for (int l = 15; l--;) {
      dcov[l] = cov1[l] - cov0[l];
    }
    for (int l = kNKinParBON; l--;) {
      dpar[l] = par1[l] - par0[l];
    } // eloss affects all parameters!
    pnt->setMatCorrExp(dpar);
    //dpar[kParQ2Pt] = par1[kParQ2Pt] - par0[kParQ2Pt]; // only e-loss expectation is non-0
    //
    if (hasMaterial) {
      // MP2 handles only scalar residuals hence correlated matrix of material effect need to be diagonalized
      bool eLossFree = pnt->getELossVaried();
      int nParFree = eLossFree ? kNKinParBON : kNKinParBOFF;
      TMatrixDSym matCov(nParFree);
      for (int i = nParFree; i--;) {
        for (int j = i + 1; j--;) {
          auto err2 = dcov[j + ((i * (i + 1)) >> 1)];
          if (i == j && err2 < 1e-20) { // meaninglessly small diagonal error
            LOGP(warn, "Material correction {}-th error too small: {}, declare no material despite x/X0={}", i, err2, matTL.getX2X0());
            hasMaterial = false;
            break;
          }
          matCov(i, j) = matCov(j, i) = err2;
        }
      }
      if (hasMaterial) {
        TMatrixDSymEigen matDiag(matCov); // find eigenvectors
        const TMatrixD& matEVec = matDiag.GetEigenVectors();
        if (!matEVec.IsValid()) {
          if (algConf.verbose > 2) {
            LOG(error) << "Failed to diagonalize covariance of material correction";
            matCov.Print();
            return false;
          }
        }
        for (int ix = 0; ix < matDiag.GetEigenValues().GetNrows(); ix++) {
          if (matDiag.GetEigenValues()[ix] < 1e-18) {
            LOG(warn) << "Too small " << ix << "-th eignevalue " << matDiag.GetEigenValues()[ix] << " for cov.matrix !!!";
            LOG(warn) << "Track with mat.corr: " << trc.asString();
            LOG(warn) << "Track w/o  mat.corr: " << tr0.asString();
            matCov.Print();
            matDiag.GetEigenValues().Print();
            hasMaterial = false;
            break;
          }
        }
        if (hasMaterial) {
          pnt->setMatCovDiagonalizationMatrix(matEVec); // store diagonalization matrix
          pnt->setMatCovDiag(matDiag.GetEigenValues()); // store E.Values: diagonalized cov.matrix
          if (!eLossFree) {
            pnt->setMatCovDiagElem(kParQ2Pt, dcov[14]);
          }
          pnt->setX2X0(matTL.getX2X0());
          pnt->setXTimesRho(matTL.getXRho());
        }
      }
      pnt->setContainsMaterial(hasMaterial);
    }
    if (pnt->containsMeasurement()) { // update track to have best possible kinematics
      const double* yz = pnt->getYZTracking();
      const double* errYZ = pnt->getYZErrTracking();
      if (!trc.update(yz, errYZ)) {
        if (algConf.verbose > 2) {
          LOGF(warn, "Failed on Update %f,%f {%f,%f,%f}", yz[0], yz[1], errYZ[0], errYZ[1], errYZ[2]);
          trc.print();
        }
        return false;
      }
      //
    }
    //
  }
  //
  return true;
}

//______________________________________________
void AlignmentTrack::sortPoints()
{
  // sort points in order against track direction: innermost point is last
  // for collision tracks.
  // For 2-leg cosmic tracks: 1st points of outgoing (lower) leg are added from large to
  // small radii, then the points of incomint (upper) leg are added in increasing R direction
  //
  // The mInnerPointID will mark the id of the innermost point, i.e. the last one for collision-like
  // tracks and in case of cosmics - the point of lower leg with smallest R
  //
  // the mDetPoints vector will not change anymore: it is safe to work with pointers
  for (size_t i = 0; i < mDetPoints.size(); i++) {
    mPoints.push_back(&mDetPoints[i]);
  }

  auto isAfter = [](const AlignmentPoint* a, const AlignmentPoint* b) {
    auto xa = a->getXTracking(), xb = b->getXTracking();
    if (!a->isInvDir()) {   // lower leg: track propagates from low to large X via this point
      if (!b->isInvDir()) { // this one also
        return xa > xb;
      } else {
        return true; // any point on the lower leg has higher priority than on the upper leg
      }
    } else { // this point is from upper cosmic leg: track propagates from large to low X
      if (b->isInvDir()) {
        return xa < xb;
      } else {
        return false;
      }
    }
  };

  std::sort(mPoints.begin(), mPoints.end(), isAfter);
  int np = getNPoints();
  mInnerPointID = np - 1;
  if (isCosmic()) {
    for (int ip = np; ip--;) {
      if (getPoint(ip)->isInvDir()) {
        continue;
      } // this is a point of upper leg
      mInnerPointID = ip;
      break;
    }
  }
  //
}

//______________________________________________
void AlignmentTrack::setLocPars(const double* pars)
{
  // store loc par corrections
  memcpy(mLocPar.data(), pars, mNLocPar * sizeof(double));
}

//______________________________________________
void AlignmentTrack::addLocPars(const double* pars)
{
  // store loc par corrections
  for (int i = 0; i < mNLocPar; i++) {
    mLocPar[i] += pars[i];
  }
}

//______________________________________________
void AlignmentTrack::checkExpandDerGloBuffer(unsigned int minSize)
{
  // if needed, expand global derivatives buffer
  if (mGloParID.size() < minSize) {
    mGloParID.resize(minSize);
    mDResDGlo[0].resize(minSize);
    mDResDGlo[1].resize(minSize);
  }
}

//____________________________________________
bool AlignmentTrack::testLocalSolution()
{
  // test track local solution
  int npnt = getNPoints();
  double mat[mNLocPar][mNLocPar], rhs[mNLocPar];
  std::memset(mat, 0, sizeof(double) * mNLocPar * mNLocPar);
  std::memset(rhs, 0, sizeof(double) * mNLocPar);
  for (int ip = npnt; ip--;) {
    if (mPoints[ip]->containsMeasurement()) {
      for (int idim = 2; idim--;) {                       // each point has 2 position residuals
        auto sg2inv = 1. / mPoints[ip]->getErrDiag(idim); // inv. error
        auto deriv = getDResDLoc(idim, ip);               // array of Dresidual/Dparams
        for (int parI = 0; parI < mNLocPar; parI++) {
          rhs[parI] -= deriv[parI] * mResid[idim][ip] * sg2inv;
          for (int parJ = parI; parJ < mNLocPar; parJ++) {
            mat[parI][parJ] += deriv[parI] * deriv[parJ] * sg2inv;
          }
        }
      } // loop over 2 orthogonal measurements at the point
    }   // derivarives at measured points
    // if the point contains material, consider its expected kinks, eloss as measurements
    if (mPoints[ip]->containsMaterial()) { // at least 4 parameters: 2 spatial + 2 angular kinks with 0 expectaction
      int npm = mPoints[ip]->getNMatPar();
      // const float* expMatCorr = mPoints[ip]->getMatCorrExp(); // expected correction (diagonalized) // RS??
      const auto expMatCov = mPoints[ip]->getMatCorrCov(); // its error
      int offs = mPoints[ip]->getMaxLocVarID() - npm;
      for (int ipar = 0; ipar < npm; ipar++) {
        int parI = offs + ipar;
        // expected
        // rhs[parI] -= expMatCorr[ipar]/expMatCov[ipar]; // consider expectation as measurement // RS??
        mat[parI][parI] += 1. / expMatCov[ipar]; // this measurement is orthogonal to all others
      }
    } // material effect descripotion params
    //
  }
  o2::math_utils::SymMatrixSolver solver(mNLocPar, 1);
  for (int i = 0; i < mNLocPar; i++) {
    for (int j = i; j < mNLocPar; j++) {
      solver.A(i, j) = mat[i][j];
    }
    solver.B(i, 0) = rhs[i];
  }
  solver.solve();
  // increment current params by new solution
  for (int i = 0; i < mNLocPar; i++) {
    mLocPar[i] += solver.B(i, 0);
  }
  return calcResiduals();
}

//______________________________________________
void AlignmentTrack::suppressLastPoints(int n)
{
  // remove last n points, must be called before sortPoints
  while (n > 0) {
    mDetPoints.pop_back();
    n--;
  }
}

} // namespace align
} // namespace o2
