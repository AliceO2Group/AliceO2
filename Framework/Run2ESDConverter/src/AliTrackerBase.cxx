/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id: AliTrackerBase.cxx 38069 2009-12-24 16:56:18Z belikov $ */

//-------------------------------------------------------------------------
//               Implementation of the AliTrackerBase class
//                that is the base for the AliTracker class    
//                     Origin: Marian.Ivanov@cern.ch
//-------------------------------------------------------------------------
#include <TClass.h>
#include <TMath.h>
#include <TGeoManager.h>

#include "AliLog.h"
#include "AliTrackerBase.h"
#include "AliExternalTrackParam.h"
#include "AliTrackPointArray.h"
#include "TVectorD.h"

extern TGeoManager *gGeoManager;

ClassImp(AliTrackerBase)

AliTrackerBase::AliTrackerBase():
  TObject(),
  fX(0),
  fY(0),
  fZ(0),
  fSigmaX(0.005),
  fSigmaY(0.005),
  fSigmaZ(0.010)
{
  //--------------------------------------------------------------------
  // The default constructor.
  //--------------------------------------------------------------------
  if (!TGeoGlobalMagField::Instance()->GetField())
    AliWarning("Field map is not set.");
}

//__________________________________________________________________________
AliTrackerBase::AliTrackerBase(const AliTrackerBase &atr):
  TObject(atr),
  fX(atr.fX),
  fY(atr.fY),
  fZ(atr.fZ),
  fSigmaX(atr.fSigmaX),
  fSigmaY(atr.fSigmaY),
  fSigmaZ(atr.fSigmaZ)
{
  //--------------------------------------------------------------------
  // The default constructor.
  //--------------------------------------------------------------------
  if (!TGeoGlobalMagField::Instance()->GetField())
    AliWarning("Field map is not set.");
}

//__________________________________________________________________________
Double_t AliTrackerBase::GetBz()
{
  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  if (!fld) {
    AliFatalClass("Field is not loaded");
    //if (!fld) 
    return  0.5*kAlmost0Field;
  }
  Double_t bz = fld->SolenoidField();
  return TMath::Sign(0.5*kAlmost0Field,bz) + bz;
}

//__________________________________________________________________________
Double_t AliTrackerBase::GetBz(const Double_t *r) {
  //------------------------------------------------------------------
  // Returns Bz (kG) at the point "r" .
  //------------------------------------------------------------------
  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  if (!fld) {
    AliFatalClass("Field is not loaded");
    //  if (!fld) 
    return  0.5*kAlmost0Field;
  }
  Double_t bz = fld->GetBz(r);
  return  TMath::Sign(0.5*kAlmost0Field,bz) + bz;
}

//__________________________________________________________________________
void AliTrackerBase::GetBxByBz(const Double_t r[3], Double_t b[3]) {
  //------------------------------------------------------------------
  // Returns Bx, By and Bz (kG) at the point "r" .
  //------------------------------------------------------------------
  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  if (!fld) {
    AliFatalClass("Field is not loaded");
    // b[0] = b[1] = 0.;
    // b[2] = 0.5*kAlmost0Field;
    return;
  }

  if (fld->IsUniform()) {
     b[0] = b[1] = 0.;
     b[2] = fld->SolenoidField();
  }  else {
     fld->Field(r,b);
  }
  b[2] = (TMath::Sign(0.5*kAlmost0Field,b[2]) + b[2]);
  return;
}

Double_t AliTrackerBase::MeanMaterialBudget(const Double_t *start, const Double_t *end, Double_t *mparam)
{
  // 
  // Calculate mean material budget and material properties between 
  //    the points "start" and "end".
  //
  // "mparam" - parameters used for the energy and multiple scattering
  //  corrections: 
  //
  // mparam[0] - mean density: sum(x_i*rho_i)/sum(x_i) [g/cm3]
  // mparam[1] - equivalent rad length fraction: sum(x_i/X0_i) [adimensional]
  // mparam[2] - mean A: sum(x_i*A_i)/sum(x_i) [adimensional]
  // mparam[3] - mean Z: sum(x_i*Z_i)/sum(x_i) [adimensional]
  // mparam[4] - length: sum(x_i) [cm]
  // mparam[5] - Z/A mean: sum(x_i*Z_i/A_i)/sum(x_i) [adimensional]
  // mparam[6] - number of boundary crosses
  //
  //  Origin:  Marian Ivanov, Marian.Ivanov@cern.ch
  //
  //  Corrections and improvements by
  //        Andrea Dainese, Andrea.Dainese@lnl.infn.it,
  //        Andrei Gheata,  Andrei.Gheata@cern.ch
  //

  mparam[0]=0; mparam[1]=1; mparam[2] =0; mparam[3] =0;
  mparam[4]=0; mparam[5]=0; mparam[6]=0;
  //
  Double_t bparam[6]; // total parameters
  Double_t lparam[6]; // local parameters

  for (Int_t i=0;i<6;i++) bparam[i]=0;

  if (!gGeoManager) {
    AliFatalClass("No TGeo\n");
    return 0.;
  }
  //
  Double_t length;
  Double_t dir[3];
  length = TMath::Sqrt((end[0]-start[0])*(end[0]-start[0])+
                       (end[1]-start[1])*(end[1]-start[1])+
                       (end[2]-start[2])*(end[2]-start[2]));
  mparam[4]=length;
  if (length<TGeoShape::Tolerance()) return 0.0;
  Double_t invlen = 1./length;
  dir[0] = (end[0]-start[0])*invlen;
  dir[1] = (end[1]-start[1])*invlen;
  dir[2] = (end[2]-start[2])*invlen;

  // Initialize start point and direction
  TGeoNode *currentnode = 0;
  TGeoNode *startnode = gGeoManager->InitTrack(start, dir);
  if (!startnode) {
    AliDebugClass(1,Form("start point out of geometry: x %f, y %f, z %f",
			 start[0],start[1],start[2]));
    return 0.0;
  }
  TGeoMaterial *material = startnode->GetVolume()->GetMedium()->GetMaterial();
  lparam[0]   = material->GetDensity();
  lparam[1]   = material->GetRadLen();
  lparam[2]   = material->GetA();
  lparam[3]   = material->GetZ();
  lparam[4]   = length;
  lparam[5]   = lparam[3]/lparam[2];
  if (material->IsMixture()) {
    TGeoMixture * mixture = (TGeoMixture*)material;
    lparam[5] =0;
    Double_t sum =0;
    for (Int_t iel=0;iel<mixture->GetNelements();iel++){
      sum  += mixture->GetWmixt()[iel];
      lparam[5]+= mixture->GetZmixt()[iel]*mixture->GetWmixt()[iel]/mixture->GetAmixt()[iel];
    }
    lparam[5]/=sum;
  }

  // Locate next boundary within length without computing safety.
  // Propagate either with length (if no boundary found) or just cross boundary
  gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
  Double_t step = 0.0; // Step made
  Double_t snext = gGeoManager->GetStep();
  // If no boundary within proposed length, return current density
  if (!gGeoManager->IsOnBoundary()) {
    mparam[0] = lparam[0];
    mparam[1] = lparam[4]/lparam[1];
    mparam[2] = lparam[2];
    mparam[3] = lparam[3];
    mparam[4] = lparam[4];
    return lparam[0];
  }
  // Try to cross the boundary and see what is next
  Int_t nzero = 0;
  while (length>TGeoShape::Tolerance()) {
    currentnode = gGeoManager->GetCurrentNode();
    if (snext<2.*TGeoShape::Tolerance()) nzero++;
    else nzero = 0;
    if (nzero>3) {
      // This means navigation has problems on one boundary
      // Try to cross by making a small step
      static int show_error = !(getenv("HLT_ONLINE_MODE") && strcmp(getenv("HLT_ONLINE_MODE"), "on") == 0);
      if (show_error) AliErrorClass("Cannot cross boundary");
      mparam[0] = bparam[0]/step;
      mparam[1] = bparam[1];
      mparam[2] = bparam[2]/step;
      mparam[3] = bparam[3]/step;
      mparam[5] = bparam[5]/step;
      mparam[4] = step;
      mparam[0] = 0.;             // if crash of navigation take mean density 0
      mparam[1] = 1000000;        // and infinite rad length
      return bparam[0]/step;
    }
    mparam[6]+=1.;
    step += snext;
    bparam[1]    += snext/lparam[1];
    bparam[2]    += snext*lparam[2];
    bparam[3]    += snext*lparam[3];
    bparam[5]    += snext*lparam[5];
    bparam[0]    += snext*lparam[0];

    if (snext>=length) break;
    if (!currentnode) break;
    length -= snext;
    material = currentnode->GetVolume()->GetMedium()->GetMaterial();
    lparam[0] = material->GetDensity();
    lparam[1]  = material->GetRadLen();
    lparam[2]  = material->GetA();
    lparam[3]  = material->GetZ();
    lparam[5]   = lparam[3]/lparam[2];
    if (material->IsMixture()) {
      TGeoMixture * mixture = (TGeoMixture*)material;
      lparam[5]=0;
      Double_t sum =0;
      for (Int_t iel=0;iel<mixture->GetNelements();iel++){
        sum+= mixture->GetWmixt()[iel];
        lparam[5]+= mixture->GetZmixt()[iel]*mixture->GetWmixt()[iel]/mixture->GetAmixt()[iel];
      }
      lparam[5]/=sum;
    }
    gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
    snext = gGeoManager->GetStep();
  }
  mparam[0] = bparam[0]/step;
  mparam[1] = bparam[1];
  mparam[2] = bparam[2]/step;
  mparam[3] = bparam[3]/step;
  mparam[5] = bparam[5]/step;
  return bparam[0]/step;
}


Bool_t 
AliTrackerBase::PropagateTrackTo(AliExternalTrackParam *track, Double_t xToGo, 
				 Double_t mass, Double_t maxStep, Bool_t rotateTo, Double_t maxSnp, Int_t sign, Bool_t addTimeStep, Bool_t correctMaterialBudget){
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm) using the magnetic field map 
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction (if <0 then q = 2)
  // maxStep  - maximal step for propagation
  //
  //  Origin: Marian Ivanov,  Marian.Ivanov@cern.ch
  //
  //----------------------------------------------------------------
  const Double_t kEpsilon = 0.00001;
  Double_t xpos     = track->GetX();
  Int_t dir         = (xpos<xToGo) ? 1:-1;
  //
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = xpos+step;
    Double_t xyz0[3],xyz1[3],param[7];
    track->GetXYZ(xyz0);   //starting global position

    Double_t bz=GetBz(xyz0); // getting the local Bz
    if (!track->GetXYZAt(x,bz,xyz1)) return kFALSE;   // no prolongation
    xyz1[2]+=kEpsilon; // waiting for bug correction in geo
    
    if (maxSnp>0 && TMath::Abs(track->GetSnpAt(x,bz)) >= maxSnp) return kFALSE;
    if (!track->PropagateTo(x,bz))  return kFALSE;

    if (correctMaterialBudget){
      MeanMaterialBudget(xyz0,xyz1,param);
      Double_t xrho=param[0]*param[4], xx0=param[1];
      if (sign) {if (sign<0) xrho = -xrho;}  // sign is imposed
      else { // determine automatically the sign from direction
        if (dir>0) xrho = -xrho; // outward should be negative
      }
      //
      if (!track->CorrectForMeanMaterial(xx0,xrho,mass)) return kFALSE;
    }
    
    if (rotateTo){
      track->GetXYZ(xyz1);   // global position
      Double_t alphan = TMath::ATan2(xyz1[1], xyz1[0]); 
      if (maxSnp>0) {
	if (TMath::Abs(track->GetSnp()) >= maxSnp) return kFALSE;
        
	//
	Double_t ca=TMath::Cos(alphan-track->GetAlpha()), sa=TMath::Sin(alphan-track->GetAlpha());
	Double_t sf=track->GetSnp(), cf=TMath::Sqrt((1.-sf)*(1.+sf));
	Double_t sinNew =  sf*ca - cf*sa;
        if (TMath::Abs(sinNew) >= maxSnp) return kFALSE;
        
      }
      if (!track->AliExternalTrackParam::Rotate(alphan)) return kFALSE;
      
    }
    xpos = track->GetX();
    if (addTimeStep && track->IsStartedTimeIntegral()) {
      if (!rotateTo) track->GetXYZ(xyz1); // if rotateTo==kTRUE, then xyz1 is already extracted
      Double_t dX=xyz0[0]-xyz1[0],dY=xyz0[1]-xyz1[1],dZ=xyz0[2]-xyz1[2]; 
      Double_t d=TMath::Sqrt(dX*dX + dY*dY + dZ*dZ);
      if (sign) {if (sign>0) d = -d;}  // step sign is imposed, positive means inward direction
      else { // determine automatically the sign from direction
	if (dir<0) d = -d;
      }
      track->AddTimeStep(d);
    }
  }
  return kTRUE;
}

Bool_t AliTrackerBase::PropagateTrackParamOnlyTo(AliExternalTrackParam *track, Double_t xToGo,Double_t maxStep, Bool_t rotateTo, Double_t maxSnp)
{
  //----------------------------------------------------------------
  //
  // Propagates in fixed step size the track params ONLY to the plane X=xk (cm) using the magnetic field map 
  // W/O correcting for the crossed material.
  // maxStep  - maximal step for propagation
  //
  //----------------------------------------------------------------
  const Double_t kEpsilon = 0.00001;
  double xpos = track->GetX();
  Int_t dir   = (xpos<xToGo) ? 1:-1;
  //
  double xyz[3];
  track->GetXYZ(xyz);
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = track->GetX()+step;
    Double_t bz=GetBz(xyz); // getting the local Bz
    if (!track->PropagateParamOnlyTo(x,bz))  return kFALSE;
    track->GetXYZ(xyz);   // global position
    if (rotateTo){
      Double_t alphan = TMath::ATan2(xyz[1], xyz[0]); 
      if (maxSnp>0 && TMath::Abs(track->GetSnp()) >= maxSnp) return kFALSE;
      if (!track->AliExternalTrackParam::RotateParamOnly(alphan)) return kFALSE;      
    }
    xpos = track->GetX();
  }
  return kTRUE;
}

Int_t AliTrackerBase::PropagateTrackTo2(AliExternalTrackParam *track, Double_t xToGo,
                                        Double_t mass, Double_t maxStep, Bool_t rotateTo, Double_t maxSnp, Int_t sign, Bool_t addTimeStep, Bool_t correctMaterialBudget){
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm) using the magnetic field map
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction
  // maxStep  - maximal step for propagation
  //
  //  Origin: Marian Ivanov,  Marian.Ivanov@cern.ch
  //
  //----------------------------------------------------------------
  const Double_t kEpsilon = 0.00001;
  Double_t xpos     = track->GetX();
  Int_t dir         = (xpos<xToGo) ? 1:-1;
  //
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = xpos+step;
    Double_t xyz0[3],xyz1[3],param[7];
    track->GetXYZ(xyz0);   //starting global position
    
    Double_t bz=GetBz(xyz0); // getting the local Bz
    if (!track->GetXYZAt(x,bz,xyz1)) return -1;   // no prolongation
    xyz1[2]+=kEpsilon; // waiting for bug correction in geo
    
    if (maxSnp>0 && TMath::Abs(track->GetSnpAt(x,bz)) >= maxSnp) return -2;
    if (!track->PropagateTo(x,bz))  return -3;
    
    if (correctMaterialBudget){
      MeanMaterialBudget(xyz0,xyz1,param);
      Double_t xrho=param[0]*param[4], xx0=param[1];
      if (sign) {if (sign<0) xrho = -xrho;}  // sign is imposed
      else { // determine automatically the sign from direction
        if (dir>0) xrho = -xrho; // outward should be negative
      }
      //
      if (!track->CorrectForMeanMaterial(xx0,xrho,mass)) return -4;
    }
    
    if (rotateTo){
      track->GetXYZ(xyz1);   // global position
      Double_t alphan = TMath::ATan2(xyz1[1], xyz1[0]);
      if (maxSnp>0) {
        if (TMath::Abs(track->GetSnp()) >= maxSnp) return -5;
        
        //
        Double_t ca=TMath::Cos(alphan-track->GetAlpha()), sa=TMath::Sin(alphan-track->GetAlpha());
        Double_t sf=track->GetSnp(), cf=TMath::Sqrt((1.-sf)*(1.+sf));
        Double_t sinNew =  sf*ca - cf*sa;
        if (TMath::Abs(sinNew) >= maxSnp) return -6;
        
      }
      if (!track->AliExternalTrackParam::Rotate(alphan)) return -7;
      
    }
    xpos = track->GetX();
    if (addTimeStep && track->IsStartedTimeIntegral()) {
      if (!rotateTo) track->GetXYZ(xyz1); // if rotateTo==kTRUE, then xyz1 is already extracted
      Double_t dX=xyz0[0]-xyz1[0],dY=xyz0[1]-xyz1[1],dZ=xyz0[2]-xyz1[2];
      Double_t d=TMath::Sqrt(dX*dX + dY*dY + dZ*dZ);
      if (sign) {if (sign>0) d = -d;}  // step sign is imposed, positive means inward direction
      else { // determine automatically the sign from direction
        if (dir<0) d = -d;
      }
      track->AddTimeStep(d);
    }
  }
  return 1;
}

Bool_t 
AliTrackerBase::PropagateTrackToBxByBz(AliExternalTrackParam *track,
				       Double_t xToGo,Double_t mass, Double_t maxStep, Bool_t rotateTo, Double_t maxSnp,Int_t sign, Bool_t addTimeStep,
				       Bool_t correctMaterialBudget){
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field 
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction (if <0 then q=2)
  // maxStep  - maximal step for propagation
  //
  //  Origin: Marian Ivanov,  Marian.Ivanov@cern.ch
  //
  //----------------------------------------------------------------
  const Double_t kEpsilon = 0.00001;
  Double_t xpos     = track->GetX();
  Int_t dir         = (xpos<xToGo) ? 1:-1;
  //
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = xpos+step;
    Double_t xyz0[3],xyz1[3],param[7];
    track->GetXYZ(xyz0);   //starting global position

    Double_t b[3]; GetBxByBz(xyz0,b); // getting the local Bx, By and Bz

    if (!track->GetXYZAt(x,b[2],xyz1)) return kFALSE;   // no prolongation
    xyz1[2]+=kEpsilon; // waiting for bug correction in geo

    //    if (maxSnp>0 && TMath::Abs(track->GetSnpAt(x,b[2])) >= maxSnp) return kFALSE;
    if (!track->PropagateToBxByBz(x,b))  return kFALSE;
    if (maxSnp>0 && TMath::Abs(track->GetSnp())>=maxSnp) return kFALSE;

    if (correctMaterialBudget) {
      MeanMaterialBudget(xyz0,xyz1,param);    
      Double_t xrho=param[0]*param[4], xx0=param[1];
      if (sign) {if (sign<0) xrho = -xrho;}  // sign is imposed
      else { // determine automatically the sign from direction
	if (dir>0) xrho = -xrho; // outward should be negative
      }    
      //
      if (!track->CorrectForMeanMaterial(xx0,xrho,mass)) return kFALSE;
    }
    if (rotateTo){
      track->GetXYZ(xyz1);   // global position
      Double_t alphan = TMath::ATan2(xyz1[1], xyz1[0]); 
      /*
	if (maxSnp>0) {
	if (TMath::Abs(track->GetSnp()) >= maxSnp) return kFALSE;
	Double_t ca=TMath::Cos(alphan-track->GetAlpha()), sa=TMath::Sin(alphan-track->GetAlpha());
	Double_t sf=track->GetSnp(), cf=TMath::Sqrt((1.-sf)*(1.+sf));
	Double_t sinNew =  sf*ca - cf*sa;
	if (TMath::Abs(sinNew) >= maxSnp) return kFALSE;
	}
      */
      if (!track->AliExternalTrackParam::Rotate(alphan)) return kFALSE;
      if (maxSnp>0 && TMath::Abs(track->GetSnp())>=maxSnp) return kFALSE;
    }
    xpos = track->GetX();    
    if (addTimeStep && track->IsStartedTimeIntegral()) {
      if (!rotateTo) track->GetXYZ(xyz1); // if rotateTo==kTRUE, then xyz1 is already extracted
      Double_t dX=xyz0[0]-xyz1[0],dY=xyz0[1]-xyz1[1],dZ=xyz0[2]-xyz1[2]; 
      Double_t d=TMath::Sqrt(dX*dX + dY*dY + dZ*dZ);
      if (sign) {if (sign>0) d = -d;}  // step sign is imposed, positive means inward direction
      else { // determine automatically the sign from direction
	if (dir<0) d = -d;
      }
      track->AddTimeStep(d);
    }
  }
  return kTRUE;
}

Bool_t AliTrackerBase::PropagateTrackParamOnlyToBxByBz(AliExternalTrackParam *track,
						       Double_t xToGo,Double_t maxStep, Bool_t rotateTo, Double_t maxSnp)
{
  //----------------------------------------------------------------
  //
  // Propagates in fixed step size the track params ONLY to the plane X=xk (cm) using the magnetic field map 
  // W/O correcting for the crossed material.
  // maxStep  - maximal step for propagation
  //
  //----------------------------------------------------------------
  const Double_t kEpsilon = 0.00001;
  Double_t xpos     = track->GetX();
  Int_t dir         = (xpos<xToGo) ? 1:-1;
  Double_t xyz[3];
  track->GetXYZ(xyz);
  //
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = xpos+step;
    Double_t b[3]; GetBxByBz(xyz,b); // getting the local Bx, By and Bz
    if (!track->PropagateParamOnlyBxByBzTo(x,b))  return kFALSE;
    if (maxSnp>0 && TMath::Abs(track->GetSnp()) >= maxSnp) return kFALSE;
    track->GetXYZ(xyz);
    if (rotateTo){
      Double_t alphan = TMath::ATan2(xyz[1], xyz[0]); 
      if (!track->AliExternalTrackParam::Rotate(alphan)) return kFALSE;
    }
    xpos = track->GetX();    
  }
  return kTRUE;
}

Double_t AliTrackerBase::GetTrackPredictedChi2(AliExternalTrackParam *track,
                                           Double_t mass, Double_t step,
			             const AliExternalTrackParam *backup) {
  //
  // This function brings the "track" with particle "mass" [GeV] 
  // to the same local coord. system and the same reference plane as 
  // of the "backup", doing it in "steps" [cm].
  // Then, it calculates the 5D predicted Chi2 for these two tracks
  //
  Double_t chi2=kVeryBig;
  Double_t alpha=backup->GetAlpha();
  if (!track->Rotate(alpha)) return chi2;

  Double_t xb=backup->GetX();
  Double_t sign=(xb < track->GetX()) ? 1. : -1.;
  if (!PropagateTrackTo(track,xb,mass,step,kFALSE,kAlmost1,sign)) return chi2;

  chi2=track->GetPredictedChi2(backup);

  return chi2;
}




Double_t AliTrackerBase::MakeC(Double_t x1,Double_t y1,
                   Double_t x2,Double_t y2,
                   Double_t x3,Double_t y3)
{
  //-----------------------------------------------------------------
  // Initial approximation of the track curvature
  //-----------------------------------------------------------------
  x3 -=x1;
  x2 -=x1;
  y3 -=y1;
  y2 -=y1;
  //  
  Double_t det = x3*y2-x2*y3;
  if (TMath::Abs(det)<1e-10) {
    return 0;
  }
  //
  Double_t u = 0.5* (x2*(x2-x3)+y2*(y2-y3))/det;
  Double_t x0 = x3*0.5-y3*u;
  Double_t y0 = y3*0.5+x3*u;
  Double_t c2 = 1/TMath::Sqrt(x0*x0+y0*y0);
  if (det>0) c2*=-1;
  return c2;
}

Double_t AliTrackerBase::MakeSnp(Double_t x1,Double_t y1,
                   Double_t x2,Double_t y2,
                   Double_t x3,Double_t y3)
{
  //-----------------------------------------------------------------
  // Initial approximation of the track snp
  //-----------------------------------------------------------------
  x3 -=x1;
  x2 -=x1;
  y3 -=y1;
  y2 -=y1;
  //  
  Double_t det = x3*y2-x2*y3;
  if (TMath::Abs(det)<1e-10) {
    return 0;
  }
  //
  Double_t u = 0.5* (x2*(x2-x3)+y2*(y2-y3))/det;
  Double_t x0 = x3*0.5-y3*u; 
  Double_t y0 = y3*0.5+x3*u;
  Double_t c2 = 1./TMath::Sqrt(x0*x0+y0*y0);
  x0*=c2;
  x0=TMath::Abs(x0);
  if (y2*x2<0.) x0*=-1;
  return x0;
}

Double_t AliTrackerBase::MakeTgl(Double_t x1,Double_t y1,
                   Double_t x2,Double_t y2,
                   Double_t z1,Double_t z2, Double_t c)
{
  //-----------------------------------------------------------------
  // Initial approximation of the tangent of the track dip angle
  //-----------------------------------------------------------------
  //
  const Double_t kEpsilon =0.00001;
  x2-=x1;
  y2-=y1;
  z2-=z1;
  Double_t d  =  TMath::Sqrt(x2*x2+y2*y2);  // distance  straight line
  if (TMath::Abs(d*c*0.5)>1) return 0;
  Double_t   angle2    = TMath::ASin(d*c*0.5); 
  if (TMath::Abs(angle2)>kEpsilon)  {
    angle2  = z2*TMath::Abs(c/(angle2*2.));
  }else{
    angle2=z2/d;
  }
  return angle2;
}


Double_t AliTrackerBase::MakeTgl(Double_t x1,Double_t y1, 
                   Double_t x2,Double_t y2,
                   Double_t z1,Double_t z2) 
{
  //-----------------------------------------------------------------
  // Initial approximation of the tangent of the track dip angle
  //-----------------------------------------------------------------
  return (z1 - z2)/sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}


AliExternalTrackParam * AliTrackerBase::MakeSeed( AliTrackPoint &point0, AliTrackPoint &point1, AliTrackPoint &point2){
  //
  // Make Seed  - AliExternalTrackParam from input 3 points   
  // returning seed in local frame of point0
  //
  Double_t xyz0[3]={0,0,0};
  Double_t xyz1[3]={0,0,0};
  Double_t xyz2[3]={0,0,0};
  Double_t alpha=point0.GetAngle();
  Double_t xyz[3]={point0.GetX(),point0.GetY(),point0.GetZ()};
  Double_t bxyz[3]; GetBxByBz(xyz,bxyz); 
  Double_t bz = bxyz[2];
  //
  // get points in frame of point 0
  //
  AliTrackPoint p0r = point0.Rotate(alpha);
  AliTrackPoint p1r = point1.Rotate(alpha);
  AliTrackPoint p2r = point2.Rotate(alpha);
  xyz0[0]=p0r.GetX();
  xyz0[1]=p0r.GetY();
  xyz0[2]=p0r.GetZ();
  xyz1[0]=p1r.GetX();
  xyz1[1]=p1r.GetY();
  xyz1[2]=p1r.GetZ();
  xyz2[0]=p2r.GetX();
  xyz2[1]=p2r.GetY();
  xyz2[2]=p2r.GetZ();
  //
  // make covariance estimate
  //  
  Double_t covar[15];
  Double_t param[5]={0,0,0,0,0};
  for (Int_t m=0; m<15; m++) covar[m]=0;
  //
  // calculate intitial param
  param[0]=xyz0[1];              
  param[1]=xyz0[2];
  param[2]=MakeSnp(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz2[0],xyz2[1]);
  param[4]=MakeC(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz2[0],xyz2[1]);
  param[3]=MakeTgl(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz0[2],xyz1[2],param[4]);

  //covariance matrix - only diagonal elements
  //Double_t dist=p0r.GetX()-p2r.GetX();
  Double_t deltaP=0;
  covar[0]= p0r.GetCov()[3];
  covar[2]= p0r.GetCov()[5];
  //sigma snp
  deltaP= (MakeSnp(xyz0[0],xyz0[1]+TMath::Sqrt(p0r.GetCov()[3]),xyz1[0],xyz1[1],xyz2[0],xyz2[1])-param[2]);
  covar[5]+= deltaP*deltaP;
  deltaP= (MakeSnp(xyz0[0],xyz0[1],xyz1[0],xyz1[1]+TMath::Sqrt(p1r.GetCov()[3]),xyz2[0],xyz2[1])-param[2]);
  covar[5]+= deltaP*deltaP;
  deltaP= (MakeSnp(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz2[0],xyz2[1]+TMath::Sqrt(p1r.GetCov()[3]))-param[2]);
  covar[5]+= deltaP*deltaP;
  //sigma tgl
  //
  deltaP=MakeTgl(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz0[2]+TMath::Sqrt(p1r.GetCov()[5]),xyz1[2],param[4])-param[3];
  covar[9]+= deltaP*deltaP;
  deltaP=MakeTgl(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz0[2],xyz1[2]+TMath::Sqrt(p1r.GetCov()[5]),param[4])-param[3];
  covar[9]+= deltaP*deltaP;
  //
  
  deltaP=MakeC(xyz0[0],xyz0[1]+TMath::Sqrt(p0r.GetCov()[3]),xyz1[0],xyz1[1],xyz2[0],xyz2[1])-param[4];
  covar[14]+= deltaP*deltaP;
  deltaP=MakeC(xyz0[0],xyz0[1],xyz1[0],xyz1[1]+TMath::Sqrt(p1r.GetCov()[3]),xyz2[0],xyz2[1])-param[4];
  covar[14]+= deltaP*deltaP;
  deltaP=MakeC(xyz0[0],xyz0[1],xyz1[0],xyz1[1],xyz2[0],xyz2[1]+TMath::Sqrt(p2r.GetCov()[3]))-param[4];
  covar[14]+= deltaP*deltaP;
  
  if (TMath::Abs(bz)>kAlmost0Field) {
    covar[14]/=(bz*kB2C)*(bz*kB2C);
    param[4]/=(bz*kB2C); // transform to 1/pt
  }
  else { // assign 0.6 GeV pT
    const double kq2pt = 1./0.6;
    param[4] = kq2pt;
    covar[14] = (0.5*0.5)*kq2pt;
  }
  AliExternalTrackParam * trackParam = new AliExternalTrackParam(xyz0[0],alpha,param, covar);
  if (0) {
    // consistency check  -to put warnings here 
    // small disagrement once Track extrapolation used 
    // nice agreement in seeds with MC track parameters - problem in extrapoloation - to be fixed
    // to check later
    Double_t y1,y2,z1,z2;
    trackParam->GetYAt(xyz1[0],bz,y1);
    trackParam->GetZAt(xyz1[0],bz,z1);
    trackParam->GetYAt(xyz2[0],bz,y2);
    trackParam->GetZAt(xyz2[0],bz,z2);
    if (TMath::Abs(y1-xyz1[1])> TMath::Sqrt(p1r.GetCov()[3]*5)){
      AliWarningClass("Seeding problem y1\n");
    }
    if (TMath::Abs(y2-xyz2[1])> TMath::Sqrt(p2r.GetCov()[3]*5)){
      AliWarningClass("Seeding problem y2\n");
    }
    if (TMath::Abs(z1-xyz1[2])> TMath::Sqrt(p1r.GetCov()[5]*5)){
      AliWarningClass("Seeding problem z1\n");
    }
  }
  return trackParam;  
} 

Double_t  AliTrackerBase::FitTrack(AliExternalTrackParam * trackParam, AliTrackPointArray *pointArray, Double_t mass, Double_t maxStep){
  //
  // refit the track  - trackParam using the points in point array  
  //
  const Double_t kMaxSnp=0.99;
  if (!trackParam) return 0;
  Int_t  npoints=pointArray->GetNPoints();
  AliTrackPoint point,point2;
  Double_t pointPos[2]={0,0};
  Double_t pointCov[3]={0,0,0};
  // choose coordinate frame
  // in standard way the coordinate frame should be changed point by point
  // Some problems with rotation observed
  // rotate method of AliExternalTrackParam should be revisited
  pointArray->GetPoint(point,0);
  pointArray->GetPoint(point2,npoints-1);
  Double_t alpha=TMath::ATan2(point.GetY()-point2.GetY(), point.GetX()-point2.GetX());
  
  for (Int_t ipoint=npoints-1; ipoint>0; ipoint-=1){
    pointArray->GetPoint(point,ipoint);
    AliTrackPoint pr = point.Rotate(alpha);
    trackParam->Rotate(alpha);
    Bool_t status = PropagateTrackTo(trackParam,pr.GetX(),mass,maxStep,kFALSE,kMaxSnp);
    if(!status){
      AliWarningClass("Problem to propagate\n");    
      break;
    }
    if (TMath::Abs(trackParam->GetSnp())>kMaxSnp){ 
      AliWarningClass("sin(phi) > kMaxSnp \n");
      break;
    }
    pointPos[0]=pr.GetY();//local y
    pointPos[1]=pr.GetZ();//local z
    pointCov[0]=pr.GetCov()[3];//simay^2
    pointCov[1]=pr.GetCov()[4];//sigmayz
    pointCov[2]=pr.GetCov()[5];//sigmaz^2
    trackParam->Update(pointPos,pointCov); 
  }
  return 0;
}



void AliTrackerBase::UpdateTrack(AliExternalTrackParam &track1, const AliExternalTrackParam &track2){
  //
  // Update track 1 with track 2
  //
  //
  //
  TMatrixD vecXk(5,1);    // X vector
  TMatrixD covXk(5,5);    // X covariance 
  TMatrixD matHk(5,5);    // vector to mesurement
  TMatrixD measR(5,5);    // measurement error 
  TMatrixD vecZk(5,1);    // measurement
  //
  TMatrixD vecYk(5,1);    // Innovation or measurement residual
  TMatrixD matHkT(5,5);
  TMatrixD matSk(5,5);    // Innovation (or residual) covariance
  TMatrixD matKk(5,5);    // Optimal Kalman gain
  TMatrixD mat1(5,5);     // update covariance matrix
  TMatrixD covXk2(5,5);   // 
  TMatrixD covOut(5,5);
  //
  Double_t *param1=(Double_t*) track1.GetParameter();
  Double_t *covar1=(Double_t*) track1.GetCovariance();
  Double_t *param2=(Double_t*) track2.GetParameter();
  Double_t *covar2=(Double_t*) track2.GetCovariance();
  //
  // copy data to the matrix
  for (Int_t ipar=0; ipar<5; ipar++){
    for (Int_t jpar=0; jpar<5; jpar++){
      covXk(ipar,jpar) = covar1[track1.GetIndex(ipar, jpar)];
      measR(ipar,jpar) = covar2[track2.GetIndex(ipar, jpar)];
      matHk(ipar,jpar)=0;
      mat1(ipar,jpar)=0;
    }
    vecXk(ipar,0) = param1[ipar];
    vecZk(ipar,0) = param2[ipar];
    matHk(ipar,ipar)=1;
    mat1(ipar,ipar)=1;
  }
  //
  //
  //
  //
  //
  vecYk = vecZk-matHk*vecXk;                 // Innovation or measurement residual
  matHkT=matHk.T(); matHk.T();
  matSk = (matHk*(covXk*matHkT))+measR;      // Innovation (or residual) covariance
  matSk.Invert();
  matKk = (covXk*matHkT)*matSk;              //  Optimal Kalman gain
  vecXk += matKk*vecYk;                      //  updated vector 
  covXk2 = (mat1-(matKk*matHk));
  covOut =  covXk2*covXk; 
  //
  //
  //
  // copy from matrix to parameters
  if (0) {
    vecXk.Print();
    vecZk.Print();
    //
    measR.Print();
    covXk.Print();
    covOut.Print();
    //
    track1.Print();
    track2.Print();
  }

  for (Int_t ipar=0; ipar<5; ipar++){
    param1[ipar]= vecXk(ipar,0) ;
    for (Int_t jpar=0; jpar<5; jpar++){
      covar1[track1.GetIndex(ipar, jpar)]=covOut(ipar,jpar);
    }
  }
}


