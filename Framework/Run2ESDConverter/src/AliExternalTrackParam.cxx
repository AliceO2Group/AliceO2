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

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Implementation of the external track parameterisation class.              //
//                                                                           //
// This parameterisation is used to exchange tracks between the detectors.   //
// A set of functions returning the position and the momentum of tracks      //
// in the global coordinate system as well as the track impact parameters    //
// are implemented.
// Origin: I.Belikov, CERN, Jouri.Belikov@cern.ch                            //
///////////////////////////////////////////////////////////////////////////////
#include <cassert>

#include <TVectorD.h>
#include <TMatrixDSym.h>
#include <TPolyMarker3D.h>
#include <TVector3.h>
#include <TMatrixD.h>

#include "AliExternalTrackParam.h"
#include "AliVVertex.h"
#include "AliLog.h"

ClassImp(AliExternalTrackParam)

Double32_t AliExternalTrackParam::fgMostProbablePt=kMostProbablePt;
Bool_t AliExternalTrackParam::fgUseLogTermMS = kFALSE;; 
//_____________________________________________________________________________
AliExternalTrackParam::AliExternalTrackParam() :
  AliVTrack(),
  fX(0),
  fAlpha(0)
{
  //
  // default constructor
  //
  for (Int_t i = 0; i < 5; i++) fP[i] = 0;
  for (Int_t i = 0; i < 15; i++) fC[i] = 0;
}

//_____________________________________________________________________________
AliExternalTrackParam::AliExternalTrackParam(const AliExternalTrackParam &track):
  AliVTrack(track),
  fX(track.fX),
  fAlpha(track.fAlpha)
{
  //
  // copy constructor
  //
  for (Int_t i = 0; i < 5; i++) fP[i] = track.fP[i];
  for (Int_t i = 0; i < 15; i++) fC[i] = track.fC[i];
  CheckCovariance();
}

//_____________________________________________________________________________
AliExternalTrackParam& AliExternalTrackParam::operator=(const AliExternalTrackParam &trkPar)
{
  //
  // assignment operator
  //
  
  if (this!=&trkPar) {
    AliVTrack::operator=(trkPar);
    fX = trkPar.fX;
    fAlpha = trkPar.fAlpha;

    for (Int_t i = 0; i < 5; i++) fP[i] = trkPar.fP[i];
    for (Int_t i = 0; i < 15; i++) fC[i] = trkPar.fC[i];
    CheckCovariance();
  }

  return *this;
}

//_____________________________________________________________________________
AliExternalTrackParam::AliExternalTrackParam(Double_t x, Double_t alpha, 
					     const Double_t param[5], 
					     const Double_t covar[15]) :
  AliVTrack(),
  fX(x),
  fAlpha(alpha)
{
  //
  // create external track parameters from given arguments
  //
  for (Int_t i = 0; i < 5; i++)  fP[i] = param[i];
  for (Int_t i = 0; i < 15; i++) fC[i] = covar[i];
  CheckCovariance();
}

//_____________________________________________________________________________
void AliExternalTrackParam::CopyFromVTrack(const AliVTrack *vTrack)
{
  //
  // Recreate TrackParams from VTrack
  // This is not a copy contructor !
  //
  if (!vTrack) {
    AliError("Source VTrack is NULL");
    return;
  }
  if (this==vTrack) {
    AliError("Copy of itself is requested");
    return;
  }
  //
  if (vTrack->InheritsFrom(AliExternalTrackParam::Class())) {
    AliDebug(1,"Source itself is AliExternalTrackParam, using assignment operator");
    *this = *(AliExternalTrackParam*)vTrack;
    return;
  }
  //
  AliVTrack::operator=( *vTrack );
  //
  Double_t xyz[3],pxpypz[3],cv[21];
  vTrack->GetXYZ(xyz);
  pxpypz[0]=vTrack->Px();
  pxpypz[1]=vTrack->Py();
  pxpypz[2]=vTrack->Pz();
  vTrack->GetCovarianceXYZPxPyPz(cv);
  Short_t sign = (Short_t)vTrack->Charge();
  Set(xyz,pxpypz,cv,sign);
}

//_____________________________________________________________________________
AliExternalTrackParam::AliExternalTrackParam(const AliVTrack *vTrack) :
  AliVTrack(),
  fX(0.),
  fAlpha(0.)
{
  //
  // Constructor from virtual track,
  // This is not a copy contructor !
  //

  if (vTrack->InheritsFrom("AliExternalTrackParam")) {
     AliError("This is not a copy constructor. Use AliExternalTrackParam(const AliExternalTrackParam &) !");
     AliWarning("Calling the default constructor...");
     AliExternalTrackParam();
     return;
  }

  Double_t xyz[3],pxpypz[3],cv[21];
  vTrack->GetXYZ(xyz);
  pxpypz[0]=vTrack->Px();
  pxpypz[1]=vTrack->Py();
  pxpypz[2]=vTrack->Pz();
  vTrack->GetCovarianceXYZPxPyPz(cv);
  Short_t sign = (Short_t)vTrack->Charge();

  Set(xyz,pxpypz,cv,sign);
}

//_____________________________________________________________________________
AliExternalTrackParam::AliExternalTrackParam(Double_t xyz[3],Double_t pxpypz[3],
					     Double_t cv[21],Short_t sign) :
  AliVTrack(),
  fX(0.),
  fAlpha(0.)
{
  //
  // constructor from the global parameters
  //

  Set(xyz,pxpypz,cv,sign);
}

/*
//_____________________________________________________________________________
void AliExternalTrackParam::Set(Double_t xyz[3],Double_t pxpypz[3],
				Double_t cv[21],Short_t sign) 
{
  //
  // create external track parameters from the global parameters
  // x,y,z,px,py,pz and their 6x6 covariance matrix
  // A.Dainese 10.10.08

  // Calculate alpha: the rotation angle of the corresponding local system.
  //
  // For global radial position inside the beam pipe, alpha is the
  // azimuthal angle of the momentum projected on (x,y).
  //
  // For global radial position outside the ITS, alpha is the
  // azimuthal angle of the centre of the TPC sector in which the point
  // xyz lies
  //
  const double kSafe = 1e-5;
  Double_t radPos2 = xyz[0]*xyz[0]+xyz[1]*xyz[1];  
  Double_t radMax  = 45.; // approximately ITS outer radius
  if (radPos2 < radMax*radMax) { // inside the ITS     
     fAlpha = TMath::ATan2(pxpypz[1],pxpypz[0]);
  } else { // outside the ITS
     Float_t phiPos = TMath::Pi()+TMath::ATan2(-xyz[1], -xyz[0]);
     fAlpha = 
     TMath::DegToRad()*(20*((((Int_t)(phiPos*TMath::RadToDeg()))/20))+10);
  }
  //
  Double_t cs=TMath::Cos(fAlpha), sn=TMath::Sin(fAlpha);
  // protection:  avoid alpha being too close to 0 or +-pi/2
  if (TMath::Abs(sn)<2*kSafe) {
    if (fAlpha>0) fAlpha += fAlpha< TMath::Pi()/2. ?  2*kSafe : -2*kSafe;
    else          fAlpha += fAlpha>-TMath::Pi()/2. ? -2*kSafe :  2*kSafe;
    cs=TMath::Cos(fAlpha);
    sn=TMath::Sin(fAlpha);
  }
  else if (TMath::Abs(cs)<2*kSafe) {
    if (fAlpha>0) fAlpha += fAlpha> TMath::Pi()/2. ? 2*kSafe : -2*kSafe;
    else          fAlpha += fAlpha>-TMath::Pi()/2. ? 2*kSafe : -2*kSafe;
    cs=TMath::Cos(fAlpha);
    sn=TMath::Sin(fAlpha);
  }
  // Get the vertex of origin and the momentum
  TVector3 ver(xyz[0],xyz[1],xyz[2]);
  TVector3 mom(pxpypz[0],pxpypz[1],pxpypz[2]);
  //
  // avoid momenta along axis
  if (TMath::Abs(mom[0])<kSafe) mom[0] = TMath::Sign(kSafe*TMath::Abs(mom[1]), mom[0]);
  if (TMath::Abs(mom[1])<kSafe) mom[1] = TMath::Sign(kSafe*TMath::Abs(mom[0]), mom[1]);

  // Rotate to the local coordinate system
  ver.RotateZ(-fAlpha);
  mom.RotateZ(-fAlpha);

  //
  // x of the reference plane
  fX = ver.X();

  Double_t charge = (Double_t)sign;

  fP[0] = ver.Y();
  fP[1] = ver.Z();
  fP[2] = TMath::Sin(mom.Phi());
  fP[3] = mom.Pz()/mom.Pt();
  fP[4] = TMath::Sign(1/mom.Pt(),charge);
  //
  if      (TMath::Abs( 1-fP[2]) < 3*kSafe) fP[2] = 1.- 3*kSafe; //Protection
  else if (TMath::Abs(-1-fP[2]) < 3*kSafe) fP[2] =-1.+ 3*kSafe; //Protection
  //
  // Covariance matrix (formulas to be simplified)
  Double_t pt=1./TMath::Abs(fP[4]);
  // avoid alpha+phi being to close to +-pi/2 in the cov.matrix evaluation
  double fp2 = fP[2];
  Double_t r=TMath::Sqrt((1.-fp2)*(1.+fp2));
  //
  Double_t m00=-sn;// m10=cs;
  Double_t m23=-pt*(sn + fp2*cs/r), m43=-pt*pt*(r*cs - fp2*sn);
  Double_t m24= pt*(cs - fp2*sn/r), m44=-pt*pt*(r*sn + fp2*cs);
  Double_t m35=pt, m45=-pt*pt*fP[3];

  m43*=GetSign();
  m44*=GetSign();
  m45*=GetSign();

  Double_t cv34 = TMath::Sqrt(cv[3 ]*cv[3 ]+cv[4 ]*cv[4 ]);
  Double_t a1=cv[13]-cv[9]*(m23*m44+m43*m24)/m23/m43;
  Double_t a2=m23*m24-m23*(m23*m44+m43*m24)/m43;
  Double_t a3=m43*m44-m43*(m23*m44+m43*m24)/m23;
  Double_t a4=cv[14]+2.*cv[9]; //cv[14]-2.*cv[9]*m24*m44/m23/m43;
  Double_t a5=m24*m24-2.*m24*m44*m23/m43;
  Double_t a6=m44*m44-2.*m24*m44*m43/m23;

  fC[0 ] = cv[0 ]+cv[2 ];  
  fC[1 ] = TMath::Sign(cv34,cv[3 ]/m00); 
  fC[2 ] = cv[5 ]; 
  fC[3 ] = (cv[10]*m43-cv[6]*m44)/(m24*m43-m23*m44)/m00; 
  fC[10] = (cv[6]/m00-fC[3 ]*m23)/m43; 
  fC[6 ] = (cv[15]/m00-fC[10]*m45)/m35; 
  fC[4 ] = (cv[12]*m43-cv[8]*m44)/(m24*m43-m23*m44); 
  fC[11] = (cv[8]-fC[4]*m23)/m43; 
  fC[7 ] = cv[17]/m35-fC[11]*m45/m35; 
  fC[5 ] = TMath::Abs((a4*a3-a6*a1)/(a5*a3-a6*a2));
  fC[14] = TMath::Abs((a1-a2*fC[5])/a3);
  fC[12] = (cv[9]-fC[5]*m23*m23-fC[14]*m43*m43)/m23/m43;
  Double_t b1=cv[18]-fC[12]*m23*m45-fC[14]*m43*m45;
  Double_t b2=m23*m35;
  Double_t b3=m43*m35;
  Double_t b4=cv[19]-fC[12]*m24*m45-fC[14]*m44*m45;
  Double_t b5=m24*m35;
  Double_t b6=m44*m35;
  fC[8 ] = (b4-b6*b1/b3)/(b5-b6*b2/b3);
  fC[13] = b1/b3-b2*fC[8]/b3;
  fC[9 ] = TMath::Abs((cv[20]-fC[14]*(m45*m45)-fC[13]*2.*m35*m45)/(m35*m35));

  CheckCovariance();

  return;
}
*/

//_____________________________________________________________________________
void AliExternalTrackParam::Set(Double_t xyz[3],Double_t pxpypz[3],
				Double_t cv[21],Short_t sign) 
{
  //
  // create external track parameters from the global parameters
  // x,y,z,px,py,pz and their 6x6 covariance matrix
  // A.Dainese 10.10.08

  // Calculate alpha: the rotation angle of the corresponding local system.
  //
  // For global radial position inside the beam pipe, alpha is the
  // azimuthal angle of the momentum projected on (x,y).
  //
  // For global radial position outside the ITS, alpha is the
  // azimuthal angle of the centre of the TPC sector in which the point
  // xyz lies
  //
  const double kSafe = 1e-5;
  Double_t radPos2 = xyz[0]*xyz[0]+xyz[1]*xyz[1];  
  Double_t radMax  = 45.; // approximately ITS outer radius
  if (radPos2 < radMax*radMax) { // inside the ITS     
     fAlpha = TMath::ATan2(pxpypz[1],pxpypz[0]);
  } else { // outside the ITS
     Float_t phiPos = TMath::Pi()+TMath::ATan2(-xyz[1], -xyz[0]);
     fAlpha = 
     TMath::DegToRad()*(20*((((Int_t)(phiPos*TMath::RadToDeg()))/20))+10);
  }
  //
  Double_t cs=TMath::Cos(fAlpha), sn=TMath::Sin(fAlpha);
  // protection:  avoid alpha being too close to 0 or +-pi/2
  if (TMath::Abs(sn)<2*kSafe) {
    if (fAlpha>0) fAlpha += fAlpha< TMath::Pi()/2. ?  2*kSafe : -2*kSafe;
    else          fAlpha += fAlpha>-TMath::Pi()/2. ? -2*kSafe :  2*kSafe;
    cs=TMath::Cos(fAlpha);
    sn=TMath::Sin(fAlpha);
  }
  else if (TMath::Abs(cs)<2*kSafe) {
    if (fAlpha>0) fAlpha += fAlpha> TMath::Pi()/2. ? 2*kSafe : -2*kSafe;
    else          fAlpha += fAlpha>-TMath::Pi()/2. ? 2*kSafe : -2*kSafe;
    cs=TMath::Cos(fAlpha);
    sn=TMath::Sin(fAlpha);
  }
  // Get the vertex of origin and the momentum
  TVector3 ver(xyz[0],xyz[1],xyz[2]);
  TVector3 mom(pxpypz[0],pxpypz[1],pxpypz[2]);
  //
  // Rotate to the local coordinate system
  ver.RotateZ(-fAlpha);
  mom.RotateZ(-fAlpha);

  //
  // x of the reference plane
  fX = ver.X();

  Double_t charge = (Double_t)sign;

  fP[0] = ver.Y();
  fP[1] = ver.Z();
  fP[2] = TMath::Sin(mom.Phi());
  fP[3] = mom.Pz()/mom.Pt();
  fP[4] = TMath::Sign(1/mom.Pt(),charge);
  //
  if      (TMath::Abs( 1-fP[2]) < kSafe) fP[2] = 1.- kSafe; //Protection
  else if (TMath::Abs(-1-fP[2]) < kSafe) fP[2] =-1.+ kSafe; //Protection
  //
  // Covariance matrix (formulas to be simplified)
  Double_t pt=1./TMath::Abs(fP[4]);
  Double_t r=TMath::Sqrt((1.-fP[2])*(1.+fP[2]));
  //
  Double_t cv34 = TMath::Sqrt(cv[3 ]*cv[3 ]+cv[4 ]*cv[4 ]);
  //
  Int_t special = 0;
  double sgcheck = r*sn + fP[2]*cs;
  if (TMath::Abs(sgcheck)>=1-kSafe) { // special case: lab phi is +-pi/2
    special = 1;
    sgcheck = TMath::Sign(1.0,sgcheck);
  }
  else if (TMath::Abs(sgcheck)<kSafe) {
    sgcheck = TMath::Sign(1.0,cs);
    special = 2;   // special case: lab phi is 0
  }
  //
  fC[0 ] = cv[0 ]+cv[2 ];  
  fC[1 ] = TMath::Sign(cv34,-cv[3 ]*sn); 
  fC[2 ] = cv[5 ]; 
  //
  if (special==1) {
    double pti = 1./pt;
    double pti2 = pti*pti;
    int q = GetSign();
    fC[3 ] = cv[6]*pti;
    fC[4 ] = -sgcheck*cv[8]*r*pti;
    fC[5 ] = TMath::Abs(cv[9]*r*r*pti2);
    fC[6 ] = (cv[10]*fP[3]-sgcheck*cv[15])*pti/r;
    fC[7 ] = (cv[17]-sgcheck*cv[12]*fP[3])*pti;
    fC[8 ] = (-sgcheck*cv[18]+cv[13]*fP[3])*r*pti2;
    fC[9 ] = TMath::Abs( cv[20]-2*sgcheck*cv[19]*fP[3]+cv[14]*fP[3]*fP[3])*pti2;
    fC[10] = cv[10]*pti2/r*q;
    fC[11] = -sgcheck*cv[12]*pti2*q;
    fC[12] = cv[13]*r*pti*pti2*q;
    fC[13] = (-sgcheck*cv[19]+cv[14]*fP[3])*r*pti2*pti;
    fC[14] = TMath::Abs(cv[14]*pti2*pti2);
  } else if (special==2) {
    double pti = 1./pt;
    double pti2 = pti*pti;
    int q = GetSign();
    fC[3 ] = -cv[10]*pti*cs/sn;
    fC[4 ] = cv[12]*cs*pti;
    fC[5 ] = TMath::Abs(cv[14]*cs*cs*pti2);
    fC[6 ] = (sgcheck*cv[6]*fP[3]-cv[15])*pti/sn;
    fC[7 ] = (cv[17]-sgcheck*cv[8]*fP[3])*pti;
    fC[8 ] = (cv[19]-sgcheck*cv[13]*fP[3])*cs*pti2;
    fC[9 ] = TMath::Abs( cv[20]-2*sgcheck*cv[18]*fP[3]+cv[9]*fP[3]*fP[3])*pti2;
    fC[10] = sgcheck*cv[6]*pti2/sn*q;
    fC[11] = -sgcheck*cv[8]*pti2*q;
    fC[12] = -sgcheck*cv[13]*cs*pti*pti2*q;
    fC[13] = (-sgcheck*cv[18]+cv[9]*fP[3])*pti2*pti*q;
    fC[14] = TMath::Abs(cv[9]*pti2*pti2);
  }
  else {
    Double_t m00=-sn;// m10=cs;
    Double_t m23=-pt*(sn + fP[2]*cs/r), m43=-pt*pt*(r*cs - fP[2]*sn);
    Double_t m24= pt*(cs - fP[2]*sn/r), m44=-pt*pt*(r*sn + fP[2]*cs);
    Double_t m35=pt, m45=-pt*pt*fP[3];
    //
    m43*=GetSign();
    m44*=GetSign();
    m45*=GetSign();
    //
    Double_t a1=cv[13]-cv[9]*(m23*m44+m43*m24)/m23/m43;
    Double_t a2=m23*m24-m23*(m23*m44+m43*m24)/m43;
    Double_t a3=m43*m44-m43*(m23*m44+m43*m24)/m23;
    Double_t a4=cv[14]+2.*cv[9]; //cv[14]-2.*cv[9]*m24*m44/m23/m43;
    Double_t a5=m24*m24-2.*m24*m44*m23/m43;
    Double_t a6=m44*m44-2.*m24*m44*m43/m23;
    //    
    fC[3 ] = (cv[10]*m43-cv[6]*m44)/(m24*m43-m23*m44)/m00; 
    fC[10] = (cv[6]/m00-fC[3 ]*m23)/m43; 
    fC[6 ] = (cv[15]/m00-fC[10]*m45)/m35; 
    fC[4 ] = (cv[12]*m43-cv[8]*m44)/(m24*m43-m23*m44); 
    fC[11] = (cv[8]-fC[4]*m23)/m43; 
    fC[7 ] = cv[17]/m35-fC[11]*m45/m35; 
    fC[5 ] = TMath::Abs((a4*a3-a6*a1)/(a5*a3-a6*a2));
    fC[14] = TMath::Abs((a1-a2*fC[5])/a3);
    fC[12] = (cv[9]-fC[5]*m23*m23-fC[14]*m43*m43)/m23/m43;
    Double_t b1=cv[18]-fC[12]*m23*m45-fC[14]*m43*m45;
    Double_t b2=m23*m35;
    Double_t b3=m43*m35;
    Double_t b4=cv[19]-fC[12]*m24*m45-fC[14]*m44*m45;
    Double_t b5=m24*m35;
    Double_t b6=m44*m35;
    fC[8 ] = (b4-b6*b1/b3)/(b5-b6*b2/b3);
    fC[13] = b1/b3-b2*fC[8]/b3;
    fC[9 ] = TMath::Abs((cv[20]-fC[14]*(m45*m45)-fC[13]*2.*m35*m45)/(m35*m35));
  }
  CheckCovariance();

  return;
}

//_____________________________________________________________________________
void AliExternalTrackParam::Reset() {
  //
  // Resets all the parameters to 0 
  //
  fX=fAlpha=0.;
  for (Int_t i = 0; i < 5; i++) fP[i] = 0;
  for (Int_t i = 0; i < 15; i++) fC[i] = 0;
}

//_____________________________________________________________________________
void AliExternalTrackParam::AddCovariance(const Double_t c[15]) {
  //
  // Add "something" to the track covarince matrix.
  // May be needed to account for unknown mis-calibration/mis-alignment
  //
    fC[0] +=c[0];
    fC[1] +=c[1];  fC[2] +=c[2];
    fC[3] +=c[3];  fC[4] +=c[4];  fC[5] +=c[5];
    fC[6] +=c[6];  fC[7] +=c[7];  fC[8] +=c[8];  fC[9] +=c[9];
    fC[10]+=c[10]; fC[11]+=c[11]; fC[12]+=c[12]; fC[13]+=c[13]; fC[14]+=c[14];
    CheckCovariance();
}


Double_t AliExternalTrackParam::GetP() const {
  //---------------------------------------------------------------------
  // This function returns the track momentum
  // Results for (nearly) straight tracks are meaningless !
  //---------------------------------------------------------------------
  if (TMath::Abs(fP[4])<=kAlmost0) return kVeryBig;
  return TMath::Sqrt(1.+ fP[3]*fP[3])/TMath::Abs(fP[4]);
}

Double_t AliExternalTrackParam::Get1P() const {
  //---------------------------------------------------------------------
  // This function returns the 1/(track momentum)
  //---------------------------------------------------------------------
  return TMath::Abs(fP[4])/TMath::Sqrt(1.+ fP[3]*fP[3]);
}

//_______________________________________________________________________
Double_t AliExternalTrackParam::GetD(Double_t x,Double_t y,Double_t b) const {
  //------------------------------------------------------------------
  // This function calculates the transverse impact parameter
  // with respect to a point with global coordinates (x,y)
  // in the magnetic field "b" (kG)
  //------------------------------------------------------------------
  if (TMath::Abs(b) < kAlmost0Field) return GetLinearD(x,y);
  Double_t rp4=GetC(b);

  Double_t xt=fX, yt=fP[0];

  Double_t sn=TMath::Sin(fAlpha), cs=TMath::Cos(fAlpha);
  Double_t a = x*cs + y*sn;
  y = -x*sn + y*cs; x=a;
  xt-=x; yt-=y;

  sn=rp4*xt - fP[2]; cs=rp4*yt + TMath::Sqrt((1.- fP[2])*(1.+fP[2]));
  a=2*(xt*fP[2] - yt*TMath::Sqrt((1.-fP[2])*(1.+fP[2])))-rp4*(xt*xt + yt*yt);
  return  -a/(1 + TMath::Sqrt(sn*sn + cs*cs));
}

//_______________________________________________________________________
void AliExternalTrackParam::
GetDZ(Double_t x, Double_t y, Double_t z, Double_t b, Float_t dz[2]) const {
  //------------------------------------------------------------------
  // This function calculates the transverse and longitudinal impact parameters
  // with respect to a point with global coordinates (x,y)
  // in the magnetic field "b" (kG)
  //------------------------------------------------------------------
  Double_t f1 = fP[2], r1 = TMath::Sqrt((1.-f1)*(1.+f1));
  Double_t xt=fX, yt=fP[0];
  Double_t sn=TMath::Sin(fAlpha), cs=TMath::Cos(fAlpha);
  Double_t a = x*cs + y*sn;
  y = -x*sn + y*cs; x=a;
  xt-=x; yt-=y;

  Double_t rp4=GetC(b);
  if ((TMath::Abs(b) < kAlmost0Field) || (TMath::Abs(rp4) < kAlmost0)) {
     dz[0] = -(xt*f1 - yt*r1);
     dz[1] = fP[1] + (dz[0]*f1 - xt)/r1*fP[3] - z;
     return;
  }

  sn=rp4*xt - f1; cs=rp4*yt + r1;
  a=2*(xt*f1 - yt*r1)-rp4*(xt*xt + yt*yt);
  Double_t rr=TMath::Sqrt(sn*sn + cs*cs);
  dz[0] = -a/(1 + rr);
  Double_t f2 = -sn/rr, r2 = TMath::Sqrt((1.-f2)*(1.+f2));
  dz[1] = fP[1] + fP[3]/rp4*TMath::ASin(f2*r1 - f1*r2) - z;
}

//_______________________________________________________________________
Double_t AliExternalTrackParam::GetLinearD(Double_t xv,Double_t yv) const {
  //------------------------------------------------------------------
  // This function calculates the transverse impact parameter
  // with respect to a point with global coordinates (xv,yv)
  // neglecting the track curvature.
  //------------------------------------------------------------------
  Double_t sn=TMath::Sin(fAlpha), cs=TMath::Cos(fAlpha);
  Double_t x= xv*cs + yv*sn;
  Double_t y=-xv*sn + yv*cs;

  Double_t d = (fX-x)*fP[2] - (fP[0]-y)*TMath::Sqrt((1.-fP[2])*(1.+fP[2]));

  return -d;
}

Bool_t AliExternalTrackParam::CorrectForMeanMaterialdEdx
(Double_t xOverX0,  Double_t xTimesRho, Double_t mass, 
 Double_t dEdx,
 Bool_t anglecorr) {
  //------------------------------------------------------------------
  // This function corrects the track parameters for the crossed material.
  // "xOverX0"   - X/X0, the thickness in units of the radiation length.
  // "xTimesRho" - is the product length*density (g/cm^2).
  //     It should be passed as negative when propagating tracks 
  //     from the intreaction point to the outside of the central barrel. 
  // "mass" - the mass of this particle (GeV/c^2). Negative mass means charge=2 particle
  // "dEdx" - mean enery loss (GeV/(g/cm^2)
  // "anglecorr" - switch for the angular correction
  //------------------------------------------------------------------
  Double_t &fP2=fP[2];
  Double_t &fP3=fP[3];
  Double_t &fP4=fP[4];

  Double_t &fC22=fC[5];
  Double_t &fC33=fC[9];
  Double_t &fC43=fC[13];
  Double_t &fC44=fC[14];

  //Apply angle correction, if requested
  if(anglecorr) {
    Double_t angle=TMath::Sqrt((1.+ fP3*fP3)/((1-fP2)*(1.+fP2)));
    xOverX0 *=angle;
    xTimesRho *=angle;
  } 

  Double_t p=GetP();
  if (mass<0) p += p; // q=2 particle 
  Double_t p2=p*p;
  Double_t beta2=p2/(p2 + mass*mass);

  //Calculating the multiple scattering corrections******************
  Double_t cC22 = 0.;
  Double_t cC33 = 0.;
  Double_t cC43 = 0.;
  Double_t cC44 = 0.;
  if (xOverX0 != 0) {
    //Double_t theta2=1.0259e-6*14*14/28/(beta2*p2)*TMath::Abs(d)*9.36*2.33;
    Double_t theta2=0.0136*0.0136/(beta2*p2)*TMath::Abs(xOverX0);
    if (GetUseLogTermMS()) {
      double lt = 1+0.038*TMath::Log(TMath::Abs(xOverX0));
      if (lt>0) theta2 *= lt*lt;
    }
    if (mass<0) theta2 *= 4; // q=2 particle
    if(theta2>TMath::Pi()*TMath::Pi()) return kFALSE;
    cC22 = theta2*((1.-fP2)*(1.+fP2))*(1. + fP3*fP3);
    cC33 = theta2*(1. + fP3*fP3)*(1. + fP3*fP3);
    cC43 = theta2*fP3*fP4*(1. + fP3*fP3);
    cC44 = theta2*fP3*fP4*fP3*fP4;
  }

  //Calculating the energy loss corrections************************
  Double_t cP4=1.;
  if ((xTimesRho != 0.) && (beta2 < 1.)) {
     Double_t dE=dEdx*xTimesRho;
     Double_t e=TMath::Sqrt(p2 + mass*mass);
     if ( TMath::Abs(dE) > 0.3*e ) return kFALSE; //30% energy loss is too much!
     if ( (1.+ dE/p2*(dE + 2*e)) < 0. ) return kFALSE;
     cP4 = 1./TMath::Sqrt(1.+ dE/p2*(dE + 2*e));  //A precise formula by Ruben !
     if (TMath::Abs(fP4*cP4)>100.) return kFALSE; //Do not track below 10 MeV/c


     // Approximate energy loss fluctuation (M.Ivanov)
     const Double_t knst=0.07; // To be tuned.  
     Double_t sigmadE=knst*TMath::Sqrt(TMath::Abs(dE)); 
     cC44 += ((sigmadE*e/p2*fP4)*(sigmadE*e/p2*fP4)); 
 
  }

  //Applying the corrections*****************************
  fC22 += cC22;
  fC33 += cC33;
  fC43 += cC43;
  fC44 += cC44;
  fP4  *= cP4;

  CheckCovariance();

  return kTRUE;
}

Bool_t AliExternalTrackParam::CorrectForMeanMaterial
(Double_t xOverX0,  Double_t xTimesRho, Double_t mass, 
 Bool_t anglecorr,
 Double_t (*Bethe)(Double_t)) {
  //------------------------------------------------------------------
  // This function corrects the track parameters for the crossed material.
  // "xOverX0"   - X/X0, the thickness in units of the radiation length.
  // "xTimesRho" - is the product length*density (g/cm^2). 
  //     It should be passed as negative when propagating tracks 
  //     from the intreaction point to the outside of the central barrel. 
  // "mass" - the mass of this particle (GeV/c^2). mass<0 means charge=2
  // "anglecorr" - switch for the angular correction
  // "Bethe" - function calculating the energy loss (GeV/(g/cm^2)) 
  //------------------------------------------------------------------

  Double_t bg=GetP()/mass;
  if (mass<0) {
    if (mass<-990) {
      AliDebug(2,Form("Mass %f corresponds to unknown PID particle",mass));
      return kFALSE;
    }
    bg = -2*bg;
  }
  Double_t dEdx=Bethe(bg);
  if (mass<0) dEdx *= 4;

  return CorrectForMeanMaterialdEdx(xOverX0,xTimesRho,mass,dEdx,anglecorr);
}

Bool_t AliExternalTrackParam::CorrectForMeanMaterialZA
(Double_t xOverX0, Double_t xTimesRho, Double_t mass,
 Double_t zOverA,
 Double_t density,
 Double_t exEnergy,
 Double_t jp1,
 Double_t jp2,
 Bool_t anglecorr) {
  //------------------------------------------------------------------
  // This function corrects the track parameters for the crossed material
  // using the full Geant-like Bethe-Bloch formula parameterization
  // "xOverX0"   - X/X0, the thickness in units of the radiation length.
  // "xTimesRho" - is the product length*density (g/cm^2). 
  //     It should be passed as negative when propagating tracks 
  //     from the intreaction point to the outside of the central barrel. 
  // "mass" - the mass of this particle (GeV/c^2). mass<0 means charge=2 particle
  // "density"  - mean density (g/cm^3)
  // "zOverA"   - mean Z/A
  // "exEnergy" - mean exitation energy (GeV)
  // "jp1"      - density effect first junction point
  // "jp2"      - density effect second junction point
  // "anglecorr" - switch for the angular correction
  //
  //  The default values of the parameters are for silicon 
  //
  //------------------------------------------------------------------

  Double_t bg=GetP()/mass;
  if (mass<0) {
    if (mass<-990) {
      AliDebug(2,Form("Mass %f corresponds to unknown PID particle",mass));
      return kFALSE;
    }
    bg = -2*bg;
  }
  Double_t dEdx=BetheBlochGeant(bg,density,jp1,jp2,exEnergy,zOverA);

  if (mass<0) dEdx *= 4;
  return CorrectForMeanMaterialdEdx(xOverX0,xTimesRho,mass,dEdx,anglecorr);
}



Bool_t AliExternalTrackParam::CorrectForMaterial
(Double_t d,  Double_t x0, Double_t mass, Double_t (*Bethe)(Double_t)) {
  //------------------------------------------------------------------
  //                    Deprecated function !   
  //       Better use CorrectForMeanMaterial instead of it.
  //
  // This function corrects the track parameters for the crossed material
  // "d"    - the thickness (fraction of the radiation length)
  //     It should be passed as negative when propagating tracks 
  //     from the intreaction point to the outside of the central barrel. 
  // "x0"   - the radiation length (g/cm^2) 
  // "mass" - the mass of this particle (GeV/c^2)
  //------------------------------------------------------------------

  return CorrectForMeanMaterial(d,x0*d,mass,kTRUE,Bethe);

}

Double_t AliExternalTrackParam::BetheBlochAleph(Double_t bg,
         Double_t kp1,
         Double_t kp2,
         Double_t kp3,
         Double_t kp4,
         Double_t kp5) {
  //
  // This is the empirical ALEPH parameterization of the Bethe-Bloch formula.
  // It is normalized to 1 at the minimum.
  //
  // bg - beta*gamma
  // 
  // The default values for the kp* parameters are for ALICE TPC.
  // The returned value is in MIP units
  //

  Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);

  Double_t aa = TMath::Power(beta,kp4);
  Double_t bb = TMath::Power(1./bg,kp5);

  bb=TMath::Log(kp3+bb);
  
  return (kp2-aa-bb)*kp1/aa;
}

Double_t AliExternalTrackParam::BetheBlochGeant(Double_t bg,
         Double_t kp0,
         Double_t kp1,
         Double_t kp2,
         Double_t kp3,
         Double_t kp4) {
  //
  // This is the parameterization of the Bethe-Bloch formula inspired by Geant.
  //
  // bg  - beta*gamma
  // kp0 - density [g/cm^3]
  // kp1 - density effect first junction point
  // kp2 - density effect second junction point
  // kp3 - mean excitation energy [GeV]
  // kp4 - mean Z/A
  //
  // The default values for the kp* parameters are for silicon. 
  // The returned value is in [GeV/(g/cm^2)].
  // 

  const Double_t mK  = 0.307075e-3; // [GeV*cm^2/g]
  const Double_t me  = 0.511e-3;    // [GeV/c^2]
  const Double_t rho = kp0;
  const Double_t x0  = kp1*2.303;
  const Double_t x1  = kp2*2.303;
  const Double_t mI  = kp3;
  const Double_t mZA = kp4;
  const Double_t bg2 = bg*bg;
  const Double_t maxT= 2*me*bg2;    // neglecting the electron mass
  
  //*** Density effect
  Double_t d2=0.; 
  const Double_t x=TMath::Log(bg);
  const Double_t lhwI=TMath::Log(28.816*1e-9*TMath::Sqrt(rho*mZA)/mI);
  if (x > x1) {
    d2 = lhwI + x - 0.5;
  } else if (x > x0) {
    const Double_t r=(x1-x)/(x1-x0);
    d2 = lhwI + x - 0.5 + (0.5 - lhwI - x0)*r*r*r;
  }

  return mK*mZA*(1+bg2)/bg2*
         (0.5*TMath::Log(2*me*bg2*maxT/(mI*mI)) - bg2/(1+bg2) - d2);
}

Double_t AliExternalTrackParam::BetheBlochSolid(Double_t bg) {
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula, 
  // reasonable for solid materials. 
  // All the parameters are, in fact, for Si.
  // The returned value is in [GeV/(g/cm^2)]
  //------------------------------------------------------------------

  return BetheBlochGeant(bg);
}

Double_t AliExternalTrackParam::BetheBlochGas(Double_t bg) {
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula, 
  // reasonable for gas materials.
  // All the parameters are, in fact, for Ne.
  // The returned value is in [GeV/(g/cm^2)]
  //------------------------------------------------------------------

  const Double_t rho = 0.9e-3;
  const Double_t x0  = 2.;
  const Double_t x1  = 4.;
  const Double_t mI  = 140.e-9;
  const Double_t mZA = 0.49555;

  return BetheBlochGeant(bg,rho,x0,x1,mI,mZA);
}

Bool_t AliExternalTrackParam::Rotate(Double_t alpha) {
  //------------------------------------------------------------------
  // Transform this track to the local coord. system rotated
  // by angle "alpha" (rad) with respect to the global coord. system. 
  //------------------------------------------------------------------
  if (TMath::Abs(fP[2]) >= kAlmost1) {
     AliError(Form("Precondition is not satisfied: |sin(phi)|>1 ! %f",fP[2])); 
     return kFALSE;
  }
 
  if      (alpha < -TMath::Pi()) alpha += 2*TMath::Pi();
  else if (alpha >= TMath::Pi()) alpha -= 2*TMath::Pi();

  Double_t &fP0=fP[0];
  Double_t &fP2=fP[2];
  Double_t &fC00=fC[0];
  Double_t &fC10=fC[1];
  Double_t &fC20=fC[3];
  Double_t &fC21=fC[4];
  Double_t &fC22=fC[5];
  Double_t &fC30=fC[6];
  Double_t &fC32=fC[8];
  Double_t &fC40=fC[10];
  Double_t &fC42=fC[12];

  Double_t x=fX;
  Double_t ca=TMath::Cos(alpha-fAlpha), sa=TMath::Sin(alpha-fAlpha);
  Double_t sf=fP2, cf=TMath::Sqrt((1.- fP2)*(1.+fP2)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle
  // direction in local frame is along the X axis
  if ((cf*ca+sf*sa)<0) {
    AliDebug(1,Form("Rotation failed: local cos(phi) would become %.2f",cf*ca+sf*sa));
    return kFALSE;
  }
  //
  Double_t tmp=sf*ca - cf*sa;

  if (TMath::Abs(tmp) >= kAlmost1) {
     if (TMath::Abs(tmp) > 1.+ Double_t(FLT_EPSILON))  
        AliWarning(Form("Rotation failed ! %.10e",tmp));
     return kFALSE;
  }
  fAlpha = alpha;
  fX =  x*ca + fP0*sa;
  fP0= -x*sa + fP0*ca;
  fP2=  tmp;

  if (TMath::Abs(cf)<kAlmost0) {
    AliError(Form("Too small cosine value %f",cf)); 
    cf = kAlmost0;
  } 

  Double_t rr=(ca+sf/cf*sa);  

  fC00 *= (ca*ca);
  fC10 *= ca;
  fC20 *= ca*rr;
  fC21 *= rr;
  fC22 *= rr*rr;
  fC30 *= ca;
  fC32 *= rr;
  fC40 *= ca;
  fC42 *= rr;

  CheckCovariance();

  return kTRUE;
}

//______________________________________________________
Bool_t AliExternalTrackParam::RotateParamOnly(Double_t alpha)
{
  // rotate to new frame, ignore covariance
  if (TMath::Abs(fP[2]) >= kAlmost1) {
    AliError(Form("Precondition is not satisfied: |sin(phi)|>1 ! %f",fP[2])); 
    return kFALSE;
  }
  //
  if      (alpha < -TMath::Pi()) alpha += 2*TMath::Pi();
  else if (alpha >= TMath::Pi()) alpha -= 2*TMath::Pi();
  //
  Double_t &fP0=fP[0];
  Double_t &fP2=fP[2];
  //
  Double_t x=fX;
  Double_t ca=TMath::Cos(alpha-fAlpha), sa=TMath::Sin(alpha-fAlpha);
  Double_t sf=fP2, cf=TMath::Sqrt((1.- fP2)*(1.+fP2)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle
  // direction in local frame is along the X axis
  if ((cf*ca+sf*sa)<0) {
    AliDebug(1,Form("Rotation failed: local cos(phi) would become %.2f",cf*ca+sf*sa));
    return kFALSE;
  }
  //
  Double_t tmp=sf*ca - cf*sa;

  if (TMath::Abs(tmp) >= kAlmost1) {
     if (TMath::Abs(tmp) > 1.+ Double_t(FLT_EPSILON))  
        AliWarning(Form("Rotation failed ! %.10e",tmp));
     return kFALSE;
  }
  fAlpha = alpha;
  fX =  x*ca + fP0*sa;
  fP0= -x*sa + fP0*ca;
  fP2=  tmp;
  return kTRUE;
}

Bool_t AliExternalTrackParam::Invert() {
  //------------------------------------------------------------------
  // Transform this track to the local coord. system rotated by 180 deg. 
  //------------------------------------------------------------------
  fX = -fX;
  fAlpha += TMath::Pi();
  while (fAlpha < -TMath::Pi()) fAlpha += 2*TMath::Pi();
  while (fAlpha >= TMath::Pi()) fAlpha -= 2*TMath::Pi();
  //
  fP[0] = -fP[0];
  //fP[2] = -fP[2];
  fP[3] = -fP[3];
  fP[4] = -fP[4];
  //
  fC[1] = -fC[1]; // since the fP1 and fP2 are not inverted, their covariances with others change sign
  fC[3] = -fC[3];
  fC[7] = -fC[7];
  fC[8] = -fC[8]; 
  fC[11] = -fC[11]; 
  fC[12] = -fC[12]; 
  //
  return kTRUE;
}

Bool_t AliExternalTrackParam::PropagateTo(Double_t xk, Double_t b) {
  //----------------------------------------------------------------
  // Propagate this track to the plane X=xk (cm) in the field "b" (kG)
  //----------------------------------------------------------------
  Double_t dx=xk-fX;
  if (TMath::Abs(dx)<=kAlmost0)  return kTRUE;

  Double_t crv=GetC(b);
  if (TMath::Abs(b) < kAlmost0Field) crv=0.;

  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + x2r;
  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  if (TMath::Abs(fP[4])< kAlmost0) return kFALSE;

  Double_t &fP0=fP[0], &fP1=fP[1], &fP2=fP[2], &fP3=fP[3], &fP4=fP[4];
  Double_t 
  &fC00=fC[0],
  &fC10=fC[1],   &fC11=fC[2],  
  &fC20=fC[3],   &fC21=fC[4],   &fC22=fC[5],
  &fC30=fC[6],   &fC31=fC[7],   &fC32=fC[8],   &fC33=fC[9],  
  &fC40=fC[10],  &fC41=fC[11],  &fC42=fC[12],  &fC43=fC[13], &fC44=fC[14];

  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  if (TMath::Abs(r1)<kAlmost0)  return kFALSE;
  if (TMath::Abs(r2)<kAlmost0)  return kFALSE;

  fX=xk;
  double dy2dx = (f1+f2)/(r1+r2);
  fP0 += dx*dy2dx;
  fP2 += x2r;
  if (TMath::Abs(x2r)<0.05) fP1 += dx*(r2 + f2*dy2dx)*fP3;  // Many thanks to P.Hristov !
  else { 
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    fP1 += rot/crv*fP3;
    // 
    double rot = TMath::ASin(r1*f2 - r2*f1); // more economic version from Yura.
    if (f1*f1+f2*f2>1 && f1*f2<0) {          // special cases of large rotations or large abs angles
      if (f2>0) rot =  TMath::Pi() - rot;    //
      else      rot = -TMath::Pi() - rot;
    }
    fP1 += fP3/crv*rot; 
  }

  //f = F - 1
  /*
  Double_t f02=    dx/(r1*r1*r1);            Double_t cc=crv/fP4;
  Double_t f04=0.5*dx*dx/(r1*r1*r1);         f04*=cc;
  Double_t f12=    dx*fP3*f1/(r1*r1*r1);
  Double_t f14=0.5*dx*dx*fP3*f1/(r1*r1*r1);  f14*=cc;
  Double_t f13=    dx/r1;
  Double_t f24=    dx;                       f24*=cc;
  */
  Double_t rinv = 1./r1;
  Double_t r3inv = rinv*rinv*rinv;
  Double_t f24=    x2r/fP4;
  Double_t f02=    dx*r3inv;
  Double_t f04=0.5*f24*f02;
  Double_t f12=    f02*fP3*f1;
  Double_t f14=0.5*f24*f02*fP3*f1;
  Double_t f13=    dx*rinv;

  //b = C*ft
  Double_t b00=f02*fC20 + f04*fC40, b01=f12*fC20 + f14*fC40 + f13*fC30;
  Double_t b02=f24*fC40;
  Double_t b10=f02*fC21 + f04*fC41, b11=f12*fC21 + f14*fC41 + f13*fC31;
  Double_t b12=f24*fC41;
  Double_t b20=f02*fC22 + f04*fC42, b21=f12*fC22 + f14*fC42 + f13*fC32;
  Double_t b22=f24*fC42;
  Double_t b40=f02*fC42 + f04*fC44, b41=f12*fC42 + f14*fC44 + f13*fC43;
  Double_t b42=f24*fC44;
  Double_t b30=f02*fC32 + f04*fC43, b31=f12*fC32 + f14*fC43 + f13*fC33;
  Double_t b32=f24*fC43;
  
  //a = f*b = f*C*ft
  Double_t a00=f02*b20+f04*b40,a01=f02*b21+f04*b41,a02=f02*b22+f04*b42;
  Double_t a11=f12*b21+f14*b41+f13*b31,a12=f12*b22+f14*b42+f13*b32;
  Double_t a22=f24*b42;

  //F*C*Ft = C + (b + bt + a)
  fC00 += b00 + b00 + a00;
  fC10 += b10 + b01 + a01; 
  fC20 += b20 + b02 + a02;
  fC30 += b30;
  fC40 += b40;
  fC11 += b11 + b11 + a11;
  fC21 += b21 + b12 + a12;
  fC31 += b31; 
  fC41 += b41;
  fC22 += b22 + b22 + a22;
  fC32 += b32;
  fC42 += b42;

  CheckCovariance();

  return kTRUE;
}

Bool_t AliExternalTrackParam::PropagateParamOnlyTo(Double_t xk, Double_t b) {
  //----------------------------------------------------------------
  // Propagate this track to the plane X=xk (cm) in the field "b" (kG)
  // Only parameters are propagated, not the matrix. To be used for small 
  // distances only (<mm, i.e. misalignment)
  //----------------------------------------------------------------
  Double_t dx=xk-fX;
  if (TMath::Abs(dx)<=kAlmost0)  return kTRUE;

  Double_t crv=GetC(b);
  if (TMath::Abs(b) < kAlmost0Field) crv=0.;

  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + x2r;
  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  if (TMath::Abs(fP[4])< kAlmost0) return kFALSE;

  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  if (TMath::Abs(r1)<kAlmost0)  return kFALSE;
  if (TMath::Abs(r2)<kAlmost0)  return kFALSE;

  fX=xk;
  double dy2dx = (f1+f2)/(r1+r2);
  fP[0] += dx*dy2dx;
  fP[2] += x2r;
  if (TMath::Abs(x2r)<0.05) fP[1] += dx*(r2 + f2*dy2dx)*fP[3];  // Many thanks to P.Hristov !
  else { 
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    fP1 += rot/crv*fP3;
    // 
    double rot = TMath::ASin(r1*f2 - r2*f1); // more economic version from Yura.
    if (f1*f1+f2*f2>1 && f1*f2<0) {          // special cases of large rotations or large abs angles
      if (f2>0) rot =  TMath::Pi() - rot;    //
      else      rot = -TMath::Pi() - rot;
    }
    fP[1] += fP[3]/crv*rot; 
  }
  return kTRUE;
}

Bool_t 
AliExternalTrackParam::Propagate(Double_t alpha, Double_t x, Double_t b) {
  //------------------------------------------------------------------
  // Transform this track to the local coord. system rotated
  // by angle "alpha" (rad) with respect to the global coord. system, 
  // and propagate this track to the plane X=xk (cm) in the field "b" (kG)
  //------------------------------------------------------------------
  
  //Save the parameters
  Double_t as=fAlpha;
  Double_t xs=fX;
  Double_t ps[5], cs[15];
  for (Int_t i=0; i<5;  i++) ps[i]=fP[i]; 
  for (Int_t i=0; i<15; i++) cs[i]=fC[i]; 

  if (Rotate(alpha))
     if (PropagateTo(x,b)) return kTRUE;

  //Restore the parameters, if the operation failed
  fAlpha=as;
  fX=xs;
  for (Int_t i=0; i<5;  i++) fP[i]=ps[i]; 
  for (Int_t i=0; i<15; i++) fC[i]=cs[i]; 
  return kFALSE;
}

Bool_t AliExternalTrackParam::PropagateBxByBz
(Double_t alpha, Double_t x, Double_t b[3]) {
  //------------------------------------------------------------------
  // Transform this track to the local coord. system rotated
  // by angle "alpha" (rad) with respect to the global coord. system, 
  // and propagate this track to the plane X=xk (cm),
  // taking into account all three components of the B field, "b[3]" (kG)
  //------------------------------------------------------------------
  
  //Save the parameters
  Double_t as=fAlpha;
  Double_t xs=fX;
  Double_t ps[5], cs[15];
  for (Int_t i=0; i<5;  i++) ps[i]=fP[i]; 
  for (Int_t i=0; i<15; i++) cs[i]=fC[i]; 

  if (Rotate(alpha))
     if (PropagateToBxByBz(x,b)) return kTRUE;

  //Restore the parameters, if the operation failed
  fAlpha=as;
  fX=xs;
  for (Int_t i=0; i<5;  i++) fP[i]=ps[i]; 
  for (Int_t i=0; i<15; i++) fC[i]=cs[i]; 
  return kFALSE;
}


void AliExternalTrackParam::Propagate(Double_t len, Double_t x[3],
Double_t p[3], Double_t bz) const {
  //+++++++++++++++++++++++++++++++++++++++++    
  // Origin: K. Shileev (Kirill.Shileev@cern.ch)
  // Extrapolate track along simple helix in magnetic field
  // Arguments: len -distance alogn helix, [cm]
  //            bz  - mag field, [kGaus]   
  // Returns: x and p contain extrapolated positon and momentum  
  // The momentum returned for straight-line tracks is meaningless !
  //+++++++++++++++++++++++++++++++++++++++++    
  GetXYZ(x);
    
  if (OneOverPt() < kAlmost0 || TMath::Abs(bz) < kAlmost0Field || GetC(bz) < kAlmost0){ //straight-line tracks
     Double_t unit[3]; GetDirection(unit);
     x[0]+=unit[0]*len;   
     x[1]+=unit[1]*len;   
     x[2]+=unit[2]*len;

     p[0]=unit[0]/kAlmost0;   
     p[1]=unit[1]/kAlmost0;   
     p[2]=unit[2]/kAlmost0;   
  } else {
     GetPxPyPz(p);
     Double_t pp=GetP();
     Double_t a = -kB2C*bz*GetSign();
     Double_t rho = a/pp;
     x[0] += p[0]*TMath::Sin(rho*len)/a - p[1]*(1-TMath::Cos(rho*len))/a;
     x[1] += p[1]*TMath::Sin(rho*len)/a + p[0]*(1-TMath::Cos(rho*len))/a;
     x[2] += p[2]*len/pp;

     Double_t p0=p[0];
     p[0] = p0  *TMath::Cos(rho*len) - p[1]*TMath::Sin(rho*len);
     p[1] = p[1]*TMath::Cos(rho*len) + p0  *TMath::Sin(rho*len);
  }
}

Bool_t AliExternalTrackParam::Intersect(Double_t pnt[3], Double_t norm[3],
Double_t bz) const {
  //+++++++++++++++++++++++++++++++++++++++++    
  // Origin: K. Shileev (Kirill.Shileev@cern.ch)
  // Finds point of intersection (if exists) of the helix with the plane. 
  // Stores result in fX and fP.   
  // Arguments: planePoint,planeNorm - the plane defined by any plane's point 
  // and vector, normal to the plane
  // Returns: kTrue if helix intersects the plane, kFALSE otherwise.
  //+++++++++++++++++++++++++++++++++++++++++    
  Double_t x0[3]; GetXYZ(x0); //get track position in MARS
  
  //estimates initial helix length up to plane
  Double_t s=
    (pnt[0]-x0[0])*norm[0] + (pnt[1]-x0[1])*norm[1] + (pnt[2]-x0[2])*norm[2];
  Double_t dist=99999,distPrev=dist;
  Double_t x[3],p[3]; 
  while(TMath::Abs(dist)>0.00001){
    //calculates helix at the distance s from x0 ALONG the helix
    Propagate(s,x,p,bz);

    //distance between current helix position and plane
    dist=(x[0]-pnt[0])*norm[0]+(x[1]-pnt[1])*norm[1]+(x[2]-pnt[2])*norm[2];

    if(TMath::Abs(dist) >= TMath::Abs(distPrev)) {return kFALSE;}
    distPrev=dist;
    s-=dist;
  }
  //on exit pnt is intersection point,norm is track vector at that point, 
  //all in MARS
  for (Int_t i=0; i<3; i++) {pnt[i]=x[i]; norm[i]=p[i];}
  return kTRUE;
}

Double_t 
AliExternalTrackParam::GetPredictedChi2(const Double_t p[2],const Double_t cov[3]) const {
  //----------------------------------------------------------------
  // Estimate the chi2 of the space point "p" with the cov. matrix "cov"
  //----------------------------------------------------------------
  Double_t sdd = fC[0] + cov[0]; 
  Double_t sdz = fC[1] + cov[1];
  Double_t szz = fC[2] + cov[2];
  Double_t det = sdd*szz - sdz*sdz;

  if (TMath::Abs(det) < kAlmost0) return kVeryBig;

  Double_t d = fP[0] - p[0];
  Double_t z = fP[1] - p[1];

  return (d*szz*d - 2*d*sdz*z + z*sdd*z)/det;
}

Double_t AliExternalTrackParam::
GetPredictedChi2(const Double_t p[3],const Double_t covyz[3],const Double_t covxyz[3]) const {
  //----------------------------------------------------------------
  // Estimate the chi2 of the 3D space point "p" and
  // the full covariance matrix "covyz" and "covxyz"
  //
  // Cov(x,x) ... :   covxyz[0]
  // Cov(y,x) ... :   covxyz[1]  covyz[0]
  // Cov(z,x) ... :   covxyz[2]  covyz[1]  covyz[2]
  //----------------------------------------------------------------

  Double_t res[3] = {
    GetX() - p[0],
    GetY() - p[1],
    GetZ() - p[2]
  };

  Double_t f=GetSnp();
  if (TMath::Abs(f) >= kAlmost1) return kVeryBig;
  Double_t r=TMath::Sqrt((1.-f)*(1.+f));
  Double_t a=f/r, b=GetTgl()/r;

  Double_t s2=333.*333.;  //something reasonably big (cm^2)
 
  TMatrixDSym v(3);
  v(0,0)=  s2;  v(0,1)=  a*s2;                 v(0,2)=  b*s2;;
  v(1,0)=a*s2;  v(1,1)=a*a*s2 + GetSigmaY2();  v(1,2)=a*b*s2 + GetSigmaZY();
  v(2,0)=b*s2;  v(2,1)=a*b*s2 + GetSigmaZY();  v(2,2)=b*b*s2 + GetSigmaZ2();

  v(0,0)+=covxyz[0]; v(0,1)+=covxyz[1]; v(0,2)+=covxyz[2];
  v(1,0)+=covxyz[1]; v(1,1)+=covyz[0];  v(1,2)+=covyz[1];
  v(2,0)+=covxyz[2]; v(2,1)+=covyz[1];  v(2,2)+=covyz[2];

  v.Invert();
  if (!v.IsValid()) return kVeryBig;

  Double_t chi2=0.;
  for (Int_t i = 0; i < 3; i++)
    for (Int_t j = 0; j < 3; j++) chi2 += res[i]*res[j]*v(i,j);

  return chi2;  
}

Double_t AliExternalTrackParam::
GetPredictedChi2(const AliExternalTrackParam *t) const {
  //----------------------------------------------------------------
  // Estimate the chi2 (5 dof) of this track with respect to the track
  // given by the argument.
  // The two tracks must be in the same reference system 
  // and estimated at the same reference plane.
  //----------------------------------------------------------------

  if (TMath::Abs(t->GetAlpha()-GetAlpha()) > FLT_EPSILON) {
      AliError("The reference systems of the tracks differ !");
      return kVeryBig;
  }
  if (TMath::Abs(t->GetX()-GetX()) > FLT_EPSILON) {
      AliError("The reference of the tracks planes differ !");
      return kVeryBig;
  }

  TMatrixDSym c(5);
    c(0,0)=GetSigmaY2(); 
    c(1,0)=GetSigmaZY();   c(1,1)=GetSigmaZ2();
    c(2,0)=GetSigmaSnpY(); c(2,1)=GetSigmaSnpZ(); c(2,2)=GetSigmaSnp2();
    c(3,0)=GetSigmaTglY(); c(3,1)=GetSigmaTglZ(); c(3,2)=GetSigmaTglSnp(); c(3,3)=GetSigmaTgl2();
    c(4,0)=GetSigma1PtY(); c(4,1)=GetSigma1PtZ(); c(4,2)=GetSigma1PtSnp(); c(4,3)=GetSigma1PtTgl(); c(4,4)=GetSigma1Pt2();

    c(0,0)+=t->GetSigmaY2(); 
    c(1,0)+=t->GetSigmaZY();  c(1,1)+=t->GetSigmaZ2();
    c(2,0)+=t->GetSigmaSnpY();c(2,1)+=t->GetSigmaSnpZ();c(2,2)+=t->GetSigmaSnp2();
    c(3,0)+=t->GetSigmaTglY();c(3,1)+=t->GetSigmaTglZ();c(3,2)+=t->GetSigmaTglSnp();c(3,3)+=t->GetSigmaTgl2();
    c(4,0)+=t->GetSigma1PtY();c(4,1)+=t->GetSigma1PtZ();c(4,2)+=t->GetSigma1PtSnp();c(4,3)+=t->GetSigma1PtTgl();c(4,4)+=t->GetSigma1Pt2();
    c(0,1)=c(1,0);
    c(0,2)=c(2,0); c(1,2)=c(2,1);
    c(0,3)=c(3,0); c(1,3)=c(3,1); c(2,3)=c(3,2);
    c(0,4)=c(4,0); c(1,4)=c(4,1); c(2,4)=c(4,2); c(3,4)=c(4,3);

  c.Invert();
  if (!c.IsValid()) return kVeryBig;


  Double_t res[5] = {
    GetY()   - t->GetY(),
    GetZ()   - t->GetZ(),
    GetSnp() - t->GetSnp(),
    GetTgl() - t->GetTgl(),
    GetSigned1Pt() - t->GetSigned1Pt()
  };

  Double_t chi2=0.;
  for (Int_t i = 0; i < 5; i++)
    for (Int_t j = 0; j < 5; j++) chi2 += res[i]*res[j]*c(i,j);

  return chi2;  
}

Bool_t AliExternalTrackParam::
PropagateTo(Double_t p[3],Double_t covyz[3],Double_t covxyz[3],Double_t bz) {
  //----------------------------------------------------------------
  // Propagate this track to the plane 
  // the 3D space point "p" (with the covariance matrix "covyz" and "covxyz")
  // belongs to.
  // The magnetic field is "bz" (kG)
  //
  // The track curvature and the change of the covariance matrix
  // of the track parameters are negleted !
  // (So the "step" should be small compared with 1/curvature)
  //----------------------------------------------------------------

  Double_t f=GetSnp();
  if (TMath::Abs(f) >= kAlmost1) return kFALSE;
  Double_t r=TMath::Sqrt((1.-f)*(1.+f));
  Double_t a=f/r, b=GetTgl()/r;

  Double_t s2=333.*333.;  //something reasonably big (cm^2)
 
  TMatrixDSym tV(3);
  tV(0,0)=  s2;  tV(0,1)=  a*s2;  tV(0,2)=  b*s2;
  tV(1,0)=a*s2;  tV(1,1)=a*a*s2;  tV(1,2)=a*b*s2;
  tV(2,0)=b*s2;  tV(2,1)=a*b*s2;  tV(2,2)=b*b*s2;

  TMatrixDSym pV(3);
  pV(0,0)=covxyz[0]; pV(0,1)=covxyz[1]; pV(0,2)=covxyz[2];
  pV(1,0)=covxyz[1]; pV(1,1)=covyz[0];  pV(1,2)=covyz[1];
  pV(2,0)=covxyz[2]; pV(2,1)=covyz[1];  pV(2,2)=covyz[2];

  TMatrixDSym tpV(tV);
  tpV+=pV;
  tpV.Invert();
  if (!tpV.IsValid()) return kFALSE;

  TMatrixDSym pW(3),tW(3);
  for (Int_t i=0; i<3; i++)
    for (Int_t j=0; j<3; j++) {
      pW(i,j)=tW(i,j)=0.;
      for (Int_t k=0; k<3; k++) {
	pW(i,j) += tV(i,k)*tpV(k,j);
	tW(i,j) += pV(i,k)*tpV(k,j);
      }
    }

  Double_t t[3] = {GetX(), GetY(), GetZ()};

  Double_t x=0.;
  for (Int_t i=0; i<3; i++) x += (tW(0,i)*t[i] + pW(0,i)*p[i]);  
  Double_t crv=GetC(bz);
  if (TMath::Abs(b) < kAlmost0Field) crv=0.;
  f += crv*(x-fX);
  if (TMath::Abs(f) >= kAlmost1) return kFALSE;
  fX=x;  

  fP[0]=0.;
  for (Int_t i=0; i<3; i++) fP[0] += (tW(1,i)*t[i] + pW(1,i)*p[i]);  
  fP[1]=0.;
  for (Int_t i=0; i<3; i++) fP[1] += (tW(2,i)*t[i] + pW(2,i)*p[i]);  

  return kTRUE;  
}

Double_t *AliExternalTrackParam::GetResiduals(
Double_t *p,Double_t *cov,Bool_t updated) const {
  //------------------------------------------------------------------
  // Returns the track residuals with the space point "p" having
  // the covariance matrix "cov".
  // If "updated" is kTRUE, the track parameters expected to be updated,
  // otherwise they must be predicted.  
  //------------------------------------------------------------------
  static Double_t res[2];

  Double_t r00=cov[0], r01=cov[1], r11=cov[2];
  if (updated) {
     r00-=fC[0]; r01-=fC[1]; r11-=fC[2];
  } else {
     r00+=fC[0]; r01+=fC[1]; r11+=fC[2];
  }
  Double_t det=r00*r11 - r01*r01;

  if (TMath::Abs(det) < kAlmost0) return 0;

  Double_t tmp=r00; r00=r11/det; r11=tmp/det;

  if (r00 < 0.) return 0;
  if (r11 < 0.) return 0;

  Double_t dy = fP[0] - p[0];
  Double_t dz = fP[1] - p[1];

  res[0]=dy*TMath::Sqrt(r00);
  res[1]=dz*TMath::Sqrt(r11);

  return res;
}

Bool_t AliExternalTrackParam::Update(const Double_t p[2], const Double_t cov[3]) {
  //------------------------------------------------------------------
  // Update the track parameters with the space point "p" having
  // the covariance matrix "cov"
  //------------------------------------------------------------------
  Double_t &fP0=fP[0], &fP1=fP[1], &fP2=fP[2], &fP3=fP[3], &fP4=fP[4];
  Double_t 
  &fC00=fC[0],
  &fC10=fC[1],   &fC11=fC[2],  
  &fC20=fC[3],   &fC21=fC[4],   &fC22=fC[5],
  &fC30=fC[6],   &fC31=fC[7],   &fC32=fC[8],   &fC33=fC[9],  
  &fC40=fC[10],  &fC41=fC[11],  &fC42=fC[12],  &fC43=fC[13], &fC44=fC[14];

  Double_t r00=cov[0], r01=cov[1], r11=cov[2];
  r00+=fC00; r01+=fC10; r11+=fC11;
  Double_t det=r00*r11 - r01*r01;

  if (TMath::Abs(det) < kAlmost0) return kFALSE;


  Double_t tmp=r00; r00=r11/det; r11=tmp/det; r01=-r01/det;
 
  Double_t k00=fC00*r00+fC10*r01, k01=fC00*r01+fC10*r11;
  Double_t k10=fC10*r00+fC11*r01, k11=fC10*r01+fC11*r11;
  Double_t k20=fC20*r00+fC21*r01, k21=fC20*r01+fC21*r11;
  Double_t k30=fC30*r00+fC31*r01, k31=fC30*r01+fC31*r11;
  Double_t k40=fC40*r00+fC41*r01, k41=fC40*r01+fC41*r11;

  Double_t dy=p[0] - fP0, dz=p[1] - fP1;
  Double_t sf=fP2 + k20*dy + k21*dz;
  if (TMath::Abs(sf) > kAlmost1) return kFALSE;  
  
  fP0 += k00*dy + k01*dz;
  fP1 += k10*dy + k11*dz;
  fP2  = sf;
  fP3 += k30*dy + k31*dz;
  fP4 += k40*dy + k41*dz;
  
  Double_t c01=fC10, c02=fC20, c03=fC30, c04=fC40;
  Double_t c12=fC21, c13=fC31, c14=fC41;

  fC00-=k00*fC00+k01*fC10; fC10-=k00*c01+k01*fC11;
  fC20-=k00*c02+k01*c12;   fC30-=k00*c03+k01*c13;
  fC40-=k00*c04+k01*c14; 

  fC11-=k10*c01+k11*fC11;
  fC21-=k10*c02+k11*c12;   fC31-=k10*c03+k11*c13;
  fC41-=k10*c04+k11*c14; 

  fC22-=k20*c02+k21*c12;   fC32-=k20*c03+k21*c13;
  fC42-=k20*c04+k21*c14; 

  fC33-=k30*c03+k31*c13;
  fC43-=k30*c04+k31*c14; 
  
  fC44-=k40*c04+k41*c14; 

  CheckCovariance();

  return kTRUE;
}

void 
AliExternalTrackParam::GetHelixParameters(Double_t hlx[6], Double_t b) const {
  //--------------------------------------------------------------------
  // External track parameters -> helix parameters 
  // "b" - magnetic field (kG)
  //--------------------------------------------------------------------
  Double_t cs=TMath::Cos(fAlpha), sn=TMath::Sin(fAlpha);
  
  hlx[0]=fP[0]; hlx[1]=fP[1]; hlx[2]=fP[2]; hlx[3]=fP[3];

  hlx[5]=fX*cs - hlx[0]*sn;               // x0
  hlx[0]=fX*sn + hlx[0]*cs;               // y0
//hlx[1]=                                 // z0
  hlx[2]=TMath::ASin(hlx[2]) + fAlpha;    // phi0
//hlx[3]=                                 // tgl
  hlx[4]=GetC(b);                         // C
}


static void Evaluate(const Double_t *h, Double_t t,
                     Double_t r[3],  //radius vector
                     Double_t g[3],  //first defivatives
                     Double_t gg[3]) //second derivatives
{
  //--------------------------------------------------------------------
  // Calculate position of a point on a track and some derivatives
  //--------------------------------------------------------------------
  Double_t phase=h[4]*t+h[2];
  Double_t sn=TMath::Sin(phase), cs=TMath::Cos(phase);

  r[0] = h[5];
  r[1] = h[0];
  if (TMath::Abs(h[4])>kAlmost0) {
     r[0] += (sn - h[6])/h[4];
     r[1] -= (cs - h[7])/h[4];  
  } else {
     r[0] += t*cs;
     r[1] -= -t*sn;  
  }
  r[2] = h[1] + h[3]*t;

  g[0] = cs; g[1]=sn; g[2]=h[3];
  
  gg[0]=-h[4]*sn; gg[1]=h[4]*cs; gg[2]=0.;
}

Double_t AliExternalTrackParam::GetDCA(const AliExternalTrackParam *p, 
Double_t b, Double_t &xthis, Double_t &xp) const {
  //------------------------------------------------------------
  // Returns the (weighed !) distance of closest approach between 
  // this track and the track "p".
  // Other returned values:
  //   xthis, xt - coordinates of tracks' reference planes at the DCA 
  //-----------------------------------------------------------
  Double_t dy2=GetSigmaY2() + p->GetSigmaY2();
  Double_t dz2=GetSigmaZ2() + p->GetSigmaZ2();
  Double_t dx2=dy2; 

  Double_t p1[8]; GetHelixParameters(p1,b);
  p1[6]=TMath::Sin(p1[2]); p1[7]=TMath::Cos(p1[2]);
  Double_t p2[8]; p->GetHelixParameters(p2,b);
  p2[6]=TMath::Sin(p2[2]); p2[7]=TMath::Cos(p2[2]);


  Double_t r1[3],g1[3],gg1[3]; Double_t t1=0.;
  Evaluate(p1,t1,r1,g1,gg1);
  Double_t r2[3],g2[3],gg2[3]; Double_t t2=0.;
  Evaluate(p2,t2,r2,g2,gg2);

  Double_t dx=r2[0]-r1[0], dy=r2[1]-r1[1], dz=r2[2]-r1[2];
  Double_t dm=dx*dx/dx2 + dy*dy/dy2 + dz*dz/dz2;

  Int_t max=27;
  while (max--) {
     Double_t gt1=-(dx*g1[0]/dx2 + dy*g1[1]/dy2 + dz*g1[2]/dz2);
     Double_t gt2=+(dx*g2[0]/dx2 + dy*g2[1]/dy2 + dz*g2[2]/dz2);
     Double_t h11=(g1[0]*g1[0] - dx*gg1[0])/dx2 + 
                  (g1[1]*g1[1] - dy*gg1[1])/dy2 +
                  (g1[2]*g1[2] - dz*gg1[2])/dz2;
     Double_t h22=(g2[0]*g2[0] + dx*gg2[0])/dx2 + 
                  (g2[1]*g2[1] + dy*gg2[1])/dy2 +
                  (g2[2]*g2[2] + dz*gg2[2])/dz2;
     Double_t h12=-(g1[0]*g2[0]/dx2 + g1[1]*g2[1]/dy2 + g1[2]*g2[2]/dz2);

     Double_t det=h11*h22-h12*h12;

     Double_t dt1,dt2;
     if (TMath::Abs(det)<1.e-33) {
        //(quasi)singular Hessian
        dt1=-gt1; dt2=-gt2;
     } else {
        dt1=-(gt1*h22 - gt2*h12)/det; 
        dt2=-(h11*gt2 - h12*gt1)/det;
     }

     if ((dt1*gt1+dt2*gt2)>0) {dt1=-dt1; dt2=-dt2;}

     //check delta(phase1) ?
     //check delta(phase2) ?

     if (TMath::Abs(dt1)/(TMath::Abs(t1)+1.e-3) < 1.e-4)
     if (TMath::Abs(dt2)/(TMath::Abs(t2)+1.e-3) < 1.e-4) {
        if ((gt1*gt1+gt2*gt2) > 1.e-4/dy2/dy2) 
	  AliDebug(1," stopped at not a stationary point !");
        Double_t lmb=h11+h22; lmb=lmb-TMath::Sqrt(lmb*lmb-4*det);
        if (lmb < 0.) 
	  AliDebug(1," stopped at not a minimum !");
        break;
     }

     Double_t dd=dm;
     for (Int_t div=1 ; ; div*=2) {
        Evaluate(p1,t1+dt1,r1,g1,gg1);
        Evaluate(p2,t2+dt2,r2,g2,gg2);
        dx=r2[0]-r1[0]; dy=r2[1]-r1[1]; dz=r2[2]-r1[2];
        dd=dx*dx/dx2 + dy*dy/dy2 + dz*dz/dz2;
	if (dd<dm) break;
        dt1*=0.5; dt2*=0.5;
        if (div>512) {
	  AliDebug(1," overshoot !"); break;
        }   
     }
     dm=dd;

     t1+=dt1;
     t2+=dt2;

  }

  if (max<=0) AliDebug(1," too many iterations !");

  Double_t cs=TMath::Cos(GetAlpha());
  Double_t sn=TMath::Sin(GetAlpha());
  xthis=r1[0]*cs + r1[1]*sn;

  cs=TMath::Cos(p->GetAlpha());
  sn=TMath::Sin(p->GetAlpha());
  xp=r2[0]*cs + r2[1]*sn;

  return TMath::Sqrt(dm*TMath::Sqrt(dy2*dz2));
}
 
Double_t AliExternalTrackParam::
PropagateToDCA(AliExternalTrackParam *p, Double_t b) {
  //--------------------------------------------------------------
  // Propagates this track and the argument track to the position of the
  // distance of closest approach.
  // Returns the (weighed !) distance of closest approach.
  //--------------------------------------------------------------
  Double_t xthis,xp;
  Double_t dca=GetDCA(p,b,xthis,xp);

  if (!PropagateTo(xthis,b)) {
    //AliWarning(" propagation failed !");
    return 1e+33;
  }

  if (!p->PropagateTo(xp,b)) {
    //AliWarning(" propagation failed !";
    return 1e+33;
  }

  return dca;
}


Bool_t AliExternalTrackParam::PropagateToDCA(const AliVVertex *vtx, 
Double_t b, Double_t maxd, Double_t dz[2], Double_t covar[3]) {
  //
  // Propagate this track to the DCA to vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //            Magnetic field is "b" (kG).
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  //
  //    In the case of success, the returned value is kTRUE
  //    (otherwise, it's kFALSE)
  //  
  Double_t alpha=GetAlpha();
  Double_t sn=TMath::Sin(alpha), cs=TMath::Cos(alpha);
  Double_t x=GetX(), y=GetParameter()[0], snp=GetParameter()[2];
  Double_t xv= vtx->GetX()*cs + vtx->GetY()*sn;
  Double_t yv=-vtx->GetX()*sn + vtx->GetY()*cs, zv=vtx->GetZ();
  x-=xv; y-=yv;

  //Estimate the impact parameter neglecting the track curvature
  Double_t d=TMath::Abs(x*snp - y*TMath::Sqrt((1.-snp)*(1.+snp)));
  if (d > maxd) return kFALSE; 

  //Propagate to the DCA
  Double_t crv=GetC(b);
  if (TMath::Abs(b) < kAlmost0Field) crv=0.;

  Double_t tgfv=-(crv*x - snp)/(crv*y + TMath::Sqrt((1.-snp)*(1.+snp)));
  sn=tgfv/TMath::Sqrt(1.+ tgfv*tgfv); cs=TMath::Sqrt((1.-sn)*(1.+sn));
  if (TMath::Abs(tgfv)>0.) cs = sn/tgfv;
  else cs=1.;

  x = xv*cs + yv*sn;
  yv=-xv*sn + yv*cs; xv=x;

  if (!Propagate(alpha+TMath::ASin(sn),xv,b)) return kFALSE;

  if (dz==0) return kTRUE;
  dz[0] = GetParameter()[0] - yv;
  dz[1] = GetParameter()[1] - zv;
  
  if (covar==0) return kTRUE;
  Double_t cov[6]; vtx->GetCovarianceMatrix(cov);

  //***** Improvements by A.Dainese
  alpha=GetAlpha(); sn=TMath::Sin(alpha); cs=TMath::Cos(alpha);
  Double_t s2ylocvtx = cov[0]*sn*sn + cov[2]*cs*cs - 2.*cov[1]*cs*sn;
  covar[0] = GetCovariance()[0] + s2ylocvtx;   // neglecting correlations
  covar[1] = GetCovariance()[1];               // between (x,y) and z
  covar[2] = GetCovariance()[2] + cov[5];      // in vertex's covariance matrix
  //*****

  return kTRUE;
}

Bool_t AliExternalTrackParam::PropagateToDCABxByBz(const AliVVertex *vtx, 
Double_t b[3], Double_t maxd, Double_t dz[2], Double_t covar[3]) {
  //
  // Propagate this track to the DCA to vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //
  // This function takes into account all three components of the magnetic
  // field given by the b[3] arument (kG)
  //
  // a) The track gets extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  //
  //    In the case of success, the returned value is kTRUE
  //    (otherwise, it's kFALSE)
  //  
  Double_t alpha=GetAlpha();
  Double_t sn=TMath::Sin(alpha), cs=TMath::Cos(alpha);
  Double_t x=GetX(), y=GetParameter()[0], snp=GetParameter()[2];
  Double_t xv= vtx->GetX()*cs + vtx->GetY()*sn;
  Double_t yv=-vtx->GetX()*sn + vtx->GetY()*cs, zv=vtx->GetZ();
  x-=xv; y-=yv;

  //Estimate the impact parameter neglecting the track curvature
  Double_t d=TMath::Abs(x*snp - y*TMath::Sqrt((1.-snp)*(1.+snp)));
  if (d > maxd) return kFALSE; 

  //Propagate to the DCA
  Double_t crv=GetC(b[2]);
  if (TMath::Abs(b[2]) < kAlmost0Field) crv=0.;

  Double_t tgfv=-(crv*x - snp)/(crv*y + TMath::Sqrt((1.-snp)*(1.+snp)));
  sn=tgfv/TMath::Sqrt(1.+ tgfv*tgfv); cs=TMath::Sqrt((1.-sn)*(1.+sn));
  if (TMath::Abs(tgfv)>0.) cs = sn/tgfv;
  else cs=1.;

  x = xv*cs + yv*sn;
  yv=-xv*sn + yv*cs; xv=x;

  if (!PropagateBxByBz(alpha+TMath::ASin(sn),xv,b)) return kFALSE;

  if (dz==0) return kTRUE;
  dz[0] = GetParameter()[0] - yv;
  dz[1] = GetParameter()[1] - zv;
  
  if (covar==0) return kTRUE;
  Double_t cov[6]; vtx->GetCovarianceMatrix(cov);

  //***** Improvements by A.Dainese
  alpha=GetAlpha(); sn=TMath::Sin(alpha); cs=TMath::Cos(alpha);
  Double_t s2ylocvtx = cov[0]*sn*sn + cov[2]*cs*cs - 2.*cov[1]*cs*sn;
  covar[0] = GetCovariance()[0] + s2ylocvtx;   // neglecting correlations
  covar[1] = GetCovariance()[1];               // between (x,y) and z
  covar[2] = GetCovariance()[2] + cov[5];      // in vertex's covariance matrix
  //*****

  return kTRUE;
}

void AliExternalTrackParam::GetDirection(Double_t d[3]) const {
  //----------------------------------------------------------------
  // This function returns a unit vector along the track direction
  // in the global coordinate system.
  //----------------------------------------------------------------
  Double_t cs=TMath::Cos(fAlpha), sn=TMath::Sin(fAlpha);
  Double_t snp=fP[2];
  Double_t csp =TMath::Sqrt((1.-snp)*(1.+snp));
  Double_t norm=TMath::Sqrt(1.+ fP[3]*fP[3]);
  d[0]=(csp*cs - snp*sn)/norm; 
  d[1]=(snp*cs + csp*sn)/norm; 
  d[2]=fP[3]/norm;
}

Bool_t AliExternalTrackParam::GetPxPyPz(Double_t p[3]) const {
  //---------------------------------------------------------------------
  // This function returns the global track momentum components
  // Results for (nearly) straight tracks are meaningless !
  //---------------------------------------------------------------------
  p[0]=fP[4]; p[1]=fP[2]; p[2]=fP[3];
  return Local2GlobalMomentum(p,fAlpha);
}

Double_t AliExternalTrackParam::Px() const {
  //---------------------------------------------------------------------
  // Returns x-component of momentum
  // Result for (nearly) straight tracks is meaningless !
  //---------------------------------------------------------------------

  Double_t p[3]={kVeryBig,kVeryBig,kVeryBig};
  GetPxPyPz(p);

  return p[0];
}

Double_t AliExternalTrackParam::Py() const {
  //---------------------------------------------------------------------
  // Returns y-component of momentum
  // Result for (nearly) straight tracks is meaningless !
  //---------------------------------------------------------------------

  Double_t p[3]={kVeryBig,kVeryBig,kVeryBig};
  GetPxPyPz(p);

  return p[1];
}

Double_t AliExternalTrackParam::Xv() const {
  //---------------------------------------------------------------------
  // Returns x-component of first track point
  //---------------------------------------------------------------------

  Double_t r[3]={0.,0.,0.};
  GetXYZ(r);

  return r[0];
}

Double_t AliExternalTrackParam::Yv() const {
  //---------------------------------------------------------------------
  // Returns y-component of first track point
  //---------------------------------------------------------------------

  Double_t r[3]={0.,0.,0.};
  GetXYZ(r);

  return r[1];
}

Double_t AliExternalTrackParam::Theta() const {
  // return theta angle of momentum

  return 0.5*TMath::Pi() - TMath::ATan(fP[3]);
}

Double_t AliExternalTrackParam::Phi() const {
  //---------------------------------------------------------------------
  // Returns the azimuthal angle of momentum
  // 0 <= phi < 2*pi
  //---------------------------------------------------------------------

  Double_t phi=TMath::ASin(fP[2]) + fAlpha;
  if (phi<0.) phi+=2.*TMath::Pi();
  else if (phi>=2.*TMath::Pi()) phi-=2.*TMath::Pi();
 
  return phi;
}

Double_t AliExternalTrackParam::PhiPos() const {
  //---------------------------------------------------------------------
  // Returns the azimuthal angle of position
  // 0 <= phi < 2*pi
  //---------------------------------------------------------------------
  Double_t r[3]={0.,0.,0.};
  GetXYZ(r);
  Double_t phi=TMath::ATan2(r[1],r[0]);
  if (phi<0.) phi+=2.*TMath::Pi();

  return phi;
}

Double_t AliExternalTrackParam::M() const {
  // return particle mass

  // No mass information available so far.
  // Redifine in derived class!

  return -999.;
}

Double_t AliExternalTrackParam::E() const {
  // return particle energy

  // No PID information available so far.
  // Redifine in derived class!

  return -999.;
}

Double_t AliExternalTrackParam::Eta() const { 
  // return pseudorapidity

  return -TMath::Log(TMath::Tan(0.5 * Theta())); 
}

Double_t AliExternalTrackParam::Y() const {
  // return rapidity

  // No PID information available so far.
  // Redifine in derived class!

  return -999.;
}

Bool_t AliExternalTrackParam::GetXYZ(Double_t *r) const {
  //---------------------------------------------------------------------
  // This function returns the global track position
  //---------------------------------------------------------------------
  r[0]=fX; r[1]=fP[0]; r[2]=fP[1];
  return Local2GlobalPosition(r,fAlpha);
}

Bool_t AliExternalTrackParam::GetCovarianceXYZPxPyPz(Double_t cv[21]) const {
  //---------------------------------------------------------------------
  // This function returns the global covariance matrix of the track params
  // 
  // Cov(x,x) ... :   cv[0]
  // Cov(y,x) ... :   cv[1]  cv[2]
  // Cov(z,x) ... :   cv[3]  cv[4]  cv[5]
  // Cov(px,x)... :   cv[6]  cv[7]  cv[8]  cv[9]
  // Cov(py,x)... :   cv[10] cv[11] cv[12] cv[13] cv[14]
  // Cov(pz,x)... :   cv[15] cv[16] cv[17] cv[18] cv[19] cv[20]
  //
  // Results for (nearly) straight tracks are meaningless !
  //---------------------------------------------------------------------
  if (TMath::Abs(fP[4])<=kAlmost0) {
     for (Int_t i=0; i<21; i++) cv[i]=0.;
     return kFALSE;
  }
  if (TMath::Abs(fP[2]) > kAlmost1) {
     for (Int_t i=0; i<21; i++) cv[i]=0.;
     return kFALSE;
  }
  Double_t pt=1./TMath::Abs(fP[4]);
  Double_t cs=TMath::Cos(fAlpha), sn=TMath::Sin(fAlpha);
  Double_t r=TMath::Sqrt((1.-fP[2])*(1.+fP[2]));

  Double_t m00=-sn, m10=cs;
  Double_t m23=-pt*(sn + fP[2]*cs/r), m43=-pt*pt*(r*cs - fP[2]*sn);
  Double_t m24= pt*(cs - fP[2]*sn/r), m44=-pt*pt*(r*sn + fP[2]*cs);
  Double_t m35=pt, m45=-pt*pt*fP[3];

  m43*=GetSign();
  m44*=GetSign();
  m45*=GetSign();

  cv[0 ] = fC[0]*m00*m00;
  cv[1 ] = fC[0]*m00*m10; 
  cv[2 ] = fC[0]*m10*m10;
  cv[3 ] = fC[1]*m00; 
  cv[4 ] = fC[1]*m10; 
  cv[5 ] = fC[2];
  cv[6 ] = m00*(fC[3]*m23 + fC[10]*m43); 
  cv[7 ] = m10*(fC[3]*m23 + fC[10]*m43); 
  cv[8 ] = fC[4]*m23 + fC[11]*m43; 
  cv[9 ] = m23*(fC[5]*m23 + fC[12]*m43)  +  m43*(fC[12]*m23 + fC[14]*m43);
  cv[10] = m00*(fC[3]*m24 + fC[10]*m44); 
  cv[11] = m10*(fC[3]*m24 + fC[10]*m44); 
  cv[12] = fC[4]*m24 + fC[11]*m44; 
  cv[13] = m23*(fC[5]*m24 + fC[12]*m44)  +  m43*(fC[12]*m24 + fC[14]*m44);
  cv[14] = m24*(fC[5]*m24 + fC[12]*m44)  +  m44*(fC[12]*m24 + fC[14]*m44);
  cv[15] = m00*(fC[6]*m35 + fC[10]*m45); 
  cv[16] = m10*(fC[6]*m35 + fC[10]*m45); 
  cv[17] = fC[7]*m35 + fC[11]*m45; 
  cv[18] = m23*(fC[8]*m35 + fC[12]*m45)  +  m43*(fC[13]*m35 + fC[14]*m45);
  cv[19] = m24*(fC[8]*m35 + fC[12]*m45)  +  m44*(fC[13]*m35 + fC[14]*m45); 
  cv[20] = m35*(fC[9]*m35 + fC[13]*m45)  +  m45*(fC[13]*m35 + fC[14]*m45);

  return kTRUE;
}


Bool_t 
AliExternalTrackParam::GetPxPyPzAt(Double_t x, Double_t b, Double_t *p) const {
  //---------------------------------------------------------------------
  // This function returns the global track momentum extrapolated to
  // the radial position "x" (cm) in the magnetic field "b" (kG)
  //---------------------------------------------------------------------
  p[0]=fP[4]; 
  p[1]=fP[2]+(x-fX)*GetC(b); 
  p[2]=fP[3];
  return Local2GlobalMomentum(p,fAlpha);
}

Bool_t AliExternalTrackParam::GetYZAt(Double_t x, Double_t b, Double_t *yz) const 
{
  //---------------------------------------------------------------------
  // This function returns the local Y,Z-coordinates of the intersection 
  // point between this track and the reference plane "x" (cm). 
  // Magnetic field "b" (kG)
  //---------------------------------------------------------------------
  double dx=x-fX;
  if(TMath::Abs(dx)<=kAlmost0) {
    yz[0] = fP[0]; 
    yz[1] = fP[1];
    return kTRUE;
  }
  double crv=GetC(b);
  double f1=fP[2], x2r = crv*dx, f2=f1 + x2r;
  if (TMath::Abs(f1) >= kAlmost1 || 
      TMath::Abs(f2) >= kAlmost1 || 
      TMath::Abs(fP[4])< kAlmost0) return kFALSE;
  double r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  double dy2dx = (f1+f2)/(r1+r2);
  yz[0] = fP[0] + dx*dy2dx;
  if (TMath::Abs(x2r)<0.05) yz[1] = fP[1] + dx*(r2 + f2*dy2dx)*fP[3];
  else {
    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    yz[1] = fP[1] + rot/crv*fP[3];    
  }
  return kTRUE;
}


Bool_t 
AliExternalTrackParam::GetYAt(Double_t x, Double_t b, Double_t &y) const {
  //---------------------------------------------------------------------
  // This function returns the local Y-coordinate of the intersection 
  // point between this track and the reference plane "x" (cm). 
  // Magnetic field "b" (kG)
  //---------------------------------------------------------------------
  Double_t dx=x-fX;
  if(TMath::Abs(dx)<=kAlmost0) {y=fP[0]; return kTRUE;}

  Double_t f1=fP[2], f2=f1 + dx*GetC(b);

  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  
  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  y = fP[0] + dx*(f1+f2)/(r1+r2);
  return kTRUE;
}

Bool_t 
AliExternalTrackParam::GetZAt(Double_t x, Double_t b, Double_t &z) const {
  //---------------------------------------------------------------------
  // This function returns the local Z-coordinate of the intersection 
  // point between this track and the reference plane "x" (cm). 
  // Magnetic field "b" (kG)
  //---------------------------------------------------------------------
  Double_t dx=x-fX;
  if(TMath::Abs(dx)<=kAlmost0) {z=fP[1]; return kTRUE;}

  Double_t crv=GetC(b);
  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + x2r;

  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  
  Double_t r1=sqrt((1.-f1)*(1.+f1)), r2=sqrt((1.-f2)*(1.+f2));
  double dy2dx = (f1+f2)/(r1+r2);
  if (TMath::Abs(x2r)<0.05) {
    z = fP[1] + dx*(r2 + f2*dy2dx)*fP[3]; // Many thanks to P.Hristov !    
  }
  else {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    // Similarly, the rotation angle in linear in dx only for dx<<R
    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    z = fP[1] + rot/crv*fP[3];    
  }
  return kTRUE;
}

Bool_t 
AliExternalTrackParam::GetXYZAt(Double_t x, Double_t b, Double_t *r) const {
  //---------------------------------------------------------------------
  // This function returns the global track position extrapolated to
  // the radial position "x" (cm) in the magnetic field "b" (kG)
  //---------------------------------------------------------------------
  Double_t dx=x-fX;
  if(TMath::Abs(dx)<=kAlmost0) return GetXYZ(r);

  Double_t crv=GetC(b);
  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + dx*crv;

  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  
  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  double dy2dx = (f1+f2)/(r1+r2);
  r[0] = x;
  r[1] = fP[0] + dx*dy2dx;
  if (TMath::Abs(x2r)<0.05) {
    r[2] = fP[1] + dx*(r2 + f2*dy2dx)*fP[3];//Thanks to Andrea & Peter
  }
  else {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    // Similarly, the rotation angle in linear in dx only for dx<<R
    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    r[2] = fP[1] + rot/crv*fP[3];
  }

  return Local2GlobalPosition(r,fAlpha);
}

//_____________________________________________________________________________
void AliExternalTrackParam::Print(Option_t* /*option*/) const
{
// print the parameters and the covariance matrix

  printf("AliExternalTrackParam: x = %-12g  alpha = %-12g\n", fX, fAlpha);
  printf("  parameters: %12g %12g %12g %12g %12g\n",
	 fP[0], fP[1], fP[2], fP[3], fP[4]);
  printf("  covariance: %12g\n", fC[0]);
  printf("              %12g %12g\n", fC[1], fC[2]);
  printf("              %12g %12g %12g\n", fC[3], fC[4], fC[5]);
  printf("              %12g %12g %12g %12g\n", 
	 fC[6], fC[7], fC[8], fC[9]);
  printf("              %12g %12g %12g %12g %12g\n", 
	 fC[10], fC[11], fC[12], fC[13], fC[14]);
}

Double_t AliExternalTrackParam::GetSnpAt(Double_t x,Double_t b) const {
  //
  // Get sinus at given x
  //
  Double_t crv=GetC(b);
  if (TMath::Abs(b) < kAlmost0Field) crv=0.;
  Double_t dx = x-fX;
  Double_t res = fP[2]+dx*crv;
  return res;
}

Bool_t AliExternalTrackParam::GetDistance(AliExternalTrackParam *param2, Double_t x, Double_t dist[3], Double_t bz){
  //------------------------------------------------------------------------
  // Get the distance between two tracks at the local position x 
  // working in the local frame of this track.
  // Origin :   Marian.Ivanov@cern.ch
  //-----------------------------------------------------------------------
  Double_t xyz[3];
  Double_t xyz2[3];
  xyz[0]=x;
  if (!GetYAt(x,bz,xyz[1])) return kFALSE;
  if (!GetZAt(x,bz,xyz[2])) return kFALSE;
  //  
  //
  if (TMath::Abs(GetAlpha()-param2->GetAlpha())<kAlmost0){
    xyz2[0]=x;
    if (!param2->GetYAt(x,bz,xyz2[1])) return kFALSE;
    if (!param2->GetZAt(x,bz,xyz2[2])) return kFALSE;
  }else{
    //
    Double_t xyz1[3];
    Double_t dfi = param2->GetAlpha()-GetAlpha();
    Double_t ca = TMath::Cos(dfi), sa = TMath::Sin(dfi);
    xyz2[0] =  xyz[0]*ca+xyz[1]*sa;
    xyz2[1] = -xyz[0]*sa+xyz[1]*ca;
    //
    xyz1[0]=xyz2[0];
    if (!param2->GetYAt(xyz2[0],bz,xyz1[1])) return kFALSE;
    if (!param2->GetZAt(xyz2[0],bz,xyz1[2])) return kFALSE;
    //
    xyz2[0] =  xyz1[0]*ca-xyz1[1]*sa;
    xyz2[1] = +xyz1[0]*sa+xyz1[1]*ca;
    xyz2[2] = xyz1[2];
  }
  dist[0] = xyz[0]-xyz2[0];
  dist[1] = xyz[1]-xyz2[1];
  dist[2] = xyz[2]-xyz2[2];

  return kTRUE;
}


//
// Draw functionality.
// Origin: Marian Ivanov, Marian.Ivanov@cern.ch
//

void  AliExternalTrackParam::DrawTrack(Float_t magf, Float_t minR, Float_t maxR, Float_t stepR){
  //
  // Draw track line
  //
  if (minR>maxR) return ;
  if (stepR<=0) return ;
  Int_t npoints = TMath::Nint((maxR-minR)/stepR)+1;
  if (npoints<1) return;
  TPolyMarker3D *polymarker = new TPolyMarker3D(npoints);
  FillPolymarker(polymarker, magf,minR,maxR,stepR);
  polymarker->Draw();
}

//
void AliExternalTrackParam::FillPolymarker(TPolyMarker3D *pol, Float_t magF, Float_t minR, Float_t maxR, Float_t stepR){
  //
  // Fill points in the polymarker
  //
  Int_t counter=0;
  for (Double_t r=minR; r<maxR; r+=stepR){
    Double_t point[3];
    GetXYZAt(r,magF,point);
    pol->SetPoint(counter,point[0],point[1], point[2]);
    //    printf("xyz\t%f\t%f\t%f\n",point[0], point[1],point[2]);
    counter++;
  }
}

Int_t AliExternalTrackParam::GetIndex(Int_t i, Int_t j){
  //
  Int_t min = TMath::Min(i,j);
  Int_t max = TMath::Max(i,j);

  return min+(max+1)*max/2;
}


void AliExternalTrackParam::g3helx3(Double_t qfield, 
                                    Double_t step,
                                    Double_t vect[7]) {
/******************************************************************
 *                                                                *
 *       GEANT3 tracking routine in a constant field oriented     *
 *       along axis 3                                             *
 *       Tracking is performed with a conventional                *
 *       helix step method                                        *
 *                                                                *
 *       Authors    R.Brun, M.Hansroul  *********                 *
 *       Rewritten  V.Perevoztchikov                              *
 *                                                                *
 *       Rewritten in C++ by I.Belikov                            *
 *                                                                *
 *  qfield (kG)       - particle charge times magnetic field      *
 *  step   (cm)       - step length along the helix               *
 *  vect[7](cm,GeV/c) - input/output x, y, z, px/p, py/p ,pz/p, p *
 *                                                                *
 ******************************************************************/
  const Int_t ix=0, iy=1, iz=2, ipx=3, ipy=4, ipz=5, ipp=6;
  const Double_t kOvSqSix=TMath::Sqrt(1./6.);

  Double_t cosx=vect[ipx], cosy=vect[ipy], cosz=vect[ipz];

  Double_t rho = qfield*kB2C/vect[ipp]; 
  Double_t tet = rho*step;

  Double_t tsint, sintt, sint, cos1t; 
  if (TMath::Abs(tet) > 0.03) {
     sint  = TMath::Sin(tet);
     sintt = sint/tet;
     tsint = (tet - sint)/tet;
     Double_t t=TMath::Sin(0.5*tet);
     cos1t = 2*t*t/tet;
  } else {
     tsint = tet*tet/6.;
     sintt = (1.-tet*kOvSqSix)*(1.+tet*kOvSqSix); // 1.- tsint;
     sint  = tet*sintt;
     cos1t = 0.5*tet; 
  }

  Double_t f1 = step*sintt;
  Double_t f2 = step*cos1t;
  Double_t f3 = step*tsint*cosz;
  Double_t f4 = -tet*cos1t;
  Double_t f5 = sint;

  vect[ix]  += f1*cosx - f2*cosy;
  vect[iy]  += f1*cosy + f2*cosx;
  vect[iz]  += f1*cosz + f3;

  vect[ipx] += f4*cosx - f5*cosy;
  vect[ipy] += f4*cosy + f5*cosx;  

}

Bool_t AliExternalTrackParam::PropagateToBxByBz(Double_t xk, const Double_t b[3]) {
  //----------------------------------------------------------------
  // Extrapolate this track to the plane X=xk in the field b[].
  //
  // X [cm] is in the "tracking coordinate system" of this track.
  // b[]={Bx,By,Bz} [kG] is in the Global coordidate system.
  //----------------------------------------------------------------

  Double_t dx=xk-fX;
  if (TMath::Abs(dx)<=kAlmost0)  return kTRUE;
  if (TMath::Abs(fP[4])<=kAlmost0) return kFALSE;
  // Do not propagate tracks outside the ALICE detector
  if (TMath::Abs(dx)>1e5 ||
      TMath::Abs(GetY())>1e5 ||
      TMath::Abs(GetZ())>1e5) {
    AliWarning(Form("Anomalous track, target X:%f",xk));
    Print();
    return kFALSE;
  }

  Double_t crv=GetC(b[2]);
  if (TMath::Abs(b[2]) < kAlmost0Field) crv=0.;

  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + x2r;
  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;


  // Estimate the covariance matrix  
  Double_t &fP3=fP[3], &fP4=fP[4];
  Double_t 
  &fC00=fC[0],
  &fC10=fC[1],   &fC11=fC[2],  
  &fC20=fC[3],   &fC21=fC[4],   &fC22=fC[5],
  &fC30=fC[6],   &fC31=fC[7],   &fC32=fC[8],   &fC33=fC[9],  
  &fC40=fC[10],  &fC41=fC[11],  &fC42=fC[12],  &fC43=fC[13], &fC44=fC[14];

  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));

  //f = F - 1
  /*
  Double_t f02=    dx/(r1*r1*r1);            Double_t cc=crv/fP4;
  Double_t f04=0.5*dx*dx/(r1*r1*r1);         f04*=cc;
  Double_t f12=    dx*fP3*f1/(r1*r1*r1);
  Double_t f14=0.5*dx*dx*fP3*f1/(r1*r1*r1);  f14*=cc;
  Double_t f13=    dx/r1;
  Double_t f24=    dx;                       f24*=cc;
  */
  Double_t rinv = 1./r1;
  Double_t r3inv = rinv*rinv*rinv;
  Double_t f24=    x2r/fP4;
  Double_t f02=    dx*r3inv;
  Double_t f04=0.5*f24*f02;
  Double_t f12=    f02*fP3*f1;
  Double_t f14=0.5*f24*f02*fP3*f1;
  Double_t f13=    dx*rinv;
 
  //b = C*ft
  Double_t b00=f02*fC20 + f04*fC40, b01=f12*fC20 + f14*fC40 + f13*fC30;
  Double_t b02=f24*fC40;
  Double_t b10=f02*fC21 + f04*fC41, b11=f12*fC21 + f14*fC41 + f13*fC31;
  Double_t b12=f24*fC41;
  Double_t b20=f02*fC22 + f04*fC42, b21=f12*fC22 + f14*fC42 + f13*fC32;
  Double_t b22=f24*fC42;
  Double_t b40=f02*fC42 + f04*fC44, b41=f12*fC42 + f14*fC44 + f13*fC43;
  Double_t b42=f24*fC44;
  Double_t b30=f02*fC32 + f04*fC43, b31=f12*fC32 + f14*fC43 + f13*fC33;
  Double_t b32=f24*fC43;
  
  //a = f*b = f*C*ft
  Double_t a00=f02*b20+f04*b40,a01=f02*b21+f04*b41,a02=f02*b22+f04*b42;
  Double_t a11=f12*b21+f14*b41+f13*b31,a12=f12*b22+f14*b42+f13*b32;
  Double_t a22=f24*b42;

  //F*C*Ft = C + (b + bt + a)
  fC00 += b00 + b00 + a00;
  fC10 += b10 + b01 + a01; 
  fC20 += b20 + b02 + a02;
  fC30 += b30;
  fC40 += b40;
  fC11 += b11 + b11 + a11;
  fC21 += b21 + b12 + a12;
  fC31 += b31; 
  fC41 += b41;
  fC22 += b22 + b22 + a22;
  fC32 += b32;
  fC42 += b42;

  CheckCovariance();
  
  // Appoximate step length
  double dy2dx = (f1+f2)/(r1+r2);
  Double_t step = (TMath::Abs(x2r)<0.05) ? dx*TMath::Abs(r2 + f2*dy2dx)  // chord
    : 2.*TMath::ASin(0.5*dx*TMath::Sqrt(1.+dy2dx*dy2dx)*crv)/crv;        // arc
  step *= TMath::Sqrt(1.+ GetTgl()*GetTgl());

  // Get the track's (x,y,z) and (px,py,pz) in the Global System
  Double_t r[3]; GetXYZ(r);
  Double_t p[3]; GetPxPyPz(p);
  Double_t pp=GetP();
  p[0] /= pp;
  p[1] /= pp;
  p[2] /= pp;


  // Rotate to the system where Bx=By=0.
  Double_t bt=TMath::Sqrt(b[0]*b[0] + b[1]*b[1]);
  Double_t cosphi=1., sinphi=0.;
  if (bt > kAlmost0) {cosphi=b[0]/bt; sinphi=b[1]/bt;}
  Double_t bb=TMath::Sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  Double_t costet=1., sintet=0.;
  if (bb > kAlmost0) {costet=b[2]/bb; sintet=bt/bb;}
  Double_t vect[7];

  vect[0] = costet*cosphi*r[0] + costet*sinphi*r[1] - sintet*r[2];
  vect[1] = -sinphi*r[0] + cosphi*r[1];
  vect[2] = sintet*cosphi*r[0] + sintet*sinphi*r[1] + costet*r[2];

  vect[3] = costet*cosphi*p[0] + costet*sinphi*p[1] - sintet*p[2];
  vect[4] = -sinphi*p[0] + cosphi*p[1];
  vect[5] = sintet*cosphi*p[0] + sintet*sinphi*p[1] + costet*p[2];

  vect[6] = pp;


  // Do the helix step
  g3helx3(GetSign()*bb,step,vect);


  // Rotate back to the Global System
  r[0] = cosphi*costet*vect[0] - sinphi*vect[1] + cosphi*sintet*vect[2];
  r[1] = sinphi*costet*vect[0] + cosphi*vect[1] + sinphi*sintet*vect[2];
  r[2] = -sintet*vect[0] + costet*vect[2];

  p[0] = cosphi*costet*vect[3] - sinphi*vect[4] + cosphi*sintet*vect[5];
  p[1] = sinphi*costet*vect[3] + cosphi*vect[4] + sinphi*sintet*vect[5];
  p[2] = -sintet*vect[3] + costet*vect[5];


  // Rotate back to the Tracking System
  Double_t cosalp = TMath::Cos(fAlpha);
  Double_t sinalp =-TMath::Sin(fAlpha);

  Double_t 
  t    = cosalp*r[0] - sinalp*r[1];
  r[1] = sinalp*r[0] + cosalp*r[1];  
  r[0] = t;

  t    = cosalp*p[0] - sinalp*p[1]; 
  p[1] = sinalp*p[0] + cosalp*p[1];
  p[0] = t; 


  // Do the final correcting step to the target plane (linear approximation)
  Double_t x=r[0], y=r[1], z=r[2];
  if (TMath::Abs(dx) > kAlmost0) {
     if (TMath::Abs(p[0]) < kAlmost0) return kFALSE;
     dx = xk - r[0];
     x += dx;
     y += p[1]/p[0]*dx;
     z += p[2]/p[0]*dx;  
  }


  // Calculate the track parameters
  t=TMath::Sqrt(p[0]*p[0] + p[1]*p[1]);
  fX    = x;
  fP[0] = y;
  fP[1] = z;
  fP[2] = p[1]/t;
  fP[3] = p[2]/t; 
  fP[4] = GetSign()/(t*pp);

  return kTRUE;
}

Bool_t AliExternalTrackParam::PropagateParamOnlyBxByBzTo(Double_t xk, const Double_t b[3]) {
  //----------------------------------------------------------------
  // Extrapolate this track params (w/o cov matrix) to the plane X=xk in the field b[].
  //
  // X [cm] is in the "tracking coordinate system" of this track.
  // b[]={Bx,By,Bz} [kG] is in the Global coordidate system.
  //----------------------------------------------------------------

  Double_t dx=xk-fX;
  if (TMath::Abs(dx)<=kAlmost0)  return kTRUE;
  if (TMath::Abs(fP[4])<=kAlmost0) return kFALSE;
  // Do not propagate tracks outside the ALICE detector
  if (TMath::Abs(dx)>1e5 ||
      TMath::Abs(GetY())>1e5 ||
      TMath::Abs(GetZ())>1e5) {
    AliWarning(Form("Anomalous track, target X:%f",xk));
    Print();
    return kFALSE;
  }

  Double_t crv=GetC(b[2]);
  if (TMath::Abs(b[2]) < kAlmost0Field) crv=0.;

  Double_t x2r = crv*dx;
  Double_t f1=fP[2], f2=f1 + x2r;
  if (TMath::Abs(f1) >= kAlmost1) return kFALSE;
  if (TMath::Abs(f2) >= kAlmost1) return kFALSE;
  //
  Double_t r1=TMath::Sqrt((1.-f1)*(1.+f1)), r2=TMath::Sqrt((1.-f2)*(1.+f2));
  //
  // Appoximate step length
  double dy2dx = (f1+f2)/(r1+r2);
  Double_t step = (TMath::Abs(x2r)<0.05) ? dx*TMath::Abs(r2 + f2*dy2dx)  // chord
    : 2.*TMath::ASin(0.5*dx*TMath::Sqrt(1.+dy2dx*dy2dx)*crv)/crv;        // arc
  step *= TMath::Sqrt(1.+ GetTgl()*GetTgl());
  
  // Get the track's (x,y,z) and (px,py,pz) in the Global System
  Double_t r[3]; GetXYZ(r);
  Double_t p[3]; GetPxPyPz(p);
  Double_t pp=GetP();
  p[0] /= pp;
  p[1] /= pp;
  p[2] /= pp;

  // Rotate to the system where Bx=By=0.
  Double_t bt=TMath::Sqrt(b[0]*b[0] + b[1]*b[1]);
  Double_t cosphi=1., sinphi=0.;
  if (bt > kAlmost0) {cosphi=b[0]/bt; sinphi=b[1]/bt;}
  Double_t bb=TMath::Sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  Double_t costet=1., sintet=0.;
  if (bb > kAlmost0) {costet=b[2]/bb; sintet=bt/bb;}
  Double_t vect[7];

  vect[0] = costet*cosphi*r[0] + costet*sinphi*r[1] - sintet*r[2];
  vect[1] = -sinphi*r[0] + cosphi*r[1];
  vect[2] = sintet*cosphi*r[0] + sintet*sinphi*r[1] + costet*r[2];

  vect[3] = costet*cosphi*p[0] + costet*sinphi*p[1] - sintet*p[2];
  vect[4] = -sinphi*p[0] + cosphi*p[1];
  vect[5] = sintet*cosphi*p[0] + sintet*sinphi*p[1] + costet*p[2];

  vect[6] = pp;

  // Do the helix step
  g3helx3(GetSign()*bb,step,vect);

  // Rotate back to the Global System
  r[0] = cosphi*costet*vect[0] - sinphi*vect[1] + cosphi*sintet*vect[2];
  r[1] = sinphi*costet*vect[0] + cosphi*vect[1] + sinphi*sintet*vect[2];
  r[2] = -sintet*vect[0] + costet*vect[2];

  p[0] = cosphi*costet*vect[3] - sinphi*vect[4] + cosphi*sintet*vect[5];
  p[1] = sinphi*costet*vect[3] + cosphi*vect[4] + sinphi*sintet*vect[5];
  p[2] = -sintet*vect[3] + costet*vect[5];

  // Rotate back to the Tracking System
  Double_t cosalp = TMath::Cos(fAlpha);
  Double_t sinalp =-TMath::Sin(fAlpha);

  Double_t 
  t    = cosalp*r[0] - sinalp*r[1];
  r[1] = sinalp*r[0] + cosalp*r[1];  
  r[0] = t;

  t    = cosalp*p[0] - sinalp*p[1]; 
  p[1] = sinalp*p[0] + cosalp*p[1];
  p[0] = t; 

  // Do the final correcting step to the target plane (linear approximation)
  Double_t x=r[0], y=r[1], z=r[2];
  if (TMath::Abs(dx) > kAlmost0) {
     if (TMath::Abs(p[0]) < kAlmost0) return kFALSE;
     dx = xk - r[0];
     x += dx;
     y += p[1]/p[0]*dx;
     z += p[2]/p[0]*dx;  
  }


  // Calculate the track parameters
  t=TMath::Sqrt(p[0]*p[0] + p[1]*p[1]);
  fX    = x;
  fP[0] = y;
  fP[1] = z;
  fP[2] = p[1]/t;
  fP[3] = p[2]/t; 
  fP[4] = GetSign()/(t*pp);

  return kTRUE;
}


Bool_t AliExternalTrackParam::Translate(Double_t *vTrasl,Double_t *covV){
  //
  //Translation: in the event mixing, the tracks can be shifted 
  //of the difference among primary vertices (vTrasl) and 
  //the covariance matrix is changed accordingly 
  //(covV = covariance of the primary vertex).
  //Origin: "Romita, Rossella" <R.Romita@gsi.de>
  // 
  TVector3 translation;
  // vTrasl coordinates in the local system
  translation.SetXYZ(vTrasl[0],vTrasl[1],vTrasl[2]);
  translation.RotateZ(-fAlpha);
  translation.GetXYZ(vTrasl);

 //compute the new x,y,z of the track
  Double_t newX=fX-vTrasl[0];
  Double_t newY=fP[0]-vTrasl[1];
  Double_t newZ=fP[1]-vTrasl[2];
  
  //define the new parameters
  Double_t newParam[5];
  newParam[0]=newY;
  newParam[1]=newZ;
  newParam[2]=fP[2];
  newParam[3]=fP[3];
  newParam[4]=fP[4];

  // recompute the covariance matrix:
  // 1. covV in the local system
  Double_t cosRot=TMath::Cos(fAlpha), sinRot=TMath::Sin(fAlpha);
  TMatrixD qQi(3,3);
  qQi(0,0) = cosRot;
  qQi(0,1) = sinRot;
  qQi(0,2) = 0.;
  qQi(1,0) = -sinRot;
  qQi(1,1) = cosRot;
  qQi(1,2) = 0.;
  qQi(2,0) = 0.;
  qQi(2,1) = 0.;
  qQi(2,2) = 1.;
  TMatrixD uUi(3,3);
  uUi(0,0) = covV[0];
  uUi(0,0) = covV[0];
  uUi(1,0) = covV[1];
  uUi(0,1) = covV[1];
  uUi(2,0) = covV[3];
  uUi(0,2) = covV[3];
  uUi(1,1) = covV[2];
  uUi(2,2) = covV[5];
  uUi(1,2) = covV[4];
  if(uUi.Determinant() <= 0.) {return kFALSE;}
  TMatrixD uUiQi(uUi,TMatrixD::kMult,qQi);
  TMatrixD m(qQi,TMatrixD::kTransposeMult,uUiQi);

  //2. compute the new covariance matrix of the track
  Double_t sigmaXX=m(0,0);
  Double_t sigmaXZ=m(2,0);
  Double_t sigmaXY=m(1,0);
  Double_t sigmaYY=GetSigmaY2()+m(1,1);
  Double_t sigmaYZ=fC[1]+m(1,2);
  Double_t sigmaZZ=fC[2]+m(2,2);
  Double_t covarianceYY=sigmaYY + (-1.)*((sigmaXY*sigmaXY)/sigmaXX);
  Double_t covarianceYZ=sigmaYZ-(sigmaXZ*sigmaXY/sigmaXX);
  Double_t covarianceZZ=sigmaZZ-((sigmaXZ*sigmaXZ)/sigmaXX);

  Double_t newCov[15];
  newCov[0]=covarianceYY;
  newCov[1]=covarianceYZ;
  newCov[2]=covarianceZZ;
  for(Int_t i=3;i<15;i++){
    newCov[i]=fC[i];
   }

  // set the new parameters

  Set(newX,fAlpha,newParam,newCov);

  return kTRUE;
 }

void AliExternalTrackParam::CheckCovariance() {

  // This function forces the diagonal elements of the covariance matrix to be positive.
  // In case the diagonal element is bigger than the maximal allowed value, it is set to
  // the limit and the off-diagonal elements that correspond to it are set to zero.

  fC[0] = TMath::Abs(fC[0]);
  if (fC[0]>kC0max) {
    double scl = TMath::Sqrt(kC0max/fC[0]);
    fC[0] = kC0max;
    fC[1] *= scl;
    fC[3] *= scl;
    fC[6] *= scl;
    fC[10] *= scl;
  }
  fC[2] = TMath::Abs(fC[2]);
  if (fC[2]>kC2max) {
    double scl = TMath::Sqrt(kC2max/fC[2]);
    fC[2] = kC2max;
    fC[1] *= scl;
    fC[4] *= scl;
    fC[7] *= scl;
    fC[11] *= scl;
  }
  fC[5] = TMath::Abs(fC[5]);
  if (fC[5]>kC5max) {
    double scl = TMath::Sqrt(kC5max/fC[5]);
    fC[5] = kC5max;
    fC[3] *= scl;
    fC[4] *= scl;
    fC[8] *= scl;
    fC[12] *= scl;
  }
  fC[9] = TMath::Abs(fC[9]);
  if (fC[9]>kC9max) {
    double scl = TMath::Sqrt(kC9max/fC[9]);
    fC[9] = kC9max;
    fC[6] *= scl;
    fC[7] *= scl;
    fC[8] *= scl;
    fC[13] *= scl;
  }
  fC[14] = TMath::Abs(fC[14]);
  if (fC[14]>kC14max) {
    double scl = TMath::Sqrt(kC14max/fC[14]);
    fC[14] = kC14max;
    fC[10] *= scl;
    fC[11] *= scl;
    fC[12] *= scl;
    fC[13] *= scl;
  }
      
    // The part below is used for tests and normally is commented out    
//     TMatrixDSym m(5);
//     TVectorD eig(5);
    
//     m(0,0)=fC[0];
//     m(1,0)=fC[1];  m(1,1)=fC[2];
//     m(2,0)=fC[3];  m(2,1)=fC[4];  m(2,2)=fC[5];
//     m(3,0)=fC[6];  m(3,1)=fC[7];  m(3,2)=fC[8];  m(3,3)=fC[9];
//     m(4,0)=fC[10]; m(4,1)=fC[11]; m(4,2)=fC[12]; m(4,3)=fC[13]; m(4,4)=fC[14];
    
//     m(0,1)=m(1,0);
//     m(0,2)=m(2,0); m(1,2)=m(2,1);
//     m(0,3)=m(3,0); m(1,3)=m(3,1); m(2,3)=m(3,2);
//     m(0,4)=m(4,0); m(1,4)=m(4,1); m(2,4)=m(4,2); m(3,4)=m(4,3);
//     m.EigenVectors(eig);

//     //    assert(eig(0)>=0 && eig(1)>=0 && eig(2)>=0 && eig(3)>=0 && eig(4)>=0);
//     if (!(eig(0)>=0 && eig(1)>=0 && eig(2)>=0 && eig(3)>=0 && eig(4)>=0)) {
//       AliWarning("Negative eigenvalues of the covariance matrix!");
//       this->Print();
//       eig.Print();
//     }
}

Bool_t AliExternalTrackParam::ConstrainToVertex(const AliVVertex* vtx, Double_t b[3])
{
  // Constrain TPC inner params constrained
  //
  if (!vtx) 
    return kFALSE;

  Double_t dz[2], cov[3];
  if (!PropagateToDCABxByBz(vtx, b, 3, dz, cov)) 
    return kFALSE; 

  Double_t covar[6]; 
  vtx->GetCovarianceMatrix(covar);
  
  Double_t p[2]= { fP[0] - dz[0], fP[1] - dz[1] };
  Double_t c[3]= { covar[2], 0., covar[5] };
  
  Double_t chi2C = GetPredictedChi2(p,c);
  if (chi2C>kVeryBig) 
    return kFALSE; 

  if (!Update(p,c)) 
    return kFALSE; 

  return kTRUE;
}

//___________________________________________________________________________________________
Bool_t AliExternalTrackParam::GetXatLabR(Double_t r,Double_t &x, Double_t bz, Int_t dir) const
{
  // Get local X of the track position estimated at the radius lab radius r. 
  // The track curvature is accounted exactly
  //
  // The flag "dir" can be used to remove the ambiguity of which intersection to take (out of 2 possible)
  // 0  - take the intersection closest to the current track position
  // >0 - go along the track (increasing fX)
  // <0 - go backward (decreasing fX)
  //
  const Double_t &fy=fP[0], &sn = fP[2];
  const double kEps = 1.e-6;
  //
  double crv = GetC(bz);
  if (TMath::Abs(crv)>kAlmost0) {                                 // helix
    // get center of the track circle
    double tR = 1./crv;   // track radius (for the moment signed)
    double cs = TMath::Sqrt((1-sn)*(1+sn));
    double x0 = fX - sn*tR;
    double y0 = fy + cs*tR;
    double r0 = TMath::Sqrt(x0*x0+y0*y0);
    //    printf("Xc:%+e Yc:%+e tR:%e r0:%e\n",x0,y0,tR,r0);
    //
    if (r0<=kAlmost0) return kFALSE;            // the track is concentric to circle
    tR = TMath::Abs(tR);
    double tR2r0=1.,g=0,tmp=0;
    if (TMath::Abs(tR-r0)>kEps) {
      tR2r0 = tR/r0;
      g = 0.5*(r*r/(r0*tR) - tR2r0 - 1./tR2r0);
      tmp = 1.+g*tR2r0;
    }
    else {
      tR2r0 = 1.0;
      g = 0.5*r*r/(r0*tR) - 1;
      tmp = 0.5*r*r/(r0*r0);
    }
    double det = (1.-g)*(1.+g);
    if (det<0) return kFALSE;         // does not reach raduis r
    det = TMath::Sqrt(det);    
    //
    // the intersection happens in 2 points: {x0+tR*C,y0+tR*S} 
    // with C=f*c0+-|s0|*det and S=f*s0-+c0 sign(s0)*det
    // where s0 and c0 make direction for the circle center (=x0/r0 and y0/r0)
    //
    x = x0*tmp; 
    double y = y0*tmp;
    if (TMath::Abs(y0)>kAlmost0) { // when y0==0 the x,y is unique
      double dfx = tR2r0*TMath::Abs(y0)*det;
      double dfy = tR2r0*x0*TMath::Sign(det,y0);
      if (dir==0) {                    // chose the one which corresponds to smallest step 
	double delta = (x-fX)*dfx-(y-fy)*dfy; // the choice of + in C will lead to smaller step if delta<0
	if (delta<0) x += dfx;
	else         x -= dfx;
      }
      else if (dir>0) {  // along track direction: x must be > fX
	x -= dfx; // try the smallest step (dfx is positive)
	double dfeps = fX-x; // handle special case of very small step
	if (dfeps<-kEps) return kTRUE;
	if (TMath::Abs(dfeps)<kEps &&  // are we already in right r?
	    TMath::Abs(fX*fX+fy*fy - r*r)<kEps) return fX;
	x += dfx+dfx;
	if (x-fX>0) return kTRUE;
	if (x-fX<-kEps) return kFALSE;
	x = fX; // don't move
      }
      else { // backward: x must be < fX
	x += dfx; // try the smallest step (dfx is positive)	
	double dfeps = x-fX; // handle special case of very small step
	if (dfeps<-kEps) return kTRUE;
	if (TMath::Abs(dfeps)<kEps &&  // are we already in right r?
	    TMath::Abs(fX*fX+fy*fy - r*r)<kEps) return fX;
	x-=dfx+dfx;
	if (x-fX<0) return kTRUE;
	if (x-fX>kEps) return kFALSE;
	x = fX; // don't move
      }
    }
    else { // special case: track touching the circle just in 1 point
      if ( (dir>0&&x<fX) || (dir<0&&x>fX) ) return kFALSE; 
    }
  }
  else { // this is a straight track
    if (TMath::Abs(sn)>=kAlmost1) { // || to Y axis
      double det = (r-fX)*(r+fX);
      if (det<0) return kFALSE;     // does not reach raduis r
      x = fX;
      if (dir==0) return kTRUE;
      det = TMath::Sqrt(det);
      if (dir>0) {                       // along the track direction
	if (sn>0) {if (fy>det)  return kFALSE;} // track is along Y axis and above the circle
	else      {if (fy<-det) return kFALSE;} // track is against Y axis amd belo the circle
      }
      else if(dir>0) {                                    // agains track direction
	if (sn>0) {if (fy<-det) return kFALSE;} // track is along Y axis
        else if (fy>det)  return kFALSE;        // track is against Y axis
      }
    }
    else if (TMath::Abs(sn)<=kAlmost0) { // || to X axis
      double det = (r-fy)*(r+fy);
      if (det<0) return kFALSE;     // does not reach raduis r
      det = TMath::Sqrt(det);
      if (!dir) {
	x = fX>0  ? det : -det;    // choose the solution requiring the smalest step
	return kTRUE;
      }
      else if (dir>0) {                    // along the track direction
	if      (fX > det) return kFALSE;  // current point is in on the right from the circle
	else if (fX <-det) x = -det;       // on the left
	else               x =  det;       // within the circle
      }
      else {                               // against the track direction
	if      (fX <-det) return kFALSE;  
	else if (fX > det) x =  det;
	else               x = -det;
      }
    }
    else {                                 // general case of straight line
      double cs = TMath::Sqrt((1-sn)*(1+sn));
      double xsyc = fX*sn-fy*cs;
      double det = (r-xsyc)*(r+xsyc);
      if (det<0) return kFALSE;    // does not reach raduis r
      det = TMath::Sqrt(det);
      double xcys = fX*cs+fy*sn;
      double t = -xcys;
      if (dir==0) t += t>0 ? -det:det;  // chose the solution requiring the smalest step
      else if (dir>0) {                 // go in increasing fX direction. ( t+-det > 0)
	if (t>=-det) t += -det;         // take minimal step giving t>0
	else return kFALSE;             // both solutions have negative t
      }
      else {                            // go in increasing fX direction. (t+-det < 0)
	if (t<det) t -= det;            // take minimal step giving t<0
	else return kFALSE;             // both solutions have positive t
      }
      x = fX + cs*t;
    }
  }
  //
  return kTRUE;
}
//_________________________________________________________
Bool_t AliExternalTrackParam::GetXYZatR(Double_t xr,Double_t bz, Double_t *xyz, Double_t* alpSect) const
{
  // This method has 3 modes of behaviour
  // 1) xyz[3] array is provided but alpSect pointer is 0: calculate the position of track intersection 
  //    with circle of radius xr and fill it in xyz array
  // 2) alpSect pointer is provided: find alpha of the sector where the track reaches local coordinate xr
  //    Note that in this case xr is NOT the radius but the local coordinate.
  //    If the xyz array is provided, it will be filled by track lab coordinates at local X in this sector
  // 3) Neither alpSect nor xyz pointers are provided: just check if the track reaches radius xr
  //
  //
  double crv = GetC(bz);
  if ( (TMath::Abs(bz))<kAlmost0Field ) crv=0.;
  const double &fy = fP[0];
  const double &fz = fP[1];
  const double &sn = fP[2];
  const double &tgl = fP[3];
  //
  // general circle parameterization:
  // x = (r0+tR)cos(phi0) - tR cos(t+phi0)
  // y = (r0+tR)sin(phi0) - tR sin(t+phi0)
  // where qb is the sign of the curvature, tR is the track's signed radius and r0 
  // is the DCA of helix to origin
  //
  double tR = 1./crv;            // track radius signed
  double cs = TMath::Sqrt((1-sn)*(1+sn));
  double x0 = fX - sn*tR;        // helix center coordinates
  double y0 = fy + cs*tR;
  double phi0 = TMath::ATan2(y0,x0);  // angle of PCA wrt to the origin
  if (tR<0) phi0 += TMath::Pi();
  if      (phi0 > TMath::Pi()) phi0 -= 2.*TMath::Pi();
  else if (phi0 <-TMath::Pi()) phi0 += 2.*TMath::Pi();
  double cs0 = TMath::Cos(phi0);
  double sn0 = TMath::Sin(phi0);
  double r0 = x0*cs0 + y0*sn0 - tR; // DCA to origin
  double r2R = 1.+r0/tR;
  //
  //
  if (r2R<kAlmost0) return kFALSE;  // helix is centered at the origin, no specific intersection with other concetric circle
  if (!xyz && !alpSect) return kTRUE;
  double xr2R = xr/tR;
  double r2Ri = 1./r2R;
  // the intersection cos(t) = [1 + (r0/tR+1)^2 - (r0/tR)^2]/[2(1+r0/tR)]
  double cosT = 0.5*(r2R + (1-xr2R*xr2R)*r2Ri);
  if ( TMath::Abs(cosT)>kAlmost1 ) {
    //    printf("Does not reach : %f %f\n",r0,tR);
    return kFALSE; // track does not reach the radius xr
  }
  //
  double t = TMath::ACos(cosT);
  if (tR<0) t = -t;
  // intersection point
  double xyzi[3];
  xyzi[0] = x0 - tR*TMath::Cos(t+phi0);
  xyzi[1] = y0 - tR*TMath::Sin(t+phi0);
  if (xyz) { // if postition is requested, then z is needed:
    double t0 = TMath::ATan2(cs,-sn) - phi0;
    double z0 = fz - t0*tR*tgl;    
    xyzi[2] = z0 + tR*t*tgl;
  }
  else xyzi[2] = 0;
  //
  Local2GlobalPosition(xyzi,fAlpha);
  //
  if (xyz) {
    xyz[0] = xyzi[0];
    xyz[1] = xyzi[1];
    xyz[2] = xyzi[2];
  }
  //
  if (alpSect) {
    double &alp = *alpSect;
    // determine the sector of crossing
    double phiPos = TMath::Pi()+TMath::ATan2(-xyzi[1],-xyzi[0]);
    int sect = ((Int_t)(phiPos*TMath::RadToDeg()))/20;
    alp = TMath::DegToRad()*(20*sect+10);
    double x2r,f1,f2,r1,r2,dx,dy2dx,yloc=0, ylocMax = xr*TMath::Tan(TMath::Pi()/18); // min max Y within sector at given X
    //
    while(1) {
      Double_t ca=TMath::Cos(alp-fAlpha), sa=TMath::Sin(alp-fAlpha);
      if ((cs*ca+sn*sa)<0) {
	AliDebug(1,Form("Rotation to target sector impossible: local cos(phi) would become %.2f",cs*ca+sn*sa));
	return kFALSE;
      }
      //
      f1 = sn*ca - cs*sa;
      if (TMath::Abs(f1) >= kAlmost1) {
	AliDebug(1,Form("Rotation to target sector impossible: local sin(phi) would become %.2f",f1));
	return kFALSE;
      }
      //
      double tmpX =  fX*ca + fy*sa;
      double tmpY = -fX*sa + fy*ca;
      //
      // estimate Y at X=xr
      dx=xr-tmpX;
      x2r = crv*dx;
      f2=f1 + x2r;
      if (TMath::Abs(f2) >= kAlmost1) {
	AliDebug(1,Form("Propagation in target sector failed ! %.10e",f2));
	return kFALSE;
      }
      r1 = TMath::Sqrt((1.-f1)*(1.+f1));
      r2 = TMath::Sqrt((1.-f2)*(1.+f2));
      dy2dx = (f1+f2)/(r1+r2);
      yloc = tmpY + dx*dy2dx;
      if      (yloc>ylocMax)  {alp += 2*TMath::Pi()/18; sect++;}
      else if (yloc<-ylocMax) {alp -= 2*TMath::Pi()/18; sect--;}
      else break;
      if      (alp >= TMath::Pi()) alp -= 2*TMath::Pi();
      else if (alp < -TMath::Pi()) alp += 2*TMath::Pi();
      //      if (sect>=18) sect = 0;
      //      if (sect<=0) sect = 17;
    }
    //
    // if alpha was requested, then recalculate the position at intersection in sector
    if (xyz) {
      xyz[0] = xr;
      xyz[1] = yloc;
      if (TMath::Abs(x2r)<0.05) xyz[2] = fz + dx*(r2 + f2*dy2dx)*tgl;
      else {
	// for small dx/R the linear apporximation of the arc by the segment is OK,
	// but at large dx/R the error is very large and leads to incorrect Z propagation
	// angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
	// The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
	// Similarly, the rotation angle in linear in dx only for dx<<R
	double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
	double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
	xyz[2] = fz + rot/crv*tgl;
      }
      Local2GlobalPosition(xyz,alp);
    }
  }
  return kTRUE;  
  //
}


Double_t  AliExternalTrackParam::GetParameterAtRadius(Double_t r, Double_t bz, Int_t parType) const
{
  //
  // Get track parameters at the radius of interest.
  // Given function is aimed to be used to interactivelly (tree->Draw())
  // access track properties at different radii
  //
  // TO BE USED WITH SPECICAL CARE - 
  //     it is aimed to be used for rough calculation as constant field and  
  //     no correction for material is used
  //  
  // r  - radius of interest
  // bz - magentic field 
  // retun values dependens on parType:
  //    parType = 0  -gx 
  //    parType = 1  -gy 
  //    parType = 2  -gz 
  //
  //    parType = 3  -pgx 
  //    parType = 4  -pgy 
  //    parType = 5  -pgz
  //
  //    parType = 6  - r
  //    parType = 7  - global position phi
  //    parType = 8  - global direction phi
  //    parType = 9  - direction phi- positionphi
  if (parType<0) {
    parType=-1;
     return 0;
  }
  Double_t xyz[3];
  Double_t pxyz[3];  
  Double_t localX=0;
  Bool_t res = GetXatLabR(r,localX,bz,1);
  if (!res) {
    parType=-1;
    return 0;
  }
  //
  // position parameters
  // 
  GetXYZAt(localX,bz,xyz); 
  if (parType<3)   {
    return xyz[parType];
  }

  if (parType==6) return TMath::Sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
  if (parType==7) return TMath::ATan2(xyz[1],xyz[0]);
  //
  // momenta parameters
  //
  GetPxPyPzAt(localX,bz,pxyz); 
  if (parType==8) return TMath::ATan2(pxyz[1],pxyz[0]);
  if (parType==9) {
    Double_t diff = TMath::ATan2(pxyz[1],pxyz[0])-TMath::ATan2(xyz[1],xyz[0]);
    if (diff>TMath::Pi()) diff-=TMath::TwoPi();
    if (diff<-TMath::Pi()) diff+=TMath::TwoPi();
    return diff;
  }
  if (parType>=3&&parType<6) {
    return pxyz[parType%3];
  }
  return 0;
}

//_______________________________________________________________________
Bool_t AliExternalTrackParam::RelateToVVertexBxByBzDCA(const AliVVertex *vtx, Double_t b[3], Double_t maxd, AliExternalTrackParam *cParam, Double_t dz[2], Double_t dzcov[3]) {
  //
  // Try to relate the track parameters to the vertex "vtx", 
  // if the (rough) transverse impact parameter is not bigger then "maxd". 
  //
  // All three components of the magnetic field ,"b[3]" (kG), 
  // are taken into account.
  //
  // a) The paramters are extapolated to the DCA to the vertex.
  // b) The impact parameters and their covariance matrix are calculated.
  // c) An attempt to constrain the params to the vertex is done.
  //    The constrained params are returned via "cParam".
  //
  // In the case of success, the returned value is kTRUE
  // otherwise, it's kFALSE)
  // 

  if (!vtx) return kFALSE;

  if (!PropagateToDCABxByBz(vtx, b, maxd, dz, dzcov)) return kFALSE;

  Double_t covar[6]; vtx->GetCovarianceMatrix(covar);
  Double_t p[2]={GetParameter()[0]-dz[0],GetParameter()[1]-dz[1]};
  Double_t c[3]={covar[2],0.,covar[5]};

  Double_t chi2=GetPredictedChi2(p,c);
  if (chi2>kVeryBig) return kFALSE;

  if (!cParam) return kTRUE;

  *cParam = *this;
  if (!cParam->Update(p,c)) return kFALSE;

  return kTRUE;
}

