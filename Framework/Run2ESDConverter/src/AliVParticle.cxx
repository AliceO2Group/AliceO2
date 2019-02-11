/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
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

//-------------------------------------------------------------------------
//     base class for ESD and AOD particles
//     Author: Markus Oldenburg, CERN
//-------------------------------------------------------------------------

#include "AliVParticle.h"
#include "TMath.h"

ClassImp(AliVParticle)

AliVParticle::AliVParticle(const AliVParticle& vPart) :
  TObject(vPart) { } // Copy constructor

AliVParticle& AliVParticle::operator=(const AliVParticle& vPart)
{ if (this!=&vPart) { 
    TObject::operator=(vPart); 
  }
  
  return *this; 
}

Bool_t AliVParticle::Local2GlobalMomentum(Double_t p[3], Double_t alpha) const {
  //----------------------------------------------------------------
  // This function performs local->global transformation of the
  // track momentum.
  // When called, the arguments are:
  //    p[0] = 1/pt * charge of the track;
  //    p[1] = sine of local azim. angle of the track momentum;
  //    p[2] = tangent of the track momentum dip angle;
  //   alpha - rotation angle. 
  // The result is returned as:
  //    p[0] = px
  //    p[1] = py
  //    p[2] = pz
  // Results for (nearly) straight tracks are meaningless !
  //----------------------------------------------------------------
  if (TMath::Abs(p[0])<=kAlmost0) return kFALSE;
  if (TMath::Abs(p[1])> kAlmost1) return kFALSE;

  Double_t pt=1./TMath::Abs(p[0]);
  Double_t cs=TMath::Cos(alpha), sn=TMath::Sin(alpha);
  Double_t r=TMath::Sqrt((1. - p[1])*(1. + p[1]));
  p[0]=pt*(r*cs - p[1]*sn); p[1]=pt*(p[1]*cs + r*sn); p[2]=pt*p[2];

  return kTRUE;
}

Bool_t AliVParticle::Local2GlobalPosition(Double_t r[3], Double_t alpha) const {
  //----------------------------------------------------------------
  // This function performs local->global transformation of the
  // track position.
  // When called, the arguments are:
  //    r[0] = local x
  //    r[1] = local y
  //    r[2] = local z
  //   alpha - rotation angle. 
  // The result is returned as:
  //    r[0] = global x
  //    r[1] = global y
  //    r[2] = global z
  //----------------------------------------------------------------
  Double_t cs=TMath::Cos(alpha), sn=TMath::Sin(alpha), x=r[0];
  r[0]=x*cs - r[1]*sn; r[1]=x*sn + r[1]*cs;

  return kTRUE;
}

Bool_t AliVParticle::Global2LocalMomentum(Double_t p[3], Short_t charge, Double_t &alpha) const {
  //----------------------------------------------------------------
  // This function performs global->local transformation of the
  // track momentum.
  // When called, the arguments are:
  //    p[0] = px
  //    p[1] = py
  //    p[2] = pz
  //   charge - of the track
  //   alpha - rotation angle. 
  // The result is returned as:
  //    p[0] = 1/pt * charge of the track;
  //    p[1] = sine of local azim. angle of the track momentum;
  //    p[2] = tangent of the track momentum dip angle;
  // Results for (nearly) straight tracks are meaningless !
  //----------------------------------------------------------------
  double pt = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
  if (pt == 0.) return kFALSE;
  alpha = TMath::Pi() + TMath::ATan2(-p[1], -p[0]);
  
  p[0] = 1./pt * (float)charge;
  p[1] = 0.;
  p[2] = p[2]/pt;

  return kTRUE;
}

Bool_t AliVParticle::Global2LocalPosition(Double_t r[3], Double_t alpha) const {
  return Local2GlobalPosition(r, -alpha);
}


Int_t AliVParticle::Compare( const TObject* obj) const {

  // 
  // see header file for class documentation
  //

  if (this == obj)
    return 0;
  // check type
  if ( Pt() < ((AliVParticle*)(obj))->Pt())
    return 1;
  else
    return -1;
}

