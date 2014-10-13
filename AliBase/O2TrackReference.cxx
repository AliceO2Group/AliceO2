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

#include "TVirtualMC.h"
#include "TParticle.h"

#include "O2TrackReference.h"
#include <Riostream.h>

// 
// Track Reference object is created every time particle is 
// crossing detector bounds. The object is created by Step Manager
//
// The class stores the following informations:
// track label, 
// track position: X,Y,X
// track momentum px, py, pz
// track length and time of fligth: both in cm
// status bits from Monte Carlo
//

using std::endl;
using std::cout;
ClassImp(O2TrackReference)

O2TrackReference::O2TrackReference():
   TObject(),
   fTrack(0),
   fX(0),
   fY(0),
   fZ(0),
   fPx(0),
   fPy(0),
   fPz(0),
   fLength(0),
   fTime(0),
   fUserId(0),
   fDetectorId(-999)
{
  //
  // Default constructor
  // Creates empty object

  for(Int_t i=0; i<16; i++) ResetBit(BIT(i));
}

O2TrackReference::O2TrackReference(const O2TrackReference &tr) :
  TObject(tr),
  fTrack(tr.fTrack),
  fX(tr.fX),
  fY(tr.fY),
  fZ(tr.fZ),
  fPx(tr.fPx),
  fPy(tr.fPy),
  fPz(tr.fPz),
  fLength(tr.fLength),
  fTime(tr.fTime),
  fUserId(tr.fUserId),
  fDetectorId(tr.fDetectorId)
{
    // Copy Constructor
}

O2TrackReference::O2TrackReference(Int_t label, Int_t id) :
  TObject(),
  fTrack(label),
  fX(0),
  fY(0),
  fZ(0),
  fPx(0),
  fPy(0),
  fPz(0),
  fLength(gMC->TrackLength()),
  fTime(gMC->TrackTime()),
  fUserId(0),
  fDetectorId(id)
{
  //
  // Create Reference object out of label and
  // data in TVirtualMC object
  //
  // Creates an object and fill all parameters 
  // from data in VirtualMC
  //
  // Sylwester Radomski, (S.Radomski@gsi.de)
  // GSI, Jan 31, 2003
  //
    
  Double_t vec[4];
  
  gMC->TrackPosition(vec[0],vec[1],vec[2]);

  fX = vec[0];
  fY = vec[1];
  fZ = vec[2];
  
  gMC->TrackMomentum(vec[0],vec[1],vec[2],vec[3]);
  
  fPx = vec[0];
  fPy = vec[1];
  fPz = vec[2];

  // Set Up status code 
  // Copy Bits from virtual MC

  for(Int_t i=14; i<22; i++) ResetBit(BIT(i));

  SetBit(BIT(14), gMC->IsNewTrack());
  SetBit(BIT(15), gMC->IsTrackAlive());
  SetBit(BIT(16), gMC->IsTrackDisappeared());
  SetBit(BIT(17), gMC->IsTrackEntering());
  SetBit(BIT(18), gMC->IsTrackExiting());
  SetBit(BIT(19), gMC->IsTrackInside());
  SetBit(BIT(20), gMC->IsTrackOut());
  SetBit(BIT(21), gMC->IsTrackStop()); 
  //
  // This particle has to be kept
}

/*
AliExternalTrackParam * O2TrackReference::MakeTrack(const O2TrackReference *ref, Double_t mass)
{
  //
  // Make dummy track from the track reference 
  // negative mass means opposite charge 
  //
  Double_t xx[5];
  Double_t cc[15];
  for (Int_t i=0;i<15;i++) cc[i]=0;
  Double_t x = ref->X(), y = ref->Y(), z = ref->Z();
  Double_t alpha = TMath::ATan2(y,x);
  Double_t xr = TMath::Sqrt(x*x+y*y);
  xx[0] = ref->LocalY();
  xx[1] = z;
  xx[3] = ref->Pz()/ref->Pt();
  xx[4] = 1./ref->Pt(); 
  if (mass<0) xx[4]*=-1.;  // negative mass - negative direction
  Double_t alphap = TMath::ATan2(ref->Py(),ref->Px())-alpha;
  if (alphap> TMath::Pi()) alphap-=TMath::Pi();
  if (alphap<-TMath::Pi()) alphap+=TMath::Pi();
  xx[2] = TMath::Sin(alphap);

  AliExternalTrackParam * track = new  AliExternalTrackParam(xr,alpha,xx,cc);
  return track;
}
*/

void O2TrackReference::Print(Option_t* /*opt*/) const
{
  cout << Form("Label %d P=%7.2f (PX,PY,PZ)=(%7.2f,%7.2f,%7.2f) (X,Y,Z)=(%7.2f,%7.2f,%7.2f)"
               " Length=%7.2f Time=%7.2f UserId=%d",
	       Label(),P(),Px(),Py(),Pz(),X(),Y(),Z(),GetLength(),GetTime(),UserId()) << endl;
}

