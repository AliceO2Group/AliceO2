#ifndef ALITRACKREFERENCE_H
#define ALITRACKREFERENCE_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

// 
// Track Reference object is created every time particle is 
// crossing detector bounds. 
// The object is created by Step Manager
//

#include "TObject.h"
#include "TMath.h"

class AliTrackReference : public TObject {
public:

    enum constants {kDisappeared = -1,
		    kITS   = 0,
		    kTPC   = 1,
		    kFRAME = 2,
		    kTRD   = 3,
		    kTOF   = 4,
		    kMUON  = 5,
		    kHMPID = 6,
		    kT0    = 7,
		    kEMCAL = 8,
		    kPMD   = 10,
		    kFMD   = 12,
		    kVZERO = 14,
		    kZDC   = 15,
		    kMFT   = 16,
		    kHALL  = 17,
		    kFIT  = 18, //alla
		    kAD = 19
    };
  AliTrackReference();
  AliTrackReference(Int_t label, Int_t id = -999);
  AliTrackReference(const AliTrackReference &tr);
  virtual ~AliTrackReference() {}

//  static AliExternalTrackParam * MakeTrack(const AliTrackReference *ref, Double_t mass);
  virtual Int_t GetTrack() const {return fTrack;}
  virtual void SetTrack(Int_t track) {fTrack=track;}
  virtual void SetLength(Float_t length){fLength=length;}
  virtual void SetTime(Float_t time) {fTime = time;}
  virtual Float_t GetLength() const {return fLength;}
  virtual Float_t GetTime() const {return fTime;}
  virtual Int_t Label() const {return fTrack;}
  virtual void SetLabel(Int_t track) {fTrack=track;}
  virtual Float_t R() const {return TMath::Sqrt(fX*fX+fY*fY);}
  virtual Float_t Pt() const {return TMath::Sqrt(fPx*fPx+fPy*fPy);}
  virtual Float_t Phi() const {return TMath::Pi()+TMath::ATan2(-fPy,-fPx);}
  virtual Float_t Theta() const {return (fPz==0)?TMath::Pi()/2:TMath::ACos(fPz/P());}
  virtual Float_t X() const {return fX;}
  virtual Float_t Y() const {return fY;}
  virtual Float_t Z() const {return fZ;}
  virtual Float_t Px() const {return fPx;}
  virtual Float_t Py() const {return fPy;}
  virtual Float_t Pz() const {return fPz;}
  virtual Float_t P() const {return TMath::Sqrt(fPx*fPx+fPy*fPy+fPz*fPz);}
  virtual Int_t   UserId() const {return fUserId;}
  virtual Int_t   DetectorId() const {return fDetectorId;}
  virtual void SetDetectorId(Int_t id){fDetectorId = id;}
  virtual void SetPosition(Float_t x, Float_t y, Float_t z){fX=x; fY=y; fZ=z;}
  virtual void SetMomentum(Float_t px, Float_t py, Float_t pz){fPx=px; fPy=py; fPz=pz;}
  virtual void SetUserId(Int_t userId){fUserId=userId;}
 
  // Methods to get position of the track reference in 
  // in the TPC/TRD/TOF Tracking coordinate system

  virtual Float_t PhiPos() const {return TMath::Pi()+TMath::ATan2(-fY, -fX);}
  virtual Float_t Alpha() const 
    {return TMath::Pi()*(20*((((Int_t)(PhiPos()*180/TMath::Pi()))/20))+10)/180.;}
  virtual Float_t LocalX() const {return fX*TMath::Cos(-Alpha()) - fY*TMath::Sin(-Alpha());}
  virtual Float_t LocalY() const {return fX*TMath::Sin(-Alpha()) + fY*TMath::Cos(-Alpha());}

  Bool_t IsSortable() const {return kTRUE;}
  Int_t Compare(const TObject *obj) const {
    Int_t ll = ((AliTrackReference*)obj)->GetTrack();
    if (ll < fTrack) return 1;
    if (ll > fTrack) return -1;
    return 0;
  }

  virtual void Print(Option_t* opt="") const;
    
protected:
  Int_t     fTrack;      // Track number
  Float_t   fX;          // X reference position of the track
  Float_t   fY;          // Y reference position of the track
  Float_t   fZ;          // Z reference position of the track
  Float_t   fPx;         // momentum
  Float_t   fPy;         // momentum
  Float_t   fPz;         // momentum
  Float_t   fLength;     // track lenght from its origin in cm
  Float_t   fTime;       // time of flight in cm  
  Int_t     fUserId;     // optional Id defined by user
  Int_t     fDetectorId; // Detector Id
  ClassDef(AliTrackReference,7)  //Base class for all Alice track references
};
#endif
