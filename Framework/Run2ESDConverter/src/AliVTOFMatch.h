#ifndef ALIVTOFMATCH_H
#define ALIVTOFMATCH_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: $ */

//----------------------------------------------------------------------//
//                                                                      //
// AliVTOFMatch Class                                                   //
//                                                                      //
//----------------------------------------------------------------------//

#include "TObject.h"
#include "AliPID.h"

class AliVTOFMatch : public TObject
{
 public:
  AliVTOFMatch() {}
  AliVTOFMatch(AliVTOFMatch &source) : TObject(source) {}
  virtual ~AliVTOFMatch() {}
  AliVTOFMatch &operator=(const AliVTOFMatch& source);
  // 
  virtual Int_t GetTrackIndex()           const {return -1;}
  virtual void  SetTrackIndex(Int_t ) {}
  //
  virtual Float_t GetDx() const {return 0;};
  virtual Float_t GetDy() const {return 0;};
  virtual Float_t GetDz() const {return 0;};
  virtual Float_t GetTrackLength() const {return 0;};
  virtual void    GetIntegratedTimes(Double_t *) const {}
  virtual Double_t GetIntegratedTimes(Int_t) const {return 0;}
  virtual void    SetIntegratedTimes(Double_t *) {}
  virtual void    SetDx(Double_t) {};
  virtual void    SetDy(Double_t) {};
  virtual void    SetDz(Double_t) {};
  virtual void    SetTrackLength(Double_t) {};
  
  
 protected:
  
  ClassDef(AliVTOFMatch, 1) // TOF matchable hit
    
};
#endif
