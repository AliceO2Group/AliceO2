#ifndef ALIVTOFHIT_H
#define ALIVTOFHIT_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: $ */

//----------------------------------------------------------------------//
//                                                                      //
// AliVTOFHit Class                                                     //
//                                                                      //
//----------------------------------------------------------------------//

#include "TObject.h"

class AliVTOFHit : public TObject 
{
 public:
  AliVTOFHit() {}
  AliVTOFHit(const AliVTOFHit &source) : TObject(source) {}  
  virtual ~AliVTOFHit() {}
  AliVTOFHit & operator=(const AliVTOFHit& source);
  //
  virtual Int_t   GetESDTOFClusterIndex()             const {return -1;}
  virtual void    SetESDTOFClusterIndex(Int_t ) {}
  
  virtual void    SetTime(Double_t ) {}
  virtual void    SetLabel(Int_t *) {}
  virtual void    SetTimeRaw(Double_t) {}
  virtual void    SetTOT(Double_t) {}
  virtual void    SetL0L1Latency(Int_t) {}
  virtual void    SetDeltaBC(Int_t) {}
  virtual void    SetTOFchannel(Int_t) {}
  virtual void    SetClusterIndex(Int_t) {}
  virtual Double_t GetTime() const {return 0;}
  virtual Double_t GetTimeRaw() const {return 0;}
  virtual Double_t GetTOT() const {return 0;};
  virtual Int_t   GetL0L1Latency() const {return 0;};
  virtual Int_t   GetTOFLabel(Int_t ) const {return -1;}
  virtual Int_t   GetDeltaBC() const {return 0;};
  virtual Int_t   GetTOFchannel() const {return -1;};
  virtual Int_t   GetClusterIndex() const {return -1;};
  //
  ClassDef(AliVTOFHit, 1) // TOF matchable hit

};
#endif
