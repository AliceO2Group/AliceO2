//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAGBHIT_H
#define ALIHLTTPCCAGBHIT_H

#include "Rtypes.h"
#include "AliHLTTPCCAHit.h"

/**
 * @class AliHLTTPCCAGBHit
 *
 * The AliHLTTPCCAGBHit class is the internal representation
 * of the TPC clusters for the AliHLTTPCCAGBTracker algorithm.
 *
 */
class AliHLTTPCCAGBHit
{
 public:
  AliHLTTPCCAGBHit()
    :fX(0),fY(0),fZ(0),fErrX(0),fErrY(0),fErrZ(0),
    fISlice(0), fIRow(0), fID(0), fIsUsed(0),fSliceHit(){}

  virtual ~AliHLTTPCCAGBHit(){}

  Float_t &X(){ return fX; }
  Float_t &Y(){ return fY; } 
  Float_t &Z(){ return fZ; } 

  Float_t &ErrX(){ return fErrX; }
  Float_t &ErrY(){ return fErrY; }
  Float_t &ErrZ(){ return fErrZ; }

  Int_t &ISlice(){ return fISlice; }
  Int_t &IRow(){ return fIRow; }
  Int_t &ID(){ return fID; }
  Bool_t &IsUsed(){ return fIsUsed; };

  AliHLTTPCCAHit &SliceHit(){ return fSliceHit; } 


  static bool Compare(const AliHLTTPCCAGBHit &a, const AliHLTTPCCAGBHit &b);

  static bool CompareRowDown(const AliHLTTPCCAGBHit &a, const AliHLTTPCCAGBHit &b){
    return ( a.fIRow>b.fIRow );
  }
  static bool ComparePRowDown(const AliHLTTPCCAGBHit *a, const AliHLTTPCCAGBHit *b){
    return ( a->fIRow>b->fIRow );
  }

 protected:

  Float_t fX; //* X position
  Float_t fY; //* Y position
  Float_t fZ; //* Z position

  Float_t fErrX; //* X position error
  Float_t fErrY; //* Y position error
  Float_t fErrZ; //* Z position error

  Int_t fISlice; //* slice number
  Int_t fIRow;   //* row number
  Int_t fID;     //* external ID (id of AliTPCcluster) 
  Bool_t fIsUsed; //* is used by GBTracks

  AliHLTTPCCAHit fSliceHit; //* corresponding slice hit

  //ClassDef(AliHLTTPCCAGBHit,1);

};

#endif
