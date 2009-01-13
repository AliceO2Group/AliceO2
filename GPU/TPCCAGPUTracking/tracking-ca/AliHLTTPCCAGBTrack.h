//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAGBTRACK_H
#define ALIHLTTPCCAGBTRACK_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATrackParam.h"

/**
 * @class AliHLTTPCCAGBTrack
 *
 *
 */
class AliHLTTPCCAGBTrack
{
 public:

  AliHLTTPCCAGBTrack():fFirstHitRef(0),fNHits(0),fParam(),fAlpha(0),fDeDx(0){ ; }
  virtual ~AliHLTTPCCAGBTrack(){ ; }

  Int_t &NHits()               { return fNHits; }
  Int_t &FirstHitRef()         { return fFirstHitRef; }
  AliHLTTPCCATrackParam &Param() { return fParam; }
  Float_t &Alpha()            { return fAlpha; }
  Float_t &DeDx()             { return fDeDx; } 
  static Bool_t ComparePNClusters( const AliHLTTPCCAGBTrack *a, const AliHLTTPCCAGBTrack *b){
    return (a->fNHits > b->fNHits);
  }

 protected:
  
  Int_t fFirstHitRef;        // index of the first hit reference in track->hit reference array
  Int_t fNHits;              // number of track hits
  AliHLTTPCCATrackParam fParam;// fitted track parameters
  Float_t fAlpha;             //* Alpha angle of the parametrerisation
  Float_t fDeDx;              //* DE/DX 

 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  ClassDef(AliHLTTPCCAGBTrack,1)
};


#endif
