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

  Int_t NHits()               const { return fNHits; }
  Int_t FirstHitRef()         const { return fFirstHitRef; }
  const AliHLTTPCCATrackParam &Param() const { return fParam; }
  Float_t Alpha()            const { return fAlpha; }
  Float_t DeDx()             const { return fDeDx; } 


  void SetNHits( Int_t v )                 {  fNHits = v; }
  void SetFirstHitRef( Int_t v )           {  fFirstHitRef = v; }
  void SetParam( const AliHLTTPCCATrackParam &v ) {  fParam = v; }
  void SetAlpha( Float_t v )               {  fAlpha = v; }
  void SetDeDx( Float_t v )                {  fDeDx = v; } 


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
