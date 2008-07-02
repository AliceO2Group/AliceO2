//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAMCTRACK_H
#define ALIHLTTPCCAMCTRACK_H

#include "Rtypes.h"

class TParticle;


/**
 * @class AliHLTTPCCAMCTrack
 * store MC track information for AliHLTTPCCAPerformance
 */
class AliHLTTPCCAMCTrack
{
 public:

  AliHLTTPCCAMCTrack();
  AliHLTTPCCAMCTrack( const TParticle *part );

  Double_t *Par()           { return fPar; }
  Double_t &P()             { return fP; }
  Double_t &Pt()            { return fPt; }
  Int_t    &NHits()         { return fNHits;}
  Int_t    &NReconstructed(){ return fNReconstructed; }
  Int_t    &Set()           { return fSet; }
  Int_t    &NTurns()        { return fNTurns; }

 protected:

  Double_t fPar[7];         //* x,y,z,ex,ey,ez,q/p
  Double_t fP, fPt;         //* momentum and transverse momentum
  Int_t    fNHits;          //* N TPC clusters
  Int_t    fNReconstructed; //* how many times is reconstructed
  Int_t    fSet;            //* set of tracks 0-OutSet, 1-ExtraSet, 2-RefSet 
  Int_t    fNTurns;         //* N of turns in the current sector

};

#endif
