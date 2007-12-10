//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCATRACK_H
#define ALIHLTTPCCATRACK_H


#include "Rtypes.h"
#include "AliHLTTPCCATrackPar.h"

/**
 * @class ALIHLTTPCCAtrack
 */
class AliHLTTPCCATrack
{
 public:
  AliHLTTPCCATrack():fUsed(0),fNCells(0),fIFirstCell(0),fParam(){}

  static bool CompareSize(const AliHLTTPCCATrack &t1, const AliHLTTPCCATrack &t2 ){
    return t2.fNCells<t1.fNCells;
  }

  Bool_t &Used()              { return fUsed; }
  Int_t  &NCells()            { return fNCells; }
  Int_t &IFirstCell()         { return fIFirstCell; }
  AliHLTTPCCATrackPar &Param(){ return fParam; }

 private:

  Bool_t fUsed;       // flag for mark tracks used by the track merger
  Int_t  fNCells;     // number of track cells
  Int_t  fIFirstCell; // index of first cell reference in track->cell reference array
  
  AliHLTTPCCATrackPar fParam; // fitted track parameters

  ClassDef(AliHLTTPCCATrack,1);
};

#endif
