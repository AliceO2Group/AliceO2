//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCATRACK_H
#define ALIHLTTPCCATRACK_H

#include "Rtypes.h"

/**
 * @class ALIHLTTPCCAtrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the AliHLTTPCCATracker algorithm.
 * The track parameters at both ends are stored separately in the AliHLTTPCCAEndPoint class
 */
class AliHLTTPCCATrack
{
 public:
  AliHLTTPCCATrack():fAlive(0),fFirstCellID(0),fNCells(0){}
  virtual ~AliHLTTPCCATrack(){}

  Bool_t &Alive()              { return fAlive; }
  Int_t  &NCells()            { return fNCells; }
  Int_t  *CellID()            { return fCellID; }
  Int_t  &FirstCellID()        { return fFirstCellID; }
  Int_t  *PointID()            { return fPointID; }

 private:

  Bool_t fAlive;       // flag for mark tracks used by the track merger
  Int_t  fFirstCellID; // index of the first track cell in the track->cell pointer array
  Int_t  fNCells;      // number of track cells
  Int_t  fCellID[3];   // ID of first,middle,last cell
  Int_t  fPointID[2];  // ID of the track endpoints
  
 private:
  void Dummy(); // to make rulechecker happy by having something in .cxx file

  //ClassDef(AliHLTTPCCATrack,1);
};

#endif
