//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCACELL_H
#define ALIHLTTPCCACELL_H


#include "Rtypes.h"

/**
 * @class AliHLTTPCCACell
 *
 * The AliHLTTPCCACell class describes the "Cell" object ---
 * the set of neghbouring clusters in the same TPC row.
 * Cells are used as the minimal data units 
 * by the Cellular Automaton tracking algorithm. 
 *
 */
class AliHLTTPCCACell
{
 public:

  //AliHLTTPCCACell(): fY(0),fZ(0),fErrY(0),fErrZ(0),fFirstHitRef(0),fNHits(0),fLink(0),fTrackID(0){}

  //virtual ~AliHLTTPCCACell(){}

  Float_t &Y(){ return fY; }
  Float_t &Z(){ return fZ; }
  Float_t &ErrY(){ return fErrY; } 
  Float_t &ErrZ(){ return fErrZ; }
  Float_t &ZMin(){ return fZMin; }
  Float_t &ZMax(){ return fZMax; }
  
  Int_t &FirstHitRef(){ return fFirstHitRef; }
  Int_t &NHits()      { return fNHits; }

  Int_t &Link()       { return fLink; }
  Int_t &Status()     { return fStatus; }
  Int_t &TrackID()    { return fTrackID; }
 
 protected:

  Float_t fY, fZ, fZMin,fZMax;       //* Y and Z coordinates
  Float_t fErrY, fErrZ; //* cell errors in Y and Z

  Int_t fFirstHitRef;   //* index of the first cell hit in the cell->hit reference array
  Int_t fNHits;         //* number of hits in the cell

  Int_t fLink;          //* link to the next cell on the track
  Int_t fStatus;        //* status flag
  Int_t fTrackID;       //* index of track

 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  //ClassDef(AliHLTTPCCACell,1);
};


#endif
