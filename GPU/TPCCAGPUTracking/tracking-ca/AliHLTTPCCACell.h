//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

//*                                                                        *
//* The AliHLTTPCCACell class describes the "Cell" object ---              *
//* the set of neghbouring clusters in the same TPC row.                   *
//* The cells are used as the minimal data units                           *
//* by the Cellular Automaton tracking algorithm.                          *
//*                                                                        *

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

  AliHLTTPCCACell(): fFirstHitRef(0),fNHits(0),fY(0),fZ(0),fErrY(0),fErrZ(0),fIDown(0),fIUp(0),fIUsed(0){}

  virtual ~AliHLTTPCCACell(){}

  Float_t &Y(){ return fY; }
  Float_t &Z(){ return fZ; }
  Float_t &ErrY(){ return fErrY; } 
  Float_t &ErrZ(){ return fErrZ; }
  
  Int_t &FirstHitRef(){ return fFirstHitRef; }
  Int_t &NHits()      { return fNHits; }
  Int_t &IDown()      { return fIDown; }
  Int_t &IUp()        { return fIUp; }
  Int_t &IUsed()      { return fIUsed; }
 
 protected:

  Int_t fFirstHitRef;   // index of the first cell hit in the cell->hit reference array
  Int_t fNHits;         // number of hits in the cell
  Float_t fY, fZ;       // Y and Z coordinates
  Float_t fErrY, fErrZ; // cell errors in Y and Z
  Int_t fIDown, fIUp;   // indices of 2 neighboring cells in up & down directions
  Int_t fIUsed;         // if it is used by a reconstructed track

 private:

  void Dummy(); // to make rulechecker happy by having something in .cxx file

  ClassDef(AliHLTTPCCACell,1);
};


#endif
