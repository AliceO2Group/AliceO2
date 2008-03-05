//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAROW_H
#define ALIHLTTPCCAROW_H


#include "Rtypes.h"

#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCACell.h"

/**
 * @class ALIHLTTPCCARow
 *
 * The ALIHLTTPCCARow class is a hit and cells container for one TPC row.
 * It is the internal class of the AliHLTTPCCATracker algorithm.
 *
 */
class AliHLTTPCCARow
{
 public: 

  AliHLTTPCCARow();
  AliHLTTPCCARow ( const AliHLTTPCCARow &);
  AliHLTTPCCARow &operator=( const AliHLTTPCCARow &);

  virtual ~AliHLTTPCCARow(){ Clear(); }

  AliHLTTPCCAHit  *&Hits() { return fHits; }
  AliHLTTPCCACell *&Cells(){ return fCells;}
  Int_t  *&CellHitPointers() { return fCellHitPointers; }
  
  Int_t &NHits()  { return fNHits; }
  Int_t &NCells() { return fNCells; }
  Float_t &X() { return fX; }

  AliHLTTPCCAHit  &GetCellHit( AliHLTTPCCACell &c, Int_t i ){ 
    //* get hit number i of the cell c
    return fHits[fCellHitPointers[c.FirstHitRef()+i]]; 
  }

  void Clear();

 private:

  AliHLTTPCCAHit *fHits;   // hit array
  AliHLTTPCCACell *fCells; // cell array
  Int_t *fCellHitPointers; // pointers cell->hits
  Int_t fNHits, fNCells;   // number of hits and cells
  Float_t fX;              // X coordinate of the row

  ClassDef(AliHLTTPCCARow,1);
};

#endif
