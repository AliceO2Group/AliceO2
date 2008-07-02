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
#include "AliHLTTPCCAEndPoint.h"

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
  AliHLTTPCCAEndPoint *&EndPoints(){ return fEndPoints;}
 
  Int_t &NHits()  { return fNHits; }
  Int_t &NCells() { return fNCells; }
  Int_t &NEndPoints() { return fNEndPoints; }
  Float_t &X() { return fX; }
  Float_t &MaxY() { return fMaxY; }
  Float_t &DeltaY() { return fDeltaY; }
  Float_t &DeltaZ() { return fDeltaZ; }

  AliHLTTPCCAHit  &GetCellHit( AliHLTTPCCACell &c, Int_t i ){ 
    //* get hit number i of the cell c
    return fHits[fCellHitPointers[c.FirstHitRef()+i]]; 
  }

  void Clear();

  static Bool_t CompareCellZMax( AliHLTTPCCACell&c, Double_t ZMax )
  {
    return (c.ZMax()<ZMax);
  }
  static Bool_t CompareEndPointZ( AliHLTTPCCAEndPoint &p, Double_t Z )
  {
    return (p.Param().GetZ()<Z);
  }

private:

  AliHLTTPCCAHit *fHits;   // hit array
  AliHLTTPCCACell *fCells; // cell array
  Int_t *fCellHitPointers; // pointers cell->hits
  AliHLTTPCCAEndPoint *fEndPoints; // array of track end points
  Int_t fNHits, fNCells, fNEndPoints;   // number of hits and cells
  Float_t fX;              // X coordinate of the row
  Float_t fMaxY;           // maximal Y coordinate of the row
  Float_t fDeltaY;         // allowed Y deviation to the next row
  Float_t fDeltaZ;         // allowed Z deviation to the next row

  ClassDef(AliHLTTPCCARow,1);
};

#endif
