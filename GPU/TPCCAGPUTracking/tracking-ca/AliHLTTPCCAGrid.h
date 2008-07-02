//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAGRID_H
#define ALIHLTTPCCAGRID_H

#include "Rtypes.h"

/**
 * @class AliHLTTPCCAGrid
 *
 * 2-dimensional grid of pointers.
 * pointers to (y,z)-like objects are assigned to the corresponding grig bin
 * used by AliHLTTPCCATracker to speed-up the hit operations
 * grid axis are named Z,Y to be similar to TPC row coordinates.
 */
class AliHLTTPCCAGrid
{
 public:
  AliHLTTPCCAGrid():fGrid(0),fNy(0),fNz(0),fN(0),
    fYMin(0),fYMax(0),fZMin(0),fZMax(0),fStepYInv(0),fStepZInv(0){}

  AliHLTTPCCAGrid(const AliHLTTPCCAGrid&);
  AliHLTTPCCAGrid &operator=(const AliHLTTPCCAGrid&);

  virtual ~AliHLTTPCCAGrid(){ 
    if( fGrid ) delete[] fGrid; 
  }

  void Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, Int_t n );

  void **Get( Float_t Y, Float_t Z ) const;

  void **GetNoCheck( Float_t Y, Float_t Z ) const {
    Int_t yBin = (Int_t) ( (Y-fYMin)*fStepYInv );
    Int_t zBin = (Int_t) ( (Z-fZMin)*fStepZInv );
    return fGrid + zBin*fNy + yBin;
  }

  Int_t N() const { return fN; }
  Int_t Ny() const { return fNy; }
  Int_t Nz() const { return fNz; }
  Float_t YMin() const { return fYMin; }
  Float_t YMax() const { return fYMax; }
  Float_t ZMin() const { return fZMin; }
  Float_t ZMax() const { return fZMax; }
  Float_t StepYInv() const { return fStepYInv; }
  Float_t StepZInv() const { return fStepZInv; }
  void **Grid(){ return fGrid; }

 protected:

  void **fGrid;      //* the grid as 1-d array
  Int_t fNy;         //* N bins in Y
  Int_t fNz;         //* N bins in Z
  Int_t fN;          //* total N bins
  Float_t fYMin;     //* minimal Y value
  Float_t fYMax;     //* maximal Y value
  Float_t fZMin;     //* minimal Z value
  Float_t fZMax;     //* maximal Z value
  Float_t fStepYInv; //* inverse bin size in Y
  Float_t fStepZInv; //* inverse bin size in Z
  
};


#endif
