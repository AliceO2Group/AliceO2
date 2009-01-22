//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAGRID_H
#define ALIHLTTPCCAGRID_H

#include "AliHLTTPCCADef.h"

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
  
  GPUd() void Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, UInt_t n );
  GPUd() void Create( Float_t yMin, Float_t yMax, Float_t zMin, Float_t zMax, Float_t sy, Float_t sz  );
  
  GPUd() UInt_t GetBin( Float_t Y, Float_t Z ) const;
  GPUd() void GetBin( Float_t Y, Float_t Z, UInt_t &bY, UInt_t &bZ ) const ;
  
  GPUd() UInt_t GetBinNoCheck( Float_t Y, Float_t Z ) const {
    UInt_t bY = (UInt_t) ( (Y-fYMin)*fStepYInv );
    UInt_t bZ = (UInt_t) ( (Z-fZMin)*fStepZInv );
    return bZ*fNy + bY;
  }
  
  GPUd() void GetBinNoCheck( Float_t Y, Float_t Z, UInt_t &bY, UInt_t &bZ ) const {
    bY = (UInt_t) ( (Y-fYMin)*fStepYInv );
    bZ = (UInt_t) ( (Z-fZMin)*fStepZInv );    
  }


  GPUd() UInt_t  N()  const { return fN;  }
  GPUd() UInt_t  Ny() const { return fNy; }
  GPUd() UInt_t  Nz() const { return fNz; }
  GPUd() Float_t YMin() const { return fYMin; }
  GPUd() Float_t YMax() const { return fYMax; }
  GPUd() Float_t ZMin() const { return fZMin; }
  GPUd() Float_t ZMax() const { return fZMax; }
  GPUd() Float_t StepYInv() const { return fStepYInv; }
  GPUd() Float_t StepZInv() const { return fStepZInv; }

  private:

  UInt_t fNy;        //* N bins in Y
  UInt_t fNz;        //* N bins in Z
  UInt_t fN;         //* total N bins
  Float_t fYMin;     //* minimal Y value
  Float_t fYMax;     //* maximal Y value
  Float_t fZMin;     //* minimal Z value
  Float_t fZMax;     //* maximal Z value
  Float_t fStepYInv; //* inverse bin size in Y
  Float_t fStepZInv; //* inverse bin size in Z

};

#endif
