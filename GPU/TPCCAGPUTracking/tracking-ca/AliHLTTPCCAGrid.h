//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

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

    GPUd() void Create( float yMin, float yMax, float zMin, float zMax, unsigned int n );
    GPUd() void Create( float yMin, float yMax, float zMin, float zMax, float sy, float sz  );

    GPUd() unsigned int GetBin( float Y, float Z ) const;
    GPUd() void GetBin( float Y, float Z, unsigned int &bY, unsigned int &bZ ) const ;

    GPUd() unsigned int GetBinNoCheck( float Y, float Z ) const {
      unsigned int bY = ( unsigned int ) ( ( Y - fYMin ) * fStepYInv );
      unsigned int bZ = ( unsigned int ) ( ( Z - fZMin ) * fStepZInv );
      return bZ*fNy + bY;
    }

    GPUd() void GetBinNoCheck( float Y, float Z, unsigned int &bY, unsigned int &bZ ) const {
      bY = ( unsigned int ) ( ( Y - fYMin ) * fStepYInv );
      bZ = ( unsigned int ) ( ( Z - fZMin ) * fStepZInv );
    }


    GPUd() unsigned int  N()        const { return fN;  }
    GPUd() unsigned int  Ny()       const { return fNy; }
    GPUd() unsigned int  Nz()       const { return fNz; }
    GPUd() float YMin()     const { return fYMin; }
    GPUd() float YMax()     const { return fYMax; }
    GPUd() float ZMin()     const { return fZMin; }
    GPUd() float ZMax()     const { return fZMax; }
    GPUd() float StepYInv() const { return fStepYInv; }
    GPUd() float StepZInv() const { return fStepZInv; }

  private:

    unsigned int fNy;        //* N bins in Y
    unsigned int fNz;        //* N bins in Z
    unsigned int fN;         //* total N bins
    float fYMin;     //* minimal Y value
    float fYMax;     //* maximal Y value
    float fZMin;     //* minimal Z value
    float fZMax;     //* maximal Z value
    float fStepYInv; //* inverse bin size in Y
    float fStepZInv; //* inverse bin size in Z

};

#endif
