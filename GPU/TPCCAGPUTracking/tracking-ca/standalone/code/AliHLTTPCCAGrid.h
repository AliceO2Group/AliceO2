//-*- Mode: C++ -*-
// $Id: AliHLTTPCCAGrid.h 36185 2009-11-02 07:19:00Z sgorbuno $
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
 * pointers to (y,z)-like objects are assigned to the corresponding grid bin
 * used by AliHLTTPCCATracker to speed-up the hit operations
 * grid axis are named Z,Y to be similar to TPC row coordinates.
 */
class AliHLTTPCCAGrid
{
  public:
    GPUd() void CreateEmpty();
    GPUd() void Create( float yMin, float yMax, float zMin, float zMax, float sy, float sz  );

    GPUd() int GetBin( float Y, float Z ) const;
    /**
     * returns -1 if the row is empty == no hits
     */
    GPUd() int GetBinBounded( float Y, float Z ) const;
    GPUd() void GetBin( float Y, float Z, int* const bY, int* const bZ ) const;

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

#endif //ALIHLTTPCCAGRID_H
