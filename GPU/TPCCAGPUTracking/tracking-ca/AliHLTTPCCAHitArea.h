//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCARow.h 27042 2008-07-02 12:06:02Z richterm $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAHITAREA_H
#define ALIHLTTPCCAHITAREA_H


#include "AliHLTTPCCADef.h"

class AliHLTTPCCAHit;
class AliHLTTPCCAGrid;
class AliHLTTPCCATracker;
class AliHLTTPCCARow;

/**
 * @class ALIHLTTPCCAHitArea
 *
 */
class AliHLTTPCCAHitArea
{
  public:

    GPUd() void Init( const AliHLTTPCCAGrid &grid, const unsigned short *content, unsigned int hitoffset, float y, float z, float dy, float dz );

    GPUd() int GetNext( AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row, const unsigned short *content, AliHLTTPCCAHit &h );

    GPUd() int GetBest( AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row, const unsigned short *content, AliHLTTPCCAHit &h );

    GPUhd() float Y() const { return fY;}
    GPUhd() float Z() const { return fZ;}
    GPUhd() float MinZ() const { return fMinZ;}
    GPUhd() float MaxZ() const { return fMaxZ;}
    GPUhd() float MinY() const { return fMinY;}
    GPUhd() float MaxY() const { return fMaxY;}
    GPUhd() unsigned int  BZmax() const { return fBZmax;}
    GPUhd() unsigned int  BDY() const { return fBDY;}
    GPUhd() unsigned int  IndYmin() const { return fIndYmin;}
    GPUhd() unsigned int  Iz() const { return fIz;}
    GPUhd() unsigned int  HitYfst() const { return fHitYfst;}
    GPUhd() unsigned int  HitYlst() const { return fHitYlst;}
    GPUhd() unsigned int  Ih() const { return fIh;}
    GPUhd() unsigned int  Ny() const { return fNy;}
    GPUhd() unsigned int  HitOffset() const { return fHitOffset;}

  protected:

    float fY, fZ, fMinZ, fMaxZ, fMinY, fMaxY;    // search coordinates
    unsigned int fBZmax, fBDY, fIndYmin, fIz, fHitYfst, fHitYlst, fIh, fNy; // !
    unsigned int fHitOffset; // global hit offset
};

#endif
