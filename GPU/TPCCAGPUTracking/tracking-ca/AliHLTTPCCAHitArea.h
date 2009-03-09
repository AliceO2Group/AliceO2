//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCARow.h 27042 2008-07-02 12:06:02Z richterm $

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

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
  
  GPUd() void Init( const AliHLTTPCCAGrid &grid, UShort_t *content, UInt_t hitoffset, Float_t y, Float_t z, Float_t dy, Float_t dz );

  GPUd() Int_t GetNext(AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row, UShort_t *content,AliHLTTPCCAHit &h);

  GPUd() Int_t GetBest(AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row, UShort_t *content,AliHLTTPCCAHit &h );

  GPUhd() Float_t Y() const { return fY;}
  GPUhd() Float_t Z() const { return fZ;}
  GPUhd() Float_t MinZ() const { return fMinZ;}
  GPUhd() Float_t MaxZ() const { return fMaxZ;}
  GPUhd() Float_t MinY() const { return fMinY;}
  GPUhd() Float_t MaxY() const { return fMaxY;}
  GPUhd() UInt_t  BZmax() const { return fBZmax;}
  GPUhd() UInt_t  BDY() const { return fBDY;}
  GPUhd() UInt_t  IndYmin() const { return fIndYmin;}
  GPUhd() UInt_t  Iz() const { return fIz;}
  GPUhd() UInt_t  HitYfst() const { return fHitYfst;}
  GPUhd() UInt_t  HitYlst() const { return fHitYlst;}
  GPUhd() UInt_t  Ih() const { return fIh;}
  GPUhd() UInt_t  Ny() const { return fNy;}
  GPUhd() UInt_t  HitOffset() const { return fHitOffset;}

  protected:

  Float_t fY, fZ, fMinZ, fMaxZ, fMinY, fMaxY;    // search coordinates
  UInt_t fBZmax, fBDY, fIndYmin, fIz, fHitYfst, fHitYlst, fIh, fNy; // !
  UInt_t fHitOffset; // global hit offset
};

#endif
