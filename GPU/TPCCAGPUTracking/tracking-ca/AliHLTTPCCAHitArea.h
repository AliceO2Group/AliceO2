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

/**
 * @class ALIHLTTPCCAHitArea
 *
 */
class AliHLTTPCCAHitArea
{
public:
  
  GPUd() void Init( AliHLTTPCCATracker &tracker, AliHLTTPCCAGrid &grid, UInt_t hitoffset, Float_t y, Float_t z, Float_t dy, Float_t dz );
  GPUd() Int_t GetNext(AliHLTTPCCATracker &tracker, AliHLTTPCCAHit &h);
  GPUd() Int_t GetBest(AliHLTTPCCATracker &tracker, AliHLTTPCCAHit &h );

  GPUhd() Float_t& Y(){ return fY;}
  GPUhd() Float_t& Z(){ return fZ;}
  GPUhd() Float_t& MinZ(){ return fMinZ;}
  GPUhd() Float_t& MaxZ(){ return fMaxZ;}
  GPUhd() Float_t& MinY(){ return fMinY;}
  GPUhd() Float_t& MaxY(){ return fMaxY;}
  GPUhd() UInt_t&  BZmax(){ return fBZmax;}
  GPUhd() UInt_t&  BDY(){ return fBDY;}
  GPUhd() UInt_t&  IndYmin(){ return fIndYmin;}
  GPUhd() UInt_t&  Iz(){ return fIz;}
  GPUhd() UInt_t&  HitYfst(){ return fHitYfst;}
  GPUhd() UInt_t&  HitYlst(){ return fHitYlst;}
  GPUhd() UInt_t&  Ih(){ return fIh;}
  GPUhd() UInt_t&  Ny(){ return fNy;}
  GPUhd() UInt_t&  N2(){ return fn2;}
  GPUhd() UInt_t&  C2(){ return fc2;}
  GPUhd() UInt_t&  HitOffset(){ return fHitOffset;}

  protected:

  Float_t fY, fZ, fMinZ, fMaxZ, fMinY, fMaxY;    // search coordinates
  UInt_t fBZmax, fBDY, fIndYmin, fIz, fHitYfst, fHitYlst, fIh, fNy, fn2, fc2; // !
  UInt_t fHitOffset; // global hit offset
};

#endif
