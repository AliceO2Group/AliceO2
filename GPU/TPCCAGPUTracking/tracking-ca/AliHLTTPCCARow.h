//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAROW_H
#define ALIHLTTPCCAROW_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGrid.h"

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

#if !defined(HLTCA_GPUCODE)
    AliHLTTPCCARow();
#endif

    GPUhd() int   FirstHit() const { return fFirstHit; }
    GPUhd() int   NHits()    const { return fNHits; }
    GPUhd() float X()        const { return fX; }
    GPUhd() float MaxY()     const { return fMaxY; }
    GPUhd() const AliHLTTPCCAGrid &Grid() const { return fGrid; }

    GPUhd() float Hy0()      const { return fHy0;}
    GPUhd() float Hz0()      const { return fHz0;}
    GPUhd() float HstepY()   const { return fHstepY;}
    GPUhd() float HstepZ()   const { return fHstepZ;}
    GPUhd() float HstepYi()  const { return fHstepYi;}
    GPUhd() float HstepZi()  const { return fHstepZi;}
    GPUhd() int   FullSize()    const { return fFullSize;}
    GPUhd() int   FullOffset()  const { return fFullOffset;}
    GPUhd() int   FullGridOffset()  const { return fFullGridOffset;}
    GPUhd() int   FullLinkOffset()  const { return fFullLinkOffset;}

    GPUhd() void SetFirstHit( int v ) { fFirstHit = v; }
    GPUhd() void SetNHits( int v )   { fNHits = v; }
    GPUhd() void SetX( float v )       { fX = v; }
    GPUhd() void SetMaxY( float v )    { fMaxY = v; }
    GPUhd() void SetGrid( const AliHLTTPCCAGrid &v ) { fGrid = v; }

    GPUhd() void SetHy0( float v ) { fHy0 = v;}
    GPUhd() void SetHz0( float v ) { fHz0 = v;}
    GPUhd() void SetHstepY( float v ) { fHstepY = v;}
    GPUhd() void SetHstepZ( float v ) { fHstepZ = v;}
    GPUhd() void SetHstepYi( float v ) { fHstepYi = v;}
    GPUhd() void SetHstepZi( float v ) { fHstepZi = v;}
    GPUhd() void SetFullSize( int v ) { fFullSize = v;}
    GPUhd() void SetFullOffset( int v ) { fFullOffset = v;}
    GPUhd() void SetFullGridOffset( int v ) { fFullGridOffset = v;}
    GPUhd() void SetFullLinkOffset( int v ) { fFullLinkOffset = v;}

  private:

    int fFirstHit;         // index of the first hit in the hit array
    int fNHits;            // number of hits
    float fX;              // X coordinate of the row
    float fMaxY;           // maximal Y coordinate of the row
    AliHLTTPCCAGrid fGrid;   // grid of hits

    float fHy0, fHz0, fHstepY, fHstepZ, fHstepYi, fHstepZi; // temporary variables
    int fFullSize, fFullOffset, fFullGridOffset, fFullLinkOffset; // temporary variables

};

#endif
