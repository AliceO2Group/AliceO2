//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

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

  GPUhd() Int_t   FirstHit() const { return fFirstHit; }
  GPUhd() Int_t   NHits()    const { return fNHits; }
  GPUhd() Float_t X()        const { return fX; }
  GPUhd() Float_t MaxY()     const { return fMaxY; }
  GPUhd() const AliHLTTPCCAGrid &Grid() const { return fGrid; }  

  GPUhd() Float_t Hy0()      const { return fHy0;}
  GPUhd() Float_t Hz0()      const { return fHz0;}
  GPUhd() Float_t HstepY()   const { return fHstepY;}
  GPUhd() Float_t HstepZ()   const { return fHstepZ;}
  GPUhd() Float_t HstepYi()  const { return fHstepYi;}
  GPUhd() Float_t HstepZi()  const { return fHstepZi;}
  GPUhd() Int_t   FullSize()    const { return fFullSize;}
  GPUhd() Int_t   FullOffset()  const { return fFullOffset;}
  GPUhd() Int_t   FullGridOffset()  const { return fFullGridOffset;}
  GPUhd() Int_t   FullLinkOffset()  const { return fFullLinkOffset;}


  GPUhd() void SetFirstHit( Int_t v ){ fFirstHit = v; }
  GPUhd() void SetNHits( Int_t v )   { fNHits = v; }
  GPUhd() void SetX( Float_t v )       { fX = v; }
  GPUhd() void SetMaxY( Float_t v )    { fMaxY = v; }
  GPUhd() void SetGrid( const AliHLTTPCCAGrid &v ){ fGrid = v; }  

  GPUhd() void SetHy0( Float_t v ) { fHy0 = v;}
  GPUhd() void SetHz0( Float_t v ) { fHz0 = v;}
  GPUhd() void SetHstepY( Float_t v ) { fHstepY = v;}
  GPUhd() void SetHstepZ( Float_t v ) { fHstepZ = v;}
  GPUhd() void SetHstepYi( Float_t v ) { fHstepYi = v;}
  GPUhd() void SetHstepZi( Float_t v ) { fHstepZi = v;}
  GPUhd() void SetFullSize( Int_t v ) { fFullSize = v;}
  GPUhd() void SetFullOffset( Int_t v ) { fFullOffset = v;}
  GPUhd() void SetFullGridOffset( Int_t v ) { fFullGridOffset = v;}
  GPUhd() void SetFullLinkOffset( Int_t v ) { fFullLinkOffset = v;}

private:

  Int_t fFirstHit;         // index of the first hit in the hit array
  Int_t fNHits;            // number of hits 
  Float_t fX;              // X coordinate of the row
  Float_t fMaxY;           // maximal Y coordinate of the row
  AliHLTTPCCAGrid fGrid;   // grid of hits

  Float_t fHy0,fHz0, fHstepY,fHstepZ, fHstepYi, fHstepZi; // temporary variables
  Int_t fFullSize, fFullOffset, fFullGridOffset,fFullLinkOffset; // temporary variables

};

#endif
