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

  //AliHLTTPCCARow &operator=( const AliHLTTPCCARow &);

  GPUhd() Int_t   &FirstHit(){ return fFirstHit; }
  GPUhd() Int_t   &NHits()   { return fNHits; }
  GPUhd() Float_t &X()       { return fX; }
  GPUhd() Float_t &MaxY()    { return fMaxY; }
  GPUhd() AliHLTTPCCAGrid &Grid(){ return fGrid; }  

  GPUhd() Float_t &Hy0() { return fHy0;}
  GPUhd() Float_t &Hz0() { return fHz0;}
  GPUhd() Float_t &HstepY() { return fHstepY;}
  GPUhd() Float_t &HstepZ() { return fHstepZ;}
  GPUhd() Float_t &HstepYi() { return fHstepYi;}
  GPUhd() Float_t &HstepZi() { return fHstepZi;}
  GPUhd() Int_t &FullSize() { return fFullSize;}
  GPUhd() Int_t &FullOffset() { return fFullOffset;}
  GPUhd() Int_t &FullGridOffset() { return fFullGridOffset;}
  GPUhd() Int_t &FullLinkOffset() { return fFullLinkOffset;}

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
