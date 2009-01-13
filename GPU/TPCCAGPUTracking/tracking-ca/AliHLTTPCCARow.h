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

  GPUhd() float &Hy0() { return fHy0;}
  GPUhd() float &Hz0() { return fHz0;}
  GPUhd() float &HstepY() { return fHstepY;}
  GPUhd() float &HstepZ() { return fHstepZ;}
  GPUhd() float &HstepYi() { return fHstepYi;}
  GPUhd() float &HstepZi() { return fHstepZi;}
  GPUhd() int &FullSize() { return fFullSize;}
  GPUhd() int &FullOffset() { return fFullOffset;}
  GPUhd() int &FullGridOffset() { return fFullGridOffset;}
  GPUhd() int &FullLinkOffset() { return fFullLinkOffset;}

private:

  Int_t fFirstHit;         // index of the first hit in the hit array
  Int_t fNHits;            // number of hits 
  Float_t fX;              // X coordinate of the row
  Float_t fMaxY;           // maximal Y coordinate of the row
  AliHLTTPCCAGrid fGrid;   // grid of hits

  float fHy0,fHz0, fHstepY,fHstepZ, fHstepYi, fHstepZi; // temporary variables
  int fFullSize, fFullOffset, fFullGridOffset,fFullLinkOffset; // temporary variables

};

#endif
