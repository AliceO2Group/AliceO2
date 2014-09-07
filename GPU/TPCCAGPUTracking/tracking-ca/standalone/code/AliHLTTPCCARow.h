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
MEM_CLASS_PRE() class AliHLTTPCCARow
{
    MEM_CLASS_PRE2() friend class AliHLTTPCCASliceData;
  public:

#if !defined(HLTCA_GPUCODE)
    AliHLTTPCCARow();
#endif //!HLTCA_GPUCODE

    GPUhd() int   NHits()    const { return fNHits; }
    GPUhd() float X()        const { return fX; }
    GPUhd() float MaxY()     const { return fMaxY; }
    GPUhd() MakeType(const AliHLTTPCCAGrid&) Grid() const { return fGrid; }

    GPUhd() float Hy0()      const { return fHy0; }
    GPUhd() float Hz0()      const { return fHz0; }
    GPUhd() float HstepY()   const { return fHstepY; }
    GPUhd() float HstepZ()   const { return fHstepZ; }
    GPUhd() float HstepYi()  const { return fHstepYi; }
    GPUhd() float HstepZi()  const { return fHstepZi; }
    GPUhd() int   FullSize() const { return fFullSize; }
    GPUhd() int   HitNumberOffset() const { return fHitNumberOffset; }
    GPUhd() unsigned int FirstHitInBinOffset() const { return fFirstHitInBinOffset; }

  private:
    int fNHits;            // number of hits
    float fX;              // X coordinate of the row
    float fMaxY;           // maximal Y coordinate of the row
    AliHLTTPCCAGrid fGrid;   // grid of hits

    // hit packing:
    float fHy0;          // offset
    float fHz0;          // offset
    float fHstepY;       // step size
    float fHstepZ;       // step size
    float fHstepYi;      // inverse step size
    float fHstepZi;      // inverse step size

    int fFullSize;       // size of this row in Tracker::fRowData
    int fHitNumberOffset;  // index of the first hit in the hit array, used as
    // offset in AliHLTTPCCASliceData::LinkUp/DownData/HitDataY/...
    unsigned int fFirstHitInBinOffset; // offset in Tracker::fRowData to find the FirstHitInBin
};

#endif //ALIHLTTPCCAROW_H
