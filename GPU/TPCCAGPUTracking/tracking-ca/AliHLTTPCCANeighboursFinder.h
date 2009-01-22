//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCANEIGHBOURSFINDER_H
#define ALIHLTTPCCANEIGHBOURSFINDER_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGrid.h"
class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCANeighboursFinder
 * 
 */
class AliHLTTPCCANeighboursFinder
{
 public:
  class AliHLTTPCCASharedMemory
    {
  friend class AliHLTTPCCANeighboursFinder;
    public:
#if !defined(HLTCA_GPUCODE)
      AliHLTTPCCASharedMemory()
	: fGridUp(), fGridDn(), fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fFirst(0), fFirstDn(0), fFirstUp(0), fNRows(0), fHitLinkUp(0), fHitLinkDn(0)
      {}

      AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/) 
	: fGridUp(), fGridDn(), fNHits(0), fUpNHits(0), fDnNHits(0), fUpDx(0), fDnDx(0), fUpTx(0), fDnTx(0), fIRow(0), fIRowUp(0), fIRowDn(0), fFirst(0), fFirstDn(0), fFirstUp(0), fNRows(0), fHitLinkUp(0), fHitLinkDn(0)
      {}
      AliHLTTPCCASharedMemory& operator=(const AliHLTTPCCASharedMemory& /*dummy*/){ return *this; }
#endif
    protected:
      AliHLTTPCCAGrid fGridUp; // grid for the next row
      AliHLTTPCCAGrid fGridDn; // grid for the previous row
      Int_t fNHits; // n hits
      Int_t fUpNHits; // n hits in the next row
      Int_t fDnNHits; // n hits in the prev row
      Float_t fUpDx; // x distance to the next row
      Float_t fDnDx; // x distance to the previous row
      Float_t fUpTx; // normalized x distance to the next row
      Float_t fDnTx; // normalized x distance to the previous row
      Int_t fIRow; // row number
      Int_t fIRowUp; // next row number 
      Int_t fIRowDn;// previous row number 
      Int_t fFirst; // index of the first hit
      Int_t fFirstDn; // index of the first hit in the next row
      Int_t fFirstUp;// index of the first hit in the previous row
      Int_t fNRows; // number of rows
      Short_t *fHitLinkUp; // links to the next row
      Short_t *fHitLinkDn; // links to the previous  row
      float2 fA[256][5]; // temp memory
      UShort_t fB[256][5]; // temp memory
      UShort_t fGridContentUp[700]; // grid content for the next row
      UShort_t fGridContentDn[700];// grid content for the previous row
    };
  
  GPUd() static Int_t NThreadSyncPoints(){ return 2; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
  
};


#endif
