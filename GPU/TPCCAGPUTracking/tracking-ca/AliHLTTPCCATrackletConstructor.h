//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCATRACKLETCONSTRUCTOR_H
#define ALIHLTTPCCATRACKLETCONSTRUCTOR_H


#include "AliHLTTPCCADef.h"

/**
 * @class AliHLTTPCCATrackletConstructor
 * 
 */
class AliHLTTPCCATrackletConstructor
{
 public:

  class   AliHLTTPCCASharedMemory
    {
      friend class AliHLTTPCCATrackletConstructor;
    public:
#if !defined(HLTCA_GPUCODE)
      AliHLTTPCCASharedMemory()
	: fItr0(0), fItr1(0), fNRows(0), fUsedHits(0), fMinStartRow(0), fMaxStartRow(0)
      {}

      AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/) 
	: fItr0(0), fItr1(0), fNRows(0), fUsedHits(0), fMinStartRow(0), fMaxStartRow(0)
      {}
      AliHLTTPCCASharedMemory& operator=(const AliHLTTPCCASharedMemory& /*dummy*/){ return *this; }
#endif
    protected:
      uint4 fData[2][(500+500+500)/4]; // temp memory
      Int_t fItr0; // start track index
      Int_t fItr1; // end track index
      Int_t fNRows; // n rows
      Int_t *fUsedHits;   // array of used hits
      Int_t fMinStartRow; // min start row
      Int_t fMinStartRow32[32]; // min start row for each thread in warp
      Int_t fMaxStartRow; // max start row
      Int_t fMaxStartRow32[32];// max start row for each thread in warp
    };

  class  AliHLTTPCCAThreadMemory
    {
      friend class AliHLTTPCCATrackletConstructor;
     public:
#if !defined(HLTCA_GPUCODE)
      AliHLTTPCCAThreadMemory()
	: fItr(0), fFirstRow(0), fLastRow(0), fCurrIH(0), fIsMemThread(0), fGo(0), fSave(0), fCurrentData(0), fStage(0), fNHits(0), fNMissed(0), fTrackStoreOffset(0), fHitStoreOffset(0), fLastY(0), fLastZ(0) 
      {}

      AliHLTTPCCAThreadMemory( const AliHLTTPCCAThreadMemory& /*dummy*/) 
	: fItr(0), fFirstRow(0), fLastRow(0), fCurrIH(0), fIsMemThread(0), fGo(0), fSave(0), fCurrentData(0), fStage(0), fNHits(0), fNMissed(0), fTrackStoreOffset(0), fHitStoreOffset(0), fLastY(0), fLastZ(0)
      {}
      AliHLTTPCCAThreadMemory& operator=(const AliHLTTPCCAThreadMemory& /*dummy*/){ return *this; }
#endif
    protected:
      Int_t fItr; // track index
      Int_t fFirstRow;  // first row index
      Int_t fLastRow; // last row index
      Int_t fCurrIH; // indef of the current hit
      Bool_t fIsMemThread; // is the thread used for memory taken
      Bool_t fGo; // do fit/searching flag
      Bool_t fSave; // save flag
      Bool_t fCurrentData; // index of the current memory array
      Int_t fStage; // reco stage
      Int_t fNHits; // n track hits
      Int_t fNMissed; // n missed hits during search
      Int_t fTrackStoreOffset; // offset in the global array
      Int_t fHitStoreOffset;   // offset in the global array
      Float_t fLastY; // Y of the last fitted cluster
      Float_t fLastZ; // Z of the last fitted cluster
    };

  GPUd() static Int_t NThreadSyncPoints(){ return 4+159*2 +1+1; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, 
			     Int_t iSync, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, 
			     AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );
  
  GPUd() static void Step0
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );
  GPUd() static void Step1
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );
  GPUd() static void Step2
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );

  GPUd() static void ReadData( Int_t iThread, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, Int_t iRow );

  GPUd() static void UpdateTracklet
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam, Int_t iRow );

  GPUd() static void StoreTracklet
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam);

  GPUd() static Bool_t SAVE(){ return 1; }
 
#if defined(HLTCA_GPUCODE)
  GPUhd() static Int_t NMemThreads(){ return 128; }
#else
  GPUhd() static Int_t NMemThreads(){ return 1; }
#endif

};


#endif
