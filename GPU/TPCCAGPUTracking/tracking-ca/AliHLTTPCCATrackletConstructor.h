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
      UInt_t fGridContent1[600]; // grid1 content
      int fItr0; // start track index
      int fItr1; // end track index
      int fNRows; // n rows
      Int_t *fUsedHits;   // array of used hits
      int fMinStartRow; // min start row
      int fMinStartRow32[32]; // min start row for each thread in warp
      int fMaxStartRow; // max start row
      int fMaxStartRow32[32];// max start row for each thread in warp
    };

  class  AliHLTTPCCAThreadMemory
    {
      friend class AliHLTTPCCATrackletConstructor;
     public:
#if !defined(HLTCA_GPUCODE)
      AliHLTTPCCAThreadMemory()
	: fItr(0), fFirstRow(0), fLastRow(0), fCurrIH(0), fIsMemThread(0), fGo(0), fSave(0), fCurrentData(0), fStage(0), fNHits(0), fNMissed(0), fTrackStoreOffset(0), fHitStoreOffset(0) 
      {}

      AliHLTTPCCAThreadMemory( const AliHLTTPCCAThreadMemory& /*dummy*/) 
	: fItr(0), fFirstRow(0), fLastRow(0), fCurrIH(0), fIsMemThread(0), fGo(0), fSave(0), fCurrentData(0), fStage(0), fNHits(0), fNMissed(0), fTrackStoreOffset(0), fHitStoreOffset(0) 
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
    };

  GPUd() static Int_t NThreadSyncPoints(){ return 4+159*4 +1+1; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, 
			     Int_t iSync, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, 
			     AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam );
  
  GPUd() static void Step0
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam );
  GPUd() static void Step1
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam );
  GPUd() static void Step2
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam );

  GPUd() static void ReadData( Int_t iThread, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, Int_t iRow );

  GPUd() static void UpdateTracklet
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam, Int_t iRow );

  GPUd() static void UnpackGrid
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam, Int_t iRow );

  GPUd() static void StoreTracklet
    ( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam1 &tParam);

  static Bool_t SAVE(){ return 1; }

};


#endif
