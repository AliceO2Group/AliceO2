//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLETCONSTRUCTOR_H
#define ALIHLTTPCCATRACKLETCONSTRUCTOR_H

#ifdef HLTCA_GPUCODE
#define HLTCA_GPU_USE_INT short
#else
#define HLTCA_GPU_USE_INT int
#endif

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"
#include "AliHLTTPCCATrackParam.h"

/**
 * @class AliHLTTPCCATrackletConstructor
 *
 */
class AliHLTTPCCATracker;

class AliHLTTPCCATrackletConstructor
{
  public:

    class   AliHLTTPCCASharedMemory
    {
        friend class AliHLTTPCCATrackletConstructor;
      public:
#if !defined(HLTCA_GPUCODE)
        AliHLTTPCCASharedMemory()
            : fItr0( 0 ), fItr1( 0 ), fNRows( 0 ), fMinStartRow( 0 ), fMaxEndRow( 0 ) {}

        AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ )
            : fItr0( 0 ), fItr1( 0 ), fNRows( 0 ), fMinStartRow( 0 ), fMaxEndRow( 0 ) {}
        AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif

#ifndef CUDA_DEVICE_EMULATION
      protected:
#endif

#ifdef HLTCA_GPU_PREFETCHDATA
        uint4 fData[2][ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM / 4]; // temp memory
		AliHLTTPCCARow fRow[2]; // row
#endif
        int fItr0; // start track index
        int fItr1; // end track index
        int fNRows; // n rows
        int fMinStartRow; // min start row
        int fMinStartRow32[32]; // min start row for each thread in warp
        int fMaxEndRow; // max start row
		int fNextTrackletFirst;
		int fNextTrackletCount;
		int fNextTrackletNoDummy;
		int fNextTrackletStupidDummy;
		int fNextTrackletFirstRun;

		int fTrackletStoreCount[2][HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1];
    };

    class  AliHLTTPCCAThreadMemory
    {
        friend class AliHLTTPCCATrackletConstructor;
      public:
#if !defined(HLTCA_GPUCODE)
        AliHLTTPCCAThreadMemory()
            : fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fIsMemThread( 0 ), fGo( 0 ), fSave( 0 ), fCurrentData( 0 ), fStage( 0 ), fNHits( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 ) {}

        AliHLTTPCCAThreadMemory( const AliHLTTPCCAThreadMemory& /*dummy*/ )
            : fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fIsMemThread( 0 ), fGo( 0 ), fSave( 0 ), fCurrentData( 0 ), fStage( 0 ), fNHits( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 ) {}
        AliHLTTPCCAThreadMemory& operator=( const AliHLTTPCCAThreadMemory& /*dummy*/ ) { return *this; }
#endif

#ifndef CUDA_DEVICE_EMULATION
      protected:
#endif
        int fItr; // track index
        int fFirstRow;  // first row index
        int fLastRow; // last row index
        int fStartRow;  // first row index
        int fEndRow;  // first row index
        int fCurrIH; // indef of the current hit
        bool fIsMemThread; // is the thread used for memory taken
        bool fGo; // do fit/searching flag
        bool fSave; // save flag
        bool fCurrentData; // index of the current memory array
        int fStage; // reco stage
        int fNHits; // n track hits
        int fNMissed; // n missed hits during search
        float fLastY; // Y of the last fitted cluster
        float fLastZ; // Z of the last fitted cluster
    };

	struct AliHLTTPCCAGPUTempMemory
	{
		AliHLTTPCCAThreadMemory fThreadMem;
		AliHLTTPCCATrackParam fParam;
	};

    GPUd() static int NThreadSyncPoints() { return 4 + 159*2 + 1 + 1; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread,
                               int iSync, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r,
                               AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );

	GPUd() static void InitTracklet	( AliHLTTPCCATrackParam &tParam );
	GPUd() static void CopyTrackletTempData( AliHLTTPCCAThreadMemory &rMemSrc, AliHLTTPCCAThreadMemory &rMemDst, AliHLTTPCCATrackParam &tParamSrc, AliHLTTPCCATrackParam &tParamDst);

    GPUd() static void Step0
    ( int nBlocks, int nThreads, int iBlock, int iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );
    GPUd() static void Step1
    ( int nBlocks, int nThreads, int iBlock, int iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );
    GPUd() static void Step2
    ( int nBlocks, int nThreads, int iBlock, int iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );

    GPUd() static void ReadData( int iThread, AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, int iRow );

    GPUd() static void UpdateTracklet
    ( int nBlocks, int nThreads, int iBlock, int iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam, int iRow );

    GPUd() static void StoreTracklet
    ( int nBlocks, int nThreads, int iBlock, int iThread,
      AliHLTTPCCASharedMemory &s, AliHLTTPCCAThreadMemory &r, AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam &tParam );

#ifdef HLTCA_GPUCODE
	GPUd() static void AliHLTTPCCATrackletConstructorNew();
	GPUd() static int FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem, int Reverse, int RowBlock);
	GPUd() static void AliHLTTPCCATrackletConstructorInit(int iTracklet, AliHLTTPCCATracker &tracke);
#endif

    GPUd() static bool SAVE() { return 1; }

#if defined(HLTCA_GPUCODE)
    //GPUhd() inline int NMemThreads() { return 128; }
#define TRACKLET_CONSTRUCTOR_NMEMTHREDS HLTCA_GPU_TRACKLET_CONSTRUCTOR_NMEMTHREDS
#else
    //GPUhd() inline int NMemThreads() { return 1; }
#define TRACKLET_CONSTRUCTOR_NMEMTHREDS 1
#endif

};



#endif
