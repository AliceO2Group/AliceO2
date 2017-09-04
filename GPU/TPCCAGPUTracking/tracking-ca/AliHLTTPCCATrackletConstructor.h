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
#endif //HLTCA_GPUCODE

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"
#include "AliHLTTPCCATrackParam.h"

/**
* @class AliHLTTPCCATrackletConstructor
*
*/
MEM_CLASS_PRE() class AliHLTTPCCATracker;

class AliHLTTPCCATrackletConstructor
{
public:

	class  AliHLTTPCCAThreadMemory
	{
		friend class AliHLTTPCCATrackletConstructor; //! friend class
	public:
#if !defined(HLTCA_GPUCODE)
		AliHLTTPCCAThreadMemory()
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}

		AliHLTTPCCAThreadMemory( const AliHLTTPCCAThreadMemory& /*dummy*/ )
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}
		AliHLTTPCCAThreadMemory& operator=( const AliHLTTPCCAThreadMemory& /*dummy*/ ) { return *this; }
#endif //!HLTCA_GPUCODE

	protected:
		//WARNING: This data is copied element by element in CopyTrackletTempData. Changes to members of this class must be reflected in CopyTrackletTempData!!!
		int fItr; // track index
		int fFirstRow;  // first row index
		int fLastRow; // last row index
		int fStartRow;  // first row index
		int fEndRow;  // first row index
		int fCurrIH; // indef of the current hit
		bool fGo; // do fit/searching flag
		int fStage; // reco stage
		int fNHits; // n track hits
		int fNMissed; // n missed hits during search
		float fLastY; // Y of the last fitted cluster
		float fLastZ; // Z of the last fitted cluster
	};

	MEM_CLASS_PRE() class AliHLTTPCCASharedMemory
	{
		friend class AliHLTTPCCATrackletConstructor; // friend class
	public:
#if !defined(HLTCA_GPUCODE)
		AliHLTTPCCASharedMemory() : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ ) : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif //HLTCA_GPUCODE

	protected:
		MEM_LG(AliHLTTPCCARow) fRows[HLTCA_ROW_COUNT]; // rows
		int fNextTrackletFirst; //First tracklet to be processed by CUDA block during next iteration
		int fNextTrackletCount; //Number of Tracklets to be processed by CUDA block during next iteration
		int fNextTrackletFirstRun; //First run for dynamic scheduler?
		int fNTracklets; // Total number of tracklets

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
		int fMaxSync; //temporary shared variable during profile creation
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	};

	MEM_CLASS_PRE2() GPUd() static void InitTracklet( register MEM_LG2(AliHLTTPCCATrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static void UpdateTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() AliHLTTPCCASharedMemory) &s, AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, MEM_LG2(AliHLTTPCCATrackParam) &tParam, int iRow );

	MEM_CLASS_PRE23() GPUd() static void StoreTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() AliHLTTPCCASharedMemory) &s, AliHLTTPCCAThreadMemory &r, GPUconstant() MEM_LG2(AliHLTTPCCATracker) &tracker, MEM_LG3(AliHLTTPCCATrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static bool CheckCov(register MEM_LG2(AliHLTTPCCATrackParam) &tParam);

#ifdef HLTCA_GPUCODE
	GPUd() static void AliHLTTPCCATrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem);
	GPUd() static void AliHLTTPCCATrackletConstructorSingleSlice(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem);
	MEM_CLASS_PRE2() GPUd() static void AliHLTTPCCATrackletConstructorGPUPP(MEM_LG2(AliHLTTPCCATracker) *pTracker);
	GPUd() static int FetchTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &sMem, AliHLTTPCCAThreadMemory &rMem, MEM_PLAIN(AliHLTTPCCATrackParam) &tParam);
	MEM_TEMPLATE4() GPUd() static void CopyTrackletTempData( MEM_TYPE(AliHLTTPCCAThreadMemory) &rMemSrc, MEM_TYPE2(AliHLTTPCCAThreadMemory) &rMemDst, MEM_TYPE3(AliHLTTPCCATrackParam) &tParamSrc, MEM_TYPE4(AliHLTTPCCATrackParam) &tParamDst);
#else
	GPUd() static void AliHLTTPCCATrackletConstructorCPU(AliHLTTPCCATracker &tracker);
	GPUd() static int AliHLTTPCCATrackletConstructorGlobalTracking(AliHLTTPCCATracker &tracker, AliHLTTPCCATrackParam& tParam, int startrow, int increment);
#endif //HLTCA_GPUCODE

	GPUd() static bool SAVE() { return 1; }
};

#endif //ALIHLTTPCCATRACKLETCONSTRUCTOR_H
