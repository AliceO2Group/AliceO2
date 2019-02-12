//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLETCONSTRUCTOR_H
#define ALIHLTTPCCATRACKLETCONSTRUCTOR_H

#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"
#include "AliGPUTPCTrackParam.h"
#include "AliGPUConstantMem.h"
/**
* @class AliGPUTPCTrackletConstructor
*
*/
MEM_CLASS_PRE() class AliGPUTPCTracker;

class AliGPUTPCTrackletConstructor
{
public:

	class  AliGPUTPCThreadMemory
	{
		friend class AliGPUTPCTrackletConstructor; //! friend class
	public:
#if !defined(GPUCA_GPUCODE)
		AliGPUTPCThreadMemory()
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNHitsEndRow( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}

		AliGPUTPCThreadMemory( const AliGPUTPCThreadMemory& /*dummy*/ )
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNHitsEndRow( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}
		AliGPUTPCThreadMemory& operator=( const AliGPUTPCThreadMemory& /*dummy*/ ) { return *this; }
#endif //!GPUCA_GPUCODE

	protected:
		//WARNING: This data is copied element by element in CopyTrackletTempData. Changes to members of this class must be reflected in CopyTrackletTempData!!!
		int fItr; // track index
		int fFirstRow;  // first row index
		int fLastRow; // last row index
		int fStartRow;  // first row index
		int fEndRow;  // first row index
		calink fCurrIH; // indef of the current hit
		char fGo; // do fit/searching flag
		int fStage; // reco stage
		int fNHits; // n track hits
		int fNHitsEndRow; // n hits at end row
		int fNMissed; // n missed hits during search
		float fLastY; // Y of the last fitted cluster
		float fLastZ; // Z of the last fitted cluster
	};

	MEM_CLASS_PRE() class AliGPUTPCSharedMemory
	{
		friend class AliGPUTPCTrackletConstructor; // friend class
	public:
#if !defined(GPUCA_GPUCODE)
		AliGPUTPCSharedMemory() : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		AliGPUTPCSharedMemory( const AliGPUTPCSharedMemory& /*dummy*/ ) : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		AliGPUTPCSharedMemory& operator=( const AliGPUTPCSharedMemory& /*dummy*/ ) { return *this; }
#endif //GPUCA_GPUCODE

	protected:
		MEM_LG(AliGPUTPCRow) fRows[GPUCA_ROW_COUNT]; // rows
		int fNextTrackletFirst; //First tracklet to be processed by CUDA block during next iteration
		int fNextTrackletCount; //Number of Tracklets to be processed by CUDA block during next iteration
		int fNextTrackletFirstRun; //First run for dynamic scheduler?
		int fNTracklets; // Total number of tracklets

#ifdef GPUCA_GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
		int fMaxSync; //temporary shared variable during profile creation
#endif //GPUCA_GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
	};

	MEM_CLASS_PRE2() GPUd() static void InitTracklet( MEM_LG2(AliGPUTPCTrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static void UpdateTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() AliGPUTPCSharedMemory) &s, AliGPUTPCThreadMemory &r, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, MEM_LG2(AliGPUTPCTrackParam) &tParam, int iRow );

	MEM_CLASS_PRE23() GPUd() static void StoreTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() AliGPUTPCSharedMemory) &s, AliGPUTPCThreadMemory &r, GPUconstant() MEM_LG2(AliGPUTPCTracker) &tracker, MEM_LG3(AliGPUTPCTrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static bool CheckCov( MEM_LG2(AliGPUTPCTrackParam) &tParam );

	GPUd() static void DoTracklet(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker)& tracker, GPUsharedref() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory)& sMem, AliGPUTPCThreadMemory& rMem);

	GPUd() static int FetchTracklet(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &sMem);
#ifdef GPUCA_GPUCODE
	GPUd() static void AliGPUTPCTrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker, GPUsharedref() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory)& sMem);
#else
	GPUd() static void AliGPUTPCTrackletConstructorCPU(AliGPUTPCTracker &tracker);
	static int AliGPUTPCTrackletConstructorGlobalTracking(AliGPUTPCTracker &tracker, AliGPUTPCTrackParam& tParam, int startrow, int increment, int iTracklet);
#endif //GPUCA_GPUCODE

	typedef GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) workerType;
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(AliGPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &smem, workerType &tracker);

};

#endif //ALIHLTTPCCATRACKLETCONSTRUCTOR_H
