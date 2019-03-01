// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTrackletConstructor.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCTRACKLETCONSTRUCTOR_H
#define GPUTPCTRACKLETCONSTRUCTOR_H

#include "GPUTPCDef.h"
#include "GPUTPCGPUConfig.h"
#include "GPUTPCTrackParam.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
/**
* @class GPUTPCTrackletConstructor
*
*/
MEM_CLASS_PRE() class GPUTPCTracker;

class GPUTPCTrackletConstructor
{
public:

	class  GPUTPCThreadMemory
	{
		friend class GPUTPCTrackletConstructor; //! friend class
	public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCThreadMemory()
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNHitsEndRow( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}

		GPUTPCThreadMemory( const GPUTPCThreadMemory& /*dummy*/ )
			: fItr( 0 ), fFirstRow( 0 ), fLastRow( 0 ), fStartRow( 0 ), fEndRow( 0 ), fCurrIH( 0 ), fGo( 0 ), fStage( 0 ), fNHits( 0 ), fNHitsEndRow( 0 ), fNMissed( 0 ), fLastY( 0 ), fLastZ( 0 )
		{}
		GPUTPCThreadMemory& operator=( const GPUTPCThreadMemory& /*dummy*/ ) { return *this; }
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

	MEM_CLASS_PRE() class GPUTPCSharedMemory
	{
		friend class GPUTPCTrackletConstructor; // friend class
	public:
#if !defined(GPUCA_GPUCODE)
		GPUTPCSharedMemory() : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		GPUTPCSharedMemory( const GPUTPCSharedMemory& /*dummy*/ ) : fNextTrackletFirst(0), fNextTrackletCount(0), fNextTrackletFirstRun(0), fNTracklets(0) {
		}

		GPUTPCSharedMemory& operator=( const GPUTPCSharedMemory& /*dummy*/ ) { return *this; }
#endif //GPUCA_GPUCODE

	protected:
		MEM_LG(GPUTPCRow) fRows[GPUCA_ROW_COUNT]; // rows
		int fNextTrackletFirst; //First tracklet to be processed by CUDA block during next iteration
		int fNextTrackletCount; //Number of Tracklets to be processed by CUDA block during next iteration
		int fNextTrackletFirstRun; //First run for dynamic scheduler?
		int fNTracklets; // Total number of tracklets

#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
		int fMaxSync; //temporary shared variable during profile creation
#endif //GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
	};

	MEM_CLASS_PRE2() GPUd() static void InitTracklet( MEM_LG2(GPUTPCTrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static void UpdateTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() GPUTPCSharedMemory) &s, GPUTPCThreadMemory &r, GPUconstantref() MEM_CONSTANT(GPUTPCTracker) &tracker, MEM_LG2(GPUTPCTrackParam) &tParam, int iRow );

	MEM_CLASS_PRE23() GPUd() static void StoreTracklet
		( int nBlocks, int nThreads, int iBlock, int iThread,
		  MEM_LOCAL(GPUsharedref() GPUTPCSharedMemory) &s, GPUTPCThreadMemory &r, GPUconstantref() MEM_LG2(GPUTPCTracker) &tracker, MEM_LG3(GPUTPCTrackParam) &tParam );

	MEM_CLASS_PRE2() GPUd() static bool CheckCov( MEM_LG2(GPUTPCTrackParam) &tParam );

	GPUd() static void DoTracklet(GPUconstantref() MEM_CONSTANT(GPUTPCTracker)& tracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUTPCSharedMemory)& sMem, GPUTPCThreadMemory& rMem);

	GPUd() static int FetchTracklet(GPUconstantref() MEM_CONSTANT(GPUTPCTracker) &tracker, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &sMem);
#ifdef GPUCA_GPUCODE
	GPUd() static void GPUTPCTrackletConstructorGPU(GPUconstantref() MEM_CONSTANT(GPUTPCTracker) *pTracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUTPCSharedMemory)& sMem);
#else
	GPUd() static void GPUTPCTrackletConstructorCPU(GPUTPCTracker &tracker);
	static int GPUTPCTrackletConstructorGlobalTracking(GPUTPCTracker &tracker, GPUTPCTrackParam& tParam, int startrow, int increment, int iTracklet);
#endif //GPUCA_GPUCODE

	typedef GPUconstantref() MEM_CONSTANT(GPUTPCTracker) workerType;
	GPUhdi() static GPUDataTypes::RecoStep GetRecoStep() {return GPUCA_RECO_STEP::TPCSliceTracking;}
	MEM_TEMPLATE() GPUhdi() static workerType* Worker(MEM_TYPE(GPUConstantMem) &workers) {return workers.tpcTrackers;}
	template <int iKernel = 0> GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &smem, workerType &tracker);

};

#endif //GPUTPCTRACKLETCONSTRUCTOR_H
