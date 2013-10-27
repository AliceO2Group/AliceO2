//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliHLTTPCCAGPUTrackerOpenCL.h
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 


#ifndef ALIHLTTPCCAGPUTRACKEROPENCL_H
#define ALIHLTTPCCAGPUTRACKEROPENCL_H

#include "AliHLTTPCCAGPUTrackerBase.h"

struct AliHLTTPCCAGPUTrackerOpenCLInternals;

class AliHLTTPCCAGPUTrackerOpenCL : public AliHLTTPCCAGPUTrackerBase
{
public:
	AliHLTTPCCAGPUTrackerOpenCL();
	virtual ~AliHLTTPCCAGPUTrackerOpenCL();

	virtual int InitGPU_Runtime(int sliceCount = -1, int forceDeviceID = -1);
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU_Runtime();
	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger);

protected:
	virtual void ActivateThreadContext();
	virtual void ReleaseThreadContext();
	virtual void SynchronizeGPU();
	virtual int GPUSync(char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0);

private:
	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);
	bool GPUFailedMsgA(int, const char* file, int line);
	AliHLTTPCCAGPUTrackerOpenCLInternals* ocl;


	// disable copy
	AliHLTTPCCAGPUTrackerOpenCL( const AliHLTTPCCAGPUTrackerOpenCL& );
	AliHLTTPCCAGPUTrackerOpenCL &operator=( const AliHLTTPCCAGPUTrackerOpenCL& );

	ClassDef( AliHLTTPCCAGPUTrackerOpenCL, 0 )
};

#ifdef R__WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" DLL_EXPORT AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate();
extern "C" DLL_EXPORT void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr);

#endif //ALIHLTTPCCAGPUTRACKER_H
