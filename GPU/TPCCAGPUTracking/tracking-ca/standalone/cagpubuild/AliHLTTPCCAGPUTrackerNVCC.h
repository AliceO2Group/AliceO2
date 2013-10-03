//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliHLTTPCCAGPUTrackerNVCC.h
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 


#ifndef ALIHLTTPCCAGPUTRACKERNVCC_H
#define ALIHLTTPCCAGPUTRACKERNVCC_H

#include "AliHLTTPCCAGPUTrackerBase.h"

class AliHLTTPCCAGPUTrackerNVCC : public AliHLTTPCCAGPUTrackerBase
{
public:
	AliHLTTPCCAGPUTrackerNVCC();
	virtual ~AliHLTTPCCAGPUTrackerNVCC();

	virtual int InitGPU_Runtime(int sliceCount = -1, int forceDeviceID = -1);
	virtual int Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int fFirstSlice, int fSliceCount = -1);
	virtual int ExitGPU_Runtime();
	virtual int RefitMergedTracks(AliHLTTPCGMMerger* Merger);
	virtual int GPUMergerAvailable();

protected:
	virtual void ActivateThreadContext();
	virtual void ReleaseThreadContext();
	virtual void SynchronizeGPU();
	virtual int GPUSync(char* state = "UNKNOWN", int sliceLocal = 0, int slice = 0);

private:
	void DumpRowBlocks(AliHLTTPCCATracker* tracker, int iSlice, bool check = true);
	void* fCudaContext; //Pointer to CUDA context
	bool GPUFailedMsgA(cudaError_t error, const char* file, int line);

	void* fpCudaStreams; //Pointer to array of CUDA Streams

	// disable copy
	AliHLTTPCCAGPUTrackerNVCC( const AliHLTTPCCAGPUTrackerNVCC& );
	AliHLTTPCCAGPUTrackerNVCC &operator=( const AliHLTTPCCAGPUTrackerNVCC& );

	ClassDef( AliHLTTPCCAGPUTrackerNVCC, 0 )
};

#ifdef R__WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" DLL_EXPORT AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate();
extern "C" DLL_EXPORT void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr);

#endif //ALIHLTTPCCAGPUTRACKER_H
