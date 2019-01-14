//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASLICEOUTPUT_H
#define ALIHLTTPCCASLICEOUTPUT_H

#include "AliGPUTPCDef.h"

#if !defined(__OPENCL__)

#include <cstdlib>
#ifndef GPUCA_GPUCODE
#include "AliGPUTPCSliceOutTrack.h"
#else
class AliGPUTPCSliceOutTrack;
#endif
#else
#define NULL 0
#endif

struct AliGPUCAOutputControl;

/**
 * @class AliGPUTPCSliceOutput
 *
 * AliGPUTPCSliceOutput class is used to store the output of AliGPUTPCTracker{Component}
 * and transport the output to AliGPUTPCGBMerger{Component}
 *
 * The class contains all the necessary information about TPC tracks, reconstructed in one slice.
 * This includes the reconstructed track parameters and some compressed information
 * about the assigned clusters: clusterId, position and amplitude.
 *
 */
class AliGPUTPCSliceOutput
{
  public:
#if !defined(__OPENCL__)
	GPUhd() int NTracks() const
	{
		return fNTracks;
	}
	GPUhd() int NLocalTracks() const { return fNLocalTracks; }
	GPUhd() int NTrackClusters() const { return fNTrackClusters; }
#ifndef GPUCA_GPUCODE
	GPUhd() const AliGPUTPCSliceOutTrack *GetFirstTrack() const
	{
		return fMemory;
	}
	GPUhd() AliGPUTPCSliceOutTrack *FirstTrack() { return fMemory; }
#endif
	GPUhd() size_t Size() const
	{
		return (fMemorySize);
	}

	static int EstimateSize(int nOfTracks, int nOfTrackClusters);
	static void Allocate(AliGPUTPCSliceOutput* &ptrOutput, int nTracks, int nTrackHits, AliGPUCAOutputControl *outputControl, void* &internalMemory);

	GPUhd() void SetNTracks(int v) { fNTracks = v; }
	GPUhd() void SetNLocalTracks(int v) { fNLocalTracks = v; }
	GPUhd() void SetNTrackClusters(int v) { fNTrackClusters = v; }

  private:
	AliGPUTPCSliceOutput()
	    : fNTracks(0), fNLocalTracks(0), fNTrackClusters(0), fMemorySize(0) {}

	~AliGPUTPCSliceOutput() {}
	AliGPUTPCSliceOutput(const AliGPUTPCSliceOutput &);
	AliGPUTPCSliceOutput &operator=(const AliGPUTPCSliceOutput &) { return *this; }

	GPUh() void SetMemorySize(size_t val) { fMemorySize = val; }

	int fNTracks; // number of reconstructed tracks
	int fNLocalTracks;
	int fNTrackClusters; // total number of track clusters
	size_t fMemorySize;  // Amount of memory really used

	//Must be last element of this class, user has to make sure to allocate anough memory consecutive to class memory!
	//This way the whole Slice Output is one consecutive Memory Segment

#ifndef GPUCA_GPUCODE
	AliGPUTPCSliceOutTrack fMemory[0]; // the memory where the pointers above point into
#endif
#endif
};
#endif
