// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutput.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCSLICEOUTPUT_H
#define GPUTPCSLICEOUTPUT_H

#include "GPUTPCDef.h"

#if !defined(GPUCA_GPUCODE_DEVICE) // clang-format off
  #include <cstdlib>
  #ifndef GPUCA_GPUCODE
    #include "GPUTPCSliceOutTrack.h"
  #else
    class GPUTPCSliceOutTrack;
  #endif
#elif defined(__OPENCL__) && !defined(__OPENCLCPP__)
  #define NULL 0
#endif // clang-format on

struct GPUOutputControl;

/**
 * @class GPUTPCSliceOutput
 *
 * GPUTPCSliceOutput class is used to store the output of GPUTPCTracker{Component}
 * and transport the output to GPUTPCGBMerger{Component}
 *
 * The class contains all the necessary information about TPC tracks, reconstructed in one slice.
 * This includes the reconstructed track parameters and some compressed information
 * about the assigned clusters: clusterId, position and amplitude.
 *
 */
class GPUTPCSliceOutput
{
  public:
#if !defined(GPUCA_GPUCODE_DEVICE)
	GPUhd() unsigned int NTracks() const
	{
		return fNTracks;
	}
	GPUhd() unsigned int NLocalTracks() const { return fNLocalTracks; }
	GPUhd() unsigned int NTrackClusters() const { return fNTrackClusters; }
#ifndef GPUCA_GPUCODE
	GPUhd() const GPUTPCSliceOutTrack *GetFirstTrack() const
	{
		return fMemory;
	}
	GPUhd() GPUTPCSliceOutTrack *FirstTrack() { return fMemory; }
#endif
	GPUhd() size_t Size() const
	{
		return (fMemorySize);
	}

	static unsigned int EstimateSize(unsigned int nOfTracks, unsigned int nOfTrackClusters);
	static void Allocate(GPUTPCSliceOutput* &ptrOutput, int nTracks, int nTrackHits, GPUOutputControl *outputControl, void* &internalMemory);

	GPUhd() void SetNTracks(unsigned int v) { fNTracks = v; }
	GPUhd() void SetNLocalTracks(unsigned int v) { fNLocalTracks = v; }
	GPUhd() void SetNTrackClusters(unsigned int v) { fNTrackClusters = v; }

  private:
	GPUTPCSliceOutput()
	    : fNTracks(0), fNLocalTracks(0), fNTrackClusters(0), fMemorySize(0) {}

	~GPUTPCSliceOutput() {}
	GPUTPCSliceOutput(const GPUTPCSliceOutput &);
	GPUTPCSliceOutput &operator=(const GPUTPCSliceOutput &) { return *this; }

	GPUh() void SetMemorySize(size_t val) { fMemorySize = val; }

	unsigned int fNTracks; // number of reconstructed tracks
	unsigned int fNLocalTracks;
	unsigned int fNTrackClusters; // total number of track clusters
	size_t fMemorySize;  // Amount of memory really used

	//Must be last element of this class, user has to make sure to allocate anough memory consecutive to class memory!
	//This way the whole Slice Output is one consecutive Memory Segment

#ifndef GPUCA_GPUCODE
	GPUTPCSliceOutTrack fMemory[0]; // the memory where the pointers above point into
#endif
#endif
};
#endif
