// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutput.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCSLICEOUTPUT_H
#define GPUTPCSLICEOUTPUT_H

#include "GPUTPCDef.h"
#include "GPUTPCTrack.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
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
  GPUhd() unsigned int NTracks() const
  {
    return mNTracks;
  }
  GPUhd() unsigned int NLocalTracks() const { return mNLocalTracks; }
  GPUhd() unsigned int NTrackClusters() const { return mNTrackClusters; }
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  GPUhd() const GPUTPCTrack* GetFirstTrack() const
  {
    return (const GPUTPCTrack*)((const char*)this + sizeof(*this));
  }
  GPUhd() GPUTPCTrack* FirstTrack()
  {
    return (GPUTPCTrack*)((char*)this + sizeof(*this));
  }
#endif
  GPUhd() size_t Size() const
  {
    return (mMemorySize);
  }

  static unsigned int EstimateSize(unsigned int nOfTracks, unsigned int nOfTrackClusters);
  static void Allocate(GPUTPCSliceOutput*& ptrOutput, int nTracks, int nTrackHits, GPUOutputControl* outputControl, void*& internalMemory);

  GPUhd() void SetNTracks(unsigned int v) { mNTracks = v; }
  GPUhd() void SetNLocalTracks(unsigned int v) { mNLocalTracks = v; }
  GPUhd() void SetNTrackClusters(unsigned int v) { mNTrackClusters = v; }

 private:
  GPUTPCSliceOutput() CON_DELETE;                                    // NOLINT: Must be private or ROOT tries to use them!
  ~GPUTPCSliceOutput() CON_DELETE;                                   // NOLINT
  GPUTPCSliceOutput(const GPUTPCSliceOutput&) CON_DELETE;            // NOLINT
  GPUTPCSliceOutput& operator=(const GPUTPCSliceOutput&) CON_DELETE; // NOLINT

  GPUhd() void SetMemorySize(size_t val) { mMemorySize = val; }

  unsigned int mNTracks; // number of reconstructed tracks
  unsigned int mNLocalTracks;
  unsigned int mNTrackClusters; // total number of track clusters
  size_t mMemorySize;           // Amount of memory really used
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif
