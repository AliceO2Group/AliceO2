// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUOutputControl.h
/// \author David Rohr

#ifndef GPUOUTPUTCONTROL_H
#define GPUOUTPUTCONTROL_H

#include "GPUCommonDef.h"
#include <cstddef>
#include <functional>
#include <new>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// This defines an output region. ptrBase points to a memory buffer, which should have a proper alignment.
// Since DPL does not respect the alignment of data types, we do not impose anything specic but just use void*, but it should be >= 64 bytes ideally.
// The size defines the maximum possible buffer size when GPUReconstruction is called, and returns the number of filled bytes when it returns.
// If the buffer size is exceeded, size is set to 1
// ptrCurrent must equal ptr if set (or nullptr), and can be incremented by GPUReconstruction step by step if multiple buffers are used.
// If ptr == nullptr, there is no region defined and GPUReconstruction will write its output to an internal buffer.
// If allocator is set, it is called as a callback to provide a ptr to the memory.

struct GPUOutputControl {
  GPUOutputControl() = default;
  void set(void* p, size_t s)
  {
    reset();
    ptrBase = ptrCurrent = p;
    size = s;
  }
  void set(const std::function<void*(size_t)>& a)
  {
    reset();
    allocator = a;
  }
  void reset()
  {
    new (this) GPUOutputControl;
  }
  bool useExternal() { return size || allocator; }
  bool useInternal() { return !useExternal(); }
  void checkCurrent()
  {
    if (ptrBase && ptrCurrent == nullptr) {
      ptrCurrent = ptrBase;
    }
  }

  void* ptrBase = nullptr;                          // Base ptr to memory pool, occupied size is ptrCurrent - ptr
  void* ptrCurrent = nullptr;                       // Pointer to free Output Space
  size_t size = 0;                                  // Max Size of Output Data if Pointer to output space is given
  std::function<void*(size_t)> allocator = nullptr; // Allocator callback
};

struct GPUTrackingOutputs {
  GPUOutputControl compressedClusters;
  GPUOutputControl clustersNative;
  GPUOutputControl tpcTracks;
  GPUOutputControl clusterLabels;
  GPUOutputControl sharedClusterMap;
  GPUOutputControl tpcTracksO2;
  GPUOutputControl tpcTracksO2ClusRefs;
  GPUOutputControl tpcTracksO2Labels;

  static constexpr size_t count() { return sizeof(GPUTrackingOutputs) / sizeof(GPUOutputControl); }
  GPUOutputControl* asArray() { return (GPUOutputControl*)this; }
  size_t getIndex(const GPUOutputControl& v) { return &v - (const GPUOutputControl*)this; }
  static int getIndex(GPUOutputControl GPUTrackingOutputs::*v) { return &(((GPUTrackingOutputs*)(nullptr))->*v) - (GPUOutputControl*)(nullptr); }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
