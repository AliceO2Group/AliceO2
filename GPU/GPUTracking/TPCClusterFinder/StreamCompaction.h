// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file StreamCompaction.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_STREAM_COMPACTION_H
#define O2_GPU_STREAM_COMPACTION_H

#include "clusterFinderDefs.h"
#include "GPUTPCClusterFinderKernels.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class StreamCompaction
{

 public:
  static GPUd() void nativeScanUpStartImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&,
                                           GPUglobalref() const uchar*, GPUglobalref() int*, GPUglobalref() int*,
                                           int);

  static GPUd() void nativeScanUpImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&,
                                      GPUglobalref() int*, GPUglobalref() int*, int);

  static GPUd() void nativeScanTopImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&,
                                       GPUglobalref() int*, int);

  static GPUd() void nativeScanDownImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&,
                                        GPUglobalref() int*, GPUglobalref() const int*, unsigned int, int);

  static GPUd() void compactDigitImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&,
                                      GPUglobalref() const deprecated::Digit*, GPUglobalref() deprecated::Digit*,
                                      GPUglobalref() const uchar*, GPUglobalref() int*, const GPUglobalref() int*,
                                      int);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
