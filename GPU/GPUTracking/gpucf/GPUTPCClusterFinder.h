// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinder.h
/// \author David Rohr

#ifndef O2_GPU_GPUTPCCLUSTERFINDER_H
#define O2_GPU_GPUTPCCLUSTERFINDER_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace gpucf
{
typedef struct PackedDigit_s PackedDigit;
typedef struct ClusterNative_s ClusterNative;
} // namespace gpucf

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCClusterFinder : public GPUProcessor
{
 public:
  struct Memory {
    size_t nDigits = 0;
    size_t nPeaks = 0;
    size_t nClusters = 0;
  };

#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersInput(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersMemory(void* mem);

  size_t getNSteps(size_t items) const;
#endif

  gpucf::PackedDigit* mPdigits = nullptr;
  gpucf::PackedDigit* mPpeaks = nullptr;
  gpucf::PackedDigit* mPfilteredPeaks = nullptr;
  unsigned char* mPisPeak = nullptr;
  ushort* mPchargeMap = nullptr;
  unsigned char* mPpeakMap = nullptr;
  uint* mPclusterInRow = nullptr;
  gpucf::ClusterNative* mPclusterByRow = nullptr;
  int* mPbuf = nullptr;
  Memory* mPmemory = nullptr;

  int mISlice = 0;
  constexpr static int mScanWorkGroupSize = GPUCA_THREAD_COUNT_SCAN;
  size_t mNMaxClusterPerRow = 0;
  size_t mNMaxDigits = 0;
  size_t mNMaxPeaks = 0;
  size_t mNMaxClusters = 0;
  size_t mBufSize = 0;
  size_t mNBufs = 0;
  
  unsigned short mMemoryId = 0;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
