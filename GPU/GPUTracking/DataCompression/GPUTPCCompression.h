// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompression.h
/// \author David Rohr

#ifndef GPUTPCCOMPRESSION_H
#define GPUTPCCOMPRESSION_H

#include "GPUDef.h"
#include "GPUProcessor.h"

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/CompressedClusters.h"
#else
namespace o2
{
namespace TPC
{
struct CompressedClusters {
};
} // namespace TPC
} // namespace o2
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
using CompressedClusters = o2::TPC::CompressedClusters;
struct ClusterNativeAccessExt;
class GPUTPCGMMerger;

class GPUTPCCompression : public GPUProcessor
{
  friend class GPUTPCCompressionKernel;
  friend class GPUChainTracking;

 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData();

  void* SetPointersOutputHost(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersMemory(void* mem);
  void SetPointersCompressedClusters(void*& mem, CompressedClusters& c, unsigned int nClA, unsigned int nTr, unsigned int nClU, bool reducedClA);
#endif

  struct memory {
    unsigned int nStoredTracks = 0;
    unsigned int nStoredAttachedClusters = 0;
    unsigned int nStoredUnattachedClusters = 0;
  };

  constexpr static unsigned int NSLICES = GPUCA_NSLICES;

  CompressedClusters mPtrs;
  CompressedClusters mOutput;
  const GPUTPCGMMerger* mMerger = nullptr;

  memory* mMemory = nullptr;
  unsigned int* mAttachedClusterFirstIndex = nullptr;

  unsigned char* mClusterStatus = nullptr;

  unsigned int mMaxTracks = 0;
  unsigned int mMaxClusters = 0;
  unsigned int mMaxTrackClusters = 0;

 protected:
  short mMemoryResOutput = -1;
  short mMemoryResOutputHost = -1;
  short mMemoryResMemory = -1;
  short mMemoryResScratch = -1;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
