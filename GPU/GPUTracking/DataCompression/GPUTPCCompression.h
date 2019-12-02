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
#include "GPUCommonMath.h"
#include "GPUParam.h"

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/CompressedClusters.h"
#else
namespace o2
{
namespace tpc
{
template <class T>
struct CompressedClustersPtrs_helper {
};
struct CompressedClustersCounters {
};
using CompressedClusters = CompressedClustersPtrs_helper<CompressedClustersCounters>;
struct CompressedClustersPtrsOnly {
};
} // namespace tpc
} // namespace o2
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMMerger;

class GPUTPCCompression : public GPUProcessor
{
  friend class GPUTPCCompressionKernels;
  friend class GPUChainTracking;

 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersOutputHost(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersMemory(void* mem);
#endif

  static constexpr unsigned int P_MAX_QMAX = 1 << 10;
  static constexpr unsigned int P_MAX_QTOT = 5 * 5 * P_MAX_QMAX;
  static constexpr unsigned int P_MAX_TIME = 1 << 24;
  static constexpr unsigned int P_MAX_PAD = 1 << 16;
  static constexpr unsigned int P_MAX_SIGMA = 1 << 8;
  static constexpr unsigned int P_MAX_FLAGS = 1 << 8;
  static constexpr unsigned int P_MAX_QPT = 1 << 8;

  GPUd() static void truncateSignificantBitsCharge(unsigned short& charge, const GPUParam& param) { truncateSignificantBits(charge, param.rec.tpcSigBitsCharge, P_MAX_QTOT); }
  GPUd() static void truncateSignificantBitsChargeMax(unsigned short& charge, const GPUParam& param) { truncateSignificantBits(charge, param.rec.tpcSigBitsCharge, P_MAX_QMAX); }
  GPUd() static void truncateSignificantBitsWidth(unsigned char& width, const GPUParam& param) { truncateSignificantBits(width, param.rec.tpcSigBitsWidth, P_MAX_SIGMA); }

 protected:
  struct memory {
    unsigned int nStoredTracks = 0;
    unsigned int nStoredAttachedClusters = 0;
    unsigned int nStoredUnattachedClusters = 0;
  };

  constexpr static unsigned int NSLICES = GPUCA_NSLICES;

  o2::tpc::CompressedClustersPtrsOnly mPtrs;
  o2::tpc::CompressedClusters mOutput;
  const GPUTPCGMMerger* mMerger = nullptr;

  memory* mMemory = nullptr;
  unsigned int* mAttachedClusterFirstIndex = nullptr;
  unsigned int* mClusterSortBuffer = nullptr;
  unsigned char* mClusterStatus = nullptr;

  unsigned int mMaxTracks = 0;
  unsigned int mMaxClusters = 0;
  unsigned int mMaxTrackClusters = 0;
  unsigned int mNMaxClusterSliceRow = 0;
  unsigned int mNGPUBlocks = 0;

  template <class T>
  void SetPointersCompressedClusters(void*& mem, T& c, unsigned int nClA, unsigned int nTr, unsigned int nClU, bool reducedClA);
  template <class T>
  GPUd() static void truncateSignificantBits(T& val, unsigned int nBits, unsigned int max);

  short mMemoryResOutput = -1;
  short mMemoryResOutputHost = -1;
  short mMemoryResMemory = -1;
  short mMemoryResScratch = -1;
};

template <class T>
GPUdi() void GPUTPCCompression::truncateSignificantBits(T& v, unsigned int nBits, unsigned int max)
{
  if (nBits == 0) {
    return;
  }

  unsigned int val = v;
  unsigned int ldz = sizeof(unsigned int) * 8 - CAMath::Clz(val);
  if (val && ldz > nBits) {
    if (val & (1 << (ldz - nBits - 1))) {
      val += (1 << (ldz - nBits - 1));
      ldz = sizeof(unsigned int) * 8 - CAMath::Clz(val);
    }
    val &= ((1 << ldz) - 1) ^ ((1 << (ldz - nBits)) - 1);
    if (val >= max) {
      val = max - 1;
    }
    // GPUInfo("CHANGING X %x --> %x", (unsigned int) v, val);
    v = val;
  }
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
