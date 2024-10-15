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

/// \file GPUTPCCompression.h
/// \author David Rohr

#ifndef GPUTPCCOMPRESSION_H
#define GPUTPCCOMPRESSION_H

#include "GPUDef.h"
#include "GPUProcessor.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"

#ifdef GPUCA_HAVE_O2HEADERS
#include "DataFormatsTPC/CompressedClusters.h"
#else
namespace o2::tpc
{
struct CompressedClustersPtrs {
};
struct CompressedClusters {
};
struct CompressedClustersFlat {
};
} // namespace o2::tpc
#endif

namespace GPUCA_NAMESPACE::gpu
{
class GPUTPCGMMerger;

class GPUTPCCompression : public GPUProcessor
{
  friend class GPUTPCCompressionKernels;
  friend class GPUTPCCompressionGatherKernels;
  friend class GPUChainTracking;

 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersOutputGPU(void* mem);
  void* SetPointersOutputHost(void* mem);
  void* SetPointersOutputPtrs(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersMemory(void* mem);
#endif

  static constexpr uint32_t P_MAX_QMAX = 1 << 10;
  static constexpr uint32_t P_MAX_QTOT = 5 * 5 * P_MAX_QMAX;
  static constexpr uint32_t P_MAX_TIME = 1 << 24;
  static constexpr uint32_t P_MAX_PAD = 1 << 16;
  static constexpr uint32_t P_MAX_SIGMA = 1 << 8;
  static constexpr uint32_t P_MAX_FLAGS = 1 << 8;
  static constexpr uint32_t P_MAX_QPT = 1 << 8;

  GPUd() static void truncateSignificantBitsCharge(uint16_t& charge, const GPUParam& param) { truncateSignificantBits(charge, param.rec.tpc.sigBitsCharge, P_MAX_QTOT); }
  GPUd() static void truncateSignificantBitsChargeMax(uint16_t& charge, const GPUParam& param) { truncateSignificantBits(charge, param.rec.tpc.sigBitsCharge, P_MAX_QMAX); }
  GPUd() static void truncateSignificantBitsWidth(uint8_t& width, const GPUParam& param) { truncateSignificantBits(width, param.rec.tpc.sigBitsWidth, P_MAX_SIGMA); }

 protected:
  struct memory {
    uint32_t nStoredTracks = 0;
    uint32_t nStoredAttachedClusters = 0;
    uint32_t nStoredUnattachedClusters = 0;
  };

  constexpr static uint32_t NSLICES = GPUCA_NSLICES;

  o2::tpc::CompressedClustersPtrs mPtrs;
  o2::tpc::CompressedClusters* mOutput = nullptr;
  o2::tpc::CompressedClusters* mOutputA = nullptr; // Always points to host buffer
  o2::tpc::CompressedClustersFlat* mOutputFlat = nullptr;

  memory* mMemory = nullptr;
  uint32_t* mAttachedClusterFirstIndex = nullptr;
  uint8_t* mClusterStatus = nullptr;

  uint32_t mMaxTracks = 0;
  uint32_t mMaxClusters = 0;
  uint32_t mMaxTrackClusters = 0;
  uint32_t mMaxClustersInCache = 0;
  size_t mMaxClusterFactorBase1024 = 0;

  template <class T>
  void SetPointersCompressedClusters(void*& mem, T& c, uint32_t nClA, uint32_t nTr, uint32_t nClU, bool reducedClA);
  template <class T>
  GPUd() static void truncateSignificantBits(T& val, uint32_t nBits, uint32_t max);

  int16_t mMemoryResOutputHost = -1;
  int16_t mMemoryResOutputGPU = -1;
};

template <class T>
GPUdi() void GPUTPCCompression::truncateSignificantBits(T& v, uint32_t nBits, uint32_t max)
{
  if (nBits == 0) {
    return;
  }

  uint32_t val = v;
  uint32_t ldz = sizeof(uint32_t) * 8 - CAMath::Clz(val);
  if (val && ldz > nBits) {
    if (val & (1 << (ldz - nBits - 1))) {
      val += (1 << (ldz - nBits - 1));
      ldz = sizeof(uint32_t) * 8 - CAMath::Clz(val);
    }
    val &= ((1 << ldz) - 1) ^ ((1 << (ldz - nBits)) - 1);
    if (val >= max) {
      val = max - 1;
    }
    // GPUInfo("CHANGING X %x --> %x", (uint32_t) v, val);
    v = val;
  }
}
} // namespace GPUCA_NAMESPACE::gpu

#endif
