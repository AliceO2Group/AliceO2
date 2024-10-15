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

/// \file DecodeZS.h
/// \author David Rohr, Felix Weiglhofer

#ifndef O2_GPU_DECODE_ZS_H
#define O2_GPU_DECODE_ZS_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "TPCBase/PadPos.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
#include "DetectorsRaw/RDHUtils.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCClusterFinder;

class GPUTPCCFDecodeZS : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory /*: public GPUKernelTemplate::GPUSharedMemoryScan64<int32_t, GPUCA_WARP_SIZE>*/ {
    CA_SHARED_STORAGE(uint32_t ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(uint32_t)]);
    uint32_t RowClusterOffset[o2::tpc::TPCZSHDR::TPC_MAX_ZS_ROW_IN_ENDPOINT];
    uint32_t nRowsRegion;
    uint32_t regionStartRow;
    uint32_t nThreadsPerRow;
    uint32_t rowStride;
    GPUAtomic(uint32_t) rowOffsetCounter;
  };

  enum K : int32_t {
    decodeZS,
  };

  static GPUd() void decode(GPUTPCClusterFinder& clusterer, GPUSharedMemory& s, int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, int32_t firstHBF);

#ifdef GPUCA_HAVE_O2HEADERS
  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }
#endif

  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);
};

class GPUTPCCFDecodeZSLinkBase : public GPUKernelTemplate
{

 public:
#ifdef GPUCA_HAVE_O2HEADERS
  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }
#endif

  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <class Decoder>
  GPUd() static void Decode(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, typename Decoder::GPUSharedMemory& smem, processorType& clusterer, int32_t firstHBF);

  GPUd() static o2::tpc::PadPos GetPadAndRowFromFEC(processorType& clusterer, int32_t cru, int32_t rawFecChannel, int32_t fecInPartition);
  GPUd() static void WriteCharge(processorType& clusterer, float charge, o2::tpc::PadPos pos, tpccf::TPCFragmentTime localTime, size_t positionOffset);
  GPUd() static uint16_t FillWithInvalid(processorType& clusterer, int32_t iThread, int32_t nThreads, uint32_t pageDigitOffset, uint16_t nSamples);

  GPUdi() static const uint8_t* ConsumeBytes(const uint8_t*& page, size_t nbytes)
  {
    const uint8_t* oldPage = page;
    page += nbytes;
    return oldPage;
  }

  GPUdi() static uint8_t ConsumeByte(const uint8_t*& page)
  {
    return *(page++);
  }

  template <typename T>
  GPUdi() static const T* ConsumeHeader(const uint8_t*& page)
  {
    assert(size_t(page) % alignof(T) == 0);
    return reinterpret_cast<const T*>(ConsumeBytes(page, sizeof(T)));
  }

  template <typename T = uint8_t>
  GPUdi() static const T* Peek(const uint8_t* page, ptrdiff_t offset = 0)
  {
    // if ((size_t(page) + offset) % alignof(T) != 0) {
    //   printf("page = %zu, offset = %zu, alignof = %zu\n", size_t(page), offset, alignof(T));
    // }
    assert((size_t(page) + offset) % alignof(T) == 0);
    return reinterpret_cast<const T*>(page + offset);
  }

  GPUdi() static float ADCToFloat(uint32_t adc, uint32_t decodeMask, float decodeBitsFactor)
  {
    return float(adc & decodeMask) * decodeBitsFactor;
  }
};

class GPUTPCCFDecodeZSLink : public GPUTPCCFDecodeZSLinkBase
{
 public:
  // constants for decoding
  static inline constexpr int32_t DECODE_BITS = o2::tpc::TPCZSHDRV2::TPC_ZS_NBITS_V34;
  static inline constexpr float DECODE_BITS_FACTOR = 1.f / (1 << (DECODE_BITS - 10));
  static inline constexpr uint32_t DECODE_MASK = (1 << DECODE_BITS) - 1;

  struct GPUSharedMemory : GPUKernelTemplate::GPUSharedMemoryWarpScan64<uint8_t, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSLink)> {
    // CA_SHARED_STORAGE(uint32_t ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(uint32_t)]);
  };

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  GPUd() static size_t DecodePage(GPUSharedMemory& smem, processorType& clusterer, int32_t iBlock, int32_t nThreads, int32_t iThread, const uint8_t* page, uint32_t pageDigitOffset, int32_t firstHBF);

  GPUd() static void GetChannelBitmask(const tpc::zerosupp_link_based::CommonHeader& tbHdr, uint32_t* chan);
  GPUd() static bool ChannelIsActive(const uint32_t* chan, uint8_t chanIndex);

  GPUd() static void DecodeTBSingleThread(processorType& clusterer, const uint8_t* adcData, uint32_t nAdc, const uint32_t* channelMask, int32_t timeBin, int32_t cru, int32_t fecInPartition, uint32_t pageDigitOffset);
  GPUd() static void DecodeTBMultiThread(processorType& clusterer, int32_t iThread, GPUSharedMemory& smem, const uint8_t* adcData, uint32_t nAdc, const uint32_t* channelMask, int32_t timeBin, int32_t cru, int32_t fecInPartition, uint32_t pageDigitOffset);
};

class GPUTPCCFDecodeZSDenseLink : public GPUTPCCFDecodeZSLinkBase
{
 public:
  // constants for decoding
  static inline constexpr int32_t DECODE_BITS = o2::tpc::TPCZSHDRV2::TPC_ZS_NBITS_V34;
  static inline constexpr float DECODE_BITS_FACTOR = 1.f / (1 << (DECODE_BITS - 10));
  static inline constexpr uint32_t DECODE_MASK = (1 << DECODE_BITS) - 1;

  static inline constexpr int32_t MaxNLinksPerTimebin = 16;

  struct GPUSharedMemory : GPUKernelTemplate::GPUSharedMemoryWarpScan64<uint8_t, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSDenseLink)> {
    // CA_SHARED_STORAGE(uint32_t ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(uint32_t)]);
    uint16_t samplesPerLinkEnd[MaxNLinksPerTimebin]; // Offset from end of TB link header to first sample not in this link
    uint8_t linkIds[MaxNLinksPerTimebin];
    uint8_t rawFECChannels[MaxNLinksPerTimebin * 80];
  };

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  GPUd() static uint32_t DecodePage(GPUSharedMemory& smem, processorType& clusterer, int32_t iBlock, int32_t nThreads, int32_t iThread, const uint8_t* page, uint32_t pageDigitOffset, int32_t firstHBF);

  GPUd() static bool ChannelIsActive(const uint8_t* chan, uint16_t chanIndex);

  template <bool DecodeInParallel, bool PayloadExtendsToNextPage>
  GPUd() static uint16_t DecodeTB(processorType& clusterer, GPUSharedMemory& smem, int32_t iThread, const uint8_t*& page, uint32_t pageDigitOffset, const header::RAWDataHeader* rawDataHeader, int32_t firstHBF, int32_t cru, const uint8_t* payloadEnd, const uint8_t* nextPage);

  template <bool PayloadExtendsToNextPage>
  GPUd() static uint16_t DecodeTBSingleThread(processorType& clusterer, const uint8_t*& page, uint32_t pageDigitOffset, const header::RAWDataHeader* rawDataHeader, int32_t firstHBF, int32_t cru, const uint8_t* payloadEnd, const uint8_t* nextPage);

  template <bool PayloadExtendsToNextPage>
  GPUd() static uint16_t DecodeTBMultiThread(processorType& clusterer, GPUSharedMemory& smem, const int32_t iThread, const uint8_t*& page, uint32_t pageDigitOffset, const header::RAWDataHeader* rawDataHeader, int32_t firstHBF, int32_t cru, const uint8_t* payloadEnd, const uint8_t* nextPage);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
