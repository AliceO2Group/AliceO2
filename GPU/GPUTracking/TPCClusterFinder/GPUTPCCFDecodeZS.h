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
  struct GPUSharedMemory /*: public GPUKernelTemplate::GPUSharedMemoryScan64<int, GPUCA_WARP_SIZE>*/ {
    CA_SHARED_STORAGE(unsigned int ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(unsigned int)]);
    unsigned int RowClusterOffset[o2::tpc::TPCZSHDR::TPC_MAX_ZS_ROW_IN_ENDPOINT];
    unsigned int nRowsRegion;
    unsigned int regionStartRow;
    unsigned int nThreadsPerRow;
    unsigned int rowStride;
    GPUAtomic(unsigned int) rowOffsetCounter;
  };

  enum K : int {
    decodeZS,
  };

  static GPUd() void decode(GPUTPCClusterFinder& clusterer, GPUSharedMemory& s, int nBlocks, int nThreads, int iBlock, int iThread, int firstHBF);

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

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);
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
  GPUd() static void Decode(int nBlocks, int nThreads, int iBlock, int iThread, typename Decoder::GPUSharedMemory& smem, processorType& clusterer, int firstHBF);

  GPUd() static o2::tpc::PadPos GetPadAndRowFromFEC(processorType& clusterer, int cru, int rawFecChannel, int fecInPartition);
  GPUd() static void WriteCharge(processorType& clusterer, float charge, o2::tpc::PadPos pos, tpccf::TPCFragmentTime localTime, size_t positionOffset);

  GPUdi() static const unsigned char* ConsumeBytes(const unsigned char*& page, size_t nbytes)
  {
    const unsigned char* oldPage = page;
    page += nbytes;
    return oldPage;
  }

  GPUdi() static unsigned char ConsumeByte(const unsigned char*& page)
  {
    return *(page++);
  }

  template <typename T>
  GPUdi() static const T* ConsumeHeader(const unsigned char*& page)
  {
    assert((size_t(page) + sizeof(T)) % alignof(T) == 0);
    return reinterpret_cast<const T*>(ConsumeBytes(page, sizeof(T)));
  }

  template <typename T = unsigned char>
  GPUdi() static const T* Peek(const unsigned char* page, ptrdiff_t offset = 0)
  {
    // if ((size_t(page) + offset) % alignof(T) != 0) {
    //   printf("page = %zu, offset = %zu, alignof = %zu\n", size_t(page), offset, alignof(T));
    // }
    assert((size_t(page) + offset) % alignof(T) == 0);
    return reinterpret_cast<const T*>(page + offset);
  }

  GPUdi() static float ADCToFloat(unsigned int adc, unsigned int decodeMask, float decodeBitsFactor)
  {
    return float(adc & decodeMask) * decodeBitsFactor;
  }
};

class GPUTPCCFDecodeZSLink : public GPUTPCCFDecodeZSLinkBase
{
 public:
  // constants for decoding
  static inline constexpr int DECODE_BITS = o2::tpc::TPCZSHDRV2::TPC_ZS_NBITS_V34;
  static inline constexpr float DECODE_BITS_FACTOR = 1.f / (1 << (DECODE_BITS - 10));
  static inline constexpr unsigned int DECODE_MASK = (1 << DECODE_BITS) - 1;

  struct GPUSharedMemory : GPUKernelTemplate::GPUSharedMemoryWarpScan64<unsigned char, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSLink)> {
    // CA_SHARED_STORAGE(unsigned int ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(unsigned int)]);
  };

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  GPUd() static size_t DecodePage(GPUSharedMemory& smem, processorType& clusterer, int iBlock, int nThreads, int iThread, const unsigned char* page, size_t pageDigitOffset, int firstHBF);

  GPUd() static void GetChannelBitmask(const tpc::zerosupp_link_based::CommonHeader& tbHdr, uint32_t* chan);
  GPUd() static bool ChannelIsActive(const uint32_t* chan, unsigned char chanIndex);

  GPUd() static void DecodeTBSingleThread(processorType& clusterer, const unsigned char* adcData, unsigned int nAdc, const uint32_t* channelMask, int timeBin, int cru, int fecInPartition, size_t pageDigitOffset);
  GPUd() static void DecodeTBMultiThread(processorType& clusterer, int iThread, GPUSharedMemory& smem, const unsigned char* adcData, unsigned int nAdc, const uint32_t* channelMask, int timeBin, int cru, int fecInPartition, size_t pageDigitOffset);
};

class GPUTPCCFDecodeZSDenseLink : public GPUTPCCFDecodeZSLinkBase
{
 public:
  // constants for decoding
  static inline constexpr int DECODE_BITS = o2::tpc::TPCZSHDRV2::TPC_ZS_NBITS_V34;
  static inline constexpr float DECODE_BITS_FACTOR = 1.f / (1 << (DECODE_BITS - 10));
  static inline constexpr unsigned int DECODE_MASK = (1 << DECODE_BITS) - 1;

  struct GPUSharedMemory : GPUKernelTemplate::GPUSharedMemoryWarpScan64<unsigned char, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSLink)> {
    // CA_SHARED_STORAGE(unsigned int ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(unsigned int)]);
  };

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  GPUd() static size_t DecodePage(GPUSharedMemory& smem, processorType& clusterer, int iBlock, int nThreads, int iThread, const unsigned char* page, size_t pageDigitOffset, int firstHBF);

  GPUd() static bool ChannelIsActive(const uint8_t* chan, unsigned short chanIndex);

  template <bool PayloadExtendsToNextPage>
  GPUd() static size_t DecodeTBSingleThread(processorType& clusterer, const unsigned char*& page, size_t pageDigitOffset, const header::RAWDataHeader* rawDataHeader, int firstHBF, int cru, const unsigned char* payloadEnd, const unsigned char* nextPage);

  template <bool PayloadExtendsToNextPage>
  GPUd() static size_t DecodeTBMultiThread(processorType& clusterer, const unsigned char*& page, size_t pageDigitOffset, const header::RAWDataHeader* rawDataHeader, int firstHBF, int cru, const unsigned char* payloadEnd, const unsigned char* nextPage);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
