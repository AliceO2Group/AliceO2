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

/// \file GPUTPCCFDecodeZS.cxx
/// \author David Rohr, Felix Weiglhofer

#include "GPUTPCCFDecodeZS.h"
#include "GPUCommonMath.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "CfUtils.h"
#include "CommonConstants/LHCConstants.h"
#include "GPUCommonAlgorithm.h"
#include "TPCPadGainCalib.h"
#include "TPCZSLinkMapping.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;
using namespace o2::tpc;
using namespace o2::tpc::constants;

// ===========================================================================
// ===========================================================================
// Decode ZS Row
// ===========================================================================
// ===========================================================================

template <>
GPUdii() void GPUTPCCFDecodeZS::Thread<GPUTPCCFDecodeZS::decodeZS>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int firstHBF)
{
  GPUTPCCFDecodeZS::decode(clusterer, smem, nBlocks, nThreads, iBlock, iThread, firstHBF);
}

GPUdii() void GPUTPCCFDecodeZS::decode(GPUTPCClusterFinder& clusterer, GPUSharedMemory& s, int nBlocks, int nThreads, int iBlock, int iThread, int firstHBF)
{
  const unsigned int slice = clusterer.mISlice;
#ifdef GPUCA_GPUCODE
  const unsigned int endpoint = clusterer.mPzsOffsets[iBlock].endpoint;
#else
  const unsigned int endpoint = iBlock;
#endif
  const GPUTrackingInOutZS::GPUTrackingInOutZSSlice& zs = clusterer.GetConstantMem()->ioPtrs.tpcZS->slice[slice];
  if (zs.count[endpoint] == 0) {
    return;
  }
  ChargePos* positions = clusterer.mPpositions;
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  const size_t nDigits = clusterer.mPzsOffsets[iBlock].offset;
  if (iThread == 0) {
    const int region = endpoint / 2;
    s.nRowsRegion = clusterer.Param().tpcGeometry.GetRegionRows(region);
    s.regionStartRow = clusterer.Param().tpcGeometry.GetRegionStart(region);
    s.nThreadsPerRow = CAMath::Max(1u, nThreads / ((s.nRowsRegion + (endpoint & 1)) / 2));
    s.rowStride = nThreads / s.nThreadsPerRow;
    s.rowOffsetCounter = 0;
  }
  GPUbarrier();
  const unsigned int myRow = iThread / s.nThreadsPerRow;
  const unsigned int mySequence = iThread % s.nThreadsPerRow;
#ifdef GPUCA_GPUCODE
  const unsigned int i = 0;
  const unsigned int j = clusterer.mPzsOffsets[iBlock].num;
  {
    {
#else
  for (unsigned int i = clusterer.mMinMaxCN[endpoint].minC; i < clusterer.mMinMaxCN[endpoint].maxC; i++) {
    const unsigned int minJ = (i == clusterer.mMinMaxCN[endpoint].minC) ? clusterer.mMinMaxCN[endpoint].minN : 0;
    const unsigned int maxJ = (i + 1 == clusterer.mMinMaxCN[endpoint].maxC) ? clusterer.mMinMaxCN[endpoint].maxN : zs.nZSPtr[endpoint][i];
    for (unsigned int j = minJ; j < maxJ; j++) {
#endif
      const unsigned int* pageSrc = (const unsigned int*)(((const unsigned char*)zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE);
      CA_SHARED_CACHE_REF(&s.ZSPage[0], pageSrc, TPCZSHDR::TPC_ZS_PAGE_SIZE, unsigned int, pageCache);
      GPUbarrier();
      const unsigned char* page = (const unsigned char*)pageCache;
      const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)page;
      if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
#ifdef GPUCA_GPUCODE
        return;
#else
        continue;
#endif
      }
      const unsigned char* pagePtr = page + sizeof(o2::header::RAWDataHeader);
      const TPCZSHDR* hdr = reinterpret_cast<const TPCZSHDR*>(pagePtr);
      pagePtr += sizeof(*hdr);
      const bool decode12bit = hdr->version == 2;
      const unsigned int decodeBits = decode12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
      const float decodeBitsFactor = 1.f / (1 << (decodeBits - 10));
      unsigned int mask = (1 << decodeBits) - 1;
      int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
      const int rowOffset = s.regionStartRow + ((endpoint & 1) ? (s.nRowsRegion / 2) : 0);
      const int nRows = (endpoint & 1) ? (s.nRowsRegion - s.nRowsRegion / 2) : (s.nRowsRegion / 2);

      for (int l = 0; l < hdr->nTimeBinSpan; l++) { // TODO: Parallelize over time bins
        pagePtr += (pagePtr - page) & 1;            // Ensure 16 bit alignment
        const TPCZSTBHDR* tbHdr = reinterpret_cast<const TPCZSTBHDR*>(pagePtr);
        if ((tbHdr->rowMask & 0x7FFF) == 0) {
          pagePtr += 2;
          continue;
        }
        const int nRowsUsed = CAMath::Popcount((unsigned int)(tbHdr->rowMask & 0x7FFF));
        pagePtr += 2 * nRowsUsed;

        GPUbarrier();
        for (int n = iThread; n < nRowsUsed; n += nThreads) {
          const unsigned char* rowData = n == 0 ? pagePtr : (page + tbHdr->rowAddr1()[n - 1]);
          s.RowClusterOffset[n] = CAMath::AtomicAddShared<unsigned int>(&s.rowOffsetCounter, rowData[2 * *rowData]);
        }
        /*if (iThread < GPUCA_WARP_SIZE) { // TODO: Seems to miscompile with HIP, CUDA performance doesn't really change, for now sticking to the AtomicAdd
          GPUSharedMemory& smem = s;
          int o;
          if (iThread < nRowsUsed) {
            const unsigned char* rowData = iThread == 0 ? pagePtr : (page + tbHdr->rowAddr1()[iThread - 1]);
            o = rowData[2 * *rowData];
          } else {
            o = 0;
          }
          int x = warp_scan_inclusive_add(o);
          if (iThread < nRowsUsed) {
            s.RowClusterOffset[iThread] = s.rowOffsetCounter + x - o;
          } else if (iThread == GPUCA_WARP_SIZE - 1) {
            s.rowOffsetCounter += x;
          }
        }*/
        GPUbarrier();

        if (myRow < s.rowStride) {
          for (int m = myRow; m < nRows; m += s.rowStride) {
            if ((tbHdr->rowMask & (1 << m)) == 0) {
              continue;
            }
            const int rowPos = CAMath::Popcount((unsigned int)(tbHdr->rowMask & ((1 << m) - 1)));
            size_t nDigitsTmp = nDigits + s.RowClusterOffset[rowPos];
            const unsigned char* rowData = rowPos == 0 ? pagePtr : (page + tbHdr->rowAddr1()[rowPos - 1]);
            const int nSeqRead = *rowData;
            const int nSeqPerThread = (nSeqRead + s.nThreadsPerRow - 1) / s.nThreadsPerRow;
            const int mySequenceStart = mySequence * nSeqPerThread;
            const int mySequenceEnd = CAMath::Min(mySequenceStart + nSeqPerThread, nSeqRead);
            if (mySequenceEnd > mySequenceStart) {
              const unsigned char* adcData = rowData + 2 * nSeqRead + 1;
              const unsigned int nSamplesStart = mySequenceStart ? rowData[2 * mySequenceStart] : 0;
              nDigitsTmp += nSamplesStart;
              unsigned int nADCStartBits = nSamplesStart * decodeBits;
              const unsigned int nADCStart = (nADCStartBits + 7) / 8;
              const int nADC = (rowData[2 * mySequenceEnd] * decodeBits + 7) / 8;
              adcData += nADCStart;
              nADCStartBits &= 0x7;
              unsigned int byte = 0, bits = 0;
              if (nADCStartBits) { // % 8 != 0
                bits = 8 - nADCStartBits;
                byte = ((*(adcData - 1) & (0xFF ^ ((1 << nADCStartBits) - 1)))) >> nADCStartBits;
              }
              int nSeq = mySequenceStart;
              int seqLen = nSeq ? (rowData[(nSeq + 1) * 2] - rowData[nSeq * 2]) : rowData[2];
              Pad pad = rowData[nSeq++ * 2 + 1];
              for (int n = nADCStart; n < nADC; n++) {
                byte |= *(adcData++) << bits;
                bits += 8;
                while (bits >= decodeBits) {
                  if (seqLen == 0) {
                    seqLen = rowData[(nSeq + 1) * 2] - rowData[nSeq * 2];
                    pad = rowData[nSeq++ * 2 + 1];
                  }
                  const CfFragment& fragment = clusterer.mPmemory->fragment;
                  TPCTime globalTime = timeBin + l;
                  bool inFragment = fragment.contains(globalTime);
                  Row row = rowOffset + m;
                  ChargePos pos(row, Pad(pad), inFragment ? fragment.toLocal(globalTime) : INVALID_TIME_BIN);
                  positions[nDigitsTmp++] = pos;

                  if (inFragment) {
                    float q = float(byte & mask) * decodeBitsFactor;
                    q *= clusterer.GetConstantMem()->calibObjects.tpcPadGain->getGainCorrection(slice, row, pad);
                    chargeMap[pos] = PackedCharge(q);
                  }
                  pad++;
                  byte = byte >> decodeBits;
                  bits -= decodeBits;
                  seqLen--;
                }
              }
            }
          }
        }
        if (nRowsUsed > 1) {
          pagePtr = page + tbHdr->rowAddr1()[nRowsUsed - 2];
        }
        pagePtr += 2 * *pagePtr;                        // Go to entry for last sequence length
        pagePtr += 1 + (*pagePtr * decodeBits + 7) / 8; // Go to beginning of next time bin
      }
    }
  }
}

// ===========================================================================
// ===========================================================================
// Decode ZS Link
// ===========================================================================
// ===========================================================================

template <>
GPUdii() void GPUTPCCFDecodeZSLink::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int firstHBF)
{
  Decode<GPUTPCCFDecodeZSLink>(nBlocks, nThreads, iBlock, iThread, smem, clusterer, firstHBF);
}

GPUd() size_t GPUTPCCFDecodeZSLink::DecodePage(GPUSharedMemory& smem, processorType& clusterer, int iBlock, int nThreads, int iThread, const unsigned char* page, uint32_t pageDigitOffset, int firstHBF)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;

  const auto* rdHdr = ConsumeHeader<header::RAWDataHeader>(page);

  if (o2::raw::RDHUtils::getMemorySize(*rdHdr) == sizeof(o2::header::RAWDataHeader)) {
    return pageDigitOffset;
  }

  int nDecoded = 0;
  const auto* decHdr = ConsumeHeader<TPCZSHDRV2>(page);
  ConsumeBytes(page, decHdr->firstZSDataOffset * 16);

  assert(decHdr->version == ZSVersionLinkBasedWithMeta);
  assert(decHdr->magicWord == o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader);

  for (unsigned int t = 0; t < decHdr->nTimebinHeaders; t++) {
    const auto* tbHdr = ConsumeHeader<zerosupp_link_based::CommonHeader>(page);
    const auto* adcData = ConsumeBytes(page, tbHdr->numWordsPayload * 16); // Page now points to next timebin or past the page

    int timeBin = (decHdr->timeOffset + tbHdr->bunchCrossing + (unsigned long)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdHdr) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;

    uint32_t channelMask[3];
    GetChannelBitmask(*tbHdr, channelMask);
    unsigned int nAdc = CAMath::Popcount(channelMask[0]) + CAMath::Popcount(channelMask[1]) + CAMath::Popcount(channelMask[2]);

    bool inFragment = fragment.contains(timeBin);
    nDecoded += nAdc;

    // TimeBin not in fragment: Skip this timebin header and fill positions with dummy values instead
    if (not inFragment) {
      pageDigitOffset += FillWithInvalid(clusterer, iThread, nThreads, pageDigitOffset, nAdc);
      continue;
    }

#ifdef GPUCA_GPUCODE
    DecodeTBMultiThread(
      clusterer,
      iThread,
      smem,
      adcData,
      nAdc,
      channelMask,
      timeBin,
      decHdr->cruID,
      tbHdr->fecInPartition,
      pageDigitOffset);
#else // CPU
    DecodeTBSingleThread(
      clusterer,
      adcData,
      nAdc,
      channelMask,
      timeBin,
      decHdr->cruID,
      tbHdr->fecInPartition,
      pageDigitOffset);
#endif
    pageDigitOffset += nAdc;
  } // for (unsigned int t = 0; t < decHdr->nTimebinHeaders; t++)
  (void)nDecoded;
#ifdef GPUCA_CHECK_TPCZS_CORRUPTION
  if (iThread == 0 && nDecoded != decHdr->nADCsamples) {
    clusterer.raiseError(GPUErrors::ERROR_TPCZS_INVALID_NADC, clusterer.mISlice, decHdr->nADCsamples, nDecoded);
    /*#ifndef GPUCA_GPUCODE
            FILE* foo = fopen("dump.bin", "w+b");
            fwrite(pageSrc, 1, o2::raw::RDHUtils::getMemorySize(*rdHdr), foo);
            fclose(foo);
    #endif*/
  }
  fwrite(pageSrc, 1, o2::raw::RDHUtils::getMemorySize(*rdHdr), foo);
  fclose(foo);
#endif
  return pageDigitOffset;
}

GPUd() void GPUTPCCFDecodeZSLink::DecodeTBSingleThread(
  processorType& clusterer,
  const unsigned char* adcData,
  unsigned int nAdc,
  const uint32_t* channelMask,
  int timeBin,
  int cru,
  int fecInPartition,
  uint32_t pageDigitOffset)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;

  if CONSTEXPR17 (TPCZSHDRV2::TIGHTLY_PACKED_V3) {

    unsigned int byte = 0, bits = 0, nSamplesWritten = 0, rawFECChannel = 0;

    // unpack adc values, assume tightly packed data
    while (nSamplesWritten < nAdc) {
      byte |= adcData[0] << bits;
      adcData++;
      bits += CHAR_BIT;
      while (bits >= DECODE_BITS) {

        // Find next channel with data
        for (; !ChannelIsActive(channelMask, rawFECChannel); rawFECChannel++) {
        }

        // Unpack data for cluster finder
        o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannel, fecInPartition);

        WriteCharge(clusterer, byte, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + nSamplesWritten);

        byte = byte >> DECODE_BITS;
        bits -= DECODE_BITS;
        nSamplesWritten++;
        rawFECChannel++; // Ensure we don't decode same channel twice
      }                  // while (bits >= DECODE_BITS)
    }                    // while (nSamplesWritten < nAdc)

  } else { // ! TPCZSHDRV2::TIGHTLY_PACKED_V3
    unsigned int rawFECChannel = 0;
    const unsigned long* adcData64 = (const unsigned long*)adcData;
    for (unsigned int j = 0; j < nAdc; j++) {
      for (; !ChannelIsActive(channelMask, rawFECChannel); rawFECChannel++) {
      }

      unsigned int adc = (adcData64[j / TPCZSHDRV2::SAMPLESPER64BIT] >> ((j % TPCZSHDRV2::SAMPLESPER64BIT) * DECODE_BITS)) & DECODE_MASK;

      o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannel, fecInPartition);
      float charge = ADCToFloat(adc, DECODE_MASK, DECODE_BITS_FACTOR);
      WriteCharge(clusterer, charge, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + j);
      rawFECChannel++;
    }
  }
}

GPUd() void GPUTPCCFDecodeZSLink::DecodeTBMultiThread(
  processorType& clusterer,
  int iThread,
  GPUSharedMemory& smem,
  const unsigned char* adcData,
  unsigned int nAdc,
  const uint32_t* channelMask,
  int timeBin,
  int cru,
  int fecInPartition,
  uint32_t pageDigitOffset)
{
  constexpr int NTHREADS = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSLink);
  static_assert(NTHREADS == GPUCA_WARP_SIZE, "Decoding TB Headers in parallel assumes block size is a single warp.");

  unsigned char blockOffset = 0;
  for (unsigned char i = iThread; blockOffset < nAdc; i += NTHREADS) {

    unsigned char rawFECChannel = i;

    unsigned char myChannelActive = ChannelIsActive(channelMask, rawFECChannel);

    unsigned char myOffset = warp_scan_inclusive_add(myChannelActive) - 1 + blockOffset;
    blockOffset = warp_broadcast(myOffset, NTHREADS - 1) + 1;

    // Decode entire timebin at once if we have enough threads
    // This should further improve performance, but code below is buggy...
    // if (nAdc <= NThreads) {
    //   for (int j = 1; blockOffset < nAdc; j++) {
    //     rawFECChannel = myChannelActive ? rawFECChannel : (iThread + j*NThreads - myOffset);

    //     bool iAmIdle = not myChannelActive;

    //     myChannelActive =
    //       rawFECChannel < zerosupp_link_based::CommonHeaderlPerTBHeader
    //         ? BitIsSet(channelMask, rawFECChannel)
    //         : false;

    //     unsigned char newOffset = warp_scan_inclusive_add(static_cast<unsigned char>(myChannelActive && iAmIdle)) - 1 + blockOffset;
    //     blockOffset = warp_broadcast(newOffset, NThreads - 1) + 1;

    //     myOffset = iAmIdle ? newOffset : myOffset;
    //   }
    // }

    if (not myChannelActive) {
      continue;
    }
    assert(myOffset < nAdc);

    unsigned int adc = 0;

    if CONSTEXPR17 (TPCZSHDRV2::TIGHTLY_PACKED_V3) {

      // Try to access adcData with 4 byte reads instead of 1 byte.
      // You'd think this would improve performace, but it's actually slower...
      // const uint32_t* adcDataU32 = reinterpret_cast<const uint32_t*>(adcData);

      unsigned int adcBitOffset = myOffset * DECODE_BITS;
      unsigned int adcByteOffset = adcBitOffset / CHAR_BIT;
      unsigned int adcOffsetInByte = adcBitOffset - adcByteOffset * CHAR_BIT;
      // unsigned int adcByteOffset = adcBitOffset / 32;
      // unsigned int adcOffsetInByte = adcBitOffset - adcByteOffset * 32;

      unsigned int byte = 0, bits = 0;

      // unsigned int byte = adcDataU32[adcByteOffset] >> adcOffsetInByte;
      // unsigned int bits = 32 - adcOffsetInByte;
      // adcByteOffset++;

      while (bits < DECODE_BITS) {
        byte |= ((unsigned int)adcData[adcByteOffset]) << bits;
        // byte |= adcDataU32[adcByteOffset] << bits;
        adcByteOffset++;
        bits += CHAR_BIT;
        // bits += 32;
      }
      adc = byte >> adcOffsetInByte;

    } else { // ! TPCZSHDRV2::TIGHTLY_PACKED_V3
      const unsigned long* adcData64 = (const unsigned long*)adcData;
      adc = (adcData64[myOffset / TPCZSHDRV2::SAMPLESPER64BIT] >> ((myOffset % TPCZSHDRV2::SAMPLESPER64BIT) * DECODE_BITS)) & DECODE_MASK;
    }

    o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannel, fecInPartition);
    const CfFragment& fragment = clusterer.mPmemory->fragment;
    float charge = ADCToFloat(adc, DECODE_MASK, DECODE_BITS_FACTOR);
    WriteCharge(clusterer, charge, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + myOffset);

  } // for (unsigned char i = iThread; blockOffset < nAdc; i += NThreads)
}

GPUd() void GPUTPCCFDecodeZSLink::GetChannelBitmask(const zerosupp_link_based::CommonHeader& tbHdr, uint32_t* chan)
{
  chan[0] = tbHdr.bitMaskLow & 0xfffffffful;
  chan[1] = tbHdr.bitMaskLow >> (sizeof(uint32_t) * CHAR_BIT);
  chan[2] = tbHdr.bitMaskHigh;
}

GPUd() bool GPUTPCCFDecodeZSLink::ChannelIsActive(const uint32_t* chan, unsigned char chanIndex)
{
  if (chanIndex >= zerosupp_link_based::ChannelPerTBHeader) {
    return false;
  }
  constexpr unsigned char N_BITS_PER_ENTRY = sizeof(*chan) * CHAR_BIT;
  const unsigned char entryIndex = chanIndex / N_BITS_PER_ENTRY;
  const unsigned char bitInEntry = chanIndex % N_BITS_PER_ENTRY;
  return chan[entryIndex] & (1 << bitInEntry);
}

// ===========================================================================
// ===========================================================================
// Decode ZS Link Base
// ===========================================================================
// ===========================================================================

template <class Decoder>
GPUd() void GPUTPCCFDecodeZSLinkBase::Decode(int nBlocks, int nThreads, int iBlock, int iThread, typename Decoder::GPUSharedMemory& smem, processorType& clusterer, int firstHBF)
{
  const unsigned int slice = clusterer.mISlice;

#ifdef GPUCA_GPUCODE
  const unsigned int endpoint = clusterer.mPzsOffsets[iBlock].endpoint;
#else // CPU
  const unsigned int endpoint = iBlock;
#endif

  const GPUTrackingInOutZS::GPUTrackingInOutZSSlice& zs = clusterer.GetConstantMem()->ioPtrs.tpcZS->slice[slice];
  if (zs.count[endpoint] == 0) {
    return;
  }

  uint32_t pageDigitOffset = clusterer.mPzsOffsets[iBlock].offset;

#ifdef GPUCA_GPUCODE
  const unsigned int i = 0;
  const unsigned int j = clusterer.mPzsOffsets[iBlock].num;
  {
    {
#else // CPU
  for (unsigned int i = clusterer.mMinMaxCN[endpoint].minC; i < clusterer.mMinMaxCN[endpoint].maxC; i++) {
    const unsigned int minJ = (i == clusterer.mMinMaxCN[endpoint].minC) ? clusterer.mMinMaxCN[endpoint].minN : 0;
    const unsigned int maxJ = (i + 1 == clusterer.mMinMaxCN[endpoint].maxC) ? clusterer.mMinMaxCN[endpoint].maxN : zs.nZSPtr[endpoint][i];
    for (unsigned int j = minJ; j < maxJ; j++) {
#endif
      const unsigned int* pageSrc = (const unsigned int*)(((const unsigned char*)zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE);
      // Cache zs page in shared memory. Curiously this actually degrades performance...
      // CA_SHARED_CACHE_REF(&smem.ZSPage[0], pageSrc, TPCZSHDR::TPC_ZS_PAGE_SIZE, unsigned int, pageCache);
      // GPUbarrier();
      // const unsigned char* page = (const unsigned char*)pageCache;
      const unsigned char* page = (const unsigned char*)pageSrc;

      const auto* rdHdr = Peek<header::RAWDataHeader>(page);

      if (o2::raw::RDHUtils::getMemorySize(*rdHdr) == sizeof(o2::header::RAWDataHeader)) {
#ifdef GPUCA_GPUCODE
        return;
#else
        continue;
#endif
      }

      pageDigitOffset = Decoder::DecodePage(smem, clusterer, iBlock, nThreads, iThread, page, pageDigitOffset, firstHBF);

    } // [CPU] for (unsigned int j = minJ; j < maxJ; j++)
  }   // [CPU] for (unsigned int i = clusterer.mMinMaxCN[endpoint].minC; i < clusterer.mMinMaxCN[endpoint].maxC; i++)
}

GPUd() o2::tpc::PadPos GPUTPCCFDecodeZSLinkBase::GetPadAndRowFromFEC(processorType& clusterer, int cru, int rawFECChannel, int fecInPartition)
{
  // Ported from tpc::Mapper (Not available on GPU...)
  const GPUTPCGeometry& geo = clusterer.Param().tpcGeometry;

  const int regionIter = cru % 2;
  const int istreamm = ((rawFECChannel % 10) / 2);
  const int partitionStream = istreamm + regionIter * 5;
  const int sampaOnFEC = geo.GetSampaMapping(partitionStream);
  const int channel = (rawFECChannel % 2) + 2 * (rawFECChannel / 10);
  const int channelOnSAMPA = channel + geo.GetChannelOffset(partitionStream);

  const int partition = (cru % 10) / 2;
  const int fecInSector = geo.GetSectorFECOffset(partition) + fecInPartition;

  const TPCZSLinkMapping* gpuMapping = clusterer.GetConstantMem()->calibObjects.tpcZSLinkMapping;
  assert(gpuMapping != nullptr);

  unsigned short globalSAMPAId = (static_cast<unsigned short>(fecInSector) << 8) + (static_cast<unsigned short>(sampaOnFEC) << 5) + static_cast<unsigned short>(channelOnSAMPA);
  const o2::tpc::PadPos pos = gpuMapping->FECIDToPadPos[globalSAMPAId];

  return pos;
}

GPUd() void GPUTPCCFDecodeZSLinkBase::WriteCharge(processorType& clusterer, float charge, PadPos padAndRow, TPCFragmentTime localTime, size_t positionOffset)
{
  const unsigned int slice = clusterer.mISlice;
  ChargePos* positions = clusterer.mPpositions;
#ifdef GPUCA_CHECK_TPCZS_CORRUPTION
  if (padAndRow.getRow() >= GPUCA_ROW_COUNT) {
    positions[positionOffset] = INVALID_CHARGE_POS;
    clusterer.raiseError(GPUErrors::ERROR_CF_ROW_CLUSTER_OVERFLOW, clusterer.mISlice * 1000 + padAndRow.getRow(), 0, 0);
    return;
  }
#endif
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  ChargePos pos(padAndRow.getRow(), padAndRow.getPad(), localTime);
  positions[positionOffset] = pos;

  charge *= clusterer.GetConstantMem()->calibObjects.tpcPadGain->getGainCorrection(slice, padAndRow.getRow(), padAndRow.getPad());
  chargeMap[pos] = PackedCharge(charge);
}

GPUd() unsigned short GPUTPCCFDecodeZSLinkBase::FillWithInvalid(processorType& clusterer, int iThread, int nThreads, uint32_t pageDigitOffset, unsigned short nSamples)
{
  for (unsigned short i = iThread; i < nSamples; i += nThreads) {
    clusterer.mPpositions[pageDigitOffset + i] = INVALID_CHARGE_POS;
  }
  return nSamples;
}

// ===========================================================================
// ===========================================================================
// Decode ZS Dense Link
// ===========================================================================
// ===========================================================================

template <>
GPUd() void GPUTPCCFDecodeZSDenseLink::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int firstHBF)
{
  Decode<GPUTPCCFDecodeZSDenseLink>(nBlocks, nThreads, iBlock, iThread, smem, clusterer, firstHBF);
}

GPUd() uint32_t GPUTPCCFDecodeZSDenseLink::DecodePage(GPUSharedMemory& smem, processorType& clusterer, int iBlock, int nThreads, int iThread, const unsigned char* page, uint32_t pageDigitOffset, int firstHBF)
{
  constexpr bool DecodeInParallel =
#ifdef GPUCA_GPUCODE
    true;
#else
    false;
#endif

  const unsigned char* const pageStart = page;

  const auto* rawDataHeader = Peek<header::RAWDataHeader>(page);
  const auto* decHeader = Peek<TPCZSHDRV2>(page, raw::RDHUtils::getMemorySize(*rawDataHeader) - sizeof(TPCZSHDRV2));
  ConsumeHeader<header::RAWDataHeader>(page);

  assert(decHeader->version == ZSVersionDenseLinkBased);
  assert(decHeader->magicWord == tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader);

  unsigned short nSamplesWritten = 0;
  const unsigned short nSamplesInPage = decHeader->nADCsamples;

  const auto* payloadEnd = Peek(pageStart, raw::RDHUtils::getMemorySize(*rawDataHeader) - sizeof(TPCZSHDRV2) - ((decHeader->flags & TPCZSHDRV2::ZSFlags::TriggerWordPresent) ? TPCZSHDRV2::TRIGGER_WORD_SIZE : 0));
  const auto* nextPage = Peek(pageStart, TPCZSHDR::TPC_ZS_PAGE_SIZE);

  ConsumeBytes(page, decHeader->firstZSDataOffset);

  for (unsigned short i = 0; i < decHeader->nTimebinHeaders; i++) {
    [[maybe_unused]] ptrdiff_t sizeLeftInPage = payloadEnd - page;
    assert(sizeLeftInPage > 0);
    assert(nSamplesWritten < nSamplesInPage);

    unsigned short nSamplesWrittenTB = 0;

    if (i == decHeader->nTimebinHeaders - 1 && decHeader->flags & o2::tpc::TPCZSHDRV2::ZSFlags::payloadExtendsToNextPage) {
      assert(o2::raw::RDHUtils::getMemorySize(*rawDataHeader) == TPCZSHDR::TPC_ZS_PAGE_SIZE);
      // Disable check for dropped pages temporarily, decoding fails on large dataset when enabled...
      if ((unsigned char)(raw::RDHUtils::getPacketCounter(rawDataHeader) + 1) == raw::RDHUtils::getPacketCounter(nextPage)) {
        nSamplesWrittenTB = DecodeTB<DecodeInParallel, true>(clusterer, smem, iThread, page, pageDigitOffset, rawDataHeader, firstHBF, decHeader->cruID, payloadEnd, nextPage);
      } else {
        nSamplesWrittenTB = FillWithInvalid(clusterer, iThread, nThreads, pageDigitOffset, nSamplesInPage - nSamplesWritten);
#ifdef GPUCA_CHECK_TPCZS_CORRUPTION
        if (iThread == 0) {
          clusterer.raiseError(GPUErrors::ERROR_TPCZS_INCOMPLETE_HBF, clusterer.mISlice, raw::RDHUtils::getPacketCounter(rawDataHeader), raw::RDHUtils::getPacketCounter(nextPage));
        }
#endif
      }
    } else {
      nSamplesWrittenTB = DecodeTB<DecodeInParallel, false>(clusterer, smem, iThread, page, pageDigitOffset, rawDataHeader, firstHBF, decHeader->cruID, payloadEnd, nextPage);
    }

    nSamplesWritten += nSamplesWrittenTB;
    pageDigitOffset += nSamplesWrittenTB;

  } // for (unsigned short i = 0; i < decHeader->nTimebinHeaders; i++)

#ifdef GPUCA_CHECK_TPCZS_CORRUPTION
  if (iThread == 0 && nSamplesWritten != nSamplesInPage) {
    clusterer.raiseError(GPUErrors::ERROR_TPCZS_INVALID_NADC, clusterer.mISlice, nSamplesInPage, nSamplesWritten);
    /*#ifndef GPUCA_GPUCODE
            FILE* foo = fopen("dump.bin", "w+b");
            fwrite(pageSrc, 1, o2::raw::RDHUtils::getMemorySize(*rdHdr), foo);
            fclose(foo);
    #endif*/
  }
#endif

  return pageDigitOffset;
}

template <bool DecodeInParallel, bool PayloadExtendsToNextPage>
GPUd() unsigned short GPUTPCCFDecodeZSDenseLink::DecodeTB(
  processorType& clusterer,
  [[maybe_unused]] GPUSharedMemory& smem,
  int iThread,
  const unsigned char*& page,
  uint32_t pageDigitOffset,
  const header::RAWDataHeader* rawDataHeader,
  int firstHBF,
  int cru,
  [[maybe_unused]] const unsigned char* payloadEnd,
  [[maybe_unused]] const unsigned char* nextPage)
{

  if CONSTEXPR17 (DecodeInParallel) {
    return DecodeTBMultiThread<PayloadExtendsToNextPage>(clusterer, smem, iThread, page, pageDigitOffset, rawDataHeader, firstHBF, cru, payloadEnd, nextPage);
  } else {
    unsigned short nSamplesWritten = 0;
    if (iThread == 0) {
      nSamplesWritten = DecodeTBSingleThread<PayloadExtendsToNextPage>(clusterer, page, pageDigitOffset, rawDataHeader, firstHBF, cru, payloadEnd, nextPage);
    }
    return warp_broadcast(nSamplesWritten, 0);
  }
}

template <bool PayloadExtendsToNextPage>
GPUd() unsigned short GPUTPCCFDecodeZSDenseLink::DecodeTBMultiThread(
  processorType& clusterer,
  GPUSharedMemory& smem,
  const int iThread,
  const unsigned char*& page,
  uint32_t pageDigitOffset,
  const header::RAWDataHeader* rawDataHeader,
  int firstHBF,
  int cru,
  [[maybe_unused]] const unsigned char* payloadEnd,
  [[maybe_unused]] const unsigned char* nextPage)
{
#define MAYBE_PAGE_OVERFLOW(pagePtr)                               \
  if CONSTEXPR17 (PayloadExtendsToNextPage) {                      \
    if (pagePtr >= payloadEnd && pagePtr < nextPage) {             \
      ptrdiff_t diff = pagePtr - payloadEnd;                       \
      pagePtr = nextPage;                                          \
      ConsumeBytes(pagePtr, sizeof(header::RAWDataHeader) + diff); \
    }                                                              \
  } else                                                           \
    assert(pagePtr <= payloadEnd)

#define PEEK_OVERFLOW(pagePtr, offset)                                                      \
  (*(PayloadExtendsToNextPage && (pagePtr) < nextPage && (pagePtr) + (offset) >= payloadEnd \
       ? nextPage + sizeof(header::RAWDataHeader) + ((pagePtr) + (offset)-payloadEnd)       \
       : (pagePtr) + (offset)))

#define TEST_BIT(x, bit) static_cast<bool>((x) & (1 << (bit)))

  constexpr int NTHREADS = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFDecodeZSDenseLink);
  static_assert(NTHREADS == GPUCA_WARP_SIZE, "Decoding TB Headers in parallel assumes block size is a single warp.");

  const CfFragment& fragment = clusterer.mPmemory->fragment;

  // Read timebin block header
  unsigned short tbbHdr = ConsumeByte(page);
  MAYBE_PAGE_OVERFLOW(page);
  tbbHdr |= static_cast<unsigned short>(ConsumeByte(page)) << CHAR_BIT;
  MAYBE_PAGE_OVERFLOW(page);

  unsigned char nLinksInTimebin = tbbHdr & 0x000F;
  unsigned short linkBC = (tbbHdr & 0xFFF0) >> 4;
  int timeBin = (linkBC + (unsigned long)(raw::RDHUtils::getHeartBeatOrbit(*rawDataHeader) - firstHBF) * constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;

  unsigned short nSamplesInTB = 0;

  // Read timebin link headers
  for (unsigned char iLink = 0; iLink < nLinksInTimebin; iLink++) {
    unsigned char timebinLinkHeaderStart = ConsumeByte(page);
    MAYBE_PAGE_OVERFLOW(page);

    smem.linkIds[iLink] = timebinLinkHeaderStart & 0b00011111;
    bool bitmaskIsFlat = timebinLinkHeaderStart & 0b00100000;

    unsigned short bitmaskL2 = 0x03FF;
    if (not bitmaskIsFlat) {
      bitmaskL2 = static_cast<unsigned short>(timebinLinkHeaderStart & 0b11000000) << 2 | static_cast<unsigned short>(ConsumeByte(page));
      MAYBE_PAGE_OVERFLOW(page);
    }

    int nBytesBitmask = CAMath::Popcount(bitmaskL2);
    assert(nBytesBitmask <= 10);

    for (int chan = iThread; chan < CAMath::nextMultipleOf<NTHREADS>(80); chan += NTHREADS) {
      int chanL2Idx = chan / 8;
      bool l2 = TEST_BIT(bitmaskL2, chanL2Idx);

      int chanByteOffset = CAMath::Popcount(bitmaskL2 >> (chanL2Idx + 1));

      unsigned char myChannelHasData = (chan < 80 && l2 ? TEST_BIT(PEEK_OVERFLOW(page, chanByteOffset), chan % 8) : 0);
      assert(myChannelHasData == 0 || myChannelHasData == 1);

      int nSamplesStep;
      int threadSampleOffset = CfUtils::warpPredicateScan(myChannelHasData, &nSamplesStep);

      if (myChannelHasData) {
        smem.rawFECChannels[nSamplesInTB + threadSampleOffset] = chan;
      }

      nSamplesInTB += nSamplesStep;
    }

    ConsumeBytes(page, nBytesBitmask);
    MAYBE_PAGE_OVERFLOW(page);

    smem.samplesPerLinkEnd[iLink] = nSamplesInTB;

  } // for (unsigned char iLink = 0; iLink < nLinksInTimebin; iLink++)

  const unsigned char* adcData = ConsumeBytes(page, (nSamplesInTB * DECODE_BITS + 7) / 8);
  MAYBE_PAGE_OVERFLOW(page); // TODO: We don't need this check?

  if (not fragment.contains(timeBin)) {
    return FillWithInvalid(clusterer, iThread, NTHREADS, pageDigitOffset, nSamplesInTB);
  }

  // Unpack ADC
  int iLink = 0;
  for (unsigned short sample = iThread; sample < nSamplesInTB; sample += NTHREADS) {
    const unsigned short adcBitOffset = sample * DECODE_BITS;
    unsigned short adcByteOffset = adcBitOffset / CHAR_BIT;
    const unsigned char adcOffsetInByte = adcBitOffset - adcByteOffset * CHAR_BIT;

    unsigned char bits = 0;
    unsigned short byte = 0;

    static_assert(DECODE_BITS <= sizeof(unsigned short) * CHAR_BIT);

    while (bits < DECODE_BITS) {
      byte |= static_cast<unsigned short>(PEEK_OVERFLOW(adcData, adcByteOffset)) << bits;
      adcByteOffset++;
      bits += CHAR_BIT;
    }
    byte >>= adcOffsetInByte;

    while (smem.samplesPerLinkEnd[iLink] <= sample) {
      iLink++;
    }

    int rawFECChannelLink = smem.rawFECChannels[sample];

    // Unpack data for cluster finder
    o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannelLink, smem.linkIds[iLink]);

    float charge = ADCToFloat(byte, DECODE_MASK, DECODE_BITS_FACTOR);
    WriteCharge(clusterer, charge, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + sample);

  } // for (unsigned short sample = iThread; sample < nSamplesInTB; sample += NTHREADS)

  assert(PayloadExtendsToNextPage || adcData <= page);
  assert(PayloadExtendsToNextPage || page <= payloadEnd);

  return nSamplesInTB;

#undef TEST_BIT
#undef PEEK_OVERFLOW
#undef MAYBE_PAGE_OVERFLOW
}

template <bool PayloadExtendsToNextPage>
GPUd() unsigned short GPUTPCCFDecodeZSDenseLink::DecodeTBSingleThread(
  processorType& clusterer,
  const unsigned char*& page,
  uint32_t pageDigitOffset,
  const header::RAWDataHeader* rawDataHeader,
  int firstHBF,
  int cru,
  [[maybe_unused]] const unsigned char* payloadEnd,
  [[maybe_unused]] const unsigned char* nextPage)
{
#define MAYBE_PAGE_OVERFLOW(pagePtr)                               \
  if CONSTEXPR17 (PayloadExtendsToNextPage) {                      \
    if (pagePtr >= payloadEnd && pagePtr < nextPage) {             \
      ptrdiff_t diff = pagePtr - payloadEnd;                       \
      pagePtr = nextPage;                                          \
      ConsumeBytes(pagePtr, sizeof(header::RAWDataHeader) + diff); \
    }                                                              \
  } else                                                           \
    assert(pagePtr <= payloadEnd)

  using zerosupp_link_based::ChannelPerTBHeader;

  const CfFragment& fragment = clusterer.mPmemory->fragment;

  unsigned char linkIds[MaxNLinksPerTimebin];
  unsigned char channelMasks[MaxNLinksPerTimebin * 10] = {0};
  unsigned short nSamplesWritten = 0;

  // Read timebin block header
  unsigned short tbbHdr = ConsumeByte(page);
  MAYBE_PAGE_OVERFLOW(page);
  tbbHdr |= static_cast<unsigned short>(ConsumeByte(page)) << CHAR_BIT;
  MAYBE_PAGE_OVERFLOW(page);

  unsigned char nLinksInTimebin = tbbHdr & 0x000F;
  unsigned short linkBC = (tbbHdr & 0xFFF0) >> 4;
  int timeBin = (linkBC + (unsigned long)(raw::RDHUtils::getHeartBeatOrbit(*rawDataHeader) - firstHBF) * constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;

  unsigned short nSamplesInTB = 0;

  // Read timebin link headers
  for (unsigned char iLink = 0; iLink < nLinksInTimebin; iLink++) {
    unsigned char timebinLinkHeaderStart = ConsumeByte(page);
    MAYBE_PAGE_OVERFLOW(page);

    linkIds[iLink] = timebinLinkHeaderStart & 0b00011111;

    bool bitmaskIsFlat = timebinLinkHeaderStart & 0b00100000;

    unsigned short bitmaskL2 = 0x0FFF;
    if (not bitmaskIsFlat) {
      bitmaskL2 = static_cast<unsigned short>(timebinLinkHeaderStart & 0b11000000) << 2 | static_cast<unsigned short>(ConsumeByte(page));
      MAYBE_PAGE_OVERFLOW(page);
    }

    for (int i = 9; i >= 0; i--) {
      if (bitmaskL2 & 1 << i) {
        nSamplesInTB += CAMath::Popcount(*Peek(page));
        channelMasks[10 * iLink + i] = ConsumeByte(page);
        MAYBE_PAGE_OVERFLOW(page);
      }
    }

  } // for (unsigned char iLink = 0; iLink < nLinksInTimebin; iLink++)

  const unsigned char* adcData = ConsumeBytes(page, (nSamplesInTB * DECODE_BITS + 7) / 8);
  MAYBE_PAGE_OVERFLOW(page);

  if (not fragment.contains(timeBin)) {
    FillWithInvalid(clusterer, 0, 1, pageDigitOffset, nSamplesInTB);
    return nSamplesInTB;
  }

  // Unpack ADC
  unsigned int byte = 0, bits = 0;
  unsigned short rawFECChannel = 0;

  // unpack adc values, assume tightly packed data
  while (nSamplesWritten < nSamplesInTB) {
    byte |= static_cast<unsigned int>(ConsumeByte(adcData)) << bits;
    MAYBE_PAGE_OVERFLOW(adcData);
    bits += CHAR_BIT;
    while (bits >= DECODE_BITS) {

      // Find next channel with data
      for (; !ChannelIsActive(channelMasks, rawFECChannel); rawFECChannel++) {
      }

      int iLink = rawFECChannel / ChannelPerTBHeader;
      int rawFECChannelLink = rawFECChannel % ChannelPerTBHeader;

      // Unpack data for cluster finder
      o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannelLink, linkIds[iLink]);

      float charge = ADCToFloat(byte, DECODE_MASK, DECODE_BITS_FACTOR);
      WriteCharge(clusterer, charge, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + nSamplesWritten);

      byte >>= DECODE_BITS;
      bits -= DECODE_BITS;
      nSamplesWritten++;
      rawFECChannel++; // Ensure we don't decode same channel twice
    }                  // while (bits >= DECODE_BITS)
  }                    // while (nSamplesWritten < nAdc)

  assert(PayloadExtendsToNextPage || adcData <= page);
  assert(PayloadExtendsToNextPage || page <= payloadEnd);
  assert(nSamplesWritten == nSamplesInTB);

  return nSamplesInTB;

#undef MAYBE_PAGE_OVERFLOW
}

GPUd() bool GPUTPCCFDecodeZSDenseLink::ChannelIsActive(const uint8_t* chan, unsigned short chanIndex)
{
  constexpr unsigned char N_BITS_PER_ENTRY = sizeof(*chan) * CHAR_BIT;
  const unsigned char entryIndex = chanIndex / N_BITS_PER_ENTRY;
  const unsigned char bitInEntry = chanIndex % N_BITS_PER_ENTRY;
  return chan[entryIndex] & (1 << bitInEntry);
}
