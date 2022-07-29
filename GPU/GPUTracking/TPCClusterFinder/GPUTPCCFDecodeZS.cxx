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
/// \author David Rohr

#include "GPUTPCCFDecodeZS.h"
#include "GPUCommonMath.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "CommonConstants/LHCConstants.h"
#include "GPUCommonAlgorithm.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCPadGainCalib.h"
#include "TPCZSLinkMapping.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;
using namespace o2::tpc;
using namespace o2::tpc::constants;

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

      for (int l = 0; l < hdr->nTimeBins; l++) { // TODO: Parallelize over time bins
        pagePtr += (pagePtr - page) & 1;         // Ensure 16 bit alignment
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

template <>
GPUdii() void GPUTPCCFDecodeZSLink::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int firstHBF)
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
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  size_t pageDigitOffset = clusterer.mPzsOffsets[iBlock].offset;

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

      const auto* rdHdr = ConsumeHeader<header::RAWDataHeader>(page);

      if (o2::raw::RDHUtils::getMemorySize(*rdHdr) == sizeof(o2::header::RAWDataHeader)) {
#ifdef GPUCA_GPUCODE
        return;
#else
        continue;
#endif
      }

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

        // TimeBin not in fragment: Skip this timebin header and fill positions with dummy values instead
        if (not inFragment) {
          for (unsigned int a = iThread; a < nAdc; a += nThreads) {
            constexpr ChargePos INVALID_POS(UCHAR_MAX, UCHAR_MAX, INVALID_TIME_BIN);
            clusterer.mPpositions[pageDigitOffset + a] = INVALID_POS;
          }
          pageDigitOffset += nAdc;
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
    }   // [CPU] for (unsigned int j = minJ; j < maxJ; j++)
  }     // [CPU] for (unsigned int i = clusterer.mMinMaxCN[endpoint].minC; i < clusterer.mMinMaxCN[endpoint].maxC; i++)
}

GPUd() void GPUTPCCFDecodeZSLink::DecodeTBSingleThread(
  processorType& clusterer,
  const unsigned char* adcData,
  unsigned int nAdc,
  const uint32_t* channelMask,
  int timeBin,
  int cru,
  int fecInPartition,
  size_t pageDigitOffset)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;

  if CONSTEXPR17 (TPCZSHDRV2::TIGHTLY_PACKED_V3) {

    unsigned int byte = 0, bits = 0, posXbits = 0, rawFECChannel = 0;

    // unpack adc values, assume tightly packed data
    while (posXbits < nAdc) {
      byte |= adcData[0] << bits;
      adcData++;
      bits += CHAR_BIT;
      while (bits >= DECODE_BITS) {

        // Find next channel with data
        for (; !ChannelIsActive(channelMask, rawFECChannel); rawFECChannel++) {
        }

        // Unpack data for cluster finder
        o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannel, fecInPartition);

        WriteCharge(clusterer, byte, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + posXbits);

        byte = byte >> DECODE_BITS;
        bits -= DECODE_BITS;
        posXbits++;
        rawFECChannel++; // Ensure we don't decode same channel twice
      }                  // while (bits >= DECODE_BITS)
    }                    // while (posXbits < nAdc)

  } else { // ! TPCZSHDRV2::TIGHTLY_PACKED_V3
    unsigned int rawFECChannel = 0;
    const unsigned long* adcData64 = (const unsigned long*)adcData;
    for (unsigned int j = 0; j < nAdc; j++) {
      for (; !ChannelIsActive(channelMask, rawFECChannel); rawFECChannel++) {
      }

      unsigned int adc = (adcData64[j / TPCZSHDRV2::SAMPLESPER64BIT] >> ((j % TPCZSHDRV2::SAMPLESPER64BIT) * DECODE_BITS)) & DECODE_MASK;

      o2::tpc::PadPos padAndRow = GetPadAndRowFromFEC(clusterer, cru, rawFECChannel, fecInPartition);
      WriteCharge(clusterer, adc, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + j);
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
  size_t pageDigitOffset)
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
    WriteCharge(clusterer, adc, padAndRow, fragment.toLocal(timeBin), pageDigitOffset + myOffset);

  } // for (unsigned char i = iThread; blockOffset < nAdc; i += NThreads)
}

GPUd() void GPUTPCCFDecodeZSLink::WriteCharge(processorType& clusterer, unsigned int adc, PadPos padAndRow, TPCFragmentTime localTime, size_t positionOffset)
{
  const unsigned int slice = clusterer.mISlice;
  ChargePos* positions = clusterer.mPpositions;
  if (padAndRow.getRow() >= GPUCA_ROW_COUNT) { // FIXME: to be removed once TPC does not send corrupt data any more
    constexpr ChargePos INVALID_POS(UCHAR_MAX, UCHAR_MAX, INVALID_TIME_BIN);
    positions[positionOffset] = INVALID_POS;
    return;
  }

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  ChargePos pos(padAndRow.getRow(), padAndRow.getPad(), localTime);
  positions[positionOffset] = pos;

  float q = float(adc & DECODE_MASK) * DECODE_BITS_FACTOR;
  q *= clusterer.GetConstantMem()->calibObjects.tpcPadGain->getGainCorrection(slice, padAndRow.getRow(), padAndRow.getPad());
  chargeMap[pos] = PackedCharge(q);
}

GPUd() o2::tpc::PadPos GPUTPCCFDecodeZSLink::GetPadAndRowFromFEC(processorType& clusterer, int cru, int rawFECChannel, int fecInPartition)
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

  unsigned int globalSAMPAId = (static_cast<uint32_t>(fecInSector) << 8) + (static_cast<uint32_t>(sampaOnFEC) << 5) + static_cast<uint32_t>(channelOnSAMPA);
  const o2::tpc::PadPos pos = gpuMapping->FECIDToPadPos[globalSAMPAId];

  return pos;
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
