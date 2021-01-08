// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsTPC/ZeroSuppression.h"
#include "CommonConstants/LHCConstants.h"
#include "GPURawData.h"
#include "GPUCommonAlgorithm.h"

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
      const RAWDataHeaderGPU* rdh = (const RAWDataHeaderGPU*)page;
      if (GPURawDataUtils::getSize(rdh) == sizeof(RAWDataHeaderGPU)) {
#ifdef GPUCA_GPUCODE
        return;
#else
        continue;
#endif
      }
      const unsigned char* pagePtr = page + sizeof(RAWDataHeaderGPU);
      const TPCZSHDR* hdr = reinterpret_cast<const TPCZSHDR*>(pagePtr);
      pagePtr += sizeof(*hdr);
      const bool decode12bit = hdr->version == 2;
      const unsigned int decodeBits = decode12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
      const float decodeBitsFactor = 1.f / (1 << (decodeBits - 10));
      unsigned int mask = (1 << decodeBits) - 1;
      int timeBin = (hdr->timeOffset + (GPURawDataUtils::getOrbit(rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
      const int rowOffset = s.regionStartRow + ((endpoint & 1) ? (s.nRowsRegion / 2) : 0);
      const int nRows = (endpoint & 1) ? (s.nRowsRegion - s.nRowsRegion / 2) : (s.nRowsRegion / 2);

      for (int l = 0; l < hdr->nTimeBins; l++) { // TODO: Parallelize over time bins
        pagePtr += (pagePtr - page) & 1; //Ensure 16 bit alignment
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
        pagePtr += 2 * *pagePtr;                          // Go to entry for last sequence length
        pagePtr += 1 + (*pagePtr * decodeBits + 7) / 8;   // Go to beginning of next time bin
      }
    }
  }
}
