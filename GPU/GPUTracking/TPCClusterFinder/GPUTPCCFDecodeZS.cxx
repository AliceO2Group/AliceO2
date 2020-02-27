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
#include "DataFormatsTPC/ZeroSuppression.h"

#ifndef __OPENCL__
#include "Headers/RAWDataHeader.h"
#else
namespace o2
{
namespace header
{
struct RAWDataHeader {
  unsigned int words[16];
};
} // namespace header
} // namespace o2

#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCCFDecodeZS::Thread<GPUTPCCFDecodeZS::decodeZS>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  GPUTPCCFDecodeZS::decode(clusterer, smem, nBlocks, nThreads, iBlock, iThread);
}

GPUd() void GPUTPCCFDecodeZS::decode(GPUTPCClusterFinder& clusterer, GPUSharedMemory& s, int nBlocks, int nThreads, int iBlock, int iThread)
{
  const unsigned int slice = clusterer.mISlice;
  const unsigned int endpoint = iBlock;
  const GPUTrackingInOutZS::GPUTrackingInOutZSSlice& zs = clusterer.GetConstantMem()->ioPtrs.tpcZS->slice[slice];
  if (zs.count[endpoint] == 0) {
    return;
  }
  deprecated::PackedDigit* digits = clusterer.mPdigits;
  const size_t nDigits = clusterer.mPmemory->nDigitsOffset[endpoint];
  unsigned int rowOffsetCounter = 0;
  if (iThread == 0) {
    const int region = endpoint / 2;
    s.nRowsRegion = clusterer.Param().tpcGeometry.GetRegionRows(region);
    s.regionStartRow = clusterer.Param().tpcGeometry.GetRegionStart(region);
    s.nThreadsPerRow = CAMath::Max(1u, nThreads / ((s.nRowsRegion + (endpoint & 1)) / 2));
    s.rowStride = nThreads / s.nThreadsPerRow;
    const unsigned char* page = (const unsigned char*)zs.zsPtr[endpoint][0];
    const TPCZSHDR* hdr = reinterpret_cast<const TPCZSHDR*>(page + sizeof(o2::header::RAWDataHeader));
    const bool decode12bit = hdr->version == 2;
    s.decodeBits = decode12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
    s.decodeBitsFactor = 1.f / (1 << (s.decodeBits - 10));
  }
  GPUbarrier();
  const unsigned int myRow = iThread / s.nThreadsPerRow;
  const unsigned int mySequence = iThread % s.nThreadsPerRow;
  for (unsigned int i = 0; i < zs.count[endpoint]; i++) {
    for (unsigned int j = 0; j < zs.nZSPtr[endpoint][i]; j++) {
      const unsigned int* pageSrc = (const unsigned int*)(((const unsigned char*)zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE);
      GPUbarrier();
      CA_SHARED_CACHE_REF(&s.ZSPage[0], pageSrc, TPCZSHDR::TPC_ZS_PAGE_SIZE, unsigned int, pageCache);
      GPUbarrier();
      const unsigned char* page = (const unsigned char*)pageCache;
      const unsigned char* pagePtr = page + sizeof(o2::header::RAWDataHeader);
      const TPCZSHDR* hdr = reinterpret_cast<const TPCZSHDR*>(pagePtr);
      pagePtr += sizeof(*hdr);
      unsigned int mask = (1 << s.decodeBits) - 1;
      int timeBin = hdr->timeOffset;
      for (int l = 0; l < hdr->nTimeBins; l++) {
        pagePtr += (pagePtr - page) & 1; //Ensure 16 bit alignment
        const TPCZSTBHDR* tbHdr = reinterpret_cast<const TPCZSTBHDR*>(pagePtr);
        if ((tbHdr->rowMask & 0x7FFF) == 0) {
          pagePtr += 2;
          continue;
        }
        const int rowOffset = s.regionStartRow + ((endpoint & 1) ? (s.nRowsRegion / 2) : 0);
        const int nRows = (endpoint & 1) ? (s.nRowsRegion - s.nRowsRegion / 2) : (s.nRowsRegion / 2);
        const int nRowsUsed = CAMath::Popcount((unsigned int)(tbHdr->rowMask & 0x7FFF));
        pagePtr += 2 * nRowsUsed;
        GPUbarrier();
        if (iThread == 0) {
          for (int n = 0; n < nRowsUsed; n++) {
            s.RowClusterOffset[n] = rowOffsetCounter;
            const unsigned char* rowData = n == 0 ? pagePtr : (page + tbHdr->rowAddr1()[n - 1]);
            rowOffsetCounter += rowData[2 * *rowData]; // Sum up number of ADC samples per row to compute offset in target buffer
          }
        }
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
              unsigned int nADCStartBits = nSamplesStart * s.decodeBits;
              const unsigned int nADCStart = (nADCStartBits + 7) / 8;
              const int nADC = (rowData[2 * mySequenceEnd] * s.decodeBits + 7) / 8;
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
                while (bits >= s.decodeBits) {
                  if (seqLen == 0) {
                    seqLen = rowData[(nSeq + 1) * 2] - rowData[nSeq * 2];
                    pad = rowData[nSeq++ * 2 + 1];
                  }
                  digits[nDigitsTmp++] = deprecated::PackedDigit{(float)(byte & mask) * s.decodeBitsFactor, (Timestamp)(timeBin + l), pad++, (Row)(rowOffset + m)};
                  byte = byte >> s.decodeBits;
                  bits -= s.decodeBits;
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
        pagePtr += 1 + (*pagePtr * s.decodeBits + 7) / 8; // Go to beginning of next time bin
      }
    }
  }
}
