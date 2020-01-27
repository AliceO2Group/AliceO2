// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DecodeZS.cxx
/// \author David Rohr

#include "DecodeZS.h"
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

GPUd() void DecodeZS::decode(GPUTPCClusterFinder& clusterer, int nBlocks, int nThreads, int iBlock, int iThread)
{
  if (iThread) {
    return;
  }
  const unsigned int slice = clusterer.mISlice;
  const unsigned int endpoint = iBlock;
  deprecated::PackedDigit* digits = clusterer.mPdigits;
  size_t& nDigits = clusterer.mPmemory->nDigits;
  GPUTrackingInOutZS::GPUTrackingInOutZSSlice& zs = clusterer.GetConstantMem()->ioPtrs.tpcZS->slice[slice];
  unsigned short streamBuffer[TPCZSHDR::TPC_MAX_SEQ_LEN];
  for (unsigned int i = 0; i < zs.count[endpoint]; i++) {
    for (unsigned int j = 0; j < zs.nZSPtr[endpoint][i]; j++) {
      unsigned char* page = ((unsigned char*)zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      unsigned char* pagePtr = page;
      pagePtr += sizeof(o2::header::RAWDataHeader);
      TPCZSHDR* hdr = reinterpret_cast<TPCZSHDR*>(pagePtr);
      pagePtr += sizeof(*hdr);
      if (hdr->version != 1 && hdr->version != 2) {
        return;
      }
      bool decode12bit = hdr->version == 2;
      unsigned int decodeBits = decode12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
      const float decodeBitsFactor = 1.f / (1 << (decodeBits - 10));
      unsigned int mask = (1 << decodeBits) - 1;
      int cruid = hdr->cruID;
      unsigned int sector = cruid / 10;
      if (sector != slice) {
        return;
      }
      int region = cruid % 10;
      int nRowsRegion = clusterer.Param().tpcGeometry.GetRegionRows(region);

      int timeBin = hdr->timeOffset;
      for (int l = 0; l < hdr->nTimeBins; l++) {
        if ((pagePtr - page) & 1) {
          pagePtr++;
        }
        TPCZSTBHDR* tbHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
        if (tbHdr->rowMask && (endpoint & 1) != (unsigned int)(tbHdr->rowMask & 0x8000) >> 15) {
          return;
        }
        const int rowOffset = clusterer.Param().tpcGeometry.GetRegionStart(region) + ((endpoint & 1) ? (nRowsRegion / 2) : 0);
        const int nRows = (endpoint & 1) ? (nRowsRegion - nRowsRegion / 2) : nRowsRegion;
        const int nRowsUsed = CAMath::Popcount((unsigned int)(tbHdr->rowMask & 0x7FFF));
        pagePtr += nRowsUsed ? (2 * nRowsUsed) : 2;
        int rowPos = 0;
        for (int m = 0; m < nRows; m++) {
          if ((tbHdr->rowMask & (1 << m)) == 0) {
            continue;
          }
          unsigned char* rowData = rowPos == 0 ? pagePtr : (page + tbHdr->rowAddr1[rowPos - 1]);
          const int nSeqRead = *rowData;
          unsigned char* adcData = rowData + 2 * nSeqRead + 1;
          int nADC = (rowData[2 * nSeqRead] * decodeBits + 7) / 8;
          pagePtr += 1 + 2 * nSeqRead + nADC;
          unsigned int byte = 0, bits = 0, pos10 = 0;
          for (int n = 0; n < nADC; n++) {
            byte |= *(adcData++) << bits;
            bits += 8;
            while (bits >= decodeBits) {
              streamBuffer[pos10++] = byte & mask;
              byte = byte >> decodeBits;
              bits -= decodeBits;
            }
          }
          pos10 = 0;
          for (int n = 0; n < nSeqRead; n++) {
            const int seqLen = rowData[(n + 1) * 2] - (n ? rowData[n * 2] : 0);
            for (int o = 0; o < seqLen; o++) {
              digits[nDigits++] = deprecated::PackedDigit{(float)streamBuffer[pos10++] * decodeBitsFactor, (Timestamp)(timeBin + l), (Pad)(rowData[n * 2 + 1] + o), (Row)(rowOffset + m)};
            }
          }
          rowPos++;
        }
      }
    }
  }
}
