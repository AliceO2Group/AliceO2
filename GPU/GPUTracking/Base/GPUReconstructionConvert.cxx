// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionConvert.cxx
/// \author David Rohr

#include "GPUReconstructionConvert.h"
#include "TPCFastTransform.h"
#include "GPUTPCClusterData.h"
#include "GPUO2DataTypes.h"
#include "GPUDataTypes.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUParam.h"
#include <algorithm>
#include <vector>

#ifdef HAVE_O2HEADERS
#include "Digit.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "Headers/RAWDataHeader.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

void GPUReconstructionConvert::ConvertNativeToClusterData(o2::tpc::ClusterNativeAccess* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin)
{
#ifdef HAVE_O2HEADERS
  memset(nClusters, 0, NSLICES * sizeof(nClusters[0]));
  unsigned int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    unsigned int nClSlice = 0;
    for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
      nClSlice += native->nClusters[i][j];
    }
    nClusters[i] = nClSlice;
    clusters[i].reset(new GPUTPCClusterData[nClSlice]);
    nClSlice = 0;
    for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
      for (unsigned int k = 0; k < native->nClusters[i][j]; k++) {
        const auto& clin = native->clusters[i][j][k];
        float x = 0, y = 0, z = 0;
        if (continuousMaxTimeBin == 0) {
          transform->Transform(i, j, clin.getPad(), clin.getTime(), x, y, z);
        } else {
          transform->TransformInTimeFrame(i, j, clin.getPad(), clin.getTime(), x, y, z, continuousMaxTimeBin);
        }
        auto& clout = clusters[i].get()[nClSlice];
        clout.x = x;
        clout.y = y;
        clout.z = z;
        clout.row = j;
        clout.amp = clin.qTot;
        clout.flags = clin.getFlags();
        clout.id = offset + k;
        nClSlice++;
      }
      native->clusterOffset[i][j] = offset;
      offset += native->nClusters[i][j];
    }
  }
#endif
}

void GPUReconstructionConvert::ConvertRun2RawToNative(o2::tpc::ClusterNativeAccess& native, std::unique_ptr<ClusterNative[]>& nativeBuffer, const AliHLTTPCRawCluster** rawClusters, unsigned int* nRawClusters)
{
#ifdef HAVE_O2HEADERS
  memset((void*)&native, 0, sizeof(native));
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < nRawClusters[i]; j++) {
      native.nClusters[i][rawClusters[i][j].GetPadRow()]++;
    }
    native.nClustersTotal += nRawClusters[i];
  }
  nativeBuffer.reset(new ClusterNative[native.nClustersTotal]);
  native.clustersLinear = nativeBuffer.get();
  native.setOffsetPtrs();
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      native.nClusters[i][j] = 0;
    }
    for (unsigned int j = 0; j < nRawClusters[i]; j++) {
      const AliHLTTPCRawCluster& org = rawClusters[i][j];
      int row = org.GetPadRow();
      ClusterNative& c = nativeBuffer[native.clusterOffset[i][row] + native.nClusters[i][row]++];
      c.setTimeFlags(org.GetTime(), org.GetFlags());
      c.setPad(org.GetPad());
      c.setSigmaTime(std::sqrt(org.GetSigmaTime2()));
      c.setSigmaPad(std::sqrt(org.GetSigmaPad2()));
      c.qMax = org.GetQMax();
      c.qTot = org.GetCharge();
    }
  }
#endif
}

int GPUReconstructionConvert::GetMaxTimeBin(const ClusterNativeAccess& native)
{
#ifdef HAVE_O2HEADERS
  float retVal = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      for (unsigned int k = 0; k < native.nClusters[i][j]; k++) {
        if (native.clusters[i][j][k].getTime() > retVal) {
          retVal = native.clusters[i][j][k].getTime();
        }
      }
    }
  }
  return ceil(retVal);
#else
  return 0;
#endif
}

int GPUReconstructionConvert::GetMaxTimeBin(const GPUTrackingInOutDigits& digits)
{
#ifdef HAVE_O2HEADERS
  float retVal = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int k = 0; k < digits.nTPCDigits[i]; k++) {
      if (digits.tpcDigits[i][k].time > retVal) {
        retVal = digits.tpcDigits[i][k].time;
      }
    }
  }
  return ceil(retVal);
#else
  return 0;
#endif
}

int GPUReconstructionConvert::GetMaxTimeBin(const GPUTrackingInOutZS& zspages)
{
#ifdef HAVE_O2HEADERS
  float retVal = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      for (unsigned int k = 0; k < zspages.slice[i].count[j]; k++) {
        const char* page = (const char*)zspages.slice[i].zsPtr[j][k];
        for (unsigned int l = 0; l < zspages.slice[i].nZSPtr[j][k]; l++) {
          TPCZSHDR* hdr = (TPCZSHDR*)(page + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
          unsigned int timeBin = (unsigned int)hdr->timeOffset + hdr->nTimeBins;
          if (timeBin > retVal) {
            retVal = timeBin;
          }
        }
      }
    }
  }
  return ceil(retVal);
#else
  return 0;
#endif
}

void GPUReconstructionConvert::ZSstreamOut(unsigned short* bufIn, unsigned int& lenIn, unsigned char* bufOut, unsigned int& lenOut, unsigned int nBits)
{
  unsigned int byte = 0, bits = 0;
  unsigned int mask = (1 << nBits) - 1;
  for (unsigned int i = 0; i < lenIn; i++) {
    byte |= (bufIn[i] & mask) << bits;
    bits += nBits;
    while (bits >= 8) {
      bufOut[lenOut++] = (unsigned char)(byte & 0xFF);
      byte = byte >> 8;
      bits -= 8;
    }
  }
  if (bits) {
    bufOut[lenOut++] = byte;
  }
  lenIn = 0;
}

void GPUReconstructionConvert::RunZSEncoder(const GPUTrackingInOutDigits* in, GPUTrackingInOutZS*& out, const GPUParam& param, bool zs12bit)
{
#ifdef GPUCA_TPC_GEOMETRY_O2
  static std::vector<std::array<long long int, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(long long int)>> buffer[NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  static GPUTrackingInOutZS tmp;
  static void* ptrs[NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  static unsigned int sizes[NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  out = &tmp;

  std::vector<deprecated::PackedDigit> tmpBuffer;
  std::array<unsigned short, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer;
  std::array<unsigned char, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer8;
  size_t nErrors = 0;
  int encodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
  const float encodeBitsFactor = (1 << (encodeBits - 10));
  for (unsigned int i = 0; i < NSLICES; i++) {
    tmpBuffer.resize(in->nTPCDigits[i]);
    std::copy(in->tpcDigits[i], in->tpcDigits[i] + in->nTPCDigits[i], tmpBuffer.begin());
    std::sort(tmpBuffer.begin(), tmpBuffer.end(), [&param](const deprecated::PackedDigit a, const deprecated::PackedDigit b) {
      int endpointa = param.tpcGeometry.GetRegion(a.row);
      int endpointb = param.tpcGeometry.GetRegion(b.row);
      endpointa = 2 * endpointa + (a.row >= param.tpcGeometry.GetRegionStart(endpointa) + param.tpcGeometry.GetRegionRows(endpointa) / 2);
      endpointb = 2 * endpointb + (b.row >= param.tpcGeometry.GetRegionStart(endpointb) + param.tpcGeometry.GetRegionRows(endpointb) / 2);
      if (endpointa != endpointb) {
        return endpointa <= endpointb;
      }
      if (a.time != b.time) {
        return a.time <= b.time;
      }
      if (a.row != b.row) {
        return a.row <= b.row;
      }
      return a.pad <= b.pad;
    });
    int lastEndpoint = -1, lastRow = GPUCA_ROW_COUNT, lastTime = -1;
    std::array<long long int, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(long long int)>* page = nullptr;
    TPCZSHDR* hdr = nullptr;
    TPCZSTBHDR* tbHdr = nullptr;
    unsigned char* pagePtr = nullptr;
    unsigned char* nSeq = nullptr;
    int nRowsInTB = 0;
    int region = 0, endpoint = 0, endpointStart = 0;
    unsigned int streamSize = 0, streamSize8 = 0;
    for (unsigned int k = 0; k <= tmpBuffer.size(); k++) {
      int seqLen = 1;
      if (k < tmpBuffer.size()) {
        if (lastRow != tmpBuffer[k].row) {
          region = param.tpcGeometry.GetRegion(tmpBuffer[k].row);
          endpointStart = param.tpcGeometry.GetRegionStart(region);
          endpoint = region * 2;
          if (tmpBuffer[k].row >= endpointStart + param.tpcGeometry.GetRegionRows(region) / 2) {
            endpoint++;
            endpointStart += param.tpcGeometry.GetRegionRows(region) / 2;
          }
        }
        for (unsigned int l = k + 1; l < tmpBuffer.size(); l++) {
          if (tmpBuffer[l].row == tmpBuffer[k].row && tmpBuffer[l].time == tmpBuffer[k].time && tmpBuffer[l].pad == tmpBuffer[l - 1].pad + 1) {
            seqLen++;
          } else {
            break;
          }
        }
        unsigned int sizeChk = (unsigned int)(pagePtr - reinterpret_cast<unsigned char*>(page));                    // already written
        sizeChk += 2 * (nRowsInTB + (tmpBuffer[k].row != lastRow));                                                 // TB HDR
        sizeChk += streamSize8;                                                                                     // in stream buffer
        sizeChk += (tmpBuffer[k].time != lastTime || tmpBuffer[k].row != lastRow) ? 3 : 0;                          // new row overhead
        sizeChk += (lastTime != -1 && tmpBuffer[k].time > lastTime) ? ((tmpBuffer[k].time - lastTime - 1) * 2) : 0; // empty time bins
        sizeChk += 2;                                                                                               // sequence metadata
        if (sizeChk + ((1 + streamSize) * encodeBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
          lastEndpoint = -1;
        } else if (sizeChk + ((seqLen + streamSize) * encodeBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
          seqLen = (TPCZSHDR::TPC_ZS_PAGE_SIZE - sizeChk) * 8 / encodeBits - streamSize;
        }
        if (lastTime != -1 && (int)hdr->nTimeBins + tmpBuffer[k].time - lastTime >= 256) {
          lastEndpoint = -1;
        }
        // printf("Endpoint %d (%d), Pos %d, Chk %d, Len %d, rows %d, StreamSize %d %d, time %d (%d), row %d (%d)\n", endpoint, lastEndpoint, (int) (pagePtr - reinterpret_cast<unsigned char*>(page)), sizeChk, seqLen, nRowsInTB, streamSize8, streamSize, (int) tmpBuffer[k].time, lastTime, (int) tmpBuffer[k].row, lastRow);
      }
      if (k >= tmpBuffer.size() || endpoint != lastEndpoint || tmpBuffer[k].time != lastTime) {
        if (pagePtr != reinterpret_cast<unsigned char*>(page)) {
          pagePtr += 2 * nRowsInTB;
          ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
          pagePtr = std::copy(streamBuffer8.data(), streamBuffer8.data() + streamSize8, pagePtr);
          streamSize8 = 0;
          for (int l = 1; l < nRowsInTB; l++) {
            tbHdr->rowAddr1()[l - 1] += 2 * nRowsInTB;
          }
        }
        if (k >= tmpBuffer.size()) {
          break;
        }
      }
      if (endpoint != lastEndpoint) {
        buffer[i][endpoint].emplace_back();
        page = &buffer[i][endpoint].back();
        pagePtr = reinterpret_cast<unsigned char*>(page);
        std::fill(page->begin(), page->end(), 0);
        pagePtr += sizeof(o2::header::RAWDataHeader);
        hdr = reinterpret_cast<TPCZSHDR*>(pagePtr);
        pagePtr += sizeof(*hdr);
        hdr->version = zs12bit ? 2 : 1;
        hdr->cruID = i * 10 + region;
        hdr->timeOffset = tmpBuffer[k].time;
        lastTime = -1;
        tbHdr = nullptr;
        lastEndpoint = endpoint;
      }
      if (tmpBuffer[k].time != lastTime) {
        if (lastTime != -1) {
          hdr->nTimeBins += tmpBuffer[k].time - lastTime - 1;
          pagePtr += (tmpBuffer[k].time - lastTime - 1) * 2;
        }
        hdr->nTimeBins++;
        lastTime = tmpBuffer[k].time;
        if ((pagePtr - reinterpret_cast<unsigned char*>(page)) & 1) {
          pagePtr++;
        }
        tbHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
        tbHdr->rowMask |= (endpoint & 1) << 15;
        nRowsInTB = 0;
        lastRow = GPUCA_ROW_COUNT;
      }
      if (tmpBuffer[k].row != lastRow) {
        tbHdr->rowMask |= 1 << (tmpBuffer[k].row - endpointStart);
        lastRow = tmpBuffer[k].row;
        ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
        if (nRowsInTB) {
          tbHdr->rowAddr1()[nRowsInTB - 1] = (pagePtr - reinterpret_cast<unsigned char*>(page)) + streamSize8;
        }
        nRowsInTB++;
        nSeq = streamBuffer8.data() + streamSize8++;
        *nSeq = 0;
      }
      (*nSeq)++;
      streamBuffer8[streamSize8++] = tmpBuffer[k].pad;
      streamBuffer8[streamSize8++] = streamSize + seqLen;
      hdr->nADCsamples += seqLen;
      for (int l = 0; l < seqLen; l++) {
        streamBuffer[streamSize++] = (unsigned short)(tmpBuffer[k + l].charge * encodeBitsFactor + 0.5f);
      }
      k += seqLen - 1;
    }

    // Verification
    std::vector<deprecated::PackedDigit> compareBuffer;
    compareBuffer.reserve(tmpBuffer.size());
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      for (unsigned int k = 0; k < buffer[i][j].size(); k++) {
        page = &buffer[i][j][k];
        pagePtr = reinterpret_cast<unsigned char*>(page);
        pagePtr += sizeof(o2::header::RAWDataHeader);
        hdr = reinterpret_cast<TPCZSHDR*>(pagePtr);
        pagePtr += sizeof(*hdr);
        if (hdr->version != 1 && hdr->version != 2) {
          throw std::runtime_error("invalid ZS version");
        }
        const bool decode12bit = hdr->version == 2;
        const unsigned int decodeBits = decode12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
        const float decodeBitsFactor = 1.f / (1 << (decodeBits - 10));
        unsigned int mask = (1 << decodeBits) - 1;
        int cruid = hdr->cruID;
        unsigned int sector = cruid / 10;
        if (sector != i) {
          throw std::runtime_error("invalid TPC sector");
        }
        region = cruid % 10;
        if ((unsigned int)region != j / 2) {
          throw std::runtime_error("CRU ID / endpoint mismatch");
        }
        int nRowsRegion = param.tpcGeometry.GetRegionRows(region);

        int timeBin = hdr->timeOffset;
        for (int l = 0; l < hdr->nTimeBins; l++) {
          if ((pagePtr - reinterpret_cast<unsigned char*>(page)) & 1) {
            pagePtr++;
          }
          tbHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
          bool upperRows = tbHdr->rowMask & 0x8000;
          if (tbHdr->rowMask != 0 && ((upperRows) ^ ((j & 1) != 0))) {
            throw std::runtime_error("invalid endpoint");
          }
          const int rowOffset = param.tpcGeometry.GetRegionStart(region) + (upperRows ? (nRowsRegion / 2) : 0);
          const int nRows = upperRows ? (nRowsRegion - nRowsRegion / 2) : (nRowsRegion / 2);
          const int nRowsUsed = __builtin_popcount((unsigned int)(tbHdr->rowMask & 0x7FFF));
          pagePtr += nRowsUsed ? (2 * nRowsUsed) : 2;
          int rowPos = 0;
          for (int m = 0; m < nRows; m++) {
            if ((tbHdr->rowMask & (1 << m)) == 0) {
              continue;
            }
            unsigned char* rowData = rowPos == 0 ? pagePtr : (reinterpret_cast<unsigned char*>(page) + tbHdr->rowAddr1()[rowPos - 1]);
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
                compareBuffer.emplace_back(deprecated::PackedDigit{(float)streamBuffer[pos10++] * decodeBitsFactor, (Timestamp)(timeBin + l), (Pad)(rowData[n * 2 + 1] + o), (Row)(rowOffset + m)});
              }
            }
            rowPos++;
          }
        }
      }
    }
    for (unsigned int j = 0; j < tmpBuffer.size(); j++) {
      const unsigned int decodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
      const float decodeBitsFactor = (1 << (decodeBits - 10));
      const float c = zs12bit ? (float)((int)(tmpBuffer[j].charge * decodeBitsFactor + 0.5f)) / decodeBitsFactor : (float)(int)(tmpBuffer[j].charge + 0.5f);
      int ok = c == compareBuffer[j].charge && (int)tmpBuffer[j].time == (int)compareBuffer[j].time && (int)tmpBuffer[j].pad == (int)compareBuffer[j].pad && (int)tmpBuffer[j].row == (int)compareBuffer[j].row;
      if (ok) {
        continue;
      }
      nErrors++;
      printf("%4u: OK %d: Charge %3d %3d Time %4d %4d Pad %3d %3d Row %3d %3d\n", j, ok,
             (int)c, (int)compareBuffer[j].charge, (int)tmpBuffer[j].time, (int)compareBuffer[j].time, (int)tmpBuffer[j].pad, (int)compareBuffer[j].pad, (int)tmpBuffer[j].row, (int)compareBuffer[j].row);
    }
  }

  if (nErrors) {
    printf("%lld ERRORS DURING ZS!", (long long int)nErrors);
  }

  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      ptrs[i][j] = buffer[i][j].data();
      sizes[i][j] = buffer[i][j].size();
      tmp.slice[i].zsPtr[j] = &ptrs[i][j];
      tmp.slice[i].nZSPtr[j] = &sizes[i][j];
      tmp.slice[i].count[j] = 1;
    }
  }
#endif
}

void GPUReconstructionConvert::RunZSFilter(std::unique_ptr<deprecated::PackedDigit[]>* buffers, const deprecated::PackedDigit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit)
{
#ifdef HAVE_O2HEADERS
  for (unsigned int i = 0; i < NSLICES; i++) {
    if (buffers[i].get() != ptrs[i] || nsb != ns) {
      throw std::runtime_error("Not owning digits");
    }
    unsigned int j = 0;
    const unsigned int decodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
    const float decodeBitsFactor = (1 << (decodeBits - 10));
    for (unsigned int k = 0; k < ns[i]; k++) {
      if (buffers[i][k].charge >= param.rec.tpcZSthreshold) {
        if (k > j) {
          buffers[i][j] = buffers[i][k];
        }
        if (zs12bit) {
          buffers[i][j].charge = (float)((int)(buffers[i][j].charge * decodeBitsFactor + 0.5f)) / decodeBitsFactor;
        } else {
          buffers[i][j].charge = (float)((int)(buffers[i][j].charge + 0.5f));
        }
        j++;
      }
    }
    nsb[i] = j;
  }
#endif
}
