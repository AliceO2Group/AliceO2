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

#ifdef GPUCA_O2_LIB
#include "DetectorsRaw/RawFileWriter.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#endif

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
#include "clusterFinderDefs.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Constants.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/RDHUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::tpc::constants;

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
      if (digits.tpcDigits[i][k].getTimeStamp() > retVal) {
        retVal = digits.tpcDigits[i][k].getTimeStamp();
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
    int firstHBF = zspages.slice[i].count[0] ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)zspages.slice[i].zsPtr[0][0]) : 0;
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      for (unsigned int k = 0; k < zspages.slice[i].count[j]; k++) {
        const char* page = (const char*)zspages.slice[i].zsPtr[j][k];
        for (unsigned int l = 0; l < zspages.slice[i].nZSPtr[j][k]; l++) {
          o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)(page + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          TPCZSHDR* hdr = (TPCZSHDR*)(page + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
          unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN + hdr->nTimeBins;
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

#ifdef HAVE_O2HEADERS
void GPUReconstructionConvert::ZSfillEmpty(void* ptr, int shift, unsigned int feeId, int orbit)
{
  o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)ptr;
  o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, orbit);
  o2::raw::RDHUtils::setHeartBeatBC(*rdh, shift);
  o2::raw::RDHUtils::setMemorySize(*rdh, sizeof(o2::header::RAWDataHeader));
  o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
  o2::raw::RDHUtils::setFEEID(*rdh, feeId);
}

static inline auto ZSEncoderGetDigits(const GPUTrackingInOutDigits& in, int i) { return in.tpcDigits[i]; }
static inline auto ZSEncoderGetNDigits(const GPUTrackingInOutDigits& in, int i) { return in.nTPCDigits[i]; }
template void GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit, GPUTrackingInOutDigits>(const GPUTrackingInOutDigits&, std::unique_ptr<unsigned long long int[]>*, unsigned int*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, bool, bool, float, bool);
#ifdef GPUCA_O2_LIB
using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::Sector::MAXSECTOR>;
template void GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit, DigitArray>(const DigitArray&, std::unique_ptr<unsigned long long int[]>*, unsigned int*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, bool, bool, float, bool);
static inline auto ZSEncoderGetDigits(const DigitArray& in, int i) { return in[i].data(); }
static inline auto ZSEncoderGetNDigits(const DigitArray& in, int i) { return in[i].size(); }
#endif
static inline auto ZSEncoderGetTime(const o2::tpc::Digit a) { return a.getTimeStamp(); }
static inline auto ZSEncoderGetPad(const o2::tpc::Digit a) { return a.getPad(); }
static inline auto ZSEncoderGetRow(const o2::tpc::Digit a) { return a.getRow(); }
static inline auto ZSEncoderGetCharge(const o2::tpc::Digit a) { return a.getChargeFloat(); }
#endif

template <class T, class S>
void GPUReconstructionConvert::RunZSEncoder(const S& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, const GPUParam& param, bool zs12bit, bool verify, float threshold, bool padding)
{
  // Pass in either outBuffer / outSizes, to fill standalone output buffers, or raw / ir to use RawFileWriter
  // ir is the interaction record for time bin 0
  if (((outBuffer == nullptr) ^ (outSizes == nullptr)) || ((raw != nullptr) && (ir == nullptr)) || !((outBuffer == nullptr) ^ (raw == nullptr)) || (raw && verify) || (threshold > 0.f && verify)) {
    throw std::runtime_error("Invalid parameters");
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::vector<std::array<long long int, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(long long int)>> buffer[NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  unsigned int totalPages = 0;
  size_t nErrors = 0;
  int encodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
  const float encodeBitsFactor = (1 << (encodeBits - 10));
  // clang-format off
  GPUCA_OPENMP(parallel for reduction(+ : totalPages) reduction(+ : nErrors))
  // clang-format on
  for (unsigned int i = 0; i < NSLICES; i++) {
    std::array<long long int, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(long long int)> singleBuffer;
#ifdef GPUCA_O2_LIB
    int rawlnk = rdh_utils::UserLogicLinkID;
    int bcShiftInFirstHBF = ir ? ir->bc : 0;
    int orbitShift = ir ? ir->orbit : 0;
#else
    int rawlnk = 15;
    int bcShiftInFirstHBF = 0;
    int orbitShift = 0;
#endif
    int rawcru = 0;
    int rawendpoint = 0;
    (void)(rawcru + rawendpoint); // avoid compiler warning

    std::vector<T> tmpBuffer;
    std::array<unsigned short, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer;
    std::array<unsigned char, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer8;
    tmpBuffer.resize(ZSEncoderGetNDigits(in, i));
    if (threshold > 0.f) {
      auto it = std::copy_if(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin(), [threshold](auto& v) { return ZSEncoderGetCharge(v) >= threshold; });
      tmpBuffer.resize(std::distance(tmpBuffer.begin(), it));
    } else {
      std::copy(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin());
    }
    std::sort(tmpBuffer.begin(), tmpBuffer.end(), [&param](const T a, const T b) {
      int endpointa = param.tpcGeometry.GetRegion(ZSEncoderGetRow(a));
      int endpointb = param.tpcGeometry.GetRegion(ZSEncoderGetRow(b));
      endpointa = 2 * endpointa + (ZSEncoderGetRow(a) >= param.tpcGeometry.GetRegionStart(endpointa) + param.tpcGeometry.GetRegionRows(endpointa) / 2);
      endpointb = 2 * endpointb + (ZSEncoderGetRow(b) >= param.tpcGeometry.GetRegionStart(endpointb) + param.tpcGeometry.GetRegionRows(endpointb) / 2);
      if (endpointa != endpointb) {
        return endpointa <= endpointb;
      }
      if (ZSEncoderGetTime(a) != ZSEncoderGetTime(b)) {
        return ZSEncoderGetTime(a) <= ZSEncoderGetTime(b);
      }
      if (ZSEncoderGetRow(a) != ZSEncoderGetRow(b)) {
        return ZSEncoderGetRow(a) <= ZSEncoderGetRow(b);
      }
      return ZSEncoderGetPad(a) < ZSEncoderGetPad(b);
    });
    int lastEndpoint = -1, lastRow = GPUCA_ROW_COUNT, lastTime = -1;
    long hbf = -1, nexthbf = 0;
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
        if (lastRow != ZSEncoderGetRow(tmpBuffer[k])) {
          region = param.tpcGeometry.GetRegion(ZSEncoderGetRow(tmpBuffer[k]));
          endpointStart = param.tpcGeometry.GetRegionStart(region);
          endpoint = region * 2;
          if (ZSEncoderGetRow(tmpBuffer[k]) >= endpointStart + param.tpcGeometry.GetRegionRows(region) / 2) {
            endpoint++;
            endpointStart += param.tpcGeometry.GetRegionRows(region) / 2;
          }
        }
        for (unsigned int l = k + 1; l < tmpBuffer.size(); l++) {
          if (ZSEncoderGetRow(tmpBuffer[l]) == ZSEncoderGetRow(tmpBuffer[k]) && ZSEncoderGetTime(tmpBuffer[l]) == ZSEncoderGetTime(tmpBuffer[k]) && ZSEncoderGetPad(tmpBuffer[l]) == ZSEncoderGetPad(tmpBuffer[l - 1]) + 1) {
            seqLen++;
          } else {
            break;
          }
        }
        if (lastTime != -1 && (int)hdr->nTimeBins + ZSEncoderGetTime(tmpBuffer[k]) - lastTime >= 256) {
          lastEndpoint = -1;
        }
        if (ZSEncoderGetTime(tmpBuffer[k]) != lastTime) {
          nexthbf = ((long)ZSEncoderGetTime(tmpBuffer[k]) * LHCBCPERTIMEBIN + bcShiftInFirstHBF) / o2::constants::lhc::LHCMaxBunches;
          if (nexthbf < 0) {
            throw std::runtime_error("Received digit before the defined first orbit");
          }
          if (hbf != nexthbf) {
            lastEndpoint = -2;
          }
        }
        if (endpoint == lastEndpoint) {
          unsigned int sizeChk = (unsigned int)(pagePtr - reinterpret_cast<unsigned char*>(page));                                              // already written
          sizeChk += 2 * (nRowsInTB + (ZSEncoderGetRow(tmpBuffer[k]) != lastRow && ZSEncoderGetTime(tmpBuffer[k]) == lastTime));                // TB HDR
          sizeChk += streamSize8;                                                                                                               // in stream buffer
          sizeChk += (lastTime != ZSEncoderGetTime(tmpBuffer[k])) && ((sizeChk + (streamSize * encodeBits + 7) / 8) & 1);                       // time bin alignment
          sizeChk += (ZSEncoderGetTime(tmpBuffer[k]) != lastTime || ZSEncoderGetRow(tmpBuffer[k]) != lastRow) ? 3 : 0;                          // new row overhead
          sizeChk += (lastTime != -1 && ZSEncoderGetTime(tmpBuffer[k]) > lastTime) ? ((ZSEncoderGetTime(tmpBuffer[k]) - lastTime - 1) * 2) : 0; // empty time bins
          sizeChk += 2;                                                                                                                         // sequence metadata
          const unsigned int streamSizeChkBits = streamSize * encodeBits + ((lastTime != ZSEncoderGetTime(tmpBuffer[k]) && (streamSize * encodeBits) % 8) ? (8 - (streamSize * encodeBits) % 8) : 0);
          if (sizeChk + (encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
            lastEndpoint = -1;
          } else if (sizeChk + (seqLen * encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
            seqLen = ((TPCZSHDR::TPC_ZS_PAGE_SIZE - sizeChk) * 8 - streamSizeChkBits) / encodeBits;
          }
          // sizeChk += (seqLen * encodeBits + streamSizeChkBits + 7) / 8;
          // printf("Endpoint %d (%d), Pos %d, Chk %d, Len %d, rows %d, StreamSize %d %d, time %d (%d), row %d (%d), pad %d\n", endpoint, lastEndpoint, (int) (pagePtr - reinterpret_cast<unsigned char*>(page)), sizeChk, seqLen, nRowsInTB, streamSize8, streamSize, (int) ZSEncoderGetTime(tmpBuffer[k]), lastTime, (int) ZSEncoderGetRow(tmpBuffer[k]), lastRow, ZSEncoderGetPad(tmpBuffer[k]));
        }
      }
      if (k >= tmpBuffer.size() || endpoint != lastEndpoint || ZSEncoderGetTime(tmpBuffer[k]) != lastTime) {
        if (pagePtr != reinterpret_cast<unsigned char*>(page)) {
          pagePtr += 2 * nRowsInTB;
          ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
          pagePtr = std::copy(streamBuffer8.data(), streamBuffer8.data() + streamSize8, pagePtr);
          if (pagePtr - reinterpret_cast<unsigned char*>(page) > 8192) {
            throw std::runtime_error("internal error during ZS encoding");
          }
          streamSize8 = 0;
          for (int l = 1; l < nRowsInTB; l++) {
            tbHdr->rowAddr1()[l - 1] += 2 * nRowsInTB;
          }
        }
        if (page && (k >= tmpBuffer.size() || endpoint != lastEndpoint)) {
          const rdh_utils::FEEIDType rawfeeid = rdh_utils::getFEEID(rawcru, rawendpoint, rawlnk);
#ifdef GPUCA_O2_LIB
          if (raw) {
            size_t size = (padding || lastEndpoint == -1) ? TPCZSHDR::TPC_ZS_PAGE_SIZE : (pagePtr - (unsigned char*)page);
            size = CAMath::nextMultipleOf<o2::raw::RDHUtils::GBTWord>(size);
            raw->addData(rawfeeid, rawcru, rawlnk, rawendpoint, *ir + (hbf - orbitShift) * o2::constants::lhc::LHCMaxBunches, gsl::span<char>((char*)page + sizeof(o2::header::RAWDataHeader), (char*)page + size), true);
          } else
#endif
          {
            o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)page;
            o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, hbf);
            o2::raw::RDHUtils::setHeartBeatBC(*rdh, bcShiftInFirstHBF);
            o2::raw::RDHUtils::setMemorySize(*rdh, TPCZSHDR::TPC_ZS_PAGE_SIZE);
            o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
            o2::raw::RDHUtils::setFEEID(*rdh, rawfeeid);
          }
        }
        if (k >= tmpBuffer.size()) {
          break;
        }
      }
      if (endpoint != lastEndpoint) {
        if (raw) {
          page = &singleBuffer;
        } else {
          if (buffer[i][endpoint].size() == 0 && nexthbf > orbitShift) {
            // Emplace empty page with RDH containing beginning of TFgpuDigitsMap
            buffer[i][endpoint].emplace_back();
            ZSfillEmpty(&buffer[i][endpoint].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(i * 10 + endpoint / 2, endpoint & 1, rawlnk), orbitShift);
            totalPages++;
          }
          buffer[i][endpoint].emplace_back();
          page = &buffer[i][endpoint].back();
        }
        hbf = nexthbf;
        pagePtr = reinterpret_cast<unsigned char*>(page);
        std::fill(page->begin(), page->end(), 0);
        pagePtr += sizeof(o2::header::RAWDataHeader);
        hdr = reinterpret_cast<TPCZSHDR*>(pagePtr);
        pagePtr += sizeof(*hdr);
        hdr->version = zs12bit ? 2 : 1;
        hdr->cruID = i * 10 + region;
        rawcru = i * 10 + region;
        rawendpoint = endpoint & 1;
        hdr->timeOffset = (long)ZSEncoderGetTime(tmpBuffer[k]) * LHCBCPERTIMEBIN - (long)hbf * o2::constants::lhc::LHCMaxBunches;
        lastTime = -1;
        tbHdr = nullptr;
        lastEndpoint = endpoint;
        totalPages++;
      }
      if (ZSEncoderGetTime(tmpBuffer[k]) != lastTime) {
        if (lastTime != -1) {
          hdr->nTimeBins += ZSEncoderGetTime(tmpBuffer[k]) - lastTime - 1;
          pagePtr += (ZSEncoderGetTime(tmpBuffer[k]) - lastTime - 1) * 2;
        }
        hdr->nTimeBins++;
        lastTime = ZSEncoderGetTime(tmpBuffer[k]);
        if ((pagePtr - reinterpret_cast<unsigned char*>(page)) & 1) {
          pagePtr++;
        }
        tbHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
        tbHdr->rowMask |= (endpoint & 1) << 15;
        nRowsInTB = 0;
        lastRow = GPUCA_ROW_COUNT;
      }
      if (ZSEncoderGetRow(tmpBuffer[k]) != lastRow) {
        tbHdr->rowMask |= 1 << (ZSEncoderGetRow(tmpBuffer[k]) - endpointStart);
        lastRow = ZSEncoderGetRow(tmpBuffer[k]);
        ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
        if (nRowsInTB) {
          tbHdr->rowAddr1()[nRowsInTB - 1] = (pagePtr - reinterpret_cast<unsigned char*>(page)) + streamSize8;
        }
        nRowsInTB++;
        nSeq = streamBuffer8.data() + streamSize8++;
        *nSeq = 0;
      }
      (*nSeq)++;
      streamBuffer8[streamSize8++] = ZSEncoderGetPad(tmpBuffer[k]);
      streamBuffer8[streamSize8++] = streamSize + seqLen;
      hdr->nADCsamples += seqLen;
      for (int l = 0; l < seqLen; l++) {
        streamBuffer[streamSize++] = (unsigned short)(ZSEncoderGetCharge(tmpBuffer[k + l]) * encodeBitsFactor + 0.5f);
      }
      k += seqLen - 1;
    }
    if (!raw) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        if (buffer[i][j].size() == 0) {
          buffer[i][j].emplace_back();
          ZSfillEmpty(&buffer[i][j].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(i * 10 + j / 2, j & 1, rawlnk), orbitShift);
          totalPages++;
        }
      }
    }

    // Verification
    if (verify) {
      std::vector<o2::tpc::Digit> compareBuffer;
      compareBuffer.reserve(tmpBuffer.size());
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        unsigned int firstOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)buffer[i][j].data());
        for (unsigned int k = 0; k < buffer[i][j].size(); k++) {
          page = &buffer[i][j][k];
          pagePtr = reinterpret_cast<unsigned char*>(page);
          const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)pagePtr;
          if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
            continue;
          }
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

          int timeBin = (hdr->timeOffset + (unsigned long)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
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
                  compareBuffer.emplace_back(o2::tpc::Digit{0, (float)streamBuffer[pos10++] * decodeBitsFactor, (tpccf::Row)(rowOffset + m), (tpccf::Pad)(rowData[n * 2 + 1] + o), timeBin + l});
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
        const float c = zs12bit ? (float)((int)(ZSEncoderGetCharge(tmpBuffer[j]) * decodeBitsFactor + 0.5f)) / decodeBitsFactor : (float)(int)(ZSEncoderGetCharge(tmpBuffer[j]) + 0.5f);
        int ok = c == ZSEncoderGetCharge(compareBuffer[j]) && (int)ZSEncoderGetTime(tmpBuffer[j]) == (int)ZSEncoderGetTime(compareBuffer[j]) && (int)ZSEncoderGetPad(tmpBuffer[j]) == (int)ZSEncoderGetPad(compareBuffer[j]) && (int)ZSEncoderGetRow(tmpBuffer[j]) == (int)ZSEncoderGetRow(compareBuffer[j]);
        if (ok) {
          continue;
        }
        nErrors++;
        printf("%4u: OK %d: Charge %3d %3d Time %4d %4d Pad %3d %3d Row %3d %3d\n", j, ok,
               (int)c, (int)ZSEncoderGetCharge(compareBuffer[j]), (int)ZSEncoderGetTime(tmpBuffer[j]), (int)ZSEncoderGetTime(compareBuffer[j]), (int)ZSEncoderGetPad(tmpBuffer[j]), (int)ZSEncoderGetPad(compareBuffer[j]), (int)ZSEncoderGetRow(tmpBuffer[j]), (int)ZSEncoderGetRow(compareBuffer[j]));
      }
    }
  }

  if (outBuffer) {
    outBuffer->reset(new unsigned long long int[totalPages * TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(unsigned long long int)]);
    unsigned long long int offset = 0;
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        memcpy((char*)outBuffer->get() + offset, buffer[i][j].data(), buffer[i][j].size() * TPCZSHDR::TPC_ZS_PAGE_SIZE);
        offset += buffer[i][j].size() * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        outSizes[i * GPUTrackingInOutZS::NENDPOINTS + j] = buffer[i][j].size();
      }
    }
  }
  if (nErrors) {
    printf("%lld ERRORS DURING ZS!", (long long int)nErrors);
  }
#endif
}

void GPUReconstructionConvert::RunZSEncoderCreateMeta(const unsigned long long int* buffer, const unsigned int* sizes, void** ptrs, GPUTrackingInOutZS* out)
{
  unsigned long long int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      ptrs[i * GPUTrackingInOutZS::NENDPOINTS + j] = (char*)buffer + offset;
      offset += sizes[i * GPUTrackingInOutZS::NENDPOINTS + j] * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      out->slice[i].zsPtr[j] = &ptrs[i * GPUTrackingInOutZS::NENDPOINTS + j];
      out->slice[i].nZSPtr[j] = &sizes[i * GPUTrackingInOutZS::NENDPOINTS + j];
      out->slice[i].count[j] = 1;
    }
  }
}

void GPUReconstructionConvert::RunZSFilter(std::unique_ptr<o2::tpc::Digit[]>* buffers, const o2::tpc::Digit* const* ptrs, size_t* nsb, const size_t* ns, const GPUParam& param, bool zs12bit, float threshold)
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
      if (buffers[i][k].getChargeFloat() >= threshold) {
        if (k > j) {
          buffers[i][j] = buffers[i][k];
        }
        if (zs12bit) {
          buffers[i][j].setCharge((float)((int)(buffers[i][j].getChargeFloat() * decodeBitsFactor + 0.5f)) / decodeBitsFactor);
        } else {
          buffers[i][j].setCharge((float)((int)(buffers[i][j].getChargeFloat() + 0.5f)));
        }
        j++;
      }
    }
    nsb[i] = j;
  }
#endif
}
