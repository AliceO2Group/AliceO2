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

/// \file GPUReconstructionConvert.cxx
/// \author David Rohr

#ifdef GPUCA_O2_LIB
#include "DetectorsRaw/RawFileWriter.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Mapper.h"
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

#ifdef GPUCA_HAVE_O2HEADERS
#include "clusterFinderDefs.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
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
#ifdef GPUCA_HAVE_O2HEADERS
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
#ifdef GPUCA_HAVE_O2HEADERS
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
#ifdef GPUCA_HAVE_O2HEADERS
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
#ifdef GPUCA_HAVE_O2HEADERS
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
#ifdef GPUCA_HAVE_O2HEADERS
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

// ------------------------------------------------- TPC ZS -------------------------------------------------

#ifdef GPUCA_TPC_GEOMETRY_O2
namespace // anonymous
{

// ------------------------------------------------- TPC ZS General -------------------------------------------------

typedef std::array<long long int, TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(long long int)> zsPage;

struct zsEncoder {
  int curRegion = 0;
  unsigned int encodeBits = 0;
  unsigned int zsVersion = 0;
  unsigned int iSector;
  o2::raw::RawFileWriter* raw;
  const o2::InteractionRecord* ir;
  const GPUParam& param;
  bool padding;
  int lastEndpoint = -2, lastTime = -1, lastRow = GPUCA_ROW_COUNT;
  int endpoint = 0;
  long hbf = -1, nexthbf = 0;
  zsPage* page = nullptr;
  unsigned char* pagePtr = nullptr;
  int bcShiftInFirstHBF = 0;
  int firstTimebinInPage = -1;
  float encodeBitsFactor = 0;
  static void ZSfillEmpty(void* ptr, int shift, unsigned int feeId, int orbit);
  static void ZSstreamOut(unsigned short* bufIn, unsigned int& lenIn, unsigned char* bufOut, unsigned int& lenOut, unsigned int nBits);
  long int getHbf(long int timestamp) { return (timestamp * LHCBCPERTIMEBIN + bcShiftInFirstHBF) / o2::constants::lhc::LHCMaxBunches; }
};

inline void zsEncoder::ZSfillEmpty(void* ptr, int shift, unsigned int feeId, int orbit)
{
  o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)ptr;
  o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, orbit);
  o2::raw::RDHUtils::setHeartBeatBC(*rdh, shift);
  o2::raw::RDHUtils::setMemorySize(*rdh, sizeof(o2::header::RAWDataHeader));
  o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
  o2::raw::RDHUtils::setFEEID(*rdh, feeId);
}

inline void zsEncoder::ZSstreamOut(unsigned short* bufIn, unsigned int& lenIn, unsigned char* bufOut, unsigned int& lenOut, unsigned int nBits)
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

static inline auto ZSEncoderGetDigits(const GPUTrackingInOutDigits& in, int i) { return in.tpcDigits[i]; }
static inline auto ZSEncoderGetNDigits(const GPUTrackingInOutDigits& in, int i) { return in.nTPCDigits[i]; }
#ifdef GPUCA_O2_LIB
using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, o2::tpc::Sector::MAXSECTOR>;
static inline auto ZSEncoderGetDigits(const DigitArray& in, int i) { return in[i].data(); }
static inline auto ZSEncoderGetNDigits(const DigitArray& in, int i) { return in[i].size(); }
#endif // GPUCA_O2_LIB

// ------------------------------------------------- TPC ZS Original Row-based ZS -------------------------------------------------

struct zsEncoderRow : public zsEncoder {
  std::array<unsigned short, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer = {};
  std::array<unsigned char, TPCZSHDR::TPC_ZS_PAGE_SIZE> streamBuffer8 = {};
  TPCZSHDR* hdr = nullptr;
  TPCZSTBHDR* curTBHdr = nullptr;
  unsigned char* nSeq = nullptr;
  int seqLen = 0;
  int endpointStart = 0;
  int nRowsInTB = 0;
  unsigned int streamSize = 0, streamSize8 = 0;

  bool checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k);
  bool writeSubPage();
  void init() {}
  void initPage() {}
  unsigned int encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k);

  bool sort(const o2::tpc::Digit a, const o2::tpc::Digit b);
  void decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* page, unsigned int endpoint, unsigned int firstOrbit);
};

inline bool zsEncoderRow::sort(const o2::tpc::Digit a, const o2::tpc::Digit b)
{
  int endpointa = param.tpcGeometry.GetRegion(a.getRow());
  int endpointb = param.tpcGeometry.GetRegion(b.getRow());
  endpointa = 2 * endpointa + (a.getRow() >= param.tpcGeometry.GetRegionStart(endpointa) + param.tpcGeometry.GetRegionRows(endpointa) / 2);
  endpointb = 2 * endpointb + (b.getRow() >= param.tpcGeometry.GetRegionStart(endpointb) + param.tpcGeometry.GetRegionRows(endpointb) / 2);
  if (endpointa != endpointb) {
    return endpointa <= endpointb;
  }
  if (a.getTimeStamp() != b.getTimeStamp()) {
    return a.getTimeStamp() < b.getTimeStamp();
  }
  if (a.getRow() != b.getRow()) {
    return a.getRow() < b.getRow();
  }
  return a.getPad() < b.getPad();
}

bool zsEncoderRow::checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k)
{
  seqLen = 1;
  if (lastRow != tmpBuffer[k].getRow()) {
    endpointStart = param.tpcGeometry.GetRegionStart(curRegion);
    endpoint = curRegion * 2;
    if (tmpBuffer[k].getRow() >= endpointStart + param.tpcGeometry.GetRegionRows(curRegion) / 2) {
      endpoint++;
      endpointStart += param.tpcGeometry.GetRegionRows(curRegion) / 2;
    }
  }
  for (unsigned int l = k + 1; l < tmpBuffer.size(); l++) {
    if (tmpBuffer[l].getRow() == tmpBuffer[k].getRow() && tmpBuffer[l].getTimeStamp() == tmpBuffer[k].getTimeStamp() && tmpBuffer[l].getPad() == tmpBuffer[l - 1].getPad() + 1) {
      seqLen++;
    } else {
      break;
    }
  }
  if (lastEndpoint >= 0 && lastTime != -1 && (int)hdr->nTimeBins + tmpBuffer[k].getTimeStamp() - lastTime >= 256) {
    lastEndpoint = -1;
  }
  if (endpoint == lastEndpoint) {
    unsigned int sizeChk = (unsigned int)(pagePtr - reinterpret_cast<unsigned char*>(page));                                        // already written
    sizeChk += 2 * (nRowsInTB + (tmpBuffer[k].getRow() != lastRow && tmpBuffer[k].getTimeStamp() == lastTime));                     // TB HDR
    sizeChk += streamSize8;                                                                                                         // in stream buffer
    sizeChk += (lastTime != tmpBuffer[k].getTimeStamp()) && ((sizeChk + (streamSize * encodeBits + 7) / 8) & 1);                    // time bin alignment
    sizeChk += (tmpBuffer[k].getTimeStamp() != lastTime || tmpBuffer[k].getRow() != lastRow) ? 3 : 0;                               // new row overhead
    sizeChk += (lastTime != -1 && tmpBuffer[k].getTimeStamp() > lastTime) ? ((tmpBuffer[k].getTimeStamp() - lastTime - 1) * 2) : 0; // empty time bins
    sizeChk += 2;                                                                                                                   // sequence metadata
    const unsigned int streamSizeChkBits = streamSize * encodeBits + ((lastTime != tmpBuffer[k].getTimeStamp() && (streamSize * encodeBits) % 8) ? (8 - (streamSize * encodeBits) % 8) : 0);
    if (sizeChk + (encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      lastEndpoint = -1;
    } else if (sizeChk + (seqLen * encodeBits + streamSizeChkBits + 7) / 8 > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      seqLen = ((TPCZSHDR::TPC_ZS_PAGE_SIZE - sizeChk) * 8 - streamSizeChkBits) / encodeBits;
    }
    // sizeChk += (seqLen * encodeBits + streamSizeChkBits + 7) / 8;
    // printf("Endpoint %d (%d), Pos %d, Chk %d, Len %d, rows %d, StreamSize %d %d, time %d (%d), row %d (%d), pad %d\n", endpoint, lastEndpoint, (int) (pagePtr - reinterpret_cast<unsigned char*>(page)), sizeChk, seqLen, nRowsInTB, streamSize8, streamSize, (int)tmpBuffer[k].getTimeStamp(), lastTime, (int)tmpBuffer[k].getRow(), lastRow, tmpBuffer[k].getPad());
  }
  return endpoint != lastEndpoint || tmpBuffer[k].getTimeStamp() != lastTime;
}

bool zsEncoderRow::writeSubPage()
{
  if (pagePtr != reinterpret_cast<unsigned char*>(page)) {
    pagePtr += 2 * nRowsInTB;
    ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
    pagePtr = std::copy(streamBuffer8.data(), streamBuffer8.data() + streamSize8, pagePtr);
    if (pagePtr - reinterpret_cast<unsigned char*>(page) > 8192) {
      throw std::runtime_error("internal error during ZS encoding");
    }
    streamSize8 = 0;
    for (int l = 1; l < nRowsInTB; l++) {
      curTBHdr->rowAddr1()[l - 1] += 2 * nRowsInTB;
    }
  }
  return endpoint != lastEndpoint;
}

unsigned int zsEncoderRow::encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k)
{
  if (tmpBuffer[k].getTimeStamp() != lastTime) {
    if (lastTime != -1) {
      hdr->nTimeBins += tmpBuffer[k].getTimeStamp() - lastTime - 1;
      pagePtr += (tmpBuffer[k].getTimeStamp() - lastTime - 1) * 2;
    }
    hdr->nTimeBins++;
    if ((pagePtr - reinterpret_cast<unsigned char*>(page)) & 1) {
      pagePtr++;
    }
    curTBHdr = reinterpret_cast<TPCZSTBHDR*>(pagePtr);
    curTBHdr->rowMask |= (endpoint & 1) << 15;
    nRowsInTB = 0;
    lastRow = GPUCA_ROW_COUNT;
  }
  if (tmpBuffer[k].getRow() != lastRow) {
    curTBHdr->rowMask |= 1 << (tmpBuffer[k].getRow() - endpointStart);
    ZSstreamOut(streamBuffer.data(), streamSize, streamBuffer8.data(), streamSize8, encodeBits);
    if (nRowsInTB) {
      curTBHdr->rowAddr1()[nRowsInTB - 1] = (pagePtr - reinterpret_cast<unsigned char*>(page)) + streamSize8;
    }
    nRowsInTB++;
    nSeq = streamBuffer8.data() + streamSize8++;
    *nSeq = 0;
  }
  (*nSeq)++;
  streamBuffer8[streamSize8++] = tmpBuffer[k].getPad();
  streamBuffer8[streamSize8++] = streamSize + seqLen;
  for (int l = 0; l < seqLen; l++) {
    streamBuffer[streamSize++] = (unsigned short)(tmpBuffer[k + l].getChargeFloat() * encodeBitsFactor + 0.5f);
  }
  return seqLen;
}

void zsEncoderRow::decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* decPage, unsigned int decEndpoint, unsigned int firstOrbit)
{
  const unsigned char* decPagePtr = reinterpret_cast<const unsigned char*>(decPage);
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)decPagePtr;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  decPagePtr += sizeof(o2::header::RAWDataHeader);
  const TPCZSHDR* decHDR = reinterpret_cast<const TPCZSHDR*>(decPagePtr);
  decPagePtr += sizeof(*decHDR);
  if (decHDR->version != 1 && decHDR->version != 2) {
    throw std::runtime_error("invalid ZS version");
  }
  const float decodeBitsFactor = 1.f / (1 << (encodeBits - 10));
  unsigned int mask = (1 << encodeBits) - 1;
  int cruid = decHDR->cruID;
  unsigned int sector = cruid / 10;
  if (sector != iSector) {
    throw std::runtime_error("invalid TPC sector");
  }
  int region = cruid % 10;
  if ((unsigned int)region != decEndpoint / 2) {
    throw std::runtime_error("CRU ID / endpoint mismatch");
  }
  int nRowsRegion = param.tpcGeometry.GetRegionRows(region);

  int timeBin = (decHDR->timeOffset + (unsigned long)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
  for (int l = 0; l < decHDR->nTimeBins; l++) {
    if ((decPagePtr - reinterpret_cast<const unsigned char*>(decPage)) & 1) {
      decPagePtr++;
    }
    const TPCZSTBHDR* tbHdr = reinterpret_cast<const TPCZSTBHDR*>(decPagePtr);
    bool upperRows = tbHdr->rowMask & 0x8000;
    if (tbHdr->rowMask != 0 && ((upperRows) ^ ((decEndpoint & 1) != 0))) {
      throw std::runtime_error("invalid endpoint");
    }
    const int rowOffset = param.tpcGeometry.GetRegionStart(region) + (upperRows ? (nRowsRegion / 2) : 0);
    const int nRows = upperRows ? (nRowsRegion - nRowsRegion / 2) : (nRowsRegion / 2);
    const int nRowsUsed = __builtin_popcount((unsigned int)(tbHdr->rowMask & 0x7FFF));
    decPagePtr += nRowsUsed ? (2 * nRowsUsed) : 2;
    int rowPos = 0;
    for (int m = 0; m < nRows; m++) {
      if ((tbHdr->rowMask & (1 << m)) == 0) {
        continue;
      }
      const unsigned char* rowData = rowPos == 0 ? decPagePtr : (reinterpret_cast<const unsigned char*>(decPage) + tbHdr->rowAddr1()[rowPos - 1]);
      const int nSeqRead = *rowData;
      const unsigned char* adcData = rowData + 2 * nSeqRead + 1;
      int nADC = (rowData[2 * nSeqRead] * encodeBits + 7) / 8;
      decPagePtr += 1 + 2 * nSeqRead + nADC;
      unsigned int byte = 0, bits = 0, posXbits = 0;
      std::array<unsigned short, TPCZSHDR::TPC_ZS_PAGE_SIZE> decBuffer;
      for (int n = 0; n < nADC; n++) {
        byte |= *(adcData++) << bits;
        bits += 8;
        while (bits >= encodeBits) {
          decBuffer[posXbits++] = byte & mask;
          byte = byte >> encodeBits;
          bits -= encodeBits;
        }
      }
      posXbits = 0;
      for (int n = 0; n < nSeqRead; n++) {
        const int decSeqLen = rowData[(n + 1) * 2] - (n ? rowData[n * 2] : 0);
        for (int o = 0; o < decSeqLen; o++) {
          outputBuffer.emplace_back(o2::tpc::Digit{0, (float)decBuffer[posXbits++] * decodeBitsFactor, (tpccf::Row)(rowOffset + m), (tpccf::Pad)(rowData[n * 2 + 1] + o), timeBin + l});
        }
      }
      rowPos++;
    }
  }
}

// ------------------------------------------------- TPC ZS Improved Link Based ZS -------------------------------------------------

#ifdef GPUCA_O2_LIB
struct zsEncoderImprovedLinkBased : public zsEncoder {
  TPCZSHDRV2* hdr = nullptr;
  int inverseChannelMapping[5][32];
  int nSamples = 0;
  int link = 0;
  bool finishPage = false;
  std::vector<unsigned short> adcValues = {};
  std::bitset<80> bitmask = {};

  bool checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k);
  bool writeSubPage();
  void init();
  void initPage();
  unsigned int encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k);
  bool sort(const o2::tpc::Digit a, const o2::tpc::Digit b);
  void decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* page, unsigned int endpoint, unsigned int firstOrbit);
};

void zsEncoderImprovedLinkBased::init()
{
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 32; j++) {
      inverseChannelMapping[i][j] = -1;
    }
  }
  for (int iCRU = 0; iCRU < 2; iCRU++) {
    for (int iChannel = 0; iChannel < 80; iChannel++) {
      int sampaOnFEC = 0, channelOnSAMPA = 0;
      Mapper::getSampaAndChannelOnFEC(iCRU, iChannel, sampaOnFEC, channelOnSAMPA);
      if (inverseChannelMapping[sampaOnFEC][channelOnSAMPA] != -1 && inverseChannelMapping[sampaOnFEC][channelOnSAMPA] != iChannel) {
        printf("ERROR: Channel conflict: %d %d: %d vs %d\n", sampaOnFEC, channelOnSAMPA, inverseChannelMapping[sampaOnFEC][channelOnSAMPA], iChannel);
        throw std::runtime_error("ZS error");
      }
      inverseChannelMapping[sampaOnFEC][channelOnSAMPA] = iChannel;
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 32; j++) {
      if (inverseChannelMapping[i][j] == -1) {
        printf("ERROR: Map missing for sampa %d channel %d\n", i, j);
        throw std::runtime_error("ZS error");
      }
    }
  }
}

void zsEncoderImprovedLinkBased::initPage()
{
  hdr->magicWord = o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader;
  hdr->nTimebinHeaders = 0;
  hdr->firstZSDataOffset = 0;
}

bool zsEncoderImprovedLinkBased::checkInput(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k)
{
  const auto& mapper = Mapper::instance();
  nSamples = 0;
  adcValues.clear();
  bitmask.reset();
  unsigned int l;
  for (l = k; l < tmpBuffer.size(); l++) {
    const auto& a = tmpBuffer[l];
    int cruinsector = param.tpcGeometry.GetRegion(a.getRow());
    o2::tpc::GlobalPadNumber pad = mapper.globalPadNumber(o2::tpc::PadPos(a.getRow(), a.getPad()));
    o2::tpc::FECInfo fec = mapper.fecInfo(pad);
    o2::tpc::CRU cru = cruinsector;
    int fecInPartition = fec.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();
    int tmpEndpoint = 2 * cruinsector + (fecInPartition >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
    if (l == k) {
      link = fecInPartition;
      endpoint = tmpEndpoint;
    } else if (endpoint != tmpEndpoint || link != fecInPartition || tmpBuffer[l].getTimeStamp() != tmpBuffer[k].getTimeStamp()) {
      break;
    }
    int channel = inverseChannelMapping[fec.getSampaChip()][fec.getSampaChannel()];
    bitmask[channel] = 1;
    adcValues.emplace_back((unsigned short)(a.getChargeFloat() * encodeBitsFactor + 0.5f));
  }
  nSamples = l - k;
  finishPage = endpoint != lastEndpoint || (firstTimebinInPage != -1 && tmpBuffer[k].getTimeStamp() - firstTimebinInPage >= (1 << 8));
  if (tmpBuffer[k].getTimeStamp() - firstTimebinInPage + 1 > (1 << (sizeof(hdr->nTimeBins) * 8)) - 1) {
    finishPage = true;
  }
  if (!finishPage) {
    unsigned int sizeChk = (unsigned int)(pagePtr - reinterpret_cast<unsigned char*>(page));
    sizeChk += sizeof(o2::tpc::zerosupp_link_based::CommonHeader);
    if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
      sizeChk += (nSamples * TPCZSHDRV2::TPC_ZS_NBITS_V3 + 127) / 128 * 16;
    } else {
      sizeChk += (nSamples + 2 * TPCZSHDRV2::SAMPLESPER64BIT - 1) / (2 * TPCZSHDRV2::SAMPLESPER64BIT) * 16;
    }
    if (sizeChk > TPCZSHDR::TPC_ZS_PAGE_SIZE) {
      finishPage = true;
    }
  }
  return finishPage;
}

bool zsEncoderImprovedLinkBased::writeSubPage()
{
  return finishPage;
}

unsigned int zsEncoderImprovedLinkBased::encodeSequence(std::vector<o2::tpc::Digit>& tmpBuffer, unsigned int k)
{
  o2::tpc::zerosupp_link_based::CommonHeader* tbHdr = (o2::tpc::zerosupp_link_based::CommonHeader*)pagePtr;
  pagePtr += sizeof(*tbHdr);
  tbHdr->bunchCrossing = (tmpBuffer[k].getTimeStamp() - firstTimebinInPage) * LHCBCPERTIMEBIN;
  tbHdr->magicWord = o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZS;
  tbHdr->bitMaskHigh = (bitmask >> 64).to_ulong();
  tbHdr->bitMaskLow = (bitmask & std::bitset<80>(0xFFFFFFFFFFFFFFFFlu)).to_ulong();
  tbHdr->syncOffsetBC = 0;
  tbHdr->fecInPartition = link;
  hdr->nTimeBins = tmpBuffer[k].getTimeStamp() - firstTimebinInPage + 1;
  hdr->nTimebinHeaders++;
  if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
    tbHdr->numWordsPayload = (nSamples * TPCZSHDRV2::TPC_ZS_NBITS_V3 + 127) / 128; // tightly packed ADC samples
    unsigned int tmp = 0;
    unsigned int tmpIn = nSamples;
    ZSstreamOut(adcValues.data(), tmpIn, pagePtr, tmp, encodeBits);
  } else {
    tbHdr->numWordsPayload = (nSamples + 2 * TPCZSHDRV2::SAMPLESPER64BIT - 1) / (2 * TPCZSHDRV2::SAMPLESPER64BIT);
    unsigned long* payloadPtr = (unsigned long*)pagePtr;
    for (unsigned int i = 0; i < 2 * tbHdr->numWordsPayload; i++) {
      payloadPtr[i] = 0;
    }
    for (unsigned int i = 0; i < nSamples; i++) {
      payloadPtr[i / TPCZSHDRV2::SAMPLESPER64BIT] |= ((unsigned long)adcValues[i]) << ((i % TPCZSHDRV2::SAMPLESPER64BIT) * TPCZSHDRV2::TPC_ZS_NBITS_V3);
    }
  }
  pagePtr += tbHdr->numWordsPayload * 16;
  return nSamples;
}

bool zsEncoderImprovedLinkBased::sort(const o2::tpc::Digit a, const o2::tpc::Digit b)
{
  // Fixme: this is blasphemy... one shoult precompute all values and sort an index array
  int cruinsectora = param.tpcGeometry.GetRegion(a.getRow());
  int cruinsectorb = param.tpcGeometry.GetRegion(b.getRow());
  if (cruinsectora != cruinsectorb) {
    return cruinsectora < cruinsectorb;
  }
  const auto& mapper = Mapper::instance();
  o2::tpc::GlobalPadNumber pada = mapper.globalPadNumber(o2::tpc::PadPos(a.getRow(), a.getPad()));
  o2::tpc::GlobalPadNumber padb = mapper.globalPadNumber(o2::tpc::PadPos(b.getRow(), b.getPad()));
  o2::tpc::FECInfo feca = mapper.fecInfo(pada);
  o2::tpc::FECInfo fecb = mapper.fecInfo(padb);
  o2::tpc::CRU cru = cruinsectora;
  int fecInPartitiona = feca.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();
  int fecInPartitionb = fecb.getIndex() - mapper.getPartitionInfo(cru.partition()).getSectorFECOffset();

  int endpointa = 2 * cruinsectora + (fecInPartitiona >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
  int endpointb = 2 * cruinsectorb + (fecInPartitionb >= (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
  if (endpointa != endpointb) {
    return endpointa < endpointb;
  }
  if (a.getTimeStamp() != b.getTimeStamp()) {
    return a.getTimeStamp() < b.getTimeStamp();
  }
  if (fecInPartitiona != fecInPartitionb) {
    return fecInPartitiona < fecInPartitionb;
  }
  return inverseChannelMapping[feca.getSampaChip()][feca.getSampaChannel()] < inverseChannelMapping[fecb.getSampaChip()][fecb.getSampaChannel()];
}

void zsEncoderImprovedLinkBased::decodePage(std::vector<o2::tpc::Digit>& outputBuffer, const zsPage* decPage, unsigned int decEndpoint, unsigned int firstOrbit)
{
  const auto& mapper = Mapper::instance();
  const unsigned char* decPagePtr = reinterpret_cast<const unsigned char*>(decPage);
  const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)decPagePtr;
  if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
    return;
  }
  decPagePtr += sizeof(o2::header::RAWDataHeader);
  const TPCZSHDRV2* decHDR = reinterpret_cast<const TPCZSHDRV2*>(decPagePtr);
  decPagePtr += sizeof(*decHDR);
  if (decHDR->version != 3) {
    throw std::runtime_error("invalid ZS version");
  }
  if (decHDR->magicWord != o2::tpc::zerosupp_link_based::CommonHeader::MagicWordLinkZSMetaHeader) {
    throw std::runtime_error("Magic word missing");
  }
  const float decodeBitsFactor = 1.f / (1 << (encodeBits - 10));
  unsigned int mask = (1 << encodeBits) - 1;
  int cruid = decHDR->cruID;
  unsigned int sector = cruid / 10;
  if (sector != iSector) {
    throw std::runtime_error("invalid TPC sector");
  }
  int region = cruid % 10;
  decPagePtr += decHDR->firstZSDataOffset * 16;
  for (unsigned int i = 0; i < decHDR->nTimebinHeaders; i++) {
    const o2::tpc::zerosupp_link_based::Header* tbHdr = (const o2::tpc::zerosupp_link_based::Header*)decPagePtr;
#if 0 // Decoding using the function for the original linkZS
    o2::tpc::CRU cru = cruid % 10;
    const int feeLink = tbHdr->fecInPartition - (decEndpoint & 1) * ((mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2);
    auto fillADC = [&outputBuffer](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) {
      outputBuffer.emplace_back(o2::tpc::Digit{0, adcValue, rowInSector, padInRow, timeBin});
      return true;
    };
    size_t size = sizeof(*tbHdr) + tbHdr->numWordsPayload * 16;
    raw_processing_helpersa::processZSdata((const char*)decPagePtr, size, rdh_utils::getFEEID(cruid, decEndpoint & 1, feeLink), o2::raw::RDHUtils::getHeartBeatOrbit(*rdh), firstOrbit, decHDR->timeOffset, fillADC);
#else // Decoding directly
    if (!tbHdr->isLinkZS()) {
      throw std::runtime_error("ZS TB Hdr does not have linkZS magic word");
    }
    int timeBin = (decHDR->timeOffset + tbHdr->bunchCrossing + (unsigned long)(o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstOrbit) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
    const unsigned char* adcData = (const unsigned char*)(decPagePtr + sizeof(*tbHdr));
    const auto& bitmask = tbHdr->getChannelBits();
    int nADC = bitmask.count();
    std::vector<unsigned short> decBuffer(nADC);
    if (TPCZSHDRV2::TIGHTLY_PACKED_V3) {
      unsigned int byte = 0, bits = 0, posXbits = 0;
      while (posXbits < nADC) {
        byte |= *(adcData++) << bits;
        bits += 8;
        while (bits >= encodeBits) {
          decBuffer[posXbits++] = byte & mask;
          byte = byte >> encodeBits;
          bits -= encodeBits;
        }
      }
    } else {
      const unsigned long* adcData64 = (const unsigned long*)adcData;
      for (int j = 0; j < nADC; j++) {
        decBuffer[j] = (adcData64[j / TPCZSHDRV2::SAMPLESPER64BIT] >> ((j % TPCZSHDRV2::SAMPLESPER64BIT) * TPCZSHDRV2::TPC_ZS_NBITS_V3)) & mask;
      }
    }
    for (int j = 0, k = 0; j < bitmask.size(); j++) {
      if (bitmask[j]) {
        int sampaOnFEC = 0, channelOnSAMPA = 0;
        mapper.getSampaAndChannelOnFEC(cruid, j, sampaOnFEC, channelOnSAMPA);
        const auto padSecPos = mapper.padSecPos(cruid, tbHdr->fecInPartition, sampaOnFEC, channelOnSAMPA);
        const auto& padPos = padSecPos.getPadPos();
        outputBuffer.emplace_back(o2::tpc::Digit{0, (float)decBuffer[k++] * decodeBitsFactor, (tpccf::Row)padPos.getRow(), (tpccf::Pad)padPos.getPad(), timeBin});
      }
    }
#endif
    decPagePtr += sizeof(*tbHdr) + tbHdr->numWordsPayload * 16;
  }
}
#endif // GPUCA_O2_LIB

// ------------------------------------------------- TPC ZS Main Encoder -------------------------------------------------

template <class T>
struct zsEncoderRun : public T {
  unsigned int run(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer);
  size_t compare(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer);

  using T::bcShiftInFirstHBF;
  using T::checkInput;
  using T::curRegion;
  using T::decodePage;
  using T::encodeBits;
  using T::encodeBitsFactor;
  using T::encodeSequence;
  using T::endpoint;
  using T::firstTimebinInPage;
  using T::getHbf;
  using T::hbf;
  using T::hdr;
  using T::init;
  using T::initPage;
  using T::ir;
  using T::iSector;
  using T::lastEndpoint;
  using T::lastRow;
  using T::lastTime;
  using T::nexthbf;
  using T::padding;
  using T::page;
  using T::pagePtr;
  using T::param;
  using T::raw;
  using T::sort;
  using T::writeSubPage;
  using T::ZSfillEmpty;
  using T::zsVersion;
};

template <class T>
inline unsigned int zsEncoderRun<T>::run(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer)
{
  unsigned int totalPages = 0;
  zsPage singleBuffer;
#ifdef GPUCA_O2_LIB
  int rawlnk = rdh_utils::UserLogicLinkID;
  int maxhbf = 0;
  int minhbf = o2::constants::lhc::LHCMaxBunches;
#else
  int rawlnk = 15;
#endif
  bcShiftInFirstHBF = ir ? ir->bc : 0;
  int orbitShift = ir ? ir->orbit : 0;
  int rawcru = 0;
  int rawendpoint = 0;
  (void)(rawcru + rawendpoint); // avoid compiler warning
  encodeBitsFactor = (1 << (encodeBits - 10));
  init();

  std::sort(tmpBuffer.begin(), tmpBuffer.end(), [this](const o2::tpc::Digit a, const o2::tpc::Digit b) { return sort(a, b); });
  for (unsigned int k = 0; k <= tmpBuffer.size(); k++) {
    bool mustWritePage = false, mustWriteSubPage = false;
    if (k < tmpBuffer.size()) {
      if (tmpBuffer[k].getTimeStamp() != lastTime) {
        nexthbf = getHbf(tmpBuffer[k].getTimeStamp());
        if (nexthbf < 0) {
          throw std::runtime_error("Received digit before the defined first orbit");
        }
        if (hbf != nexthbf) {
          lastEndpoint = -2;
          mustWritePage = true;
        }
      }
      if (lastRow != tmpBuffer[k].getRow()) {
        curRegion = param.tpcGeometry.GetRegion(tmpBuffer[k].getRow());
      }
      mustWriteSubPage = checkInput(tmpBuffer, k);
    } else {
      nexthbf = -1;
      mustWritePage = true;
    }
    if (mustWritePage || mustWriteSubPage) {
      mustWritePage |= writeSubPage();

      if (page && mustWritePage) {
        const rdh_utils::FEEIDType rawfeeid = rdh_utils::getFEEID(rawcru, rawendpoint, rawlnk);
        size_t size = (padding || lastEndpoint == -1 || hbf == nexthbf) ? TPCZSHDR::TPC_ZS_PAGE_SIZE : (pagePtr - (unsigned char*)page);
        size = CAMath::nextMultipleOf<o2::raw::RDHUtils::GBTWord>(size);
#ifdef GPUCA_O2_LIB
        if (raw) {
          raw->addData(rawfeeid, rawcru, rawlnk, rawendpoint, *ir + hbf * o2::constants::lhc::LHCMaxBunches, gsl::span<char>((char*)page + sizeof(o2::header::RAWDataHeader), (char*)page + size), true, 0, 2);
          maxhbf = std::max<int>(maxhbf, hbf);
          minhbf = std::min<int>(minhbf, hbf);
        } else
#endif
        {
          o2::header::RAWDataHeader* rdh = (o2::header::RAWDataHeader*)page;
          o2::raw::RDHUtils::setHeartBeatOrbit(*rdh, hbf + orbitShift);
          o2::raw::RDHUtils::setHeartBeatBC(*rdh, bcShiftInFirstHBF);
          o2::raw::RDHUtils::setMemorySize(*rdh, size);
          o2::raw::RDHUtils::setVersion(*rdh, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
          o2::raw::RDHUtils::setFEEID(*rdh, rawfeeid);
          o2::raw::RDHUtils::setDetectorField(*rdh, 2);
        }
      }
      if (k >= tmpBuffer.size()) {
        break;
      }
    }
    if (mustWritePage) {
      if (raw) {
        page = &singleBuffer;
      } else {
        if (buffer[endpoint].size() == 0 && nexthbf > orbitShift) {
          buffer[endpoint].emplace_back();
          ZSfillEmpty(&buffer[endpoint].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(iSector * 10 + endpoint / 2, endpoint & 1, rawlnk), orbitShift); // Emplace empty page with RDH containing beginning of TF
          totalPages++;
        }
        buffer[endpoint].emplace_back();
        page = &buffer[endpoint].back();
      }
      hbf = nexthbf;
      pagePtr = reinterpret_cast<unsigned char*>(page);
      std::fill(page->begin(), page->end(), 0);
      pagePtr += sizeof(o2::header::RAWDataHeader);
      hdr = reinterpret_cast<decltype(hdr)>(pagePtr);
      pagePtr += sizeof(*hdr);
      hdr->version = zsVersion;
      hdr->cruID = iSector * 10 + curRegion;
      rawcru = iSector * 10 + curRegion;
      rawendpoint = endpoint & 1;
      hdr->timeOffset = (long)tmpBuffer[k].getTimeStamp() * LHCBCPERTIMEBIN - (long)hbf * o2::constants::lhc::LHCMaxBunches;
      firstTimebinInPage = tmpBuffer[k].getTimeStamp();
      lastTime = -1;
      lastEndpoint = endpoint;
      initPage();
      totalPages++;
    }
    unsigned int nEncoded = encodeSequence(tmpBuffer, k);
    lastTime = tmpBuffer[k].getTimeStamp();
    lastRow = tmpBuffer[k].getRow();
    hdr->nADCsamples += nEncoded;
    k += nEncoded - 1;
  }
  if (raw) {
#ifdef GPUCA_O2_LIB
    if (iSector == 0) {
      for (int i = minhbf; i <= maxhbf; i++) {
        raw->addData(46208, 360, rawlnk, 0, *ir + i * o2::constants::lhc::LHCMaxBunches, gsl::span<char>((char*)&singleBuffer, (char*)&singleBuffer), true, 0, 4);
      }
    }
#endif
  } else {
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      if (buffer[j].size() == 0) {
        buffer[j].emplace_back();
        ZSfillEmpty(&buffer[j].back(), bcShiftInFirstHBF, rdh_utils::getFEEID(iSector * 10 + j / 2, j & 1, rawlnk), orbitShift);
        totalPages++;
      }
    }
  }
  return totalPages;
}

template <class T>
size_t zsEncoderRun<T>::compare(std::vector<zsPage>* buffer, std::vector<o2::tpc::Digit>& tmpBuffer)
{
  size_t nErrors = 0;
  std::vector<o2::tpc::Digit> compareBuffer;
  compareBuffer.reserve(tmpBuffer.size());
  for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    unsigned int firstOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)buffer[j].data());
    for (unsigned int k = 0; k < buffer[j].size(); k++) {
      zsPage* decPage = &buffer[j][k];
      decodePage(compareBuffer, decPage, j, firstOrbit);
    }
  }
  if (tmpBuffer.size() != compareBuffer.size()) {
    nErrors += tmpBuffer.size();
    printf("Number of clusters mismatch %d %d\n", (int)tmpBuffer.size(), (int)compareBuffer.size());
  } else {
    for (unsigned int j = 0; j < tmpBuffer.size(); j++) {
      const float decodeBitsFactor = (1 << (encodeBits - 10));
      const float c = (float)((int)(tmpBuffer[j].getChargeFloat() * decodeBitsFactor + 0.5f)) / decodeBitsFactor;
      int ok = c == compareBuffer[j].getChargeFloat() && (int)tmpBuffer[j].getTimeStamp() == (int)compareBuffer[j].getTimeStamp() && (int)tmpBuffer[j].getPad() == (int)compareBuffer[j].getPad() && (int)tmpBuffer[j].getRow() == (int)compareBuffer[j].getRow();
      if (ok) {
        continue;
      }
      nErrors++;
      printf("%4u: OK %d: Charge %3d %3d Time %4d %4d Pad %3d %3d Row %3d %3d\n", j, ok,
             (int)c, (int)compareBuffer[j].getChargeFloat(), (int)tmpBuffer[j].getTimeStamp(), (int)compareBuffer[j].getTimeStamp(), (int)tmpBuffer[j].getPad(), (int)compareBuffer[j].getPad(), (int)tmpBuffer[j].getRow(), (int)compareBuffer[j].getRow());
    }
  }
  return nErrors;
}

} // anonymous namespace
#endif // GPUCA_TPC_GEOMETRY_O2

template <class S>
void GPUReconstructionConvert::RunZSEncoder(const S& in, std::unique_ptr<unsigned long long int[]>* outBuffer, unsigned int* outSizes, o2::raw::RawFileWriter* raw, const o2::InteractionRecord* ir, const GPUParam& param, int version, bool verify, float threshold, bool padding)
{
  // Pass in either outBuffer / outSizes, to fill standalone output buffers, or raw to use RawFileWriter
  // ir is the interaction record for time bin 0
  if (((outBuffer == nullptr) ^ (outSizes == nullptr)) || ((raw != nullptr) && (ir == nullptr)) || !((outBuffer == nullptr) ^ (raw == nullptr)) || (raw && verify) || (threshold > 0.f && verify)) {
    throw std::runtime_error("Invalid parameters");
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  std::vector<zsPage> buffer[NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  unsigned int totalPages = 0;
  size_t nErrors = 0;
  // clang-format off
  GPUCA_OPENMP(parallel for reduction(+ : totalPages) reduction(+ : nErrors))
  // clang-format on
  for (unsigned int i = 0; i < NSLICES; i++) {
    std::vector<o2::tpc::Digit> tmpBuffer;
    tmpBuffer.resize(ZSEncoderGetNDigits(in, i));
    if (threshold > 0.f) {
      auto it = std::copy_if(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin(), [threshold](auto& v) { return v.getChargeFloat() >= threshold; });
      tmpBuffer.resize(std::distance(tmpBuffer.begin(), it));
    } else {
      std::copy(ZSEncoderGetDigits(in, i), ZSEncoderGetDigits(in, i) + ZSEncoderGetNDigits(in, i), tmpBuffer.begin());
    }

    auto runZS = [&](auto& encoder) {
      encoder.zsVersion = version;
      totalPages += encoder.run(buffer[i], tmpBuffer);
      if (verify) {
        nErrors += encoder.compare(buffer[i], tmpBuffer); // Verification
      }
    };

    if (version >= 1 && version <= 2) {
      bool zs12bit = version == 2;
      zsEncoderRun<zsEncoderRow> enc{{{.iSector = i, .raw = raw, .ir = ir, .param = param, .padding = padding}}};
      enc.encodeBits = zs12bit ? TPCZSHDR::TPC_ZS_NBITS_V2 : TPCZSHDR::TPC_ZS_NBITS_V1;
      runZS(enc);
    } else if (version == 3) {
#ifdef GPUCA_O2_LIB
      zsEncoderRun<zsEncoderImprovedLinkBased> enc{{{.iSector = i, .raw = raw, .ir = ir, .param = param, .padding = padding}}};
      enc.encodeBits = TPCZSHDRV2::TPC_ZS_NBITS_V3;
      runZS(enc);
#else
      throw std::runtime_error("Link based ZS encoding not supported in standalone build");
#endif
    } else {
      throw std::runtime_error("Invalid ZS version");
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
    printf("ERROR: %lld INCORRECT SAMPLES DURING ZS ENCODING VERIFICATION!!!\n", (long long int)nErrors);
  }
#endif
}

#ifdef GPUCA_HAVE_O2HEADERS
template void GPUReconstructionConvert::RunZSEncoder<GPUTrackingInOutDigits>(const GPUTrackingInOutDigits&, std::unique_ptr<unsigned long long int[]>*, unsigned int*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, int, bool, float, bool);
#ifdef GPUCA_O2_LIB
template void GPUReconstructionConvert::RunZSEncoder<DigitArray>(const DigitArray&, std::unique_ptr<unsigned long long int[]>*, unsigned int*, o2::raw::RawFileWriter*, const o2::InteractionRecord*, const GPUParam&, int, bool, float, bool);
#endif
#endif

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
#ifdef GPUCA_HAVE_O2HEADERS
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
