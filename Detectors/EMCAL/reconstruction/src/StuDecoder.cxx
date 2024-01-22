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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <boost/format.hpp>
#include "InfoLogger/InfoLogger.hxx"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "EMCALReconstruction/StuDecoder.h"
#include "EMCALReconstruction/STUDecoderError.h"

using namespace o2::emcal;
using namespace o2::emcal::STUparam;

StuDecoder::StuDecoder(RawReaderMemory& reader) : mRawReader(reader)
{
}

void StuDecoder::init()
{

  auto& header = mRawReader.getRawHeader();
  auto feeID = o2::raw::RDHUtils::getFEEID(header);

  if (!((feeID == FeeID[STUtype_t::ESTU]) || (feeID == FeeID[STUtype_t::DSTU]))) {
    throw STUDecoderError(feeID, STUDecoderError::ErrorCode_t::FEEID_UNEXPECTED);
  }

  mSTU = STUtype_t::ESTU;
  if (feeID == FeeID[STUtype_t::DSTU]) {
    mSTU = STUtype_t::DSTU;
  }

  //  STU payload structure is described in JIRA-EMCAL-562
  auto& buffer = mRawReader.getPayload().getPayloadWords();
  bool vbit = ((buffer[0] >> 31) & 0x1); // payload version control bit
  if (!vbit) {                           // old payloads version cannot be decoded with this method; discard 2022 early data
    throw STUDecoderError(feeID, STUDecoderError::ErrorCode_t::OLD_PAYLOAD_VERSION);
  }

  mIsFullPayload = (buffer[1] & 0x1);

  auto payloadsize = mRawReader.getPayload().getPayloadWords().size();

  if (mIsFullPayload) {
    if (payloadsize != getPaloadSizeFull()) {
      //      std::cout << "ERROR: Wrong payload size (=" << std::dec << payloadsize << ") for FullPaload \n";
      throw STUDecoderError(feeID, STUDecoderError::ErrorCode_t::FULL_PAYLOAD_SIZE_UNEXPECTED);
    }
  }

  else {
    if (payloadsize != getPaloadSizeShort()) {
      //      std::cout << "ERROR: Wrong payload size (=" << std::dec << payloadsize << ") for ShortPaload \n";
      throw STUDecoderError(feeID, STUDecoderError::ErrorCode_t::SHORT_PAYLOAD_SIZE_UNEXPECTED);
    }
  }

  if (mDebug >= 3) {
    std::cout << "==== paylaod size ===\n";
    int currentpos = 0;
    while ((currentpos < buffer.size())) {
      auto currentword = buffer[currentpos++];
      std::cout << std::dec << (currentpos - 1) << ")   0x" << std::hex << currentword << std::endl;
    }
  }
}

void StuDecoder::decode()
{

  init();

  auto& buffer = mRawReader.getPayload().getPayloadWords();

  mCFGWord0 = buffer[0];
  mCFGWord1 = buffer[1];
  mL0mask = buffer[2];
  mL1GammaHighThreshold = buffer[3];
  mShortPayloadRate = buffer[4];
  mL0bits = buffer[5];
  mL1JetHighThreshold = buffer[6];
  mL1GammaLowThreshold = buffer[9];
  mL1JetLowThreshold = buffer[12];
  mCFGword13 = buffer[13];
  mRegionEnable = buffer[14];
  mFrameReceived = buffer[15];
  mCFGword16 = buffer[16];

  // decode L1Jet High/Low indices
  int offset = getCFGWords();
  decodeL1JetPatchIndices(&buffer[offset]);

  // decode L1Gamma High/Low indices
  offset += (2 * getL1JetIndexWords() + getL0indexWords());
  decodeL1GammaPatchIndices(&buffer[offset]);

  if (!isFullPayload()) {
    return;
  }

  // decode FastOR data
  offset += (2 * getL1GammaIndexWords());
  decodeFastOrADC(&buffer[offset]);

  //  std::cout << "lastWord = 0x" << std::hex << buffer[offset + getRawWords()] << std::endl;
  return;
}

void StuDecoder::decodeL1JetPatchIndices(const uint32_t* buffer)
{
  int offset = getL1JetIndexWords(); // offset for Jet Low threshold

  if (mDebug >= 2) {
    for (int i = 0; i < offset; i++) {
      std::cout << std::dec << i << ")  0x" << std::hex << buffer[i] << "   <- Jet-high-word\n";
    }
    for (int i = 0; i < offset; i++) {
      std::cout << std::dec << i << ")  0x" << std::hex << buffer[offset + i] << "   <- Jet-low-word\n";
    }
  }

  int nSubregionEta = getSubregionsEta();
  int nSubregionPhi = getSubregionsPhi();
  bool isFired = false;

  int jetSize = 2 + getParchSize(); // 2 for 2x2-patch, 4 for 4x4=patch

  for (Int_t ix = 0; ix < nSubregionEta - (jetSize - 1); ix++) {
    uint32_t row_JetHigh = buffer[ix];
    uint32_t row_JetLow = buffer[ix + offset];
    for (Int_t iy = 0; iy < nSubregionPhi - (jetSize - 1); iy++) {
      isFired = (mSTU == STUtype_t::ESTU) ? (row_JetHigh & (1 << (nSubregionPhi - jetSize - iy))) : (row_JetHigh & (1 << iy));
      if (isFired) {
        mL1JetHighPatchIndex.push_back(((ix << 8) & 0xFF00) | (iy & 0xFF));
      }
      isFired = (mSTU == STUtype_t::ESTU) ? (row_JetLow & (1 << (nSubregionPhi - jetSize - iy))) : (row_JetLow & (1 << iy));
      if (isFired) {
        mL1JetLowPatchIndex.push_back(((ix << 8) & 0xFF00) | (iy & 0xFF));
      }
    }
  }

  return;
}

void StuDecoder::decodeL1GammaPatchIndices(const uint32_t* buffer)
{
  const int offset = getL1GammaIndexWords(); // offset for Gamma Low threshold

  if (mDebug >= 2) {
    for (int i = 0; i < offset; i++) {
      std::cout << std::dec << i << ")  0x" << std::hex << buffer[i] << "   <- Gamma-high-word\n";
    }
    for (int i = 0; i < offset; i++) {
      std::cout << std::dec << i << ")  0x" << std::hex << buffer[offset + i] << "   <- Gamma-low-word\n";
    }
  }

  const int nTRU = getNumberOfTRUs();
  const int nTRU2 = getNumberOfTRUs() / 2;

  unsigned short gammaHigh[nTRU][TRUparam::NchannelsOverPhi];
  unsigned short gammaLow[nTRU][TRUparam::NchannelsOverPhi];

  for (Int_t iphi = 0; iphi < TRUparam::NchannelsOverPhi / 2; iphi++) {
    for (Int_t itru = 0; itru < nTRU2; itru++) {
      gammaHigh[2 * itru][2 * iphi] = (buffer[iphi * nTRU2 + itru] >> 0 & 0xFF);
      gammaHigh[2 * itru][2 * iphi + 1] = (buffer[iphi * nTRU2 + itru] >> 8 & 0xFF);
      gammaHigh[2 * itru + 1][2 * iphi] = (buffer[iphi * nTRU2 + itru] >> 16 & 0xFF);
      gammaHigh[2 * itru + 1][2 * iphi + 1] = (buffer[iphi * nTRU2 + itru] >> 24 & 0xFF);

      gammaLow[2 * itru][2 * iphi] = (buffer[offset + iphi * nTRU2 + itru] >> 0 & 0xFF);
      gammaLow[2 * itru][2 * iphi + 1] = (buffer[offset + iphi * nTRU2 + itru] >> 8 & 0xFF);
      gammaLow[2 * itru + 1][2 * iphi] = (buffer[offset + iphi * nTRU2 + itru] >> 16 & 0xFF);
      gammaLow[2 * itru + 1][2 * iphi + 1] = (buffer[offset + iphi * nTRU2 + itru] >> 24 & 0xFF);
    }
  }

  for (Int_t iphi = 0; iphi < TRUparam::NchannelsOverPhi; iphi++) {
    for (Int_t ieta = 0; ieta < TRUparam::NchannelsOverEta; ieta++) {
      // loop over TRUs of Full or 2/3-size SMs
      for (Int_t itru = 0; itru < (nTRU - 2); itru++) {
        if ((gammaHigh[itru][iphi] >> ieta) & 0x1) {
          mL1GammaHighPatchIndex.push_back(((iphi << 10) & 0x7C00) | ((ieta << 5) & 0x03E0) | ((itru << 0) & 0x001F));
        }
        if ((gammaLow[itru][iphi] >> ieta) & 0x1) {
          mL1GammaLowPatchIndex.push_back(((iphi << 10) & 0x7C00) | ((ieta << 5) & 0x03E0) | ((itru << 0) & 0x001F));
        }
      }
      // loop over TRUs of 1/3-size SMs
      for (Int_t itru = (nTRU - 2); itru < nTRU; itru++) {
        short iphi_tmp = (iphi % 2 + 2 * (int)(iphi / 6));
        short ieta_tmp = (ieta + 8 * ((int)(iphi / 2) % 3));
        if ((gammaHigh[itru][iphi] >> ieta) & 0x1) {
          mL1GammaHighPatchIndex.push_back(((iphi_tmp << 10) & 0x7C00) | ((ieta_tmp << 5) & 0x03E0) | ((itru << 0) & 0x001F));
        }
        if ((gammaLow[itru][iphi] >> ieta) & 0x1) {
          mL1GammaLowPatchIndex.push_back(((iphi_tmp << 10) & 0x7C00) | ((ieta_tmp << 5) & 0x03E0) | ((itru << 0) & 0x001F));
        }
      }
    }
  }

  return;
}

void StuDecoder::decodeFastOrADC(const uint32_t* buffer)
{
  for (int i = 0; i < getRawWords(); i++) {
    if (mDebug >= 2) {
      std::cout << std::dec << i << ")  0x" << std::hex << buffer[i] << "   <- FasrOR-word\n";
    }
    mFastOrADC.push_back(int16_t(buffer[i] & 0xFFFF));
    mFastOrADC.push_back(int16_t(buffer[i] >> 16) & 0xFFFF);
  }

  return;
}

void StuDecoder::dumpSTUcfg() const
{

  std::cout << "L1GammaHighThreshold:" << std::dec << getL1GammaHighThreshold() << std::endl;
  std::cout << "L1JetHighThreshold  :" << std::dec << getL1JetHighThreshold() << std::endl;
  std::cout << "L1GammaLowThreshold :" << std::dec << getL1GammaLowThreshold() << std::endl;
  std::cout << "L1JetLowThreshold   :" << std::dec << getL1JetLowThreshold() << std::endl;
  std::cout << "Rho                 :" << std::dec << getRho() << std::endl;
  std::cout << "FrameReceivedSTU    :" << std::dec << getFrameReceivedSTU() << std::endl;
  std::cout << "RegionEnable        :" << std::hex << "0x" << getRegionEnable() << std::endl;
  std::cout << "FrameReceived       :" << std::dec << getFrameReceived() << std::endl;
  std::cout << "ParchSize           :" << std::dec << getParchSize() << std::endl;
  std::cout << "FWversion           :" << std::hex << "0x" << getFWversion() << std::endl;

  std::cout << "\n";
}
