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
#include <fairlogger/Logger.h>
#include "CPVReconstruction/RawReaderMemory.h"
#include "CPVReconstruction/RawDecoder.h"
#include "DataFormatsCPV/RawFormats.h"
#include "InfoLogger/InfoLogger.hxx"
#include "DetectorsRaw/RDHUtils.h"
#include "CPVBase/Geometry.h"

using namespace o2::cpv;

RawDecoder::RawDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                  mChannelsInitialized(false),
                                                  mIsMuteErrors(false)
{
}

RawErrorType_t RawDecoder::decode()
{
  // auto& rdh = mRawReader.getRawHeader();
  //    short linkID = o2::raw::RDHUtils::getLinkID(rdh);
  mDigits.clear();
  mBCRecords.clear();

  auto payloadWords = mRawReader.getPayload();
  if (payloadWords.size() == 0) {
    return kOK_NO_PAYLOAD;
  }

  return readChannels();
}

RawErrorType_t RawDecoder::readChannels()
{
  mChannelsInitialized = false;
  // // test error
  // if (!mIsMuteErrors) {
  //   LOG(error) << "RawDecoder::readChannels() : "
  //             << "test error";
  // }
  // mErrors.emplace_back(-1, 0, 0, 0, kOK); //5 is non-existing link with general errors

  uint8_t dataFormat = mRawReader.getDataFormat();
  int wordLength;
  if (dataFormat == 0x0) {
    wordLength = 16; // 128 bits word with padding
  } else if (dataFormat == 0x2) {
    wordLength = 10; // 80 bits word without padding
  } else {
    return RawErrorType_t::kWRONG_DATAFORMAT;
  }
  auto& payloadWords = mRawReader.getPayload();
  uint32_t wordCountFromLastHeader = 1; // header word is included
  int nDigitsAddedFromLastHeader = 0;
  bool isHeaderExpected = true;    // true if we expect to read header, false otherwise
  bool skipUntilNextHeader = true; // true if something wrong with data format, try to read next header
  uint16_t currentBC;
  uint32_t currentOrbit = mRawReader.getCurrentHBFOrbit();
  auto b = payloadWords.cbegin();
  auto e = payloadWords.cend();
  while (b != e) { // payload must start with cpvheader folowed by cpvwords and finished with cpvtrailer
    CpvHeader header(b, e);
    if (header.isOK()) {
      LOG(debug) << "RawDecoder::readChannels() : "
                 << "I read cpv header for orbit = " << header.orbit()
                 << " and BC = " << header.bc();
      if (!isHeaderExpected) { // actually, header was not expected
        if (!mIsMuteErrors) {
          LOG(error) << "RawDecoder::readChannels() : "
                     << "header was not expected";
        }
        removeLastNDigits(nDigitsAddedFromLastHeader); // remove previously added digits as they are bad
        mErrors.emplace_back(-1, 0, 0, 0, kNO_CPVTRAILER);
      }
      skipUntilNextHeader = false;
      currentBC = header.bc();
      wordCountFromLastHeader = 0;
      nDigitsAddedFromLastHeader = 0;
      if (currentOrbit != header.orbit()) { // bad cpvheader
        if (!mIsMuteErrors) {
          LOG(error) << "RawDecoder::readChannels() : "
                     << "currentOrbit(=" << currentOrbit
                     << ") != header.orbit()(=" << header.orbit() << ")";
        }
        mErrors.emplace_back(-1, 0, 0, 0, kCPVHEADER_INVALID); // 5 is non-existing link with general errors
        skipUntilNextHeader = true;
      }
    } else {
      if (skipUntilNextHeader) {
        b += wordLength;
        continue; // continue while'ing until it's not header
      }
      CpvWord word(b, e);
      if (word.isOK()) {
        wordCountFromLastHeader++;
        for (int i = 0; i < 3; i++) {
          PadWord pw = {word.cpvPadWord(i)};
          if (pw.zero == 0) { // cpv pad word, not control or empty
            if (addDigit(pw.mDataWord, word.ccId(), currentBC)) {
              nDigitsAddedFromLastHeader++;
            } else {
              if (!mIsMuteErrors) {
                LOG(debug) << "RawDecoder::readChannels() : "
                           << "read pad word with non-valid pad address";
              }
              unsigned int dil = pw.dil, gas = pw.gas, address = pw.address;
              mErrors.emplace_back(word.ccId(), dil, gas, address, kPadAddress);
            }
          }
        }
      } else { // this may be trailer
        CpvTrailer trailer(b, e);
        if (trailer.isOK()) {
          int diffInCount = wordCountFromLastHeader - trailer.wordCounter();
          if (diffInCount > 1 ||
              diffInCount < -1) {
            // some words lost?
            if (!mIsMuteErrors) {
              LOG(error) << "RawDecoder::readChannels() : "
                         << "Read " << wordCountFromLastHeader << " words, expected " << trailer.wordCounter();
            }
            mErrors.emplace_back(-1, 0, 0, 0, kCPVTRAILER_INVALID);
            // throw all previous data and go to next header
            removeLastNDigits(nDigitsAddedFromLastHeader);
            skipUntilNextHeader = true;
          }
          if (trailer.bc() != currentBC) {
            // trailer does not fit header
            if (!mIsMuteErrors) {
              LOG(error) << "RawDecoder::readChannels() : "
                         << "CPVHeader BC(" << currentBC << ") != CPVTrailer BC(" << trailer.bc() << ")";
            }
            mErrors.emplace_back(-1, 0, 0, 0, kCPVTRAILER_INVALID);
            removeLastNDigits(nDigitsAddedFromLastHeader);
            skipUntilNextHeader = true;
          }
          isHeaderExpected = true;
        } else {
          uint8_t unknownWord[10];
          bool isPadding = isHeaderExpected && dataFormat == 0x2; // may this be padding?
          for (int i = 0; i < 10 && (b + i) != e; i++) {          // read up to 10 mBytes
            unknownWord[i] = *(b + i);
            if (unknownWord[i] != 0xff) { // padding
              isPadding = false;
            }
          }
          if (!isPadding) { // this is unknown word error
            if (!mIsMuteErrors) {
              LOGF(info, "RawDecoder::readChannels() : Read unknown word  0x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                   unknownWord[9], unknownWord[8], unknownWord[7], unknownWord[6], unknownWord[5], unknownWord[4], unknownWord[3],
                   unknownWord[2], unknownWord[1], unknownWord[0]);
            }
            mErrors.emplace_back(-1, 0, 0, 0, kUNKNOWN_WORD); // add error for non-existing row
            wordCountFromLastHeader++;
          }
        }
      }
    }
    b += wordLength;
  }
  mChannelsInitialized = true;
  return kOK;
}

bool RawDecoder::addDigit(uint32_t w, short ccId, uint16_t bc)
{
  // add digit
  PadWord pad = {w};
  unsigned short absId;
  if (!o2::cpv::Geometry::hwaddressToAbsId(ccId, pad.dil, pad.gas, pad.address, absId)) {
    return false;
  }

  // new bc -> add bc reference
  if (mBCRecords.empty() || (mBCRecords.back().bc != bc)) {
    mBCRecords.push_back(BCRecord(bc, mDigits.size(), mDigits.size()));
  } else {
    mBCRecords.back().lastDigit++;
  }

  AddressCharge ac = {0};
  ac.Address = absId;
  ac.Charge = pad.charge;
  mDigits.push_back(ac.mDataWord);

  return true;
}

void RawDecoder::removeLastNDigits(int n)
{
  if (n < 0) {
    return;
  }
  int nRemoved = 0;
  while (nRemoved < n) {
    if (mDigits.size() > 0) { // still has digits to remove
      mDigits.pop_back();
      if (mBCRecords.back().lastDigit == mBCRecords.back().firstDigit) {
        mBCRecords.pop_back();
      } else {
        mBCRecords.back().lastDigit--;
      }
      nRemoved++;
    } else { // has nothing to remove already
      break;
    }
  }
}
