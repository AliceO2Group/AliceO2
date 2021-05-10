// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <FairLogger.h>
#include "CPVReconstruction/RawReaderMemory.h"
#include "CPVReconstruction/RawDecoder.h"
#include "DataFormatsCPV/RawFormats.h"
#include "InfoLogger/InfoLogger.hxx"
#include "DetectorsRaw/RDHUtils.h"
#include "CPVBase/Geometry.h"

using namespace o2::cpv;

RawDecoder::RawDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                  mChannelsInitialized(false)
{
}

RawErrorType_t RawDecoder::decode()
{

  auto& rdh = mRawReader.getRawHeader();
  short linkID = o2::raw::RDHUtils::getLinkID(rdh);
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

  auto& payloadWords = mRawReader.getPayload();
  uint32_t wordCountFromLastHeader = 1; //header word is included
  int nDigitsAddedFromLastHeader = 0;
  bool isHeaderExpected = true;    //true if we expect to read header, false otherwise
  bool skipUntilNextHeader = true; //true if something wrong with data format, try to read next header
  uint16_t currentBC;
  uint32_t currentOrbit = mRawReader.getCurrentHBFOrbit();
  auto b = payloadWords.cbegin();
  auto e = payloadWords.cend();
  while (b != e) { //payload must start with cpvheader folowed by cpvwords and finished with cpvtrailer
    CpvHeader header(b, e);
    if (header.isOK()) {
      if (!isHeaderExpected) { //actually, header was not expected
        LOG(ERROR) << "RawDecoder::readChannels() : "
                   << "header was not expected";
        removeLastNDigits(nDigitsAddedFromLastHeader); //remove previously added digits as they are bad
        mErrors.emplace_back(5, 0, 0, 0, kNO_CPVTRAILER);
      }
      skipUntilNextHeader = false;
      currentBC = header.bc();
      wordCountFromLastHeader = 0;
      nDigitsAddedFromLastHeader = 0;
      if (currentOrbit != header.orbit()) { //bad cpvheader
        LOG(ERROR) << "RawDecoder::readChannels() : "
                   << "currentOrbit != header.orbit()";
        mErrors.emplace_back(5, 0, 0, 0, kCPVHEADER_INVALID); //5 is non-existing link with general errors
        skipUntilNextHeader = true;
      }
    } else {
      if (skipUntilNextHeader) {
        b += 16;
        continue; //continue while'ing until it's not header
      }
      CpvWord word(b, e);
      if (word.isOK()) {
        wordCountFromLastHeader++;
        for (int i = 0; i < 3; i++) {
          PadWord pw = {word.cpvPadWord(i)};
          if (pw.zero == 0) {
            addDigit(pw.mDataWord, word.ccId(), currentBC);
            nDigitsAddedFromLastHeader++;
          }
        }
      } else { //this may be trailer
        CpvTrailer trailer(b, e);
        if (trailer.isOK()) {
          int diffInCount = wordCountFromLastHeader - trailer.wordCounter();
          if (diffInCount > 1 ||
              diffInCount < -1) {
            //some words lost?
            LOG(ERROR) << "RawDecoder::readChannels() : "
                       << "Read " << wordCountFromLastHeader << " words, expected " << trailer.wordCounter();
            mErrors.emplace_back(5, 0, 0, 0, kCPVTRAILER_INVALID);
            //throw all previous data and go to next header
            removeLastNDigits(nDigitsAddedFromLastHeader);
            skipUntilNextHeader = true;
          }
          if (trailer.bc() != currentBC) {
            //trailer does not fit header
            LOG(ERROR) << "RawDecoder::readChannels() : "
                       << "CPVHeader BC is " << currentBC << " but CPVTrailer BC is " << trailer.bc();
            mErrors.emplace_back(5, 0, 0, 0, kCPVTRAILER_INVALID);
            removeLastNDigits(nDigitsAddedFromLastHeader);
            skipUntilNextHeader = true;
          }
          isHeaderExpected = true;
        } else {
          wordCountFromLastHeader++;
          //error
          LOG(ERROR) << "RawDecoder::readChannels() : "
                     << "Read unknown word";
          mErrors.emplace_back(5, 0, 0, 0, kUNKNOWN_WORD); //add error for non-existing row
          //what to do?
        }
      }
    }
    b += 16;
  }
  mChannelsInitialized = true;
  return kOK;
}

void RawDecoder::addDigit(uint32_t w, short ccId, uint16_t bc)
{
  //new bc -> add bc reference
  if (mBCRecords.empty() || (mBCRecords.back().bc != bc)) {
    mBCRecords.push_back(BCRecord(bc, mDigits.size(), mDigits.size()));
  } else {
    mBCRecords.back().lastDigit++;
  }

  //add digit
  PadWord pad = {w};
  unsigned short absId;
  o2::cpv::Geometry::hwaddressToAbsId(ccId, pad.dil, pad.gas, pad.address, absId);

  AddressCharge ac = {0};
  ac.Address = absId;
  ac.Charge = pad.charge;
  mDigits.push_back(ac.mDataWord);
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