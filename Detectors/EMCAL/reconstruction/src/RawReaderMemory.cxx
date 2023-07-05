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

#include <sstream>
#include <string>
#include "EMCALReconstruction/RawReaderMemory.h"
#include "EMCALReconstruction/RawDecodingError.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::emcal;

using RDHDecoder = o2::raw::RDHUtils;

RawReaderMemory::RawReaderMemory(gsl::span<const char> rawmemory) : mRawMemoryBuffer(rawmemory)
{
  init();
}

void RawReaderMemory::setRawMemory(const gsl::span<const char> rawmemory)
{
  mRawMemoryBuffer = rawmemory;
  init();
}

o2::header::RDHAny RawReaderMemory::decodeRawHeader(const void* payloadwords)
{
  auto headerversion = RDHDecoder::getVersion(payloadwords);
  if (headerversion < RDHDecoder::getVersion<o2::header::RDHLowest>() || headerversion > RDHDecoder::getVersion<o2::header::RDHHighest>()) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING, mCurrentFEE);
  }
  return {*reinterpret_cast<const o2::header::RDHAny*>(payloadwords)};
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mRawBuffer.flush();
}

void RawReaderMemory::next()
{
  mRawPayload.reset();
  mCurrentTrailer.reset();
  bool isDataTerminated = false;
  do {
    nextPage(false);
    if (hasNext()) {
      auto nextheader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
      /**
       * eventually in the future check continuing payload based on the bc/orbit ID
      auto currentbc = RDHDecoder::getTriggerBC(mRawHeader),
           nextbc = RDHDecoder::getTriggerBC(nextheader);
      auto currentorbit = RDHDecoder::getTriggerOrbit(mRawHeader),
           nextorbit = RDHDecoder::getTriggerOrbit(nextheader);
      **/
      auto nextpagecounter = RDHDecoder::getPageCounter(nextheader);
      if (nextpagecounter == 0) {
        isDataTerminated = true;
      } else {
        isDataTerminated = false;
      }
    } else {
      isDataTerminated = true;
    }
    // Check if the data continues
  } while (!isDataTerminated);
  // add combined trailer to payload (in case the FEE is a SRU DDL)
  if (mCurrentTrailer.isInitialized()) {
    mRawPayload.appendPayloadWords(mCurrentTrailer.encode());
  }
}

void RawReaderMemory::nextPage(bool doResetPayload)
{
  if (!hasNext()) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_NOTFOUND, mCurrentFEE);
  }
  if (doResetPayload) {
    mRawPayload.reset();
  }
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  // Read header
  try {
    mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    auto feeID = RDHDecoder::getFEEID(mRawHeader);
    if (mCurrentFEE < 0 || mCurrentFEE != feeID) {
      // update current FEE ID
      mCurrentFEE = feeID;
    }
    // RDHDecoder::printRDH(mRawHeader);
    if (RDHDecoder::getOffsetToNext(mRawHeader) == RDHDecoder::getHeaderSize(mRawHeader)) {
      // No Payload - jump to next rawheader
      // This will eventually move, depending on whether for events without payload in the SRU we send the RCU trailer
      mCurrentPosition += RDHDecoder::getHeaderSize(mRawHeader);
      mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
      feeID = RDHDecoder::getFEEID(mRawHeader);
      if (mCurrentFEE < 0 || mCurrentFEE != feeID) {
        // update current FEE ID
        mCurrentFEE = feeID;
      }
      // RDHDecoder::printRDH(mRawHeader);
    }
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING, mCurrentFEE);
  }
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_DECODING, mCurrentFEE);
  } else if (mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Start position of the payload is outside the payload range
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_START_INVALID, mCurrentFEE);
  } else {
    mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader), RDHDecoder::getMemorySize(mRawHeader) - RDHDecoder::getHeaderSize(mRawHeader)));

    gsl::span<const uint32_t> payloadWithoutTrailer;
    auto feeID = RDHDecoder::getFEEID(mRawHeader);
    if (feeID >= mMinSRUDDL && feeID <= mMaxSRUDDL) {
      // Read off and chop trailer (if required)
      //
      // In case every page gets a trailer (intermediate format). The trailers from the single
      // pages need to be removed. There will be a combined trailer which keeps the sum of the
      // payloads for all trailers. This will be appended to the chopped payload.
      //
      // Trailer only at the last page (new format): Only last page gets trailer. The trailer is
      // also chopped from the payload as it will be added later again.
      //
      // The trailer is only decoded if the DDL is in the range of SRU DDLs. STU pages are propagated
      // 1-1 without trailer parsing
      auto lastword = *(mRawBuffer.getDataWords().rbegin());
      if (lastword >> 30 == 3) {
        // lastword is a trailer word
        // decode trailer and chop
        try {
          auto trailer = RCUTrailer::constructFromPayloadWords(mRawBuffer.getDataWords());
          if (!mCurrentTrailer.isInitialized()) {
            mCurrentTrailer = trailer;
          } else {
            mCurrentTrailer.setPayloadSize(mCurrentTrailer.getPayloadSize() + trailer.getPayloadSize());
          }
          payloadWithoutTrailer = gsl::span<const uint32_t>(mRawBuffer.getDataWords().data(), mRawBuffer.getDataWords().size() - trailer.getTrailerSize());
        } catch (RCUTrailer::Error& e) {
          throw RawDecodingError(RawDecodingError::ErrorType_t::TRAILER_DECODING, mCurrentFEE);
        }
      } else {
        // Not a trailer word = copy page as it is
        payloadWithoutTrailer = mRawBuffer.getDataWords(); // No trailer to be chopped
      }
    } else {
      payloadWithoutTrailer = mRawBuffer.getDataWords(); // No trailer to be chopped
    }

    mRawPayload.appendPayloadWords(payloadWithoutTrailer);
    mRawPayload.increasePageCount();
  }
  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
}

const o2::header::RDHAny& RawReaderMemory::getRawHeader() const
{
  if (!mRawHeaderInitialized) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_INVALID, mCurrentFEE);
  }
  return mRawHeader;
}

const RawBuffer& RawReaderMemory::getRawBuffer() const
{
  if (!mPayloadInitialized) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_INVALID, mCurrentFEE);
  }
  return mRawBuffer;
}
