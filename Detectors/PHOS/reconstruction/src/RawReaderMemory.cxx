// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <sstream>
#include <string>
#include "PHOSReconstruction/RawReaderMemory.h"
#include "PHOSReconstruction/RawDecodingError.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::phos;

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
  if (headerversion == 4) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(payloadwords));
  } else if (headerversion == 5) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(payloadwords));
  } else if (headerversion == 6) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV6*>(payloadwords));
  }
  throw RawDecodingError::ErrorType_t::HEADER_DECODING;
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mRawBuffer.flush();
  mNumData = mRawMemoryBuffer.size() / 8192; // assume fixed 8 kB pages
}

void RawReaderMemory::next()
{
  //Reads pages till RCU trailer found
  //Several 8kB pages can be concatenated
  //RCU trailer expected at the end of payload
  //but not at the end of each page
  mRawPayload.reset();
  mCurrentTrailer.reset();
  bool isDataTerminated = false;
  do {
    try {
      nextPage(false);
    } catch (RawDecodingError::ErrorType_t e) {
      throw e;
    }
    if (hasNext()) {
      auto nextheader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
      // check continuing payload based on the bc/orbit ID
      auto currentbc = RDHDecoder::getTriggerBC(mRawHeader),
           nextbc = RDHDecoder::getTriggerBC(nextheader);
      auto currentorbit = RDHDecoder::getTriggerOrbit(mRawHeader),
           nextorbit = RDHDecoder::getTriggerOrbit(nextheader);
      if (currentbc != nextbc || currentorbit != nextorbit) {
        isDataTerminated = true;
      } else {
        auto nextpagecounter = RDHDecoder::getPageCounter(nextheader);
        if (nextpagecounter == 0) {
          isDataTerminated = true;
        } else {
          isDataTerminated = false;
        }
      }
    } else {
      isDataTerminated = true;
    }
    // Check if the data continues
  } while (!isDataTerminated);
  try {
    mCurrentTrailer.constructFromPayloadWords(mRawBuffer.getDataWords());
  } catch (...) {
    throw RawDecodingError::ErrorType_t::HEADER_DECODING;
  }
}

void RawReaderMemory::nextPage(bool doResetPayload)
{
  if (!hasNext()) {
    throw RawDecodingError::ErrorType_t::PAGE_NOTFOUND;
  }
  if (doResetPayload) {
    mRawPayload.reset();
  }
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;

  // Read RDH header
  try {
    mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    while (RDHDecoder::getOffsetToNext(mRawHeader) == RDHDecoder::getHeaderSize(mRawHeader) &&
           mCurrentPosition < mRawMemoryBuffer.size()) {
      // No Payload - jump to next rawheader
      // This will eventually move, depending on whether for events without payload in the SRU we send the RCU trailer
      mCurrentPosition += RDHDecoder::getHeaderSize(mRawHeader);
      mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    }
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError::ErrorType_t::HEADER_DECODING;
  }
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError::ErrorType_t::PAYLOAD_DECODING;
  }

  mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader),
                                                        RDHDecoder::getMemorySize(mRawHeader) - RDHDecoder::getHeaderSize(mRawHeader)));
  gsl::span<const uint32_t> payloadWithoutTrailer(mRawBuffer.getDataWords().data(), mRawBuffer.getNDataWords());
  mRawPayload.appendPayloadWords(payloadWithoutTrailer);
  mRawPayload.increasePageCount();

  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
}

const o2::header::RDHAny& RawReaderMemory::getRawHeader() const
{
  if (!mRawHeaderInitialized) {
    throw RawDecodingError::ErrorType_t::HEADER_INVALID;
  }
  return mRawHeader;
}

const RawBuffer& RawReaderMemory::getRawBuffer() const
{
  if (!mPayloadInitialized) {
    throw RawDecodingError::ErrorType_t::PAYLOAD_INVALID;
  }
  return mRawBuffer;
}
