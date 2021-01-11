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
#include "FairLogger.h"
#include "CPVReconstruction/RawReaderMemory.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::cpv;

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
  LOG(ERROR) << "Unknown RDH version";
  return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV6*>(payloadwords));
  ;
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
}

RawErrorType_t RawReaderMemory::next()
{
  mRawPayload.clear();
  mCurrentTrailer.reset();
  bool isDataTerminated = false;
  do {
    RawErrorType_t e = nextPage();
    if (e != RawErrorType_t::kOK) {
      return e;
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
    mCurrentTrailer.constructFromPayloadWords(mRawPayload);
  } catch (...) {
    return RawErrorType_t::kHEADER_DECODING;
  }
  return RawErrorType_t::kOK;
}

RawErrorType_t RawReaderMemory::nextPage()
{
  if (!hasNext()) {
    return RawErrorType_t::kPAGE_NOTFOUND;
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
    return RawErrorType_t::kHEADER_DECODING;
  }
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Payload incomplete
    return RawErrorType_t::kPAYLOAD_DECODING;
  }

  auto tmp = reinterpret_cast<const uint32_t*>(mRawMemoryBuffer.data());
  int start = (mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader)) / sizeof(uint32_t);
  int end = start + (RDHDecoder::getMemorySize(mRawHeader) - RDHDecoder::getHeaderSize(mRawHeader)) / sizeof(uint32_t);
  for (auto iword = start; iword < end; iword++) {
    mRawPayload.push_back(tmp[iword]);
  }

  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
                                                               /*
      mCurrentTrailer.setPayloadSize(mCurrentTrailer.getPayloadSize() + trailer.getPayloadSize());
      tralersize = trailer.getTrailerSize();
    }

    gsl::span<const uint32_t> payloadWithoutTrailer(mRawBuffer.getDataWords().data(), mRawBuffer.getDataWords().size() - tralersize);

    mRawPayload.appendPayloadWords(payloadWithoutTrailer);
    mRawPayload.increasePageCount();
  }
  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
*/
  return RawErrorType_t::kOK;
}
