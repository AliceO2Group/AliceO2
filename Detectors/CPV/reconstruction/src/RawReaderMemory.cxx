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
  LOG(error) << "RawReaderMemory::decodeRawHeader() : Unknown RDH version";
  throw RawErrorType_t::kRDH_DECODING;
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mCurrentHBFOrbit = 0;
  mStopBitWasNotFound = false;
  mIsJustInited = true;
}

// Read the next pages until the stop bit is found or new HBF reached
// it means we read 1 HBF per next() call
RawErrorType_t RawReaderMemory::next()
{
  mRawPayload.clear();
  bool isStopBitFound = false;
  do {
    RawErrorType_t e = nextPage();
    if (e == RawErrorType_t::kPAGE_NOTFOUND ||       // nothing left to read...
        e == RawErrorType_t::kRDH_DECODING ||        // incorrect rdh -> fatal error
        e == RawErrorType_t::kPAYLOAD_INCOMPLETE ||  // we reached end of mRawMemoryBuffer but payload size from rdh tells to read more
        e == RawErrorType_t::kSTOPBIT_NOTFOUND ||    // new HBF orbit started but no stop bit found, need to return
        e == RawErrorType_t::kNOT_CPV_RDH ||         // not cpv rdh -> most probably
        e == RawErrorType_t::kOFFSET_TO_NEXT_IS_0) { // offset to next package is 0 -> do not know how to read next
      throw e;                                       // some principal error occured -> stop reading.
    }
    isStopBitFound = RDHDecoder::getStop(mRawHeader);
  } while (!isStopBitFound);

  return RawErrorType_t::kOK;
}

// Read the next ONLY ONE page from the stream (single DMA page)
// note: 1 raw header per page
RawErrorType_t RawReaderMemory::nextPage()
{
  if (!hasNext()) {
    return RawErrorType_t::kPAGE_NOTFOUND;
  }
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;

  // Read RDH header
  o2::header::RDHAny rawHeader;
  try {
    rawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    if (RDHDecoder::getOffsetToNext(rawHeader) == 0) { // dont' know how to read next -> skip to next HBF
      return RawErrorType_t::kOFFSET_TO_NEXT_IS_0;
    }
    if (RDHDecoder::getSourceID(rawHeader) != 0x8) {
      // Not a CPV RDH
      mCurrentPosition += RDHDecoder::getOffsetToNext(rawHeader); // not cpv rdh -> skip to next HBF
      return RawErrorType_t::kNOT_CPV_RDH;
    }
    if (mIsJustInited || mStopBitWasNotFound) { // reading first time after init() or stopbit was not found
      mCurrentHBFOrbit = RDHDecoder::getHeartBeatOrbit(rawHeader);
      mRawHeader = rawHeader; // save RDH of first page as mRawHeader
      mRawHeaderInitialized = true;
      mStopBitWasNotFound = false; // reset this flag as we start to read again
      mIsJustInited = false;
    } else if (mCurrentHBFOrbit != RDHDecoder::getHeartBeatOrbit(rawHeader)) {
      // next HBF started but we didn't find stop bit.
      mStopBitWasNotFound = true;
      mCurrentPosition += RDHDecoder::getOffsetToNext(rawHeader); // moving on
      return RawErrorType_t::kSTOPBIT_NOTFOUND;                   // Stop bit was not found -> skip to next HBF
    }
  } catch (...) {
    return RawErrorType_t::kRDH_DECODING; // this is fatal error -> skip whole TF
  }
  mRawHeader = rawHeader; // save RDH of current page as mRawHeader
  mRawHeaderInitialized = true;

  auto tmp = mRawMemoryBuffer.data();
  int start = (mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader));
  int end = (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader));
  bool isPayloadIncomplete = false;
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Payload incomplete
    isPayloadIncomplete = true;
    end = mRawMemoryBuffer.size(); // OK, lets read it anyway. Maybe there still are some completed events...
  }
  for (auto iword = start; iword < end; iword++) {
    mRawPayload.push_back(tmp[iword]);
  }
  mPayloadInitialized = true;

  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
  if (isPayloadIncomplete) {
    return RawErrorType_t::kPAYLOAD_INCOMPLETE; // skip to next HBF
  }
  return RawErrorType_t::kOK;
}
