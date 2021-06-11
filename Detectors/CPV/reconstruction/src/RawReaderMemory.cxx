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
  LOG(ERROR) << "RawReaderMemory::decodeRawHeader() : Unknown RDH version";
  return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV6*>(payloadwords));
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mCurrentHBFOrbit = 0;
  mStopBitWasNotFound = false;
}

//Read the next pages until the stop bit is found or new HBF reached
//it means we read 1 HBF per next() call
RawErrorType_t RawReaderMemory::next()
{
  mRawPayload.clear();
  bool isStopBitFound = false;
  do {
    RawErrorType_t e = nextPage();
    if (e == RawErrorType_t::kPAGE_NOTFOUND ||      // nothing left to read...
        e == RawErrorType_t::kRDH_DECODING ||       // incorrect rdh -> fatal error
        e == RawErrorType_t::kPAYLOAD_INCOMPLETE || // we reached end of mRawMemoryBuffer but payload size from rdh tells to read more
        e == RawErrorType_t::kSTOPBIT_NOTFOUND) {   //new HBF orbit started but no stop bit found, need to return
      return e;                                     //some principal error occured -> stop reading.
    }
    isStopBitFound = RDHDecoder::getStop(mRawHeader);
  } while (!isStopBitFound);

  return RawErrorType_t::kOK;
}

//Read the next ONLY ONE page from the stream (single DMA page)
//note: 1 raw header per page
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
  } catch (...) {
    return RawErrorType_t::kRDH_DECODING; //this is fatal error
  }
  if (RDHDecoder::getSourceID(rawHeader) != 0x8) {
    // Not a CPV RDH
    mCurrentPosition += RDHDecoder::getOffsetToNext(rawHeader); //moving on
    return RawErrorType_t::kNOT_CPV_RDH;
  }
  if (mCurrentHBFOrbit != 0 || mStopBitWasNotFound) { //reading first time after init() or stopbit was not found
    mCurrentHBFOrbit = RDHDecoder::getHeartBeatOrbit(rawHeader);
    mRawHeader = rawHeader; //save RDH of first page as mRawHeader
    mRawHeaderInitialized = true;
    mStopBitWasNotFound = false; //reset this flag as we start to read again
  } else if (mCurrentHBFOrbit != RDHDecoder::getHeartBeatOrbit(rawHeader)) {
    //next HBF started but we didn't find stop bit.
    mStopBitWasNotFound = true;
    return RawErrorType_t::kSTOPBIT_NOTFOUND; //Stop reading, this will be read again by calling next()
  }
  mRawHeader = rawHeader; //save RDH of current page as mRawHeader
  mRawHeaderInitialized = true;

  auto tmp = mRawMemoryBuffer.data();
  int start = (mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader));
  int end = (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader));
  bool isPayloadIncomplete = false;
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) > mRawMemoryBuffer.size()) {
    // Payload incomplete
    end = mRawMemoryBuffer.size(); //OK, lets read it anyway. Maybe there still are some completed events...
  }
  for (auto iword = start; iword < end; iword++) {
    mRawPayload.push_back(tmp[iword]);
  }
  mPayloadInitialized = true;

  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
  if (isPayloadIncomplete) {
    return RawErrorType_t::kPAYLOAD_INCOMPLETE; //return error so we can it handle later
  }
  return RawErrorType_t::kOK;
}
