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
#include "EMCALBase/RCUTrailer.h"
#include "EMCALReconstruction/RawHeaderStream.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "EMCALReconstruction/RawDecodingError.h"

using namespace o2::emcal;

template <class RawHeader>
RawReaderMemory<RawHeader>::RawReaderMemory(gsl::span<const char> rawmemory) : mRawMemoryBuffer(rawmemory)
{
  init();
}

template <class RawHeader>
void RawReaderMemory<RawHeader>::setRawMemory(const gsl::span<const char> rawmemory)
{
  mRawMemoryBuffer = rawmemory;
  init();
}

template <class RawHeader>
void RawReaderMemory<RawHeader>::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mRawBuffer.flush();
  mNumData = mRawMemoryBuffer.size() / 8192; // assume fixed 8 kB pages
}

template <class RawHeader>
void RawReaderMemory<RawHeader>::next()
{
  mRawPayload.reset();
  bool isDataTerminated = false;
  do {
    nextPage(false);
    // check if we find a valid RCU trailer
    // the payload must be at the end of the buffer
    // if not present and error will be thrown
    try {
      RCUTrailer::constructFromPayloadWords(mRawBuffer.getDataWords());
      isDataTerminated = true;
    } catch (...) {
    }
  } while (!isDataTerminated);
}

template <class RawHeader>
void RawReaderMemory<RawHeader>::nextPage(bool doResetPayload)
{
  if (!hasNext())
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_NOTFOUND);
  if (doResetPayload)
    mRawPayload.reset();
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  // Use std::string stream as byte stream
  std::string headerwords(mRawMemoryBuffer.data() + mCurrentPosition, sizeof(RawHeader) / sizeof(char));
  std::stringstream headerstream(headerwords);
  // Read header
  try {
    headerstream >> mRawHeader;
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING);
  }
  if (mCurrentPosition + sizeof(RawHeader) + mRawHeader.memorySize >= mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_DECODING);
  } else {
    mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + mCurrentPosition + sizeof(RawHeader), mRawHeader.memorySize));
    mRawPayload.appendPayloadWords(mRawBuffer.getDataWords());
    mRawPayload.increasePageCount();
  }
  mCurrentPosition += mRawHeader.offsetToNext; /// Assume fixed 8 kB page size
}

template <class RawHeader>
void RawReaderMemory<RawHeader>::readPage(int page)
{
  int currentposition = 8192 * page;
  if (currentposition >= mRawMemoryBuffer.size())
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_NOTFOUND);
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  // Use std::string stream as byte stream
  std::string headerwords(mRawMemoryBuffer.data() + currentposition, sizeof(RawHeader) / sizeof(char));
  std::stringstream headerstream(headerwords);
  // Read header
  try {
    headerstream >> mRawHeader;
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING);
  }
  if (currentposition + sizeof(RawHeader) + mRawHeader.memorySize >= mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_DECODING);
  } else {
    mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + currentposition + sizeof(RawHeader), mRawHeader.memorySize));
  }
}

template <class RawHeader>
const RawHeader& RawReaderMemory<RawHeader>::getRawHeader() const
{
  if (!mRawHeaderInitialized)
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_INVALID);
  return mRawHeader;
}

template <class RawHeader>
const RawBuffer& RawReaderMemory<RawHeader>::getRawBuffer() const
{
  if (!mPayloadInitialized)
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_INVALID);
  return mRawBuffer;
}

template class o2::emcal::RawReaderMemory<o2::emcal::RAWDataHeader>;
template class o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4>;