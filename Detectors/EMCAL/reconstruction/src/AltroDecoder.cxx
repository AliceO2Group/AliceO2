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
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fmt/format.h>
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "Framework/Logger.h"

using namespace o2::emcal;

AltroDecoder::AltroDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                      mRCUTrailer(),
                                                      mChannels(),
                                                      mChannelsInitialized(false)
{
}

void AltroDecoder::decode()
{
  mMinorDecodingErrors.clear();
  readRCUTrailer();
  checkRCUTrailer();
  readChannels();
}

void AltroDecoder::readRCUTrailer()
{
  try {
    auto payloadwordsOrig = mRawReader.getPayload().getPayloadWords();
    gsl::span<const uint32_t> payloadwords(payloadwordsOrig.data(), payloadwordsOrig.size());
    mRCUTrailer.constructFromRawPayload(payloadwords);
  } catch (RCUTrailer::Error& e) {
    LOG(error) << "Error while decoding RCU trailer: " << e.what();
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR, fmt::format("{} {}", AltroDecoderError::getErrorTypeDescription(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR), e.what()));
  }
}

void AltroDecoder::checkRCUTrailer()
{
  int trailersize = mRCUTrailer.getTrailerSize();
  int buffersize = mRawReader.getPayload().getPayloadWords().size();
  if (trailersize > buffersize) {
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_SIZE_ERROR, fmt::format("{}: Trailer size {} exceeding buffer size {}", AltroDecoderError::getErrorTypeDescription(AltroDecoderError::ErrorType_t::RCU_TRAILER_SIZE_ERROR), trailersize, buffersize));
  }
}

void AltroDecoder::readChannels()
{
  mChannelsInitialized = false;
  mChannels.clear();
  int currentpos = 0;
  auto& buffer = mRawReader.getPayload().getPayloadWords();
  auto maxpayloadsize = buffer.size() - mRCUTrailer.getTrailerSize();
  while (currentpos < maxpayloadsize) {
    auto currentword = buffer[currentpos++];
    if (currentword >> 30 != 1) {
      continue;
    }

    // decode channel header
    auto channelheader = currentword;
    int32_t hwaddress = channelheader & 0xFFF;
    uint16_t payloadsize = (channelheader >> 16) & 0x3FF;
    bool badchannel = (channelheader >> 29) & 0x1;

    /// decode all words for channel
    bool foundChannelError = false;
    int numberofwords = (payloadsize + 2) / 3;
    std::vector<uint16_t> bunchwords;
    for (int iword = 0; iword < numberofwords; iword++) {
      if (currentpos >= maxpayloadsize) {
        mMinorDecodingErrors.emplace_back(MinorAltroDecodingError::ErrorType_t::CHANNEL_PAYLOAD_EXCEED, channelheader, currentword);
        foundChannelError = true;
        break; // Must break here in order not to prevent a buffer overrun
      }
      currentword = buffer[currentpos++];
      if ((currentword >> 30) != 0) {
        // word is a new channel header
        mMinorDecodingErrors.emplace_back(MinorAltroDecodingError::ErrorType_t::CHANNEL_END_PAYLOAD_UNEXPECT, channelheader, currentword);
        foundChannelError = true;
        currentpos--;
        continue;
      }
      bunchwords.push_back((currentword >> 20) & 0x3FF);
      bunchwords.push_back((currentword >> 10) & 0x3FF);
      bunchwords.push_back(currentword & 0x3FF);
    }
    if (foundChannelError) {
      // do not decode bunch if channel payload is corrupted
      continue;
    }
    // Payload decoding for channel good - starting a new channel object
    mChannels.emplace_back(hwaddress, payloadsize);
    auto& currentchannel = mChannels.back();
    currentchannel.setBadChannel(badchannel);

    // decode bunches
    int currentsample = 0;
    while (currentsample < currentchannel.getPayloadSize() && bunchwords.size() > currentsample + 2) {
      // Check if bunch word is 0 - if yes skip all following bunches as they can no longer be reliably decoded
      if (bunchwords[currentsample] == 0) {
        mMinorDecodingErrors.emplace_back(MinorAltroDecodingError::ErrorType_t::BUNCH_HEADER_NULL, channelheader, 0);
        break;
      }
      int bunchlength = bunchwords[currentsample] - 2, // remove words for bunchlength and starttime
        starttime = bunchwords[currentsample + 1];
      // Raise minor decoding error in case the bunch length exceeds the channel payload and skip the bunch
      if ((unsigned long)bunchlength > bunchwords.size() - currentsample - 2) {
        mMinorDecodingErrors.emplace_back(MinorAltroDecodingError::ErrorType_t::BUNCH_LENGTH_EXCEED, channelheader, 0);
        // we must break here as well, the bunch is cut and the pointer would be set to invalid memory
        break;
      }
      if (bunchlength == 0) {
        // skip bunches with bunch size 0, they don't contain any payload
        // Map error type to NULL header since header word is null
        mMinorDecodingErrors.emplace_back(MinorAltroDecodingError::ErrorType_t::BUNCH_HEADER_NULL, channelheader, 0);
        // Forward position by bunch header size
        currentsample += bunchlength + 2;
        continue;
      }
      auto& currentbunch = currentchannel.createBunch(bunchlength, starttime);
      currentbunch.initFromRange(gsl::span<uint16_t>(&bunchwords[currentsample + 2], std::min((unsigned long)bunchlength, bunchwords.size() - currentsample - 2)));
      currentsample += bunchlength + 2;
    }
  }
  mChannelsInitialized = true;
}

const RCUTrailer& AltroDecoder::getRCUTrailer() const
{
  if (!mRCUTrailer.isInitialized()) {
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR, fmt::format("{}: RCU trailer was not initialized", AltroDecoderError::getErrorTypeDescription(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR)));
  }
  return mRCUTrailer;
}

const std::vector<Channel>& AltroDecoder::getChannels() const
{
  if (!mChannelsInitialized) {
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::CHANNEL_ERROR, AltroDecoderError::getErrorTypeDescription(AltroDecoderError::ErrorType_t::CHANNEL_ERROR));
  }
  return mChannels;
}

using AltroErrType = o2::emcal::AltroDecoderError::ErrorType_t;

int AltroDecoderError::errorTypeToInt(AltroErrType errortype)
{

  int errorNumber = -1;

  switch (errortype) {
    case AltroErrType::RCU_TRAILER_ERROR:
      errorNumber = 0;
      break;
    case AltroErrType::RCU_VERSION_ERROR:
      errorNumber = 1;
      break;
    case AltroErrType::RCU_TRAILER_SIZE_ERROR:
      errorNumber = 2;
      break;
    case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
      errorNumber = 3;
      break;
    case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
      errorNumber = 4;
      break;
    case AltroErrType::ALTRO_PAYLOAD_ERROR:
      errorNumber = 5;
      break;
    case AltroErrType::ALTRO_MAPPING_ERROR:
      errorNumber = 6;
      break;
    case AltroErrType::CHANNEL_ERROR:
      errorNumber = 7;
      break;
    default:
      break;
  }

  return errorNumber;
}

AltroErrType AltroDecoderError::intToErrorType(int errornumber)
{
  assert(errornumber < getNumberOfErrorTypes());

  AltroErrType errorType;

  switch (errornumber) {
    case 0:
      errorType = AltroErrType::RCU_TRAILER_ERROR;
      break;
    case 1:
      errorType = AltroErrType::RCU_VERSION_ERROR;
      break;
    case 2:
      errorType = AltroErrType::RCU_TRAILER_SIZE_ERROR;
      break;
    case 3:
      errorType = AltroErrType::ALTRO_BUNCH_HEADER_ERROR;
      break;
    case 4:
      errorType = AltroErrType::ALTRO_BUNCH_LENGTH_ERROR;
      break;
    case 5:
      errorType = AltroErrType::ALTRO_PAYLOAD_ERROR;
      break;
    case 6:
      errorType = AltroErrType::ALTRO_MAPPING_ERROR;
      break;
    case 7:
      errorType = AltroErrType::CHANNEL_ERROR;
      break;
    default:
      break;
  }

  return errorType;
}

const char* AltroDecoderError::getErrorTypeName(ErrorType_t errortype)
{
  switch (errortype) {
    case AltroErrType::RCU_TRAILER_ERROR:
      return "RCUTrailerError";
    case AltroErrType::RCU_VERSION_ERROR:
      return "RCUTrailerVersionError";
    case AltroErrType::RCU_TRAILER_SIZE_ERROR:
      return "RCUTrailerSizeError";
    case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
      return "BunchHeaderError";
    case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
      return "BunchLengthError";
    case AltroErrType::ALTRO_PAYLOAD_ERROR:
      return "ALTROPayloadError";
    case AltroErrType::ALTRO_MAPPING_ERROR:
      return "ALTROMappingError";
    case AltroErrType::CHANNEL_ERROR:
      return "ChannelError";
  };
  return "";
}

const char* AltroDecoderError::getErrorTypeTitle(ErrorType_t errortype)
{
  switch (errortype) {
    case AltroErrType::RCU_TRAILER_ERROR:
      return "RCU Trailer";
    case AltroErrType::RCU_VERSION_ERROR:
      return "RCU Version";
    case AltroErrType::RCU_TRAILER_SIZE_ERROR:
      return "RCU Trailer Size";
    case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
      return "ALTRO Bunch Header";
    case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
      return "ALTRO Bunch Length";
    case AltroErrType::ALTRO_PAYLOAD_ERROR:
      return "ALTRO Payload";
    case AltroErrType::ALTRO_MAPPING_ERROR:
      return "ALTRO Mapping";
    case AltroErrType::CHANNEL_ERROR:
      return "Channel";
  };
  return "";
}

const char* AltroDecoderError::getErrorTypeDescription(ErrorType_t errortype)
{
  switch (errortype) {
    case AltroErrType::RCU_TRAILER_ERROR:
      return "RCU trailer decoding error";
    case AltroErrType::RCU_VERSION_ERROR:
      return "Inconsistent RCU trailer version";
    case AltroErrType::RCU_TRAILER_SIZE_ERROR:
      return "Invalid RCU trailer size";
    case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
      return "Inconsistent bunch header";
    case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
      return "Bunch length exceeding payload size";
    case AltroErrType::ALTRO_PAYLOAD_ERROR:
      return "Payload could not be decoded";
    case AltroErrType::ALTRO_MAPPING_ERROR:
      return "Invalid hardware address in ALTRO mapping";
    case AltroErrType::CHANNEL_ERROR:
      return "Channels not initizalized";
  };
  return "";
}

std::string MinorAltroDecodingError::what() const noexcept
{
  std::stringstream result;
  result << getErrorTypeDescription(mErrorType);
  auto address = mChannelHeader & 0xFFF,
       payload = (mChannelHeader >> 16) & 0x3FF;
  bool good = (mChannelHeader >> 29) & 0x1;

  result << " Channel header=0x" << std::hex << mChannelHeader
         << " (Address=0x" << address << ", payload " << std::dec << payload << ", good " << (good ? "yes" : "no") << ")"
         << ", word=0x" << std::hex << mPayloadWord << std::dec;
  return result.str();
}

using MinorAltroErrType = o2::emcal::MinorAltroDecodingError::ErrorType_t;

int MinorAltroDecodingError::errorTypeToInt(MinorAltroErrType errortype)
{

  int errorNumber = -1;

  switch (errortype) {
    case MinorAltroErrType::CHANNEL_END_PAYLOAD_UNEXPECT:
      errorNumber = 0;
      break;
    case MinorAltroErrType::CHANNEL_PAYLOAD_EXCEED:
      errorNumber = 1;
      break;
    case MinorAltroErrType::BUNCH_HEADER_NULL:
      errorNumber = 2;
      break;
    case MinorAltroErrType::BUNCH_LENGTH_EXCEED:
      errorNumber = 3;
      break;
  };

  return errorNumber;
}

MinorAltroErrType MinorAltroDecodingError::intToErrorType(int errornumber)
{
  assert(errornumber < getNumberOfErrorTypes());
  MinorAltroErrType errorType;

  switch (errornumber) {
    case 0:
      errorType = MinorAltroErrType::CHANNEL_END_PAYLOAD_UNEXPECT;
      break;
    case 1:
      errorType = MinorAltroErrType::CHANNEL_PAYLOAD_EXCEED;
      break;
    case 2:
      errorType = MinorAltroErrType::BUNCH_HEADER_NULL;
      break;
    case 3:
      errorType = MinorAltroErrType::BUNCH_LENGTH_EXCEED;
      break;
    default:
      break;
  }

  return errorType;
}

const char* MinorAltroDecodingError::getErrorTypeName(ErrorType_t errortype)
{
  switch (errortype) {
    case MinorAltroErrType::CHANNEL_END_PAYLOAD_UNEXPECT:
      return "ChannelEndPayloadUnexpected";
    case MinorAltroErrType::CHANNEL_PAYLOAD_EXCEED:
      return "ChannelPayloadExceed";
    case MinorAltroErrType::BUNCH_HEADER_NULL:
      return "BunchHeaderNull";
    case MinorAltroErrType::BUNCH_LENGTH_EXCEED:
      return "BunchLengthExceed";
  };
  return "";
}

const char* MinorAltroDecodingError::getErrorTypeTitle(ErrorType_t errortype)
{
  switch (errortype) {
    case MinorAltroErrType::CHANNEL_END_PAYLOAD_UNEXPECT:
      return "Channel end unexpected";
    case MinorAltroErrType::CHANNEL_PAYLOAD_EXCEED:
      return "Channel exceed";
    case MinorAltroErrType::BUNCH_HEADER_NULL:
      return "Bunch header null";
    case MinorAltroErrType::BUNCH_LENGTH_EXCEED:
      return "Bunch length exceed";
  };
  return "";
}

const char* MinorAltroDecodingError::getErrorTypeDescription(ErrorType_t errortype)
{
  switch (errortype) {
    case MinorAltroErrType::CHANNEL_END_PAYLOAD_UNEXPECT:
      return "Unexpected end of payload in altro channel payload!";
    case MinorAltroErrType::CHANNEL_PAYLOAD_EXCEED:
      return "Trying to access out-of-bound payload!";
    case MinorAltroErrType::BUNCH_HEADER_NULL:
      return "Bunch header 0 or not configured!";
    case MinorAltroErrType::BUNCH_LENGTH_EXCEED:
      return "Bunch length exceeding channel payload size!";
  };
  return "";
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const AltroDecoderError& error)
{
  stream << error.what();
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const AltroDecoderError::ErrorType_t& errortype)
{
  stream << AltroDecoderError::getErrorTypeName(errortype);
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const MinorAltroDecodingError& error)
{
  stream << error.what();
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const MinorAltroDecodingError::ErrorType_t& errortype)
{
  stream << MinorAltroDecodingError::getErrorTypeName(errortype);
  return stream;
}
