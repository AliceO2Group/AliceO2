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
#include <array>
#include <iostream>
#include "EMCALReconstruction/STUDecoderError.h"

using namespace o2::emcal;

STUDecoderError::STUDecoderError(int ddlID, ErrorCode_t errcode) : mDDLId(ddlID),
                                                                   mErrorCode(errcode)
{
  mMessage = "STU decoding error of type " + getErrorCodeTitle(mErrorCode) + " in DDL " + std::to_string(mDDLId);
}

void STUDecoderError::printStream(std::ostream& stream) const
{
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const STUDecoderError& error)
{
  error.printStream(stream);
  return stream;
}

int STUDecoderError::errorCodeToInt(ErrorCode_t errorcode)
{
  switch (errorcode) {
    case ErrorCode_t::PAGE_ERROR:
      return 0;
    case ErrorCode_t::WORD_UNEXPECTED:
      return 1;
    case ErrorCode_t::INDEX_UNEXPECTED:
      return 2;
    case ErrorCode_t::ADC_OVERFLOW:
      return 3;
    case ErrorCode_t::FEEID_UNEXPECTED:
      return 4;
    case ErrorCode_t::OLD_PAYLOAD_VERSION:
      return 5;
    case ErrorCode_t::FULL_PAYLOAD_SIZE_UNEXPECTED:
      return 6;
    case ErrorCode_t::SHORT_PAYLOAD_SIZE_UNEXPECTED:
      return 7;
    default:
      return -1;
  }
}
STUDecoderError::ErrorCode_t STUDecoderError::intToErrorCode(int errorcode)
{
  static constexpr std::size_t NUMERRORCODES = 8;
  static constexpr std::array<ErrorCode_t, NUMERRORCODES> errorcodes = {{ErrorCode_t::PAGE_ERROR, ErrorCode_t::WORD_UNEXPECTED, ErrorCode_t::INDEX_UNEXPECTED, ErrorCode_t::ADC_OVERFLOW, ErrorCode_t::FEEID_UNEXPECTED, ErrorCode_t::OLD_PAYLOAD_VERSION, ErrorCode_t::FULL_PAYLOAD_SIZE_UNEXPECTED, ErrorCode_t::SHORT_PAYLOAD_SIZE_UNEXPECTED}};
  if (errorcode < 0 || errorcode >= NUMERRORCODES) {
    return ErrorCode_t::UNKNOWN;
  }
  return errorcodes[errorcode];
}
std::string STUDecoderError::getErrorCodeName(ErrorCode_t errorcode)
{
  switch (errorcode) {
    case ErrorCode_t::PAGE_ERROR:
      return "PageDecoding";
    case ErrorCode_t::WORD_UNEXPECTED:
      return "WordUnexpected";
    case ErrorCode_t::INDEX_UNEXPECTED:
      return "IndexUnexpected";
    case ErrorCode_t::ADC_OVERFLOW:
      return "ADCOverflow";
    case ErrorCode_t::FEEID_UNEXPECTED:
      return "FeeIdUnexpected";
    case ErrorCode_t::OLD_PAYLOAD_VERSION:
      return "OldPayloadVersion";
    case ErrorCode_t::FULL_PAYLOAD_SIZE_UNEXPECTED:
      return "FullPayloadSizeUnexpected";
    case ErrorCode_t::SHORT_PAYLOAD_SIZE_UNEXPECTED:
      return "ShortPayloadSizeUnexpected";
    default:
      return "Unknown";
  }
}
std::string STUDecoderError::getErrorCodeTitle(ErrorCode_t errorcode)
{
  switch (errorcode) {
    case ErrorCode_t::PAGE_ERROR:
      return "page decoding";
    case ErrorCode_t::WORD_UNEXPECTED:
      return "unexpected word";
    case ErrorCode_t::INDEX_UNEXPECTED:
      return "invalid patch index";
    case ErrorCode_t::ADC_OVERFLOW:
      return "ADC overflow";
    case ErrorCode_t::FEEID_UNEXPECTED:
      return "invalid FeeID";
    case ErrorCode_t::OLD_PAYLOAD_VERSION:
      return "unsupported old payload version";
    case ErrorCode_t::FULL_PAYLOAD_SIZE_UNEXPECTED:
      return "unexpected full payload size";
    case ErrorCode_t::SHORT_PAYLOAD_SIZE_UNEXPECTED:
      return "unexpected short payload size";
    default:
      return "Unknown";
  }
}