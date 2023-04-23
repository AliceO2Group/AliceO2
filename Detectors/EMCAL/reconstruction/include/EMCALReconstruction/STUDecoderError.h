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
#ifndef ALICEO2_EMCAL_STUDECODERERROR_H
#define ALICEO2_EMCAL_STUDECODERERROR_H

#include <exception>
#include <iosfwd>
#include <string>

namespace o2::emcal
{

class STUDecoderError : public std::exception
{
 public:
  enum class ErrorCode_t {
    PAGE_ERROR,
    WORD_UNEXPECTED,
    INDEX_UNEXPECTED,
    ADC_OVERFLOW
  };
  STUDecoderError() = default;
  ~STUDecoderError() noexcept final = default;

  const char* what() const noexcept final { return mMessage.data(); }

  int getDDLID() const noexcept { return mDDLId; }

  void printStream(std::ostream& stream) const;

 private:
  int mDDLId;             ///< ID of the DDL raising the exception
  ErrorCode_t mErrorCode; ///< Error code of the exception
  std::string mMessage;   ///< Message buffer
};

std::ostream& operator<<(std::ostream& stream, const STUDecoderError& error);

} // namespace o2::emcal

#endif