// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   exceptions.h
/// @author Michael Lettrich
/// @brief  rans exceptions

#ifndef RANS_INTERNAL_COMMON_EXCEPTIONS_
#define RANS_INTERNAL_COMMON_EXCEPTIONS_

#include <string>
#include <stdexcept>

namespace o2::rans
{
class Exception : public std::exception
{
 public:
  Exception() = default;
  Exception(const Exception& other) = default;
  Exception(Exception&& other) = default;
  Exception& operator=(const Exception& other) = default;
  Exception& operator=(Exception&& other) = default;
  ~Exception() override = default;

  Exception(const std::string& msg) : mMsg{msg} {};
  Exception(const char* msg) : mMsg{msg} {};

  const char* what() const noexcept override
  {
    return mMsg.c_str();
  };

  std::string mMsg{};
};

class RuntimeError : public Exception
{
  using Exception::Exception;
};

class IOError : public Exception
{
  using Exception::Exception;
};

class ParsingError : public IOError
{
  using IOError::IOError;
};

class ValueError : public Exception
{
  using Exception::Exception;
};

class OutOfBoundsError : public Exception
{
  using Exception::Exception;
};

class OverflowError : public Exception
{
  using Exception::Exception;
};

class HistogramError : public Exception
{
  using Exception::Exception;
};

class CodingError : public Exception
{
  using Exception::Exception;
};

class EncodingError : public CodingError
{
  using CodingError::CodingError;
};

class DecodingError : public CodingError
{
  using CodingError::CodingError;
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_COMMON_EXCEPTIONS_ */