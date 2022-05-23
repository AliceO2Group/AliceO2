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
#ifndef ALICEO2_EMCAL_RAWDECODINGERROR_H
#define ALICEO2_EMCAL_RAWDECODINGERROR_H

#include <exception>

namespace o2
{

namespace emcal
{

/// \class RawDecodingError
/// \brief Error handling of the raw reader
/// \ingroup EMCALreconstruction
///
/// The following error types are defined:
/// - Page not found
/// - Raw header decoding error
/// - Payload decoding error
/// - Out-of-bounds errors
/// In addition to the error type the FEE ID obtained from the
/// current raw header is propagated.
class RawDecodingError : public std::exception
{
 public:
  /// \enum ErrorType_t
  /// \brief Codes for different error types
  enum class ErrorType_t {
    PAGE_NOTFOUND,      ///< Page was not found (page index outside range)
    HEADER_DECODING,    ///< Header cannot be decoded (format incorrect)
    PAYLOAD_DECODING,   ///< Payload cannot be decoded (format incorrect)
    HEADER_INVALID,     ///< Header in memory not belonging to requested superpage
    PAGE_START_INVALID, ///< Page position starting outside payload size
    PAYLOAD_INVALID,    ///< Payload in memory not belonging to requested superpage
    TRAILER_DECODING    ///< Inconsistent trailer in memory (several trailer words missing the trailer marker)
  };

  /// \brief Constructor
  /// \param errtype Identifier code of the error type
  ///
  /// Constructing the error with error code. To be called when the
  /// exception is thrown.
  RawDecodingError(ErrorType_t errtype, int fecID) : mErrorType(errtype), mFecID(fecID)
  {
  }

  /// \brief destructor
  ~RawDecodingError() noexcept override = default;

  /// \brief Providing error message of the exception
  /// \return Error message of the exception
  const char* what() const noexcept override
  {
    switch (mErrorType) {
      case ErrorType_t::PAGE_NOTFOUND:
        return "Page with requested index not found";
      case ErrorType_t::HEADER_DECODING:
        return "RDH of page cannot be decoded";
      case ErrorType_t::PAYLOAD_DECODING:
        return "Payload of page cannot be decoded";
      case ErrorType_t::HEADER_INVALID:
        return "Access to header not belonging to requested superpage";
      case ErrorType_t::PAGE_START_INVALID:
        return "Page decoding starting outside payload size";
      case ErrorType_t::PAYLOAD_INVALID:
        return "Access to payload not belonging to requested superpage";
      case ErrorType_t::TRAILER_DECODING:
        return "Inconsistent trailer in memory";
    };
    return "Undefined error";
  }

  /// \brief Get the type identifier of the error handled with this exception
  /// \return Error code of the exception
  ErrorType_t getErrorType() const { return mErrorType; }

  /// \brief Get the ID of the frontend electronics responsible for the error
  /// \return ID of the frontend electronics
  int getFECID() const { return mFecID; }

  /// \brief Convert error type to error code number
  /// \return Numeric representation of the error type
  static int ErrorTypeToInt(RawDecodingError::ErrorType_t errortype)
  {
    switch (errortype) {
      case ErrorType_t::PAGE_NOTFOUND:
        return 0;
      case ErrorType_t::HEADER_DECODING:
        return 1;
      case ErrorType_t::PAYLOAD_DECODING:
        return 2;
      case ErrorType_t::HEADER_INVALID:
        return 3;
      case ErrorType_t::PAGE_START_INVALID:
        return 4;
      case ErrorType_t::PAYLOAD_INVALID:
        return 5;
      case ErrorType_t::TRAILER_DECODING:
        return 6;
    };
    // can never reach this, due to enum class
    // just to make Werror happy
    return -1;
  }

 private:
  ErrorType_t mErrorType; ///< Type of the error
  int mFecID;             ///< ID of the FEC responsible for the ERROR
};

} // namespace emcal

} // namespace o2

#endif