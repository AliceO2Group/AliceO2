// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
class RawDecodingError : public std::exception
{
 public:
  /// \enum ErrorType_t
  /// \brief Codes for different error types
  enum class ErrorType_t {
    PAGE_NOTFOUND,    ///< Page was not found (page index outside range)
    HEADER_DECODING,  ///< Header cannot be decoded (format incorrect)
    PAYLOAD_DECODING, ///< Payload cannot be decoded (format incorrect)
    HEADER_INVALID,   ///< Header in memory not belonging to requested superpage
    PAYLOAD_INVALID,  ///< Payload in memory not belonging to requested superpage
  };

  /// \brief Constructor
  /// \param errtype Identifier code of the error type
  ///
  /// Constructing the error with error code. To be called when the
  /// exception is thrown.
  RawDecodingError(ErrorType_t errtype) : mErrorType(errtype)
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
      case ErrorType_t::PAYLOAD_INVALID:
        return "Access to payload not belonging to requested superpage";
    };
    return "Undefined error";
  }

  /// \brief Get the type identifier of the error handled with this exception
  /// \return Error code of the exception
  ErrorType_t getErrorType() const { return mErrorType; }

 private:
  ErrorType_t mErrorType; ///< Type of the error
};

} // namespace emcal

} // namespace o2

#endif