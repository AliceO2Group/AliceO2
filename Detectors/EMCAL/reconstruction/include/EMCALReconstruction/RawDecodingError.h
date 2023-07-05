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

#include <array>
#include <cassert>
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
    return getErrorCodeDescription(mErrorType);
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

  /// \brief Get the number of error codes
  /// \return Number of error codes
  static constexpr int getNumberOfErrorTypes() { return 7; }

  static ErrorType_t intToErrorType(unsigned int errortype)
  {
    assert(errortype < getNumberOfErrorTypes());
    static constexpr std::array<ErrorType_t, getNumberOfErrorTypes()> errortypes = {{ErrorType_t::PAGE_NOTFOUND, ErrorType_t::HEADER_DECODING,
                                                                                     ErrorType_t::PAYLOAD_DECODING, ErrorType_t::HEADER_INVALID,
                                                                                     ErrorType_t::PAGE_START_INVALID, ErrorType_t::PAYLOAD_INVALID,
                                                                                     ErrorType_t::TRAILER_DECODING}};
    return errortypes[errortype];
  }

  /// \brief Get name of error type
  ///
  /// A single word descriptor i.e. to be used in object names
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (symbolic representation)
  /// \return Name of the error type
  static const char* getErrorCodeNames(ErrorType_t errortype)
  {
    switch (errortype) {
      case ErrorType_t::PAGE_NOTFOUND:
        return "PageNotFound";
      case ErrorType_t::HEADER_DECODING:
        return "HeaderDecoding";
      case ErrorType_t::PAYLOAD_DECODING:
        return "PayloadDecoding";
      case ErrorType_t::HEADER_INVALID:
        return "HeaderCorruption";
      case ErrorType_t::PAGE_START_INVALID:
        return "PageStartInvalid";
      case ErrorType_t::PAYLOAD_INVALID:
        return "PayloadCorruption";
      case ErrorType_t::TRAILER_DECODING:
        return "TrailerDecoding";
    };
    return "Undefined error";
  }

  /// \brief Get name of error type
  ///
  /// A single word descriptor i.e. to be used in object names
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Name of the error type
  static const char* getErrorCodeNames(unsigned int errortype)
  {
    return getErrorCodeNames(intToErrorType(errortype));
  }

  /// \brief Get title of error type
  ///
  /// A short description i.e. to be used in histogram titles
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (symbolic representation)
  /// \return Title of the error type
  static const char* getErrorCodeTitles(ErrorType_t errortype)
  {
    switch (errortype) {
      case ErrorType_t::PAGE_NOTFOUND:
        return "Page not found";
      case ErrorType_t::HEADER_DECODING:
        return "Header decoding";
      case ErrorType_t::PAYLOAD_DECODING:
        return "Payload decoding";
      case ErrorType_t::HEADER_INVALID:
        return "Header corruption";
      case ErrorType_t::PAGE_START_INVALID:
        return "Page start invalid";
      case ErrorType_t::PAYLOAD_INVALID:
        return "Payload corruption";
      case ErrorType_t::TRAILER_DECODING:
        return "Trailer decoding";
    };
    return "Undefined error";
  }

  /// \brief Get title of error type
  ///
  /// A short description i.e. to be used in histogram titles
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Title of the error type
  static const char* getErrorCodeTitles(unsigned int errortype)
  {
    return getErrorCodeTitles(intToErrorType(errortype));
  }

  /// \brief Get description of error type
  ///
  /// A dedicated description is created which can be used i.e. in the
  /// what() function of the exception.
  ///
  /// \param errortype Error type raising the exceptio (symbolic representation)
  /// \return Description text for the error type
  static const char* getErrorCodeDescription(ErrorType_t errortype)
  {
    switch (errortype) {
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

  /// \brief Get description of error type
  ///
  /// A dedicated description is created which can be used i.e. in the
  /// what() function of the exception.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Description text for the error type
  static const char* getErrorCodeDescription(unsigned int errortype)
  {
    return getErrorCodeDescription(intToErrorType(errortype));
  }

 private:
  ErrorType_t mErrorType; ///< Type of the error
  int mFecID;             ///< ID of the FEC responsible for the ERROR
};

/// \brief Streaming operator for RawDecodingError
/// \param stream Stream to print on
/// \param error Error to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const RawDecodingError& error);

/// \brief Streaming operator for RawDecodingError's ErrorType_t
/// \param stream Stream to print on
/// \param error Error to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const RawDecodingError::ErrorType_t& error);

} // namespace emcal

} // namespace o2

#endif