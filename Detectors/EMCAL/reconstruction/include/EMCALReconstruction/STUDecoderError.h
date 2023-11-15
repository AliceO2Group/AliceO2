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

/// \class STUDecoderError
/// \brief Handling of STU reconstruction errors
/// \ingroup EMCALreconstruction
/// \since April 25, 2023
///
/// In order to distinguish different error types the STUDecoder error
/// carries an error code which can be uniquely identified with a
/// condition raising the excpetion. Source is always the DDL of the
/// STU raising the exception.
class STUDecoderError final : public std::exception
{
 public:
  /// \enum ErrorCode_t
  /// \brief Error codes of STU decoding
  enum class ErrorCode_t {
    PAGE_ERROR,       ///< Page decoding failed (missing header)
    WORD_UNEXPECTED,  ///< Word unexpected
    INDEX_UNEXPECTED, ///< Patch index unexpected
    ADC_OVERFLOW,     ///< ADC overflow
    UNKNOWN           ///< Unknown error code (needed for conversion to int)
  };

  /// \brief Get integer representation of error code
  /// \param errorcode Error code
  /// \return Integer representation
  static int errorCodeToInt(ErrorCode_t errorcode);

  /// \brief Convert integer to error code
  /// \param errorcode Error code as integer
  /// \return Error code (symbolic) - UNKNOWN for invalid integer error codes
  static ErrorCode_t intToErrorCode(int errorcode);

  /// \brief Get the name of the error code
  /// \param errorcode Error code
  /// \return Name of the error code
  static std::string getErrorCodeName(ErrorCode_t errorcode);

  /// \brief Get the name of the error code
  /// \param errorcode Error code (integer representation)
  /// \return Name of the error code
  static std::string getErrorCodeName(int errorcode) { return getErrorCodeName(intToErrorCode(errorcode)); }

  /// \brief Get the title of the error code
  /// \param errorcode Error code
  /// \return Title of the error code
  static std::string getErrorCodeTitle(ErrorCode_t errorcode);

  /// \brief Get the title of the error code
  /// \param errorcode Error code (integer representation)
  /// \return Title of the error code
  static std::string getErrorCodeTitle(int errorcode) { return getErrorCodeTitle(intToErrorCode(errorcode)); }

  /// \brief Constructor
  /// \param ddlID ID of the DDL for which the exception is raised
  /// \param errcode Error code of the exception
  STUDecoderError(int ddlID, ErrorCode_t errcode);

  /// \brief Destructor
  ~STUDecoderError() noexcept final = default;

  /// \brief Access to error message
  /// \return Error message
  const char* what() const noexcept final { return mMessage.data(); }

  /// \brief Get the ID of the DDL in which the exception is raised
  /// \return ID of the DDL
  int getDDLID() const noexcept { return mDDLId; }

  /// \brief Get error code of the exception
  /// \return Error code
  ErrorCode_t getErrorCode() const noexcept { return mErrorCode; }

  /// \brief Print details of the error on the stream
  /// \param stream Stream to print on
  ///
  /// Helper function for streaming operator
  void printStream(std::ostream& stream) const;

 private:
  int mDDLId;             ///< ID of the DDL raising the exception
  ErrorCode_t mErrorCode; ///< Error code of the exception
  std::string mMessage;   ///< Message buffer
};

/// \brief Streaming operator of STU decoding errors
/// \param stream Stream to print the error on
/// \param error Error to be streamed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const STUDecoderError& error);

} // namespace o2::emcal

#endif