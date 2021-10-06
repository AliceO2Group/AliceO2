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

#ifndef ALICEO2_EMCAL_ERRORTYPEFEE_H
#define ALICEO2_EMCAL_ERRORTYPEFEE_H

#include <iostream>
#include "Rtypes.h"

namespace o2
{

namespace emcal
{

/// \class ErrorTypeFEE
/// \brief Errors per FEE information
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since March 04, 2021
/// \ingroup EMCALDataFormat
///
/// # Container for errors in the raw decoding
///
/// Various errors can happen during the raw decoding. In order not only to print
/// the errors in the output stream or infoLogger the ErrorTypeFEE can be used to
/// store the error type persistently or to send them to further components, i.e.
/// the QC for monitoring. Several error categories are supported:
///
/// - PAGE_ERROR: This type handles all errors related to raw page decoding (not ALTRO payload)
/// - ALTRO_ERROR: This type handles all errors related to decoding of the ALTRO payload
/// - MINOR_ALTRO_ERROR: This type handles all errors related to decoding of the ALTRO payload
///   which are not considered as fatal
/// - FIT_ERROR: This type handles all error appearing during the raw fitting procedure
/// - GEOMETRY_ERROR: This type handles all errors related to the calculation of the module position
///   using the geometry
///
/// In addition to the error type a numeric code to further distinguish the type of
/// the error within the category. A ErrorTypeFEE object can handle only one error,
/// in case multiple errors occur a separate object is needed for each error appearing.
class ErrorTypeFEE
{
 public:
  /// \enum ErrorSource_t
  /// \brief Source of the error
  enum ErrorSource_t {
    PAGE_ERROR,        ///< Raw page decoding failed
    ALTRO_ERROR,       ///< Decoding of the ALTRO payload failed
    MINOR_ALTRO_ERROR, ///< Non-fatal error in decoding of the ALTRO payload
    FIT_ERROR,         ///< Raw fit failed
    GEOMETRY_ERROR,    ///< Decoded position outside EMCAL
    GAIN_ERROR,        ///< Error due to gain type
    UNDEFINED          ///< Error source undefined
  };
  /// \brief Constructor
  ErrorTypeFEE() = default;

  /// \brief Constructor initializing the object
  /// \param FEEID ID of the FEE responsible for the error
  /// \param errortype Type of the error
  /// \param errorCode Error code for the given error type
  ErrorTypeFEE(int FEEID, ErrorSource_t errortype, int errorCode) : mFEEID(FEEID), mErrorSource(errortype), mErrorCode(errorCode) {}

  /// \brief Destructor
  ~ErrorTypeFEE() = default;

  /// \brief Set the ID of the FEE responsible for the error
  /// \param feeid ID of the FEE
  void setFEEID(int feeid) { mFEEID = feeid; }

  /// \brief Set the error as decoding error and store the error code
  /// \param decodeError Error code of the decoding error
  void setDecodeErrorType(int decodeError) { setError(ErrorSource_t::ALTRO_ERROR, decodeError); }

  /// \brief Set the error as minor (non-fatal) decoding error and store the error code
  /// \param decodeError Error code of the decoding error
  void setMinorDecodingErrorType(int decodeError) { setError(ErrorSource_t::MINOR_ALTRO_ERROR, decodeError); }

  /// \brief Set the error as raw fitter error and store the error code
  /// \param rawfitterError Error code of the raw fitter error
  void setRawFitErrorType(int rawfitterError) { setError(ErrorSource_t::FIT_ERROR, rawfitterError); }

  /// \brief Set the error as page parser error and store the error code
  /// \param pageError Error code of the page parser error
  void setPageErrorType(int pageError) { setError(ErrorSource_t::PAGE_ERROR, pageError); }

  /// \brief Set the error as gain type error and store the error code
  /// \param gainError Error code of the gain type error
  void setGainErrorType(int gainError) { setError(ErrorSource_t::GAIN_ERROR, gainError); }

  /// \brief Set the error type of the object
  /// \param errorsource Error type of the object
  void setErrorType(ErrorSource_t errorsource) { mErrorSource = errorsource; }

  /// \brief Set the error code of the object
  /// \param errorcode Error code of the object
  void setErrorCode(int errorcode) { mErrorCode = errorcode; }

  /// \brief Set source and code of the error
  /// \param errorsource Type of the error
  /// \param errorcode Code of the error
  void setError(ErrorSource_t errorsource, int errorcode)
  {
    setErrorType(errorsource);
    setErrorCode(errorcode);
  }

  /// \brief Get the FEE ID of the electronics responsible for the error
  /// \return ID of the FEE component
  int getFEEID() const { return mFEEID; }

  /// \brief Get the type of the error handled by the object
  /// \return Error type
  ErrorSource_t getErrorType() const { return mErrorSource; }

  /// \brief Get the error code of the object
  /// \return Error code
  int getErrorCode() const { return mErrorCode; }

  /// \brief Get the error code of the obect in case the object is a decoding error
  /// \return Error code (-1 in case the object is not a decoding error)
  int getDecodeErrorType() const { return getRawErrorForType(ErrorSource_t::ALTRO_ERROR); }

  /// \brief Get the error code of the obect in case the object is a decoding error
  /// \return Error code (-1 in case the object is not a decoding error)
  int getMinorDecodeErrorType() const { return getRawErrorForType(ErrorSource_t::MINOR_ALTRO_ERROR); }

  /// \brief Get the error code of the obect in case the object is a raw fitter error
  /// \return Error code (-1 in case the object is not a raw fitter error)
  int getRawFitErrorType() const { return getRawErrorForType(ErrorSource_t::FIT_ERROR); }

  /// \brief Get the error code of the obect in case the object is a page parsing error
  /// \return Error code (-1 in case the object is not a page parsing error)
  int getRawPageErrorType() const { return getRawErrorForType(ErrorSource_t::PAGE_ERROR); }

  /// \brief Get the error code of the obect in case the object is a gain type error
  /// \return Error code (-1 in case the object is not a gain type error)
  int getGainTypeErrorType() const { return getRawErrorForType(ErrorSource_t::GAIN_ERROR); }

  /// \brief Printing information of the error type
  /// \param stream Output stream where to print the error
  ///
  /// Helper function, called in the output stream operator for the ErrorTypeFEE
  void PrintStream(std::ostream& stream) const;

 private:
  /// \brief Helper function getting the error code under condition that the error is of a certain type
  /// \return Error code (-1 in case the error handle by the object is not of the given type)
  int getRawErrorForType(ErrorSource_t source) const { return mErrorSource == source ? mErrorCode : -1; }

  int mFEEID = -1;                                       ///< FEE ID of the SM responsible for the error
  ErrorSource_t mErrorSource = ErrorSource_t::UNDEFINED; ///< Source of the error
  int mErrorCode = -1;                                   ///< Raw page error type

  ClassDefNV(ErrorTypeFEE, 1);
};

/// \brief Stream operator for FEE and it errors
/// \param stream Stream where to print the fee and its errors
/// \param errorType error type to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const ErrorTypeFEE& errorType);

} // namespace emcal

} // namespace o2

#endif
