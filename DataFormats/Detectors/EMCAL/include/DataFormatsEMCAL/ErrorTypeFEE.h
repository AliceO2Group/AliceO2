// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
/// Class containing decoding and raw fitter errors per FEE
///

class ErrorTypeFEE
{
 public:
  /// \brief Constructor
  ErrorTypeFEE() = default;

  ErrorTypeFEE(int FEEID, int decoderError, int rawFitError) : mFEEID(FEEID), mDecodeError(decoderError), mRawFitterError(rawFitError) {}

  /// \brief
  ~ErrorTypeFEE() = default;

  void setFEEID(int feeid) { mFEEID = feeid; }
  void setDecodeErrorType(int decodeError) { mDecodeError = decodeError; }
  void setRawFitErrorType(int rawfitterError) { mRawFitterError = rawfitterError; }

  int getFEEID() const { return mFEEID; }
  int getDecodeErrorType() const { return mDecodeError; }
  int getRawFitErrorType() const { return mRawFitterError; }

  void PrintStream(std::ostream& stream) const;

 private:
  int mFEEID = -1;          ///< FEE ID of the SM responsible for the error
  int mDecodeError = -1;    ///< Decoding error type
  int mRawFitterError = -1; ///< RawFitter error type

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
