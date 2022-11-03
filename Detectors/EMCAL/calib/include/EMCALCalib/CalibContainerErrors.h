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
#ifndef ALICEO2_EMCAL_CALIBCONTAINERERORS_H
#define ALICEO2_EMCAL_CALIBCONTAINERERORS_H

#include <iosfwd>
#include <exception>
#include <string>

namespace o2
{

namespace emcal
{

/// \class CalibContainerIndexException
/// \brief Error handling for invalid index in calibration request
/// \ingroup EMCALCalib
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Sept 15, 2022
class CalibContainerIndexException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param index Index in container raising the exception
  CalibContainerIndexException(unsigned int index);

  /// \brief Destructor
  ~CalibContainerIndexException() noexcept final = default;

  /// \brief Access to error message of the exception
  /// \return Error message
  const char* what() const noexcept final { return mErrorMessage.data(); }

  /// \brief Access to index raising the exception
  /// \return Index raising the exception
  unsigned int getIndex() const noexcept { return mIndex; }

 private:
  unsigned int mIndex;       ///< Index raising the error message
  std::string mErrorMessage; ///< Buffer for error message
};

/// \brief Output stream operator for CalibContainerIndexException
/// \param stream Stream where the error message should be displayed
/// \param obj Exception object to be streamed
/// \return Stream after the message
std::ostream& operator<<(std::ostream& stream, const CalibContainerIndexException& obj);

} // namespace emcal

} // namespace o2

#endif // !ALICEO2_EMCAL_CALIBCONTAINERERORS_H