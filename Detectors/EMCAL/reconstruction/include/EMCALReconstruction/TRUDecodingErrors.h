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

#ifndef ALICEO2_EMCAL_TRUDECODINGERRORS_H
#define ALICEO2_EMCAL_TRUDECODINGERRORS_H

#include <exception>
#include <string>

namespace o2
{

namespace emcal
{

/// \class FastOrStartTimeInvalidException
/// \brief Handling of error if starttime is to large (>=14). This is most likely caused by a corrupted channel header where a FEC channel is identified as a TRU channel
/// \ingroup EMCALbase
class FastOrStartTimeInvalidException : public std::exception
{
 public:
  /// \brief Constructor
  /// \param l0size Size of the L0 patch
  FastOrStartTimeInvalidException(int time) : std::exception(), mErrorMessage(), mStartTime(time)
  {
    mErrorMessage = "FastOr starttime invalid: " + std::to_string(time);
  }

  /// \brief Destructor
  ~FastOrStartTimeInvalidException() noexcept final = default;

  /// \brief Access to error message
  /// \return Error message
  const char* what() const noexcept final
  {
    return mErrorMessage.data();
  }

  /// \brief Get the size of the L0 patch
  /// \return Size of the L0 patch
  int getStartTime() const noexcept { return mStartTime; }

 private:
  std::string mErrorMessage; ///< Buffer for error message
  int mStartTime;            ///< Size of the L0 patch
};

} // namespace emcal

} // namespace o2

#endif //  ALICEO2_EMCAL_TRUDECODINGERRORS_H