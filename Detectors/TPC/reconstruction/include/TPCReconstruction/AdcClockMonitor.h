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

/// \file AdcClockMonitor.h
/// \brief Class to monitor the ADC smapling clock contained in the GBT frame
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_ADCCLOCKMONITOR_H_
#define ALICEO2_TPC_ADCCLOCKMONITOR_H_

#include <fairlogger/Logger.h>
#include <iosfwd>
#include <iomanip>

namespace o2
{
namespace tpc
{

/// \class AdcClockMonitor
/// \brief Class to monitor the ADC smapling clock contained in the GBT frame

class AdcClockMonitor
{
 public:
  /// Default Constructor
  AdcClockMonitor();

  /// Constructor
  AdcClockMonitor(int sampa);

  /// Copy Constructor
  AdcClockMonitor(const AdcClockMonitor& other);

  /// Destructor
  ~AdcClockMonitor();

  /// Reset function to clear history
  void reset();

  /// Checks whether new sequence is valid
  /// @param seq New sequence of 4 bits
  /// @return Returns outcome of the check, 1 for an error, 0 otherwise
  int addSequence(const short seq);

  /// Get the state of the ADC monitor
  /// @return Returns outcome of monitor, 1 for an error, 0 otherwise
  int getState() { return ((mState == state::error) || (mSequenceCompleted == false)) ? 1 : 0; };

 private:
  enum state { locked = 0,
               error = 1 };

  int mSampa;              ///< Store SAMPA ID
  short mPrevSequence;     ///< Store previous 4 bits
  short mTransition0;      ///< 1st transition to look for
  short mTransition1;      ///< 2nd transition to look for
  bool mSequenceCompleted; ///< Store whether sequence was completed at least once
  int mSequencePosition;   ///< current position in sequence
  state mState;            ///< current state
};
} // namespace tpc
} // namespace o2
#endif
