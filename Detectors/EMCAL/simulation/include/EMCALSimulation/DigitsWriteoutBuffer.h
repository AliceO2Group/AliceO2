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

#ifndef ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_
#define ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <gsl/span>
#include "DataFormatsEMCAL/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsVectorStream.h"

namespace o2
{
namespace emcal
{

/// \class DigitsWriteoutBuffer
/// \brief Container class for time sampled digits
/// \ingroup EMCALsimulation
/// \author Hadi Hassan, ORNL
/// \author Markus Fasel, ORNL
/// \date 08/03/2021

class DigitsWriteoutBuffer
{
 public:
  /// Default constructor
  DigitsWriteoutBuffer(unsigned int nTimeBins = 15);

  /// Destructor
  ~DigitsWriteoutBuffer() = default;

  /// clear the container
  void clear();

  /// clear DigitsVectorStream
  void flush() { mDigitStream.clear(); }

  void init();

  /// Reserve space for the future container
  void reserve();

  /// This is for the readout window that was interrupted by the end of the run
  void finish();

  double getTriggerTime() const { return mTriggerTime; }
  double getEventTime() const { return mLastEventTime; }
  bool isLive(double t) const { return ((t - mTriggerTime) < mLiveTime || (t - mTriggerTime) >= (mLiveTime + mBusyTime - mPreTriggerTime)); }
  bool isLive() const { return ((mLastEventTime - mTriggerTime) < mLiveTime || (mLastEventTime - mTriggerTime) >= (mLiveTime + mBusyTime - mPreTriggerTime)); }

  // function returns true if the collision occurs 600ns before the readout window is open
  // Look here for more details https://alice.its.cern.ch/jira/browse/EMCAL-681
  bool preTriggerCollision() const { return ((mLastEventTime - mTriggerTime) >= (mLiveTime + mBusyTime - mPreTriggerTime)); }

  /// Add digit to the container
  /// \param towerID Cell ID
  /// \param dig Labaled digit to add
  /// \param eventTime The time of the event (w.r.t Tigger time)
  void addDigits(unsigned int towerID, std::vector<LabeledDigit>& digList);

  /// This function sets the right time for all digits in the buffer
  void setSampledDigitsTime();

  /// Setting the buffer size
  void setBufferSize(unsigned int nsamples) { mBufferSize = nsamples; }
  unsigned int getBufferSize() const { return mBufferSize; }

  /// forward the marker for every 100 ns
  void forwardMarker(o2::InteractionTimeRecord record, bool trigger);

  /// Setters for the live time, busy time, pre-trigger time
  void setLiveTime(unsigned int liveTime) { mLiveTime = liveTime; }
  void setBusyTime(unsigned int busyTime) { mBusyTime = busyTime; }
  void setPreTriggerTime(unsigned int pretriggerTime) { mPreTriggerTime = pretriggerTime; }

  unsigned int getPhase() const { return mPhase; }

  const std::vector<o2::emcal::Digit>& getDigits() const { return mDigitStream.getDigits(); }
  const std::vector<o2::emcal::TriggerRecord>& getTriggerRecords() const { return mDigitStream.getTriggerRecords(); }
  const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& getMCLabels() const { return mDigitStream.getMCLabels(); }

 private:
  unsigned int mBufferSize = 15;                          ///< The size of the buffer
  unsigned int mLiveTime = 1500;                          ///< EMCal live time (ns)
  unsigned int mBusyTime = 35000;                         ///< EMCal busy time (ns)
  unsigned int mPreTriggerTime = 600;                     ///< EMCal pre-trigger time (ns)
  unsigned long mTriggerTime = 0;                         ///< Time of the collision that fired the trigger (ns)
  unsigned long mLastEventTime = 0;                       ///< The event time of last collisions in the readout window
  unsigned int mPhase = 0;                                ///< The event L1 phase
  unsigned int mSwapPhase = 0;                            ///< BC phase swap
  bool mFirstEvent = true;                                ///< Flag to the first event in the run
  std::deque<o2::emcal::DigitTimebin> mTimedDigitsFuture; ///< Container for time sampled digits per tower ID for future digits
  std::deque<o2::emcal::DigitTimebin> mTimedDigitsPast;   ///< Container for time sampled digits per tower ID for past digits

  o2::emcal::DigitsVectorStream mDigitStream; ///< Output vector streamer

  ClassDefNV(DigitsWriteoutBuffer, 1);
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITSWRITEOUTBUFFER_H_ */
