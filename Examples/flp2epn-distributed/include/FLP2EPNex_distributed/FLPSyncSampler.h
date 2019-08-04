// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * FLPSyncSampler.h
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#ifndef ALICEO2_DEVICES_FLPSYNCSAMPLER_H_
#define ALICEO2_DEVICES_FLPSYNCSAMPLER_H_

#include <string>
#include <cstdint> // UINT64_MAX

#include <thread>
#include <atomic>
#include <chrono>

#include <FairMQDevice.h>

namespace o2
{
namespace devices
{

/// Stores measurment for roundtrip time of a timeframe

struct timeframeDuration {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
};

/// Publishes timeframes IDs for flpSenders (used only in test mode)

class FLPSyncSampler : public FairMQDevice
{
 public:
  /// Default constructor
  FLPSyncSampler();

  /// Default destructor
  ~FLPSyncSampler() override;

  /// Controls the send rate of the timeframe IDs
  void ResetEventCounter();

  /// Listens for acknowledgements from the epnReceivers when they collected full timeframe
  void ListenForAcks();

 protected:
  /// Overloads the InitTask() method of FairMQDevice
  void InitTask() override;

  /// Overloads the Run() method of FairMQDevice
  bool ConditionalRun() override;
  void PreRun() override;
  void PostRun() override;

  std::array<timeframeDuration, UINT16_MAX> mTimeframeRTT; ///< Container for the roundtrip values per timeframe ID
  int mEventRate;                                          ///< Publishing rate of the timeframe IDs
  int mMaxEvents;                                          ///< Maximum number of events to send (0 - unlimited)
  int mStoreRTTinFile;                                     ///< Store round trip time measurements in a file.
  int mEventCounter;                                       ///< Controls the send rate of the timeframe IDs
  uint16_t mTimeFrameId;
  std::thread mAckListener;
  std::thread mResetEventCounter;
  std::atomic<bool> mLeaving;

  std::string mAckChannelName;
  std::string mOutChannelName;
};

} // namespace devices
} // namespace o2

#endif
