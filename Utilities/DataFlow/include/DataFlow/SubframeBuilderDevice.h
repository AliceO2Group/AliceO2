// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-

#ifndef SUBFRAMEBUILDERDEVICE_H
#define SUBFRAMEBUILDERDEVICE_H

/// @file   SubframeBuilderDevice.h
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Demonstrator device for a subframe builder

#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "O2Device/O2Device.h"
#include "DataFlow/PayloadMerger.h"
#include "DataFlow/SubframeUtils.h"
#include <cstring>
#include <map>

class FairMQParts;

namespace o2 {
namespace DataFlow {

/// @class SubframeBuilderDevice
/// A demonstrator device for building of sub timeframes
///
/// The scheme in the demonstrator chain assumes multiple data publishers
/// for different input chains of the CRU. The output of these publishers
/// is assembled to a sub timeframe by this device.
///
/// The device implements the high-level API of the FairMQDevice
/// run loop. The registered method is called once per message
/// which is in itself a multipart message. It;s parts are passed
/// with a FairMQParts object. Not yet clear if this approach is
/// suited for the frame builder as it needs the data from multiple
/// channels with the same channel name. Depends on the validity of
/// the message data. Very likely the message parts are no longer
/// valid after leaving the handler method. But one could apply a
/// scheme with unique pointers and move semantics
///
/// The duration should be with respect to a time constant, which in
/// itself needs to be configurable, now the time constant is
/// hard-coded microseconds
class SubframeBuilderDevice : public Base::O2Device
{
public:
  using O2Message = o2::Base::O2Message;
  using SubframeId = o2::dataflow::SubframeId;
  using Merger = dataflow::PayloadMerger<SubframeId>;

  static constexpr const char* OptionKeyInputChannelName = "in-chan-name";
  static constexpr const char* OptionKeyOutputChannelName = "out-chan-name";
  static constexpr const char* OptionKeyOrbitDuration = "orbit-duration";
  static constexpr const char* OptionKeyOrbitsPerTimeframe = "orbits-per-timeframe";
  static constexpr const char* OptionKeyInDataFile = "indatafile-name";
  static constexpr const char* OptionKeyDetector = "detector-name";
  static constexpr const char* OptionKeyFLPId = "flp-id";
  static constexpr const char* OptionKeyStripHBF = "strip-hbf";

  // TODO: this is just a first mockup, remove it
  // Default start time for all the producers is 8/4/1977
  // Timeframe start time will be ((N * duration) + start time) where
  // N is the incremental number of timeframes being sent out.
  // TODO: replace this with a unique Heartbeat from a common device.
  static constexpr uint32_t DefaultOrbitDuration = 88924;
  static constexpr uint32_t DefaultOrbitsPerTimeframe = 256;
  static constexpr uint64_t DefaultHeartbeatStart = 229314600000000000LL;

  /// Default constructor
  SubframeBuilderDevice();

  /// Default destructor
  ~SubframeBuilderDevice() final;

protected:
  /// overloading the InitTask() method of FairMQDevice
  void InitTask() final;

  /// data handling method to be registered as handler in the
  /// FairMQDevice API method OnData
  /// The device base class handles the state loop in the RUNNING
  /// state and calls the handler when receiving a message on one channel
  /// The multiple parts included in one message are provided in the
  /// FairMQParts object.
  bool HandleData(FairMQParts& msgParts, int /*index*/);

  /// Build the frame and send it
  /// For the moment a simple mockup composing a DataHeader and adding it
  /// to the multipart message together with the SubframeMetadata as payload
  bool BuildAndSendFrame(FairMQParts &parts);

private:
  uint32_t mOrbitsPerTimeframe;
  // FIXME: lookup the actual value
  uint32_t mOrbitDuration;
  std::string mInputChannelName = "";
  std::string mOutputChannelName = "";
  size_t mFLPId = 0;
  bool mStripHBF = false;
  std::unique_ptr<Merger> mMerger;

  uint64_t mHeartbeatStart = DefaultHeartbeatStart;

  template <typename T>
  size_t fakeHBHPayloadHBT(char **buffer, std::function<void(T&,int)> filler, int numOfElements) {
    // LOG(INFO) << "SENDING TPC PAYLOAD\n";
    auto payloadSize = sizeof(header::HeartbeatHeader)+sizeof(T)*numOfElements+sizeof(header::HeartbeatTrailer);
    *buffer = new char[payloadSize];
    auto *hbh = reinterpret_cast<header::HeartbeatHeader*>(*buffer);
    assert(payloadSize > 0);
    assert(payloadSize - sizeof(header::HeartbeatTrailer) > 0);
    auto *hbt = reinterpret_cast<header::HeartbeatTrailer*>(payloadSize - sizeof(header::HeartbeatTrailer));

    T *payload = reinterpret_cast<T*>(*buffer + sizeof(header::HeartbeatHeader));
    for (int i = 0; i < numOfElements; ++i) {
      new (payload + i) T;
      // put some random toy time stamp to each cluster
      filler(payload[i], i);
    }
    return payloadSize;
  }
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
