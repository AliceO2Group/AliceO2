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
#include <cstring>

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
  typedef o2::Base::O2Message O2Message;

  static constexpr const char* OptionKeyInputChannelName = "in-chan-name";
  static constexpr const char* OptionKeyOutputChannelName = "out-chan-name";
  static constexpr const char* OptionKeyDuration = "duration";
  static constexpr const char* OptionKeySelfTriggered = "self-triggered";
  static constexpr const char* OptionKeyInDataFile = "indatafile-name";
  static constexpr const char* OptionKeyDetector = "detector-name";

  // TODO: this is just a first mockup, remove it
  // Default duration is for now harcoded to 22 milliseconds.
  // Default start time for all the producers is 8/4/1977
  // Timeframe start time will be ((N * duration) + start time) where
  // N is the incremental number of timeframes being sent out.
  // TODO: replace this with a unique Heartbeat from a common device.
  static constexpr uint32_t DefaultDuration = 22000000;
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
  unsigned mFrameNumber = 0;
  constexpr static uint32_t mOrbitsPerTimeframe = 1;
  constexpr static uint32_t mOrbitDuration = 1000000000;
  constexpr static uint32_t mDuration = mOrbitsPerTimeframe * mOrbitDuration;
  std::string mInputChannelName = "";
  std::string mOutputChannelName = "";
  bool mIsSelfTriggered = false;
  uint64_t mHeartbeatStart = DefaultHeartbeatStart;

  template <typename T>
  size_t fakeHBHPayloadHBT(char **buffer, std::function<void(T&,int)> filler, int numOfElements) {
    // LOG(INFO) << "SENDING TPC PAYLOAD\n";
    auto payloadSize = sizeof(Header::HeartbeatHeader)+sizeof(T)*numOfElements+sizeof(Header::HeartbeatTrailer);
    *buffer = new char[payloadSize];
    auto *hbh = reinterpret_cast<Header::HeartbeatHeader*>(*buffer);
    assert(payloadSize > 0);
    assert(payloadSize - sizeof(Header::HeartbeatTrailer) > 0);
    auto *hbt = reinterpret_cast<Header::HeartbeatTrailer*>(payloadSize - sizeof(Header::HeartbeatTrailer));

    T *payload = reinterpret_cast<T*>(*buffer + sizeof(Header::HeartbeatHeader));
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
