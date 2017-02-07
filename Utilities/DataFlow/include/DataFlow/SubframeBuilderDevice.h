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

namespace AliceO2 {
namespace DataFlow {

// TODO: this definition has to go to some common place
// maybe also an identifier for the time constant should be added
struct SubframeMetadata
{
  // TODO: replace with timestamp struct
  uint64_t startTime = ~(uint64_t)0;
  uint64_t duration = ~(uint64_t)0;

  //further meta data to be added
};

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
  typedef AliceO2::Base::O2Message O2Message;

  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyDuration = "duration";
  static constexpr const char* OptionKeySelfTriggered = "self-triggered";
  // TODO: this is just a first mockup, remove it
  static constexpr uint32_t DefaultDuration = 10000;

  /// Default constructor
  SubframeBuilderDevice();

  /// Default destructor
  virtual ~SubframeBuilderDevice() final;

protected:
  /// overloading the InitTask() method of FairMQDevice
  void InitTask() final;

   /// overloading ConditionalRun method of FairMQDevice
  virtual bool ConditionalRun() final;

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
  bool BuildAndSendFrame();

private:
  unsigned mFrameNumber;
  uint32_t mDuration;
  std::string mInputChannelName;
  std::string mOutputChannelName;
  bool mIsSelfTriggered;
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
