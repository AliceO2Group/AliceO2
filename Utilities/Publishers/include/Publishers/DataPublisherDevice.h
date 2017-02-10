//-*- Mode: C++ -*-

#ifndef DATAPUBLISHERDEVICE_H
#define DATAPUBLISHERDEVICE_H

/// @file   DataPublisherDevice.h
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-10
/// @brief  Utility device for data publishing

#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "O2Device/O2Device.h"
#include <cstring>

class FairMQParts;

namespace AliceO2 {
namespace Utilities {

/// @class DataPublisherDevice
/// Utility device for data publishing
///
class DataPublisherDevice : public Base::O2Device
{
public:
  typedef AliceO2::Base::O2Message O2Message;
  /// TODO: use type alias when it has been added to DataHeader.h
  typedef uint64_t SubSpecificationT;

  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyDataDescription = "data-description";
  static constexpr const char* OptionKeyDataOrigin = "data-origin";
  static constexpr const char* OptionKeySubspecification = "sub-specification";

  /// Default constructor
  DataPublisherDevice();

  /// Default destructor
  virtual ~DataPublisherDevice() final;

protected:
  /// overloading the InitTask() method of FairMQDevice
  void InitTask() final;

  /// data handling method to be registered as handler in the
  /// FairMQDevice API method OnData
  /// The device base class handles the state loop in the RUNNING
  /// state and calls the handler when receiving a message on one channel
  /// The multiple parts included in one message are provided in the
  /// FairMQParts object.
  bool HandleData(FairMQParts& msgParts, int index);

private:
  /// configurable name of input channel
  std::string mInputChannelName;
  /// configurable name of output channel
  std::string mOutputChannelName;
  /// index of the previously handled data channel in HandleData
  int mLastIndex;
  /// the default data description
  AliceO2::Header::DataDescription mDataDescription;
  /// the default data description
  AliceO2::Header::DataOrigin mDataOrigin;
  /// the default data sub specification
  SubSpecificationT mSubSpecification;
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
