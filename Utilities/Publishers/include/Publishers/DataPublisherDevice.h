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
#include <vector>

class FairMQParts;

namespace o2 {
namespace utilities {

/// @class DataPublisherDevice
/// Utility device for data publishing
///
/// TODO: Generalize with an input policy
class DataPublisherDevice : public Base::O2Device
{
public:
  typedef o2::Base::O2Message O2Message;
  /// TODO: use type alias when it has been added to DataHeader.h
  typedef uint64_t SubSpecificationT;

  static constexpr const char* OptionKeyInputChannelName = "in-chan-name";
  static constexpr const char* OptionKeyOutputChannelName = "out-chan-name";
  static constexpr const char* OptionKeyDataDescription = "data-description";
  static constexpr const char* OptionKeyDataOrigin = "data-origin";
  static constexpr const char* OptionKeySubspecification = "sub-specification";
  static constexpr const char* OptionKeyFileName = "filename";

  /// Default constructor
  DataPublisherDevice();

  /// Default destructor
  ~DataPublisherDevice() final;

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

  /// handle one logical O2 block of the input, consists of header and payload
  bool HandleO2LogicalBlock(const byte* headerBuffer, size_t headerBufferSize,
			    const byte* dataBuffer, size_t dataBufferSize);

  /// Read file and append to the buffer
  static bool AppendFile(const char* name, std::vector<byte>& buffer);

private:
  /// configurable name of input channel
  std::string mInputChannelName;
  /// configurable name of output channel
  std::string mOutputChannelName;
  /// index of the previously handled data channel in HandleData
  int mLastIndex;
  /// the default data description
  o2::header::DataDescription mDataDescription;
  /// the default data description
  o2::header::DataOrigin mDataOrigin;
  /// the default data sub specification
  SubSpecificationT mSubSpecification;
  /// buffer for the file to read
  /// Note the shift by sizeof(HeartbeatHeader)
  std::vector<byte> mFileBuffer;
  std::string mFileName;
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
