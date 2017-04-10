/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @headerfile O2Device.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2DEVICE_H_
#define O2DEVICE_H_

#include <FairMQDevice.h>
#include "Headers/DataHeader.h"
#include <stdexcept>

namespace o2 {
namespace Base {

/// just a typedef to express the fact that it is not just a FairMQParts vector,
/// it has to follow the O2 convention of header-payload-header-payload
using O2Message = FairMQParts;

class O2Device : public FairMQDevice
{
public:
  using FairMQDevice::FairMQDevice;
  ~O2Device() override = default;

  /// Here is how to add an annotated data part (with header);
  /// @param[in,out] parts is a reference to the message;
  /// @param[] incomingStack header block must be MOVED in (rvalue ref)
  /// @param[] dataMessage the data message must be MOVED in (unique_ptr by value)
  bool AddMessage(O2Message& parts,
                  o2::Header::Stack&& incomingStack,
                  FairMQMessagePtr incomingDataMessage) {

    //we have to move the incoming data
    o2::Header::Stack headerStack{std::move(incomingStack)};
    FairMQMessagePtr dataMessage{std::move(incomingDataMessage)};

    FairMQMessagePtr headerMessage = NewMessage(headerStack.buffer.get(),
                                                headerStack.bufferSize,
                                                &o2::Header::Stack::freefn,
                                                headerStack.buffer.get());
    headerStack.buffer.release();

    parts.AddPart(std::move(headerMessage));
    parts.AddPart(std::move(dataMessage));
    return true;
  }

  /// The user needs to define a member function with correct signature
  /// currently this is old school: buf,len pairs;
  /// In the end I'd like to move to array_view
  /// when this becomes available (either with C++17 or via GSL)
  template <typename T>
  bool ForEach(O2Message& parts, bool (T::*memberFunction)(const byte* headerBuffer, size_t headerBufferSize,
                                                           const byte* dataBuffer, size_t dataBufferSize))
  {
    if ((parts.Size() % 2) != 0)
      throw std::invalid_argument("number of parts in message not even (n%2 != 0)");

    for (auto it = parts.fParts.begin(); it != parts.fParts.end(); ++it) {
      byte* headerBuffer = nullptr;
      size_t headerBufferSize = 0;
      if (*it != nullptr) {
        headerBuffer = reinterpret_cast<byte*>((*it)->GetData());
        headerBufferSize = (*it)->GetSize();
      }
      ++it;
      byte* dataBuffer = nullptr;
      size_t dataBufferSize = 0;
      if (*it != nullptr) {
        dataBuffer = reinterpret_cast<byte*>((*it)->GetData());
        dataBufferSize = (*it)->GetSize();
      }

      // call the user provided function
      (static_cast<T*>(this)->*memberFunction)
        (headerBuffer, headerBufferSize, dataBuffer, dataBufferSize);
    }
    return true;
  }
};

}
}
#endif /* O2DEVICE_H_ */
