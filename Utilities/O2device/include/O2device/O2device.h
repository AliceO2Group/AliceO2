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

/// @headerfile O2device.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2DEVICE_H_
#define O2DEVICE_H_

#include "FairMQDevice.h"
#include "Headers/DataHeader.h"
#include <stdexcept>

namespace AliceO2 {
namespace Base {

/// just a typedef to express the fact that it is not just a FairMQParts vector,
/// it has to follow the O2 convention of header-payload-header-payload
using O2message = FairMQParts;

class O2device : public FairMQDevice
{
public:
  using FairMQDevice::FairMQDevice;
  virtual ~O2device() {}

  /// Here is how to add an annotated data part (with header);
  /// @param[in,out] parts is a reference to the message;
  /// @param[] incomingBlock header block must be MOVED in (rvalue ref)
  /// @param[] dataMessage the data message must be MOVED in (unique_ptr by value)
  bool AddMessage(O2message& parts,
                  AliceO2::Header::Block&& incomingBlock,
                  FairMQMessagePtr incomingDataMessage) {

    //we have to move the incoming data
    AliceO2::Header::Block headerBlock{std::move(incomingBlock)};
    FairMQMessagePtr dataMessage{std::move(incomingDataMessage)};

    FairMQMessagePtr headerMessage = NewMessage(headerBlock.buffer.get(),
                                                headerBlock.bufferSize,
                                                &AliceO2::Header::Block::freefn,
                                                headerBlock.buffer.get());
    headerBlock.buffer.release();

    parts.AddPart(std::move(headerMessage));
    parts.AddPart(std::move(dataMessage));
    return true;
  }

  /// The user needs to define a member function with correct signature
  /// currently this is old school: buf,len pairs;
  /// In the end I'd like to move to array_view
  /// when this becomes available (either with C++17 or via GSL)
  template <typename T>
  bool ForEach(O2message& parts, bool (T::*memberFunction)(const byte* headerBuffer, size_t headerBufferSize,
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
