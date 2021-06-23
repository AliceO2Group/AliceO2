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

#ifndef FRAMEWORK_FAIRMQRESIZABLEBUFFER_H
#define FRAMEWORK_FAIRMQRESIZABLEBUFFER_H

#include <memory>
#include <functional>
#include <arrow/buffer.h>

class FairMQMessage;

namespace o2
{
namespace framework
{

/// An arrow::ResizableBuffer implemented on top of a FairMQMessage
/// FIXME: this is an initial attempt to integrate arrow and FairMQ
/// a proper solution probably involves writing a `arrow::MemoryPool`
/// using a `FairMQUnmanagedRegion`. This will come at a later stage.
class FairMQResizableBuffer : public ::arrow::ResizableBuffer
{
 public:
  using Creator = std::function<std::unique_ptr<FairMQMessage>(size_t)>;

  FairMQResizableBuffer(Creator);
  ~FairMQResizableBuffer() override;

  /// Resize the buffer
  ///
  /// * If new size is larger than the backing message size, a new message
  ///   will be created.
  /// * If new size is smaller than the backing message. We will use
  ///   FairMQMessage::SetUsedSize() accordingly when finalising the message.
  arrow::Status Resize(const int64_t new_size, bool shrink_to_fit) override;
  /// Reserve behaves as std::vector<T>::reserve()
  ///
  /// * If new capacity is greater than old capacity, reallocation happens
  /// * If new capacity is smaller than the old one, nothing happens.
  arrow::Status Reserve(const int64_t capacity) override;

  /// @return the message to be sent. This will make the buffer lose ownership
  /// of the backing store, so you will have to either create a new one or
  /// in order to use it again.
  std::unique_ptr<FairMQMessage> Finalise();

 private:
  std::unique_ptr<FairMQMessage> mMessage;
  int64_t mSize;
  Creator mCreator;
};

} // namespace framework
} // namespace o2

#endif
