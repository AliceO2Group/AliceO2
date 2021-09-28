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

#include "FairMQResizableBuffer.h"
#include <fairmq/FairMQMessage.h>
#include <arrow/status.h>
#include <arrow/util/config.h>
#include <cassert>

namespace o2::framework
{

FairMQResizableBuffer::~FairMQResizableBuffer() = default;

// Creates an empty message
FairMQResizableBuffer::FairMQResizableBuffer(Creator creator)
  : ResizableBuffer(nullptr, 0),
    mMessage{nullptr},
    mCreator{creator}
{
  this->data_ = nullptr;
  this->capacity_ = 0;
  this->size_ = 0;
}

arrow::Status FairMQResizableBuffer::Resize(const int64_t newSize, bool shrink_to_fit)
{
  // NOTE: we ignore "shrink_to_fit" because in any case we
  // invoke SetUsedSize when we send the message. This
  // way we avoid unneeded copies at the arrow level.
  if (newSize > this->capacity_) {
    auto status = this->Reserve(newSize);
    if (status.ok() == false) {
      return status;
    }
  }
  assert(newSize <= this->capacity_);

  this->size_ = newSize;
  assert(this->size_ <= mMessage->GetSize());
  return arrow::Status::OK();
}

arrow::Status FairMQResizableBuffer::Reserve(const int64_t capacity)
{
  assert(!mMessage || this->size_ <= mMessage->GetSize());
  assert(!mMessage || this->capacity_ == mMessage->GetSize());
  if (capacity <= this->capacity_) {
    return arrow::Status::OK();
  }
  auto newMessage = mCreator(capacity);
  assert(!mMessage || capacity > mMessage->GetSize());
  if (mMessage) {
    memcpy(newMessage->GetData(), mMessage->GetData(), mMessage->GetSize());
  }
  mMessage = std::move(newMessage);
  assert(mMessage);
  this->data_ = reinterpret_cast<uint8_t*>(mMessage->GetData());
  this->capacity_ = static_cast<int64_t>(mMessage->GetSize());
  assert(this->data_);
  return arrow::Status::OK();
}

std::unique_ptr<FairMQMessage> FairMQResizableBuffer::Finalise()
{
  mMessage->SetUsedSize(this->size_);
  this->data_ = nullptr;
  this->capacity_ = 0;
  this->size_ = 0;
  return std::move(mMessage);
}

} // namespace o2::framework
