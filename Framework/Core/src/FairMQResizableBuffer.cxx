// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairMQResizableBuffer.h"
#include <fairmq/FairMQMessage.h>
#include <arrow/status.h>
#include <cassert>

namespace o2
{
namespace framework
{

FairMQResizableBuffer::~FairMQResizableBuffer() = default;

// Creates an empty message
FairMQResizableBuffer::FairMQResizableBuffer(Creator creator)
  : ResizableBuffer(nullptr, 0),
    mMessage{std::move(creator(4096))},
    mCreator{creator}
{
  this->mutable_data_ = reinterpret_cast<uint8_t*>(mMessage->GetData());
  this->data_ = this->mutable_data_;
  assert(this->data_);
  this->capacity_ = static_cast<int64_t>(mMessage->GetSize());
  this->size_ = 0;
}

arrow::Status FairMQResizableBuffer::Resize(const int64_t new_size, bool shrink_to_fit)
{
  if (new_size < this->capacity_ && shrink_to_fit == true) {
    auto newMessage = mCreator(new_size);
    memcpy(newMessage->GetData(), mMessage->GetData(), new_size);
    mMessage = std::move(newMessage);
    this->mutable_data_ = reinterpret_cast<uint8_t*>(mMessage->GetData());
    this->data_ = this->mutable_data_;
    assert(this->data_);
    this->capacity_ = static_cast<int64_t>(mMessage->GetSize());
    assert(new_size == this->capacity_);
  } else if (new_size > this->capacity_) {
    auto status = this->Reserve(new_size);
    if (status.ok() == false) {
      return status;
    }
  }
  assert(new_size <= this->capacity_);

  this->size_ = new_size;
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
  this->mutable_data_ = reinterpret_cast<uint8_t*>(mMessage->GetData());
  this->data_ = this->mutable_data_;
  this->capacity_ = static_cast<int64_t>(mMessage->GetSize());
  assert(this->data_);
  return arrow::Status::OK();
}

std::unique_ptr<FairMQMessage> FairMQResizableBuffer::Finalise()
{
  mMessage->SetUsedSize(this->size_);
  this->data_ = nullptr;
  this->mutable_data_ = nullptr;
  this->capacity_ = 0;
  this->size_ = 0;
  return std::move(mMessage);
}

} // namespace framework
} // namespace o2
