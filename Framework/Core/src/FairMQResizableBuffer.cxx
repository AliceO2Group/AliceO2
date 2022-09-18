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

#include "Framework/FairMQResizableBuffer.h"
#include <fairmq/Message.h>
#include <arrow/status.h>
#include <arrow/util/config.h>
#include <cassert>

namespace arrow::io::internal
{
void CloseFromDestructor(FileInterface* file);
}

namespace o2::framework
{
static constexpr int64_t kBufferMinimumSize = 256;

FairMQOutputStream::FairMQOutputStream()
  : is_open_(false), capacity_(0), position_(0), mutable_data_(nullptr) {}

FairMQOutputStream::FairMQOutputStream(const std::shared_ptr<ResizableBuffer>& buffer)
  : buffer_(buffer),
    is_open_(true),
    capacity_(buffer->size()),
    position_(0),
    mutable_data_(buffer->mutable_data()) {}

Result<std::shared_ptr<FairMQOutputStream>> FairMQOutputStream::Create(
  int64_t initial_capacity, MemoryPool* pool)
{
  // ctor is private, so cannot use make_shared
  auto ptr = std::shared_ptr<FairMQOutputStream>(new FairMQOutputStream);
  RETURN_NOT_OK(ptr->Reset(initial_capacity, pool));
  return ptr;
}

Status FairMQOutputStream::Reset(int64_t initial_capacity, MemoryPool* pool)
{
  ARROW_ASSIGN_OR_RAISE(buffer_, AllocateResizableBuffer(initial_capacity, pool));
  is_open_ = true;
  capacity_ = initial_capacity;
  position_ = 0;
  mutable_data_ = buffer_->mutable_data();
  return Status::OK();
}

Status FairMQOutputStream::Close()
{
  if (is_open_) {
    is_open_ = false;
    if (position_ < capacity_) {
      RETURN_NOT_OK(buffer_->Resize(position_, false));
    }
  }
  return Status::OK();
}

bool FairMQOutputStream::closed() const { return !is_open_; }

Result<std::shared_ptr<Buffer>> FairMQOutputStream::Finish()
{
  RETURN_NOT_OK(Close());
  buffer_->ZeroPadding();
  is_open_ = false;
  return std::move(buffer_);
}

Result<int64_t> FairMQOutputStream::Tell() const { return position_; }

Status FairMQOutputStream::Write(const void* data, int64_t nbytes)
{
  if (ARROW_PREDICT_FALSE(!is_open_)) {
    return Status::IOError("OutputStream is closed");
  }
  if (ARROW_PREDICT_TRUE(nbytes > 0)) {
    if (ARROW_PREDICT_FALSE(position_ + nbytes >= capacity_)) {
      RETURN_NOT_OK(Reserve(nbytes));
    }
    memcpy(mutable_data_ + position_, data, nbytes);
    position_ += nbytes;
  }
  return Status::OK();
}

Status FairMQOutputStream::Reserve(int64_t nbytes)
{
  // Always overallocate by doubling.  It seems that it is a better growth
  // strategy, at least for memory_benchmark.cc.
  // This may be because it helps match the allocator's allocation buckets
  // more exactly.  Or perhaps it hits a sweet spot in jemalloc.
  int64_t new_capacity = std::max(kBufferMinimumSize, capacity_);
  new_capacity = position_ + nbytes;
  if (new_capacity > capacity_) {
    RETURN_NOT_OK(buffer_->Resize(new_capacity));
    capacity_ = new_capacity;
    mutable_data_ = buffer_->mutable_data();
  }
  return Status::OK();
}

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

std::unique_ptr<fair::mq::Message> FairMQResizableBuffer::Finalise()
{
  auto oldSize = mMessage->GetSize();
  bool resized = mMessage->SetUsedSize(this->size_);
  auto newSize = mMessage->GetSize();

  this->data_ = nullptr;
  this->capacity_ = 0;
  this->size_ = 0;
  return std::move(mMessage);
}

} // namespace o2::framework
