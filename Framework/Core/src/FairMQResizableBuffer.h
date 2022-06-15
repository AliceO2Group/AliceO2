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

#ifndef O2_FRAMEWORK_FAIRMQRESIZABLEBUFFER_H_
#define O2_FRAMEWORK_FAIRMQRESIZABLEBUFFER_H_

#include <memory>
#include <functional>
#include <arrow/buffer.h>
#include "arrow/io/interfaces.h"
#include "arrow/status.h"
#include "arrow/util/future.h"

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

using namespace arrow;
using namespace arrow::io;

class FairMQOutputStream : public OutputStream
{
 public:
  explicit FairMQOutputStream(const std::shared_ptr<ResizableBuffer>& buffer);

  /// \brief Create in-memory output stream with indicated capacity using a
  /// memory pool
  /// \param[in] initial_capacity the initial allocated internal capacity of
  /// the OutputStream
  /// \param[in,out] pool a MemoryPool to use for allocations
  /// \return the created stream
  static Result<std::shared_ptr<FairMQOutputStream>> Create(
    int64_t initial_capacity = 4096, MemoryPool* pool = default_memory_pool());

  // By the time we call the destructor, the contents
  // of the buffer are already moved to fairmq
  // for being sent.
  ~FairMQOutputStream() override = default;

  // Implement the OutputStream interface

  /// Close the stream, preserving the buffer (retrieve it with Finish()).
  Status Close() override;
  bool closed() const override;
  Result<int64_t> Tell() const override;
  Status Write(const void* data, int64_t nbytes) override;

  /// \cond FALSE
  using OutputStream::Write;
  /// \endcond

  /// Close the stream and return the buffer
  Result<std::shared_ptr<Buffer>> Finish();

  /// \brief Initialize state of OutputStream with newly allocated memory and
  /// set position to 0
  /// \param[in] initial_capacity the starting allocated capacity
  /// \param[in,out] pool the memory pool to use for allocations
  /// \return Status
  Status Reset(int64_t initial_capacity = 1024, MemoryPool* pool = default_memory_pool());

  int64_t capacity() const { return capacity_; }

 private:
  FairMQOutputStream();

  // Ensures there is sufficient space available to write nbytes
  Status Reserve(int64_t nbytes);

  std::shared_ptr<ResizableBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};

/// An arrow::ResizableBuffer implemented on top of a fair::mq::Message
/// FIXME: this is an initial attempt to integrate arrow and FairMQ
/// a proper solution probably involves writing a `arrow::MemoryPool`
/// using a `fair::mq::UnmanagedRegion`. This will come at a later stage.
class FairMQResizableBuffer : public ::arrow::ResizableBuffer
{
 public:
  using Creator = std::function<std::unique_ptr<fair::mq::Message>(size_t)>;

  FairMQResizableBuffer(Creator);
  ~FairMQResizableBuffer() override;

  /// Resize the buffer
  ///
  /// * If new size is larger than the backing message size, a new message
  ///   will be created.
  /// * If new size is smaller than the backing message. We will use
  ///   fair::mq::Message::SetUsedSize() accordingly when finalising the message.
  arrow::Status Resize(const int64_t new_size, bool shrink_to_fit) override;
  /// Reserve behaves as std::vector<T>::reserve()
  ///
  /// * If new capacity is greater than old capacity, reallocation happens
  /// * If new capacity is smaller than the old one, nothing happens.
  arrow::Status Reserve(const int64_t capacity) override;

  /// @return the message to be sent. This will make the buffer lose ownership
  /// of the backing store, so you will have to either create a new one or
  /// in order to use it again.
  std::unique_ptr<fair::mq::Message> Finalise();

 private:
  std::unique_ptr<fair::mq::Message> mMessage;
  int64_t mSize;
  Creator mCreator;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_FAIRMQRESIZABLEBUFFER_H_
