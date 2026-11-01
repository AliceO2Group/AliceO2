// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_AODNATIVE_READPLAN_H_
#define O2_FRAMEWORK_AODNATIVE_READPLAN_H_

#include "AODNative/ByteSource.h"
#include "AODNative/Objects.h"
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace o2::aod::native
{

/// Memory for the reads of ReadPlans, kept from one DF to the next: a new plan
/// takes the buffers of the plans which are gone instead of allocating, and
/// faulting in, fresh ones.
class ReadBufferPool
{
 public:
  struct Buffer {
    std::unique_ptr<uint8_t[]> data;
    size_t capacity = 0;
  };
  /// The smallest free buffer of at least @a size bytes, or a new one.
  Buffer get(size_t size);
  void put(Buffer buffer);

 private:
  static constexpr size_t kMaxFree = 2;
  std::mutex mMutex;
  std::vector<Buffer> mFree;
};

/// The baskets of the trees a DF is read from, fetched in as few reads as
/// possible: ranges are registered first, and the first lookup of one which is
/// not in memory yet fetches every pending range, merging those closer than
/// Options::coalesceGap. The fetched bytes are held until the plan goes away.
class ReadPlan
{
 public:
  /// Without a @a pool, the plan allocates its own buffers.
  ReadPlan(std::shared_ptr<ByteSource> source, Options const& options, std::shared_ptr<ReadBufferPool> pool = nullptr);
  ~ReadPlan();
  ReadPlan(ReadPlan const&) = delete;
  ReadPlan& operator=(ReadPlan const&) = delete;

  void add(uint64_t offset, size_t length);
  /// Registers every basket of @a tree which is not embedded in its TBranch.
  void add(TreeInfo const& tree);
  /// The bytes of [offset, offset + length), if a registered range covers
  /// them, fetching the pending ranges first if needed. Empty otherwise. The
  /// span stays valid as long as the plan.
  std::span<const uint8_t> find(uint64_t offset, size_t length);
  /// Fetches the pending ranges now, rather than on the first find().
  void fetch();
  size_t bytesHeld() const;

 private:
  struct Chunk {
    uint64_t offset;
    uint8_t* data;
    size_t size;
  };
  std::span<const uint8_t> lookup(uint64_t offset, size_t length) const;
  void fetchPending();

  std::shared_ptr<ByteSource> mSource;
  std::shared_ptr<ReadBufferPool> mPool;
  size_t mCoalesceGap;
  std::vector<std::pair<uint64_t, size_t>> mPending;
  std::vector<Chunk> mChunks; // sorted by offset, not overlapping
  std::vector<ReadBufferPool::Buffer> mBuffers;
  mutable std::mutex mMutex;
};

/// A ByteSource which serves the ranges of a ReadPlan from memory and
/// everything else from the source underneath.
class PlannedByteSource : public ByteSource
{
 public:
  PlannedByteSource(std::shared_ptr<ByteSource> source, std::shared_ptr<ReadPlan> plan);
  uint64_t size() const override { return mSource->size(); }
  void readInto(uint64_t off, size_t len, void* dst) override;
  void readvInto(std::span<ReadRequest> requests) override;
  std::span<const uint8_t> tryMap(uint64_t off, size_t len) override;
  uint64_t bytesRead() const override { return mSource->bytesRead(); }
  uint64_t readCalls() const override { return mSource->readCalls(); }
  std::string url() const override { return mSource->url(); }

 private:
  std::shared_ptr<ByteSource> mSource;
  std::shared_ptr<ReadPlan> mPlan;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_READPLAN_H_
