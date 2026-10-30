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

#include "AODNative/ReadPlan.h"
#include <algorithm>
#include <cstring>

namespace o2::aod::native
{

ReadBufferPool::Buffer ReadBufferPool::get(size_t size)
{
  std::lock_guard lock{mMutex};
  auto best = mFree.end();
  for (auto it = mFree.begin(); it != mFree.end(); ++it) {
    if (it->capacity >= size && (best == mFree.end() || it->capacity < best->capacity)) {
      best = it;
    }
  }
  if (best != mFree.end()) {
    auto buffer = std::move(*best);
    mFree.erase(best);
    return buffer;
  }
  // All the free buffers are too small: give them back rather than keep them
  // resident next to the new one. Some headroom, so that the next DF, a bit
  // larger, still fits.
  mFree.clear();
  auto capacity = size + size / 8;
  return {std::make_unique_for_overwrite<uint8_t[]>(capacity), capacity};
}

void ReadBufferPool::put(Buffer buffer)
{
  std::lock_guard lock{mMutex};
  mFree.push_back(std::move(buffer));
  if (mFree.size() > kMaxFree) {
    mFree.erase(std::ranges::min_element(mFree, {}, &Buffer::capacity));
  }
}

ReadPlan::ReadPlan(std::shared_ptr<ByteSource> source, Options const& options, std::shared_ptr<ReadBufferPool> pool)
  : mSource(std::move(source)), mPool(pool ? std::move(pool) : std::make_shared<ReadBufferPool>()), mCoalesceGap(options.coalesceGap)
{
}

ReadPlan::~ReadPlan()
{
  for (auto& buffer : mBuffers) {
    mPool->put(std::move(buffer));
  }
}

void ReadPlan::add(uint64_t offset, size_t length)
{
  if (length == 0 || offset > mSource->size() || length > mSource->size() - offset) {
    // Left to the decoder, which reports it with the branch it belongs to.
    return;
  }
  std::lock_guard lock{mMutex};
  mPending.emplace_back(offset, length);
}

void ReadPlan::add(TreeInfo const& tree)
{
  for (auto const& branch : tree.branches) {
    for (auto const& basket : branch.baskets) {
      if (!basket.embedded) {
        add(basket.seek, basket.nbytes);
      }
    }
  }
}

std::span<const uint8_t> ReadPlan::lookup(uint64_t offset, size_t length) const
{
  auto it = std::upper_bound(mChunks.begin(), mChunks.end(), offset, [](uint64_t off, Chunk const& c) { return off < c.offset; });
  if (it == mChunks.begin()) {
    return {};
  }
  --it;
  if (offset - it->offset + length > it->size) {
    return {};
  }
  return {it->data + (offset - it->offset), length};
}

void ReadPlan::fetchPending()
{
  std::ranges::sort(mPending);
  std::vector<Chunk> fetched;
  for (auto [offset, length] : mPending) {
    if (!lookup(offset, length).empty()) {
      continue;
    }
    auto end = offset + length;
    if (!fetched.empty()) {
      auto& last = fetched.back();
      auto lastEnd = last.offset + last.size;
      auto next = std::lower_bound(mChunks.begin(), mChunks.end(), lastEnd, [](Chunk const& c, uint64_t off) { return c.offset < off; });
      bool bridgesFetched = next != mChunks.end() && next->offset < offset;
      if (offset <= lastEnd + mCoalesceGap && !bridgesFetched) {
        if (end > lastEnd) {
          last.size = end - last.offset;
        }
        continue;
      }
    }
    fetched.push_back({offset, nullptr, length});
  }
  mPending.clear();
  if (fetched.empty()) {
    return;
  }
  // One buffer for all the chunks of this fetch. Earlier buffers stay where
  // they are, so spans handed out before remain valid.
  size_t total = 0;
  for (auto& chunk : fetched) {
    total += chunk.size;
  }
  auto buffer = mPool->get(total);
  std::vector<ReadRequest> requests;
  auto* at = buffer.data.get();
  for (auto& chunk : fetched) {
    chunk.data = at;
    at += chunk.size;
    requests.push_back({chunk.offset, chunk.size, chunk.data});
  }
  mBuffers.push_back(std::move(buffer));
  mSource->readvInto(requests);
  mChunks.insert(mChunks.end(), fetched.begin(), fetched.end());
  std::ranges::sort(mChunks, {}, &Chunk::offset);
}

std::span<const uint8_t> ReadPlan::find(uint64_t offset, size_t length)
{
  std::lock_guard lock{mMutex};
  auto bytes = lookup(offset, length);
  if (bytes.empty() && !mPending.empty()) {
    fetchPending();
    bytes = lookup(offset, length);
  }
  return bytes;
}

void ReadPlan::fetch()
{
  std::lock_guard lock{mMutex};
  if (!mPending.empty()) {
    fetchPending();
  }
}

size_t ReadPlan::bytesHeld() const
{
  std::lock_guard lock{mMutex};
  size_t total = 0;
  for (auto const& chunk : mChunks) {
    total += chunk.size;
  }
  return total;
}

PlannedByteSource::PlannedByteSource(std::shared_ptr<ByteSource> source, std::shared_ptr<ReadPlan> plan)
  : mSource(std::move(source)), mPlan(std::move(plan))
{
}

std::span<const uint8_t> PlannedByteSource::tryMap(uint64_t off, size_t len)
{
  auto bytes = mSource->tryMap(off, len);
  return bytes.empty() ? mPlan->find(off, len) : bytes;
}

void PlannedByteSource::readInto(uint64_t off, size_t len, void* dst)
{
  auto bytes = mPlan->find(off, len);
  if (bytes.empty()) {
    mSource->readInto(off, len, dst);
    return;
  }
  std::memcpy(dst, bytes.data(), len);
}

void PlannedByteSource::readvInto(std::span<ReadRequest> requests)
{
  mSource->readvInto(requests);
}

} // namespace o2::aod::native
