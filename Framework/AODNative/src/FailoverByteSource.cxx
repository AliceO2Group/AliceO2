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

#include "FailoverByteSource.h"
#include "AODNative/Errors.h"
#include <chrono>
#include <cmath>
#include <thread>

namespace o2::aod::native
{

namespace
{
std::string labelOf(Replica const& r)
{
  return r.label.empty() ? r.url.substr(0, r.url.find('?')) : r.label;
}
} // namespace

FailoverByteSource::FailoverByteSource(std::vector<Replica> replicas, Options const& options, Opener opener, Log log)
  : mReplicas(std::move(replicas)), mOptions(options), mOpener(std::move(opener)), mLog(std::move(log))
{
  if (mReplicas.empty()) {
    throw IoError("(no replica)", "no replica to open");
  }
  for (mIndex = 0; mIndex < mReplicas.size(); ++mIndex) {
    try {
      mSource = openReplica(mIndex);
      mSize = mSource->size();
      mRemote = mSource->isRemote();
      return;
    } catch (IoError const& e) {
      report("Could not open replica on " + labelOf(mReplicas[mIndex]) + ": " + e.what());
      mErrors += " " + labelOf(mReplicas[mIndex]) + ": " + e.what() + ";";
    }
  }
  throw IoError(labelOf(mReplicas.front()), "no replica could be opened:" + mErrors);
}

std::shared_ptr<ByteSource> FailoverByteSource::openReplica(size_t index)
{
  std::shared_ptr<ByteSource> source = mOpener(mReplicas[index].url, mOptions);
  if (mSource && source->size() != mSize) {
    throw IoError(source->url(), "size " + std::to_string(source->size()) + " differs from the " + std::to_string(mSize) + " bytes of the replica read so far");
  }
  return source;
}

std::shared_ptr<ByteSource> FailoverByteSource::current() const
{
  std::lock_guard lock{mMutex};
  return mSource;
}

std::string FailoverByteSource::replicaLabel() const
{
  std::lock_guard lock{mMutex};
  return mReplicas[mIndex].label;
}

uint64_t FailoverByteSource::bytesRead() const
{
  return mRetiredBytes.load(std::memory_order_relaxed) + current()->bytesRead();
}

uint64_t FailoverByteSource::readCalls() const
{
  return mRetiredCalls.load(std::memory_order_relaxed) + current()->readCalls();
}

void FailoverByteSource::report(std::string const& message) const
{
  if (mLog) {
    mLog(message);
  }
}

void FailoverByteSource::readInto(uint64_t off, size_t len, void* dst)
{
  withFailover([&](ByteSource& s) { s.readInto(off, len, dst); });
}

void FailoverByteSource::readvInto(std::span<ReadRequest> requests)
{
  withFailover([&](ByteSource& s) { s.readvInto(requests); });
}

void FailoverByteSource::withFailover(std::function<void(ByteSource&)> const& read)
{
  while (true) {
    std::shared_ptr<ByteSource> source;
    uint64_t generation;
    {
      std::lock_guard lock{mMutex};
      source = mSource;
      generation = mGeneration;
    }
    try {
      read(*source);
      return;
    } catch (IoError const& e) {
      std::lock_guard lock{mMutex};
      if (generation != mGeneration) {
        continue; // another read already replaced the replica: try the new one
      }
      mRetiredBytes.fetch_add(source->bytesRead(), std::memory_order_relaxed);
      mRetiredCalls.fetch_add(source->readCalls(), std::memory_order_relaxed);
      std::string error = e.what();
      // Reopen the same replica, after a growing delay, or move on to the next
      // one, until a replica opens. Other readers wait on mMutex meanwhile.
      while (true) {
        if (mAttempts < mOptions.readRetries) {
          double delay = mOptions.retryDelay * std::pow(2., mAttempts);
          ++mAttempts;
          report("Read error on " + labelOf(mReplicas[mIndex]) + ", retry " + std::to_string(mAttempts) + " of " +
                 std::to_string(mOptions.readRetries) + " in " + std::to_string((int)std::lround(delay)) + " s: " + error);
          std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        } else {
          mErrors += " " + labelOf(mReplicas[mIndex]) + ": " + error + ";";
          if (mIndex + 1 >= mReplicas.size()) {
            throw IoError(labelOf(mReplicas.front()), "no replica left to read from:" + mErrors);
          }
          ++mIndex;
          mAttempts = 0;
          report("Moving to replica on " + labelOf(mReplicas[mIndex]) + " after: " + error);
        }
        try {
          mSource = openReplica(mIndex);
          ++mGeneration;
          break;
        } catch (IoError const& reopen) {
          error = reopen.what();
          if (mAttempts == 0) {
            mAttempts = mOptions.readRetries; // a replica that does not open is not retried
          }
        }
      }
    }
  }
}

} // namespace o2::aod::native
