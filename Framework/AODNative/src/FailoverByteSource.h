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

#ifndef O2_FRAMEWORK_AODNATIVE_FAILOVERBYTESOURCE_H_
#define O2_FRAMEWORK_AODNATIVE_FAILOVERBYTESOURCE_H_

#include "AODNative/ByteSource.h"
#include <atomic>
#include <functional>
#include <mutex>

namespace o2::aod::native
{

/// The replicas of one file, read as one ByteSource. A read which fails with
/// an IoError is retried on the same replica, reopened, Options::readRetries
/// times with a growing delay, and then on the next replica of the same size.
/// Opening goes through the replicas in order too. Reads already in flight on
/// a replica that is being replaced finish there, or fail and are retried.
class FailoverByteSource : public ByteSource
{
 public:
  using Opener = std::function<std::unique_ptr<ByteSource>(std::string const& url, Options const&)>;
  using Log = std::function<void(std::string const&)>;

  /// Throws IoError when no replica opens.
  FailoverByteSource(std::vector<Replica> replicas, Options const& options, Opener opener, Log log);

  uint64_t size() const override { return mSize; }
  void readInto(uint64_t off, size_t len, void* dst) override;
  void readvInto(std::span<ReadRequest> requests) override;
  std::span<const uint8_t> tryMap(uint64_t off, size_t len) override { return current()->tryMap(off, len); }
  bool canMap() const override { return current()->canMap(); }
  bool isRemote() const override { return mRemote; }
  void willNeed(uint64_t off, size_t len) override { current()->willNeed(off, len); }
  void dontNeed(uint64_t off, size_t len) override { current()->dontNeed(off, len); }
  uint64_t bytesRead() const override;
  uint64_t readCalls() const override;
  std::string url() const override { return current()->url(); }
  std::string replicaLabel() const override;

 private:
  std::shared_ptr<ByteSource> current() const;
  /// Runs @a read on the current replica, retrying and failing over on IoError.
  void withFailover(std::function<void(ByteSource&)> const& read);
  /// Opens replica @a index; throws IoError, also when its size differs.
  std::shared_ptr<ByteSource> openReplica(size_t index);
  void report(std::string const& message) const;

  std::vector<Replica> mReplicas;
  Options mOptions;
  Opener mOpener;
  Log mLog;

  mutable std::mutex mMutex;
  std::shared_ptr<ByteSource> mSource; // guarded by mMutex
  size_t mIndex = 0;                   // guarded by mMutex
  uint64_t mGeneration = 0;            // guarded by mMutex, bumped at every reopen
  int mAttempts = 0;                   // retries done on the current replica, guarded by mMutex
  std::string mErrors;                 // what went wrong on the replicas left behind

  uint64_t mSize = 0;
  bool mRemote = false;
  std::atomic<uint64_t> mRetiredBytes = 0;
  std::atomic<uint64_t> mRetiredCalls = 0;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_FAILOVERBYTESOURCE_H_
