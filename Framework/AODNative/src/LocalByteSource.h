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

#ifndef O2_FRAMEWORK_AODNATIVE_LOCALBYTESOURCE_H_
#define O2_FRAMEWORK_AODNATIVE_LOCALBYTESOURCE_H_

#include "AODNative/ByteSource.h"
#include <atomic>

namespace o2::aod::native
{

/// A local file, mmapped read-only on 64-bit hosts, with pread as fallback
/// (mmap failure, or --aod-reader-local-io=pread).
class LocalByteSource : public ByteSource
{
 public:
  LocalByteSource(std::string path, Options const& options);
  ~LocalByteSource() override;
  uint64_t size() const override { return mSize; }
  void readInto(uint64_t off, size_t len, void* dst) override;
  void readvInto(std::span<ReadRequest> requests) override;
  std::span<const uint8_t> tryMap(uint64_t off, size_t len) override;
  bool canMap() const override { return mMap != nullptr; }
  void willNeed(uint64_t off, size_t len) override;
  void dontNeed(uint64_t off, size_t len) override;
  uint64_t bytesRead() const override { return mBytesRead.load(std::memory_order_relaxed); }
  uint64_t readCalls() const override { return mReadCalls.load(std::memory_order_relaxed); }
  std::string url() const override { return mPath; }

 private:
  void checkRange(uint64_t off, size_t len) const;
  void advise(uint64_t off, size_t len, int advice);

  std::string mPath;
  int mFd = -1;
  uint64_t mSize = 0;
  uint8_t const* mMap = nullptr;
  std::atomic<uint64_t> mBytesRead = 0;
  std::atomic<uint64_t> mReadCalls = 0;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_LOCALBYTESOURCE_H_
