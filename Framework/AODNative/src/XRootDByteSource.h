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

#ifndef O2_FRAMEWORK_AODNATIVE_XROOTDBYTESOURCE_H_
#define O2_FRAMEWORK_AODNATIVE_XROOTDBYTESOURCE_H_

#include "AODNative/ByteSource.h"
#include <atomic>
#include <memory>

namespace XrdCl
{
class File;
}

namespace o2::aod::native
{

/// A file read with the XRootD client (root://, roots://, xroot://, xroots://).
/// readvInto() sends all its ranges at once, as vectored reads for the small
/// ones and plain reads for the large ones, and waits for all of them: one
/// round trip for a whole read plan.
class XRootDByteSource : public ByteSource
{
 public:
  XRootDByteSource(std::string url, Options const& options);
  ~XRootDByteSource() override;
  uint64_t size() const override { return mSize; }
  void readInto(uint64_t off, size_t len, void* dst) override;
  void readvInto(std::span<ReadRequest> requests) override;
  uint64_t bytesRead() const override { return mBytesRead.load(std::memory_order_relaxed); }
  /// Requests sent to the server: a vectored read counts once.
  uint64_t readCalls() const override { return mReadCalls.load(std::memory_order_relaxed); }
  /// Without the query, which may hold an access token.
  std::string url() const override { return mName; }
  bool isRemote() const override { return true; }

 private:
  void checkRange(uint64_t off, size_t len) const;

  std::string mName;
  std::unique_ptr<XrdCl::File> mFile;
  uint64_t mSize = 0;
  std::atomic<uint64_t> mBytesRead = 0;
  std::atomic<uint64_t> mReadCalls = 0;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_XROOTDBYTESOURCE_H_
