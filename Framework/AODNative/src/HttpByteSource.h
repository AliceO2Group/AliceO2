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

#ifndef O2_FRAMEWORK_AODNATIVE_HTTPBYTESOURCE_H_
#define O2_FRAMEWORK_AODNATIVE_HTTPBYTESOURCE_H_

#include "AODNative/ByteSource.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace o2::aod::native
{

/// A file read over public HTTP(S) with ranged GETs (no authentication yet).
/// readvInto() sends all its ranges at once as parallel requests, multiplexed
/// on one connection with HTTP/2, or spread over Options::httpConnections
/// connections with HTTP/1.1. Connections, TLS sessions and DNS are shared by
/// all the calls on the source.
class HttpByteSource : public ByteSource
{
 public:
  HttpByteSource(std::string url, Options const& options);
  ~HttpByteSource() override;
  uint64_t size() const override { return mSize; }
  void readInto(uint64_t off, size_t len, void* dst) override;
  void readvInto(std::span<ReadRequest> requests) override;
  uint64_t bytesRead() const override { return mBytesRead.load(std::memory_order_relaxed); }
  /// HTTP requests sent.
  uint64_t readCalls() const override { return mReadCalls.load(std::memory_order_relaxed); }
  /// Without the query, which may hold a token.
  std::string url() const override { return mName; }
  bool isRemote() const override { return true; }

  struct Shared;

 private:
  void checkRange(uint64_t off, size_t len) const;

  std::string mUrl;
  std::string mName;
  Options mOptions;
  std::unique_ptr<Shared> mShared;
  uint64_t mSize = 0;
  std::atomic<uint64_t> mBytesRead = 0;
  std::atomic<uint64_t> mReadCalls = 0;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_HTTPBYTESOURCE_H_
