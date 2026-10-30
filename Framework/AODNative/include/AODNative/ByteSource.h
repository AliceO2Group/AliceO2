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

#ifndef O2_FRAMEWORK_AODNATIVE_BYTESOURCE_H_
#define O2_FRAMEWORK_AODNATIVE_BYTESOURCE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace o2::aod::native
{

/// All the reader settings, collected from the --aod-reader-* / --aod-http-*
/// workflow options.
struct Options {
  enum struct LocalIO { MMap,
                        PRead };
  enum struct RemoteIO { Prefetch,
                         Stream };
  enum struct HttpVersion { Auto,
                            Http1_1,
                            Http2 };
  LocalIO localIO = LocalIO::MMap;
  RemoteIO remoteIO = RemoteIO::Prefetch;
  int threads = 1;
  int prefetchDepth = 1;
  size_t coalesceGap = 64 * 1024;
  double spanFraction = 0.8;
  int streamWindows = 4;
  size_t streamWindowBytes = 2 * 1024 * 1024;
  HttpVersion httpVersion = HttpVersion::Auto;
  int httpConnections = 4;
};

struct ReadRequest {
  uint64_t offset;
  size_t length;
  void* dst;
};

/// Random access to the bytes of a file. Everything above this is I/O-agnostic.
/// The caller always owns the destination. Implementations are thread-safe
/// for concurrent reads.
class ByteSource
{
 public:
  virtual ~ByteSource();
  virtual uint64_t size() const = 0;
  /// Throws IoError, or CorruptData when the range is past the end of the file.
  virtual void readInto(uint64_t off, size_t len, void* dst) = 0;
  virtual void readvInto(std::span<ReadRequest> requests) = 0;
  /// A view of the bytes without copying them, if the backend can provide it.
  virtual std::span<const uint8_t> tryMap(uint64_t off, size_t len) { return {}; }
  /// Whether tryMap() provides every range, so that nothing needs reading.
  virtual bool canMap() const { return false; }
  /// Whether each read is a network round trip, worth hiding behind decoding.
  virtual bool isRemote() const { return false; }
  /// Hints that a range will be needed soon, or is no longer needed.
  virtual void willNeed(uint64_t off, size_t len) {}
  virtual void dontNeed(uint64_t off, size_t len) {}
  virtual uint64_t bytesRead() const = 0;
  virtual uint64_t readCalls() const = 0;
  virtual std::string url() const = 0;
  static std::unique_ptr<ByteSource> open(std::string url, const Options&);
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_BYTESOURCE_H_
