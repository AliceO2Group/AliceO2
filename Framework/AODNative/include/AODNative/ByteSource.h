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
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

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
  /// Most bytes the read plans of a file hold at once, 0 for no limit. A DF
  /// larger than this is read in groups of trees. kAutoBufferBudget: 16 MB for
  /// a local file, 64 MB for a remote one, where each read is a round trip.
  static constexpr size_t kAutoBufferBudget = SIZE_MAX;
  size_t bufferBudget = kAutoBufferBudget;
  double spanFraction = 0.8;
  int streamWindows = 4;
  size_t streamWindowBytes = 2 * 1024 * 1024;
  HttpVersion httpVersion = HttpVersion::Auto;
  int httpConnections = 4;
  /// For a source opened with ByteSource::openReplicas: how many times, at
  /// most, failed reads are retried on a replica (reopened) during the life of
  /// the file before moving on to the next replica, and the seconds waited
  /// before the first retry (doubled for each further one).
  int readRetries = 2;
  double retryDelay = 2.0;
  /// Most bytes of one read plan in flight at once on a remote file, so that
  /// each request finishes within the server timeout on a slow link (XRootD
  /// expires requests after XRD_REQUESTTIMEOUT, 150 s in the ALICE setup).
  size_t remoteInFlightBytes = 32 * 1024 * 1024;
};

/// One copy of a file, for ByteSource::openReplicas: where it is, and a label
/// (e.g. the storage element) to report which one is being read.
struct Replica {
  std::string url;
  std::string label;
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
  /// The label of the replica being read, for a source opened with openReplicas().
  virtual std::string replicaLabel() const { return {}; }
  static std::unique_ptr<ByteSource> open(std::string url, const Options&);
  /// Opens the first replica which opens, and on a read error retries it
  /// (Options::readRetries times, reopened) and then moves on to the next
  /// replica of the same size. @a log, if given, is told about every retry
  /// and every change of replica. Throws IoError when no replica is left.
  static std::unique_ptr<ByteSource> openReplicas(std::vector<Replica> replicas, const Options&,
                                                  std::function<void(std::string const&)> log = {});
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_BYTESOURCE_H_
