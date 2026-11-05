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

#include "XRootDByteSource.h"
#include "AODNative/Errors.h"
#include <XrdCl/XrdClFile.hh>
#include <XrdCl/XrdClFileSystem.hh>
#include <XrdCl/XrdClURL.hh>
#include <XrdCl/XrdClXRootDResponses.hh>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <vector>

namespace o2::aod::native
{

namespace
{
// XrdCl's defaults for kXR_readv, used when the server does not tell its own
// limits: chunks of at most 2097136 bytes, at most 1024 of them per request.
// Larger ranges are read with plain reads, in pieces of at most kMaxPlainRead:
// small enough that one request completes well within the request timeout on
// a slow link, as ROOT's basket-sized reads do.
constexpr size_t kDefaultVectorChunk = 2097136;
constexpr size_t kDefaultVectorChunks = 1024;
constexpr size_t kMaxPlainRead = 8 * 1024 * 1024;

/// Counts down the replies of the requests sent by one readvInto(), keeping
/// the first error.
class Completion
{
 public:
  explicit Completion(size_t pending) : mPending(pending) {}

  void done(std::string error)
  {
    std::lock_guard lock{mMutex};
    if (!error.empty() && mError.empty()) {
      mError = std::move(error);
    }
    if (--mPending == 0) {
      mDone.notify_all();
    }
  }

  /// Waits for every reply, including those of requests which were never
  /// sent (counted down by the caller), and returns the first error.
  std::string wait()
  {
    std::unique_lock lock{mMutex};
    mDone.wait(lock, [this] { return mPending == 0; });
    return mError;
  }

 private:
  std::mutex mMutex;
  std::condition_variable mDone;
  size_t mPending;
  std::string mError;
};

/// One request: checks that the server returned exactly the bytes asked for.
class Reply : public XrdCl::ResponseHandler
{
 public:
  Reply(Completion& completion, size_t expected) : mCompletion(completion), mExpected(expected) {}

  void HandleResponse(XrdCl::XRootDStatus* status, XrdCl::AnyObject* response) override
  {
    std::string error;
    if (!status->IsOK()) {
      error = status->ToString();
    } else if (auto got = received(response); got != mExpected) {
      error = "short read: " + std::to_string(got) + " of " + std::to_string(mExpected) + " bytes";
    }
    delete status;
    delete response;
    mCompletion.done(std::move(error));
    delete this;
  }

 private:
  static size_t received(XrdCl::AnyObject* response)
  {
    if (!response) {
      return 0;
    }
    XrdCl::ChunkInfo* chunk = nullptr;
    response->Get(chunk);
    if (chunk) {
      return chunk->length;
    }
    XrdCl::VectorReadInfo* vector = nullptr;
    response->Get(vector);
    return vector ? vector->GetSize() : 0;
  }

  Completion& mCompletion;
  size_t mExpected;
};

/// One request to send: a plain read when it has one chunk, a vectored one otherwise.
struct Request {
  XrdCl::ChunkList chunks;
  size_t bytes = 0;
};

/// Asks the data server for its kXR_readv limits ("readv_ior_max" and
/// "readv_iov_max"), as ROOT's TNetXNGFile does. Leaves the defaults when the
/// server does not answer.
void queryVectorLimits(XrdCl::File& file, size_t& chunkMax, size_t& chunksMax)
{
  std::string server;
  if (!file.GetProperty("DataServer", server) || server.empty()) {
    return;
  }
  XrdCl::FileSystem fs{XrdCl::URL{server}};
  XrdCl::Buffer arg;
  arg.FromString("readv_ior_max readv_iov_max");
  XrdCl::Buffer* response = nullptr;
  auto status = fs.Query(XrdCl::QueryCode::Config, arg, response);
  if (!status.IsOK() || !response) {
    delete response;
    return;
  }
  std::istringstream lines{response->ToString()};
  delete response;
  std::string ior, iov;
  if (!std::getline(lines, ior) || !std::getline(lines, iov)) {
    return;
  }
  // An unknown setting is echoed back by name instead of a number.
  try {
    if (auto v = std::stoull(ior); v > 0) {
      chunkMax = std::min<size_t>(v, kDefaultVectorChunk);
    }
    if (auto v = std::stoull(iov); v > 0) {
      chunksMax = std::min<size_t>(v, kDefaultVectorChunks);
    }
  } catch (std::exception const&) {
  }
}
} // namespace

XRootDByteSource::XRootDByteSource(std::string url, Options const& options)
  : mName(url.substr(0, url.find('?'))),
    mFile(std::make_unique<XrdCl::File>()),
    mVectorChunkMax(kDefaultVectorChunk),
    mVectorChunksMax(kDefaultVectorChunks),
    mInFlightMax(options.remoteInFlightBytes)
{
  auto status = mFile->Open(url, XrdCl::OpenFlags::Read);
  if (!status.IsOK()) {
    throw IoError(mName, status.ToString());
  }
  XrdCl::StatInfo* info = nullptr;
  status = mFile->Stat(false, info);
  if (!status.IsOK() || !info) {
    delete info;
    throw IoError(mName, "stat failed: " + status.ToString());
  }
  mSize = info->GetSize();
  delete info;
  queryVectorLimits(*mFile, mVectorChunkMax, mVectorChunksMax);
}

XRootDByteSource::~XRootDByteSource()
{
  if (mFile->IsOpen()) {
    [[maybe_unused]] auto status = mFile->Close();
  }
}

void XRootDByteSource::checkRange(uint64_t off, size_t len) const
{
  if (off > mSize || len > mSize - off) {
    throw CorruptData(mName, off, "read of " + std::to_string(len) + " bytes past the end of a " + std::to_string(mSize) + " byte file");
  }
}

void XRootDByteSource::readInto(uint64_t off, size_t len, void* dst)
{
  ReadRequest request{off, len, dst};
  readvInto({&request, 1});
}

void XRootDByteSource::readvInto(std::span<ReadRequest> requests)
{
  // Ranges larger than a vector chunk are read with plain reads of at most
  // kMaxPlainRead; the others are gathered into vectored reads within the
  // server's limits. No request carries more than the in-flight limit.
  size_t window = mInFlightMax > 0 ? mInFlightMax : SIZE_MAX;
  std::vector<Request> out;
  Request vector;
  size_t total = 0;
  auto flushVector = [&] {
    if (!vector.chunks.empty()) {
      out.push_back(std::move(vector));
      vector = Request{};
    }
  };
  for (auto const& r : requests) {
    checkRange(r.offset, r.length);
    total += r.length;
    if (r.length == 0) {
      continue;
    }
    if (r.length > mVectorChunkMax) {
      for (size_t done = 0; done < r.length; done += kMaxPlainRead) {
        auto n = std::min(kMaxPlainRead, r.length - done);
        Request plain;
        plain.chunks.emplace_back(r.offset + done, (uint32_t)n, (uint8_t*)r.dst + done);
        plain.bytes = n;
        out.push_back(std::move(plain));
      }
      continue;
    }
    if (vector.chunks.size() == mVectorChunksMax || (!vector.chunks.empty() && vector.bytes + r.length > window)) {
      flushVector();
    }
    vector.chunks.emplace_back(r.offset, (uint32_t)r.length, r.dst);
    vector.bytes += r.length;
  }
  flushVector();

  // Send the requests in rounds of at most `window` bytes, waiting for each
  // round: on a slow link every request then gets its share of the bandwidth,
  // instead of all of them competing until some expire.
  size_t calls = 0;
  for (size_t first = 0; first < out.size();) {
    size_t last = first, bytes = 0;
    while (last < out.size() && (last == first || bytes + out[last].bytes <= window)) {
      bytes += out[last++].bytes;
    }
    Completion completion{last - first};
    for (size_t i = first; i < last; ++i) {
      auto& request = out[i];
      auto* reply = new Reply(completion, request.bytes);
      XrdCl::XRootDStatus status;
      if (request.chunks.size() == 1) {
        auto& c = request.chunks.front();
        status = mFile->Read(c.offset, c.length, c.buffer, reply);
      } else {
        status = mFile->VectorRead(request.chunks, nullptr, reply);
      }
      if (!status.IsOK()) {
        delete reply;
        completion.done(status.ToString());
      }
    }
    auto error = completion.wait();
    if (!error.empty()) {
      throw IoError(mName, error);
    }
    calls += last - first;
    first = last;
  }
  mReadCalls.fetch_add(calls, std::memory_order_relaxed);
  mBytesRead.fetch_add(total, std::memory_order_relaxed);
}

} // namespace o2::aod::native
