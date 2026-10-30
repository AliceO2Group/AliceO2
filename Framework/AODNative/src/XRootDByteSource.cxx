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
#include <XrdCl/XrdClXRootDResponses.hh>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace o2::aod::native
{

namespace
{
// XrdCl's defaults for kXR_readv: chunks of at most 2097136 bytes, at most 1024
// of them per request. Larger ranges are read with plain reads, in pieces small
// enough for the uint32_t size of XrdCl::File::Read.
constexpr size_t kMaxVectorChunk = 2097136;
constexpr size_t kMaxVectorChunks = 1024;
constexpr size_t kMaxPlainRead = 64 * 1024 * 1024;

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

struct PlainRead {
  uint64_t offset;
  uint32_t length;
  void* dst;
};

struct VectorRead {
  XrdCl::ChunkList chunks;
  size_t bytes = 0;
};
} // namespace

XRootDByteSource::XRootDByteSource(std::string url, Options const& options)
  : mName(url.substr(0, url.find('?'))), mFile(std::make_unique<XrdCl::File>())
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
  std::vector<PlainRead> plain;
  std::vector<VectorRead> vectors;
  size_t total = 0;
  for (auto const& r : requests) {
    checkRange(r.offset, r.length);
    total += r.length;
    if (r.length > kMaxVectorChunk) {
      for (size_t done = 0; done < r.length; done += kMaxPlainRead) {
        auto n = std::min(kMaxPlainRead, r.length - done);
        plain.push_back({r.offset + done, (uint32_t)n, (uint8_t*)r.dst + done});
      }
      continue;
    }
    if (r.length == 0) {
      continue;
    }
    if (vectors.empty() || vectors.back().chunks.size() == kMaxVectorChunks) {
      vectors.emplace_back();
    }
    vectors.back().chunks.emplace_back(r.offset, (uint32_t)r.length, r.dst);
    vectors.back().bytes += r.length;
  }
  // A vector of one chunk is a plain read.
  for (auto& v : vectors) {
    if (v.chunks.size() == 1) {
      auto& c = v.chunks.front();
      plain.push_back({c.offset, c.length, c.buffer});
      v.chunks.clear();
    }
  }
  std::erase_if(vectors, [](VectorRead const& v) { return v.chunks.empty(); });
  if (plain.empty() && vectors.empty()) {
    return;
  }

  Completion completion{plain.size() + vectors.size()};
  for (auto const& p : plain) {
    auto* reply = new Reply(completion, p.length);
    auto status = mFile->Read(p.offset, p.length, p.dst, reply);
    if (!status.IsOK()) {
      delete reply;
      completion.done(status.ToString());
    }
  }
  for (auto const& v : vectors) {
    auto* reply = new Reply(completion, v.bytes);
    auto status = mFile->VectorRead(v.chunks, nullptr, reply);
    if (!status.IsOK()) {
      delete reply;
      completion.done(status.ToString());
    }
  }
  auto error = completion.wait();
  if (!error.empty()) {
    throw IoError(mName, error);
  }
  mReadCalls.fetch_add(plain.size() + vectors.size(), std::memory_order_relaxed);
  mBytesRead.fetch_add(total, std::memory_order_relaxed);
}

} // namespace o2::aod::native
