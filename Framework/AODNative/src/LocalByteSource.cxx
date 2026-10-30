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

#include "LocalByteSource.h"
#include "AODNative/Errors.h"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace o2::aod::native
{

namespace
{
size_t pageSize()
{
  static size_t size = sysconf(_SC_PAGESIZE);
  return size;
}
} // namespace

LocalByteSource::LocalByteSource(std::string path, Options const& options)
  : mPath(std::move(path))
{
  mFd = ::open(mPath.c_str(), O_RDONLY | O_CLOEXEC);
  if (mFd < 0) {
    throw IoError(mPath, std::strerror(errno));
  }
  struct stat st;
  if (fstat(mFd, &st) != 0) {
    auto err = errno;
    ::close(mFd);
    throw IoError(mPath, std::strerror(err));
  }
  mSize = st.st_size;
  if (options.localIO == Options::LocalIO::MMap && sizeof(void*) == 8 && mSize > 0) {
    void* p = mmap(nullptr, mSize, PROT_READ, MAP_PRIVATE, mFd, 0);
    if (p != MAP_FAILED) {
      mMap = (uint8_t const*)p;
    }
  }
}

LocalByteSource::~LocalByteSource()
{
  if (mMap) {
    munmap((void*)mMap, mSize);
  }
  ::close(mFd);
}

void LocalByteSource::checkRange(uint64_t off, size_t len) const
{
  if (off > mSize || len > mSize - off) {
    throw CorruptData(mPath, off, "read of " + std::to_string(len) + " bytes past the end of a " + std::to_string(mSize) + " byte file");
  }
}

void LocalByteSource::readInto(uint64_t off, size_t len, void* dst)
{
  checkRange(off, len);
  if (len == 0) {
    return;
  }
  mReadCalls.fetch_add(1, std::memory_order_relaxed);
  mBytesRead.fetch_add(len, std::memory_order_relaxed);
  if (mMap) {
    std::memcpy(dst, mMap + off, len);
    return;
  }
  auto* out = (uint8_t*)dst;
  while (len > 0) {
    auto n = ::pread(mFd, out, len, off);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      throw IoError(mPath, n == 0 ? "unexpected end of file" : std::strerror(errno));
    }
    out += n;
    off += n;
    len -= n;
  }
}

void LocalByteSource::readvInto(std::span<ReadRequest> requests)
{
  for (auto& r : requests) {
    readInto(r.offset, r.length, r.dst);
  }
}

std::span<const uint8_t> LocalByteSource::tryMap(uint64_t off, size_t len)
{
  if (!mMap) {
    return {};
  }
  checkRange(off, len);
  // Mapped bytes are accounted as read, so that monitoring is the same for both modes.
  mReadCalls.fetch_add(1, std::memory_order_relaxed);
  mBytesRead.fetch_add(len, std::memory_order_relaxed);
  return {mMap + off, len};
}

void LocalByteSource::advise(uint64_t off, size_t len, int advice)
{
  if (!mMap || off >= mSize) {
    return;
  }
  len = std::min<uint64_t>(len, mSize - off);
  auto start = off & ~(uint64_t)(pageSize() - 1);
  madvise((void*)(mMap + start), len + (off - start), advice);
}

void LocalByteSource::willNeed(uint64_t off, size_t len) { advise(off, len, MADV_WILLNEED); }
void LocalByteSource::dontNeed(uint64_t off, size_t len) { advise(off, len, MADV_DONTNEED); }

} // namespace o2::aod::native
