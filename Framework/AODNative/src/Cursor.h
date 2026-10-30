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

#ifndef O2_FRAMEWORK_AODNATIVE_CURSOR_H_
#define O2_FRAMEWORK_AODNATIVE_CURSOR_H_

#include "AODNative/Errors.h"
#include "Endian.h"
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>

namespace o2::aod::native
{

/// Bounds-checked big-endian reader over a buffer. Every read past the end
/// raises CorruptData, so malformed input can never cause an out-of-bounds read.
/// @a fileOffset is where the buffer starts in the file, used in error messages.
/// @a origin is added to positions to get ROOT's buffer displacement, which is
/// what object references are keyed on (fKeylen for an object read from a key).
class Cursor
{
 public:
  Cursor(std::span<const uint8_t> buf, std::string const& file, uint64_t fileOffset = 0, int64_t origin = 0)
    : mBuf(buf), mFile(file), mFileOffset(fileOffset), mOrigin(origin)
  {
  }

  size_t pos() const { return mPos; }
  int64_t displacement() const { return (int64_t)mPos + mOrigin; }
  size_t remaining() const { return mBuf.size() - mPos; }
  std::span<const uint8_t> buffer() const { return mBuf; }

  void seek(size_t p)
  {
    if (p > mBuf.size()) {
      fail("seek past the end of the buffer");
    }
    mPos = p;
  }
  void skip(size_t n) { seek(need(n) + n); }
  void back(size_t n)
  {
    if (n > mPos) {
      fail("seek before the start of the buffer");
    }
    mPos -= n;
  }

  std::span<const uint8_t> bytes(size_t n)
  {
    auto p = need(n);
    mPos += n;
    return mBuf.subspan(p, n);
  }

  uint8_t u8() { return bytes(1)[0]; }
  uint16_t u16() { return load<uint16_t>(); }
  uint32_t u32() { return load<uint32_t>(); }
  uint64_t u64() { return load<uint64_t>(); }
  int16_t i16() { return (int16_t)u16(); }
  int32_t i32() { return (int32_t)u32(); }
  int64_t i64() { return (int64_t)u64(); }
  double f64() { return std::bit_cast<double>(u64()); }

  /// TString: 1-byte length, or 0xFF followed by a 4-byte length.
  std::string string()
  {
    uint32_t n = u8();
    if (n == 255) {
      n = u32();
    }
    auto b = bytes(n);
    return {(char const*)b.data(), b.size()};
  }

  /// NUL-terminated string, as used for class names after kNewClassTag.
  std::string cstring()
  {
    auto start = mPos;
    while (u8() != 0) {
    }
    return {(char const*)mBuf.data() + start, mPos - start - 1};
  }

  [[noreturn]] void fail(std::string const& what) const
  {
    throw CorruptData(mFile, mFileOffset + mPos, what);
  }

  std::string const& file() const { return mFile; }

 private:
  size_t need(size_t n) const
  {
    if (n > mBuf.size() - mPos) {
      fail("read of " + std::to_string(n) + " bytes past the end of a " + std::to_string(mBuf.size()) + " byte buffer");
    }
    return mPos;
  }

  template <typename T>
  T load()
  {
    T v;
    std::memcpy(&v, bytes(sizeof(T)).data(), sizeof(T));
    if constexpr (std::endian::native == std::endian::little) {
      v = bswap(v);
    }
    return v;
  }

  std::span<const uint8_t> mBuf;
  std::string const& mFile;
  uint64_t mFileOffset;
  int64_t mOrigin;
  size_t mPos = 0;
};

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_CURSOR_H_
