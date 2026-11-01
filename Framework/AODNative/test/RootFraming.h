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

#ifndef O2_FRAMEWORK_AODNATIVE_TEST_ROOTFRAMING_H_
#define O2_FRAMEWORK_AODNATIVE_TEST_ROOTFRAMING_H_

// Helpers to build ROOT-framed compressed payloads and TBasket keys in memory,
// so that codecs and the decoder can be tested without ROOT or a file.

#include "AODNative/ByteSource.h"
#include "AODNative/Codecs.h"
#include "AODNative/Errors.h"
#include <cstring>
#include <lz4.h>
#include <lzma.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>
#include <zstd.h>

namespace o2::aod::native::test
{

enum struct Algo { ZSTD,
                   ZLIB,
                   LZMA,
                   LZ4,
                   None };

inline std::vector<uint8_t> compressChunk(Algo algo, std::span<const uint8_t> in)
{
  std::vector<uint8_t> out;
  switch (algo) {
    case Algo::ZSTD: {
      out.resize(ZSTD_compressBound(in.size()));
      out.resize(ZSTD_compress(out.data(), out.size(), in.data(), in.size(), 5));
    } break;
    case Algo::ZLIB: {
      uLongf n = compressBound(in.size());
      out.resize(n);
      compress2(out.data(), &n, in.data(), in.size(), 6);
      out.resize(n);
    } break;
    case Algo::LZMA: {
      out.resize(lzma_stream_buffer_bound(in.size()));
      size_t pos = 0;
      lzma_easy_buffer_encode(6, LZMA_CHECK_CRC32, nullptr, in.data(), in.size(), out.data(), &pos, out.size());
      out.resize(pos);
    } break;
    case Algo::LZ4: {
      std::vector<uint8_t> block(LZ4_compressBound(in.size()));
      block.resize(LZ4_compress_default((char const*)in.data(), (char*)block.data(), in.size(), block.size()));
      auto checksum = xxh64(block);
      for (int i = 7; i >= 0; --i) {
        out.push_back((checksum >> (8 * i)) & 0xff);
      }
      out.insert(out.end(), block.begin(), block.end());
    } break;
    case Algo::None:
      throw std::logic_error("no chunk for uncompressed data");
  }
  return out;
}

/// A ROOT compressed payload: chunks of at most @a chunkSize uncompressed bytes,
/// each with its 9-byte header.
inline std::vector<uint8_t> rootCompress(Algo algo, std::span<const uint8_t> in, size_t chunkSize = 0xffffff)
{
  static const char* tags[] = {"ZS", "ZL", "XZ", "L4"};
  std::vector<uint8_t> out;
  for (size_t pos = 0; pos < in.size(); pos += chunkSize) {
    auto piece = in.subspan(pos, std::min(chunkSize, in.size() - pos));
    auto c = compressChunk(algo, piece);
    out.push_back(tags[(int)algo][0]);
    out.push_back(tags[(int)algo][1]);
    out.push_back(algo == Algo::ZLIB ? 8 : 1);
    for (auto v : {c.size(), piece.size()}) {
      out.push_back(v & 0xff);
      out.push_back((v >> 8) & 0xff);
      out.push_back((v >> 16) & 0xff);
    }
    out.insert(out.end(), c.begin(), c.end());
  }
  return out;
}

struct BigEndianWriter {
  std::vector<uint8_t> bytes;
  template <typename T>
  void put(T v)
  {
    for (int i = sizeof(T) - 1; i >= 0; --i) {
      bytes.push_back((uint64_t(v) >> (8 * i)) & 0xff);
    }
  }
  void string(std::string const& s)
  {
    bytes.push_back(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};

/// A TBasket key as written on disk: key header, TBasket fields, then the
/// payload holding @a values followed by a fake entry-offset tail.
inline std::vector<uint8_t> makeBasket(Algo algo, std::span<const uint8_t> values, int32_t nevbuf, size_t tailBytes = 0, size_t chunkSize = 0xffffff)
{
  std::vector<uint8_t> raw(values.begin(), values.end());
  // The tail is poisoned: a decoder that reads past the value region sees it.
  raw.resize(values.size() + tailBytes, 0xEE);
  std::vector<uint8_t> payload = algo == Algo::None ? raw : rootCompress(algo, raw, chunkSize);

  BigEndianWriter h;
  auto build = [&](uint16_t keylen) {
    h.bytes.clear();
    h.put<uint32_t>(keylen + payload.size()); // fNbytes
    h.put<int16_t>(4);                        // key version
    h.put<uint32_t>(raw.size());              // fObjlen
    h.put<uint32_t>(0);                       // fDatime
    h.put<uint16_t>(keylen);
    h.put<uint16_t>(1);  // fCycle
    h.put<uint32_t>(0);  // fSeekKey
    h.put<uint32_t>(0);  // fSeekPdir
    h.string("TBasket"); // class, name, title
    h.string("fX");
    h.string("O2track");
    h.put<uint16_t>(3);                     // fVersion
    h.put<int32_t>(32000);                  // fBufferSize
    h.put<int32_t>(0);                      // fNevBufSize
    h.put<int32_t>(nevbuf);                 // fNevBuf
    h.put<int32_t>(keylen + values.size()); // fLast
    h.put<uint8_t>(tailBytes ? 1 : 2);      // flag
  };
  build(0);
  auto keylen = (uint16_t)h.bytes.size();
  build(keylen);
  h.bytes.insert(h.bytes.end(), payload.begin(), payload.end());
  return h.bytes;
}

/// A file made of concatenated in-memory buffers.
class MemoryByteSource : public ByteSource
{
 public:
  explicit MemoryByteSource(std::vector<uint8_t> data, bool mappable = true) : mData(std::move(data)), mMappable(mappable) {}
  uint64_t size() const override { return mData.size(); }
  void readInto(uint64_t off, size_t len, void* dst) override
  {
    if (off > mData.size() || len > mData.size() - off) {
      throw CorruptData("memory", off, "read past the end");
    }
    if (len == 0) {
      return;
    }
    std::memcpy(dst, mData.data() + off, len);
    ++mCalls;
    mBytes += len;
  }
  void readvInto(std::span<ReadRequest> requests) override
  {
    for (auto& r : requests) {
      readInto(r.offset, r.length, r.dst);
    }
  }
  std::span<const uint8_t> tryMap(uint64_t off, size_t len) override
  {
    if (!mMappable) {
      return {};
    }
    if (off > mData.size() || len > mData.size() - off) {
      throw CorruptData("memory", off, "map past the end");
    }
    return std::span<const uint8_t>(mData).subspan(off, len);
  }
  uint64_t bytesRead() const override { return mBytes; }
  uint64_t readCalls() const override { return mCalls; }
  std::string url() const override { return "memory"; }

 private:
  std::vector<uint8_t> mData;
  bool mMappable;
  uint64_t mBytes = 0, mCalls = 0;
};

} // namespace o2::aod::native::test

#endif // O2_FRAMEWORK_AODNATIVE_TEST_ROOTFRAMING_H_
