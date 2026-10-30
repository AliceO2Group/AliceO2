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

#include "AODNative/Codecs.h"
#include "AODNative/Errors.h"
#include "Endian.h"
#include <bit>
#include <cstring>
#include <lz4.h>
#include <lzma.h>
#include <stdexcept>
#include <zlib.h>
// For the buffer-less decompression API, exported by the shared library too.
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

namespace o2::aod::native
{

namespace
{
constexpr size_t kChunkHeaderSize = 9;
constexpr size_t kLZ4ChecksumSize = 8;

uint32_t le24(uint8_t const* p) { return p[0] | (p[1] << 8) | (p[2] << 16); }
} // namespace

// One decoder state per codec, created on first use and reset for each chunk.
struct PayloadDecompressor::Codec {
  ZSTD_DCtx* zstd = nullptr;
  z_stream zlib{};
  bool zlibInit = false;
  lzma_stream lzma = LZMA_STREAM_INIT;
  bool lzmaInit = false;
  bool lz4Partial = false; // the current LZ4 chunk was decoded partially, straight into the destination

  ~Codec()
  {
    if (zstd) {
      ZSTD_freeDCtx(zstd);
    }
    if (zlibInit) {
      inflateEnd(&zlib);
    }
    if (lzmaInit) {
      lzma_end(&lzma);
    }
  }
};

PayloadDecompressor::PayloadDecompressor(std::span<const uint8_t> src, uint32_t objLen, std::string const& file, uint64_t fileOffset)
  : mSrc(src), mObjLen(objLen), mFile(file), mFileOffset(fileOffset), mCodec(std::make_unique<Codec>())
{
}

PayloadDecompressor::~PayloadDecompressor() = default;

void PayloadDecompressor::nextChunk()
{
  auto fail = [this](std::string const& what) { throw CorruptData(mFile, mFileOffset + mSrcPos, what); };
  if (mSrc.size() - mSrcPos < kChunkHeaderSize) {
    fail("compressed payload ends after " + std::to_string(mProduced) + " of " + std::to_string(mObjLen) + " bytes");
  }
  auto* h = mSrc.data() + mSrcPos;
  mTag[0] = (char)h[0];
  mTag[1] = (char)h[1];
  size_t cSize = le24(h + 3);
  mChunkOut = le24(h + 6);
  mChunkDone = 0;
  if (cSize > mSrc.size() - mSrcPos - kChunkHeaderSize) {
    fail("compressed chunk of " + std::to_string(cSize) + " bytes past the end of the payload");
  }
  if (mChunkOut == 0 || mChunkOut > mObjLen - mProduced) {
    fail("chunk of " + std::to_string(mChunkOut) + " uncompressed bytes does not fit the object length");
  }
  mChunkIn = mSrc.subspan(mSrcPos + kChunkHeaderSize, cSize);
  auto& codec = *mCodec;
  mScratch.clear();
  if (mTag[0] == 'Z' && mTag[1] == 'S') {
    if (!codec.zstd) {
      codec.zstd = ZSTD_createDCtx();
    }
    ZSTD_DCtx_reset(codec.zstd, ZSTD_reset_session_only);
  } else if (mTag[0] == 'Z' && mTag[1] == 'L') {
    if (codec.zlibInit) {
      inflateReset(&codec.zlib);
    } else {
      if (inflateInit(&codec.zlib) != Z_OK) {
        fail("cannot initialise zlib");
      }
      codec.zlibInit = true;
    }
    codec.zlib.next_in = (Bytef*)mChunkIn.data();
    codec.zlib.avail_in = mChunkIn.size();
  } else if (mTag[0] == 'X' && mTag[1] == 'Z') {
    if (codec.lzmaInit) {
      lzma_end(&codec.lzma);
    }
    codec.lzma = LZMA_STREAM_INIT;
    if (lzma_stream_decoder(&codec.lzma, UINT64_MAX, 0) != LZMA_OK) {
      fail("cannot initialise liblzma");
    }
    codec.lzmaInit = true;
    codec.lzma.next_in = mChunkIn.data();
    codec.lzma.avail_in = mChunkIn.size();
  } else if (mTag[0] == 'L' && mTag[1] == '4') {
    if (cSize < kLZ4ChecksumSize) {
      fail("LZ4 chunk too small for its checksum");
    }
    uint64_t expected = 0;
    for (size_t i = 0; i < kLZ4ChecksumSize; ++i) {
      expected = (expected << 8) | mChunkIn[i];
    }
    mChunkIn = mChunkIn.subspan(kLZ4ChecksumSize);
    if (xxh64(mChunkIn) != expected) {
      fail("LZ4 checksum mismatch");
    }
    codec.lz4Partial = false;
  } else {
    throw UnsupportedLayout(std::string("compression algorithm \"") + mTag[0] + mTag[1] + "\" in " + mFile);
  }
  mSrcPos += kChunkHeaderSize + cSize;
}

size_t PayloadDecompressor::produceFromChunk(uint8_t* dst, size_t n)
{
  auto fail = [this](std::string const& what) { throw CorruptData(mFile, mFileOffset + mSrcPos, what); };
  auto& codec = *mCodec;
  bool wholeChunk = mChunkDone == 0 && n == mChunkOut;
  if (mTag[0] == 'Z' && mTag[1] == 'S') {
    if (wholeChunk) {
      auto r = ZSTD_decompressDCtx(codec.zstd, dst, n, mChunkIn.data(), mChunkIn.size());
      if (ZSTD_isError(r) || r != n) {
        fail(std::string("zstd: ") + (ZSTD_isError(r) ? ZSTD_getErrorName(r) : "size mismatch"));
      }
      return n;
    }
    if (mPrefixOnly && mChunkDone == 0) {
      return zstdPrefix(dst, n);
    }
    // Resume where the previous read stopped: the input position is kept in mChunkIn.
    // The streaming API keeps its own window buffer, sized by the writer's window.
    ZSTD_outBuffer out{dst, n, 0};
    while (out.pos < out.size) {
      ZSTD_inBuffer in{mChunkIn.data(), mChunkIn.size(), 0};
      auto r = ZSTD_decompressStream(codec.zstd, &out, &in);
      mChunkIn = mChunkIn.subspan(in.pos);
      if (ZSTD_isError(r)) {
        fail(std::string("zstd: ") + ZSTD_getErrorName(r));
      }
      if (out.pos < out.size && (r == 0 || (in.pos == 0 && mChunkIn.empty()))) {
        fail("zstd chunk ended early");
      }
    }
    return n;
  }
  if (mTag[0] == 'Z' && mTag[1] == 'L') {
    auto& z = codec.zlib;
    z.next_out = dst;
    z.avail_out = n;
    while (z.avail_out > 0) {
      auto r = inflate(&z, Z_NO_FLUSH);
      if (r == Z_STREAM_END && z.avail_out > 0) {
        fail("zlib chunk ended early");
      }
      if (r != Z_OK && r != Z_STREAM_END) {
        fail(std::string("zlib: ") + (z.msg ? z.msg : std::to_string(r)));
      }
    }
    return n;
  }
  if (mTag[0] == 'X' && mTag[1] == 'Z') {
    auto& s = codec.lzma;
    s.next_out = dst;
    s.avail_out = n;
    while (s.avail_out > 0) {
      auto r = lzma_code(&s, LZMA_RUN);
      if (r == LZMA_STREAM_END && s.avail_out > 0) {
        fail("xz chunk ended early");
      }
      if (r != LZMA_OK && r != LZMA_STREAM_END) {
        fail("liblzma error " + std::to_string(r));
      }
    }
    return n;
  }
  // LZ4 block format cannot be resumed. Decode partially straight into the
  // destination when this read starts the chunk, which is the common case of
  // stopping at the end of the value region. Should a later read need the
  // rest, decode the whole chunk into scratch memory.
  if (wholeChunk) {
    // A whole chunk must use all of its input and produce exactly its size.
    auto r = LZ4_decompress_safe((char const*)mChunkIn.data(), (char*)dst, mChunkIn.size(), n);
    if (r < 0 || (size_t)r != n) {
      fail("LZ4 chunk size mismatch");
    }
    return n;
  }
  if (mScratch.empty() && mChunkDone == 0) {
    auto r = LZ4_decompress_safe_partial((char const*)mChunkIn.data(), (char*)dst, mChunkIn.size(), n, n);
    if (r < 0 || (size_t)r != n) {
      fail("LZ4 decompression failed");
    }
    codec.lz4Partial = true;
    return n;
  }
  if (mScratch.empty()) {
    mScratch.resize(mChunkOut);
    auto r = LZ4_decompress_safe((char const*)mChunkIn.data(), (char*)mScratch.data(), mChunkIn.size(), mChunkOut);
    if (r < 0 || (size_t)r != mChunkOut) {
      fail("LZ4 decompression failed");
    }
  }
  std::memcpy(dst, mScratch.data() + mChunkDone, n);
  return n;
}

// Decodes the first @a n bytes of a zstd chunk straight into @a dst, with the
// buffer-less API: the destination itself is zstd's history, so no window
// buffer is allocated. Blocks are written in place while a whole block (at most
// ZSTD_BLOCKSIZE_MAX) still fits; the last ones go to a fixed scratch, contiguous
// with each other, and only the needed bytes are copied out.
size_t PayloadDecompressor::zstdPrefix(uint8_t* dst, size_t n)
{
  auto fail = [this](std::string const& what) { throw CorruptData(mFile, mFileOffset + mSrcPos, what); };
  auto* dctx = mCodec->zstd;
  if (ZSTD_isError(ZSTD_decompressBegin(dctx))) {
    fail("cannot initialise zstd");
  }
  auto in = mChunkIn;
  size_t written = 0;
  bool inScratch = false;
  size_t scratchUsed = 0;
  while (written < n) {
    auto need = ZSTD_nextSrcSizeToDecompress(dctx);
    if (need == 0) {
      fail("zstd chunk ended early");
    }
    if (need > in.size()) {
      fail("truncated zstd chunk");
    }
    size_t r;
    if (!inScratch && n - written >= ZSTD_BLOCKSIZE_MAX) {
      r = ZSTD_decompressContinue(dctx, dst + written, n - written, in.data(), need);
      if (!ZSTD_isError(r)) {
        written += r;
      }
    } else {
      if (!inScratch) {
        mScratch.resize(2 * ZSTD_BLOCKSIZE_MAX);
        inScratch = true;
      }
      r = ZSTD_decompressContinue(dctx, mScratch.data() + scratchUsed, mScratch.size() - scratchUsed, in.data(), need);
      if (!ZSTD_isError(r)) {
        scratchUsed += r;
        if (written + scratchUsed >= n) {
          std::memcpy(dst + written, mScratch.data(), n - written);
          written = n;
        }
      }
    }
    if (ZSTD_isError(r)) {
      fail(std::string("zstd: ") + ZSTD_getErrorName(r));
    }
    in = in.subspan(need);
  }
  mScratch.clear();
  mScratch.shrink_to_fit();
  return n;
}

void PayloadDecompressor::readPrefix(uint8_t* dst, size_t n)
{
  if (mProduced != 0) {
    throw std::logic_error("readPrefix must be the first read of a payload");
  }
  mPrefixOnly = true;
  read(dst, n);
}

void PayloadDecompressor::read(uint8_t* dst, size_t n)
{
  if (mPrefixOnly && mProduced != 0) {
    throw std::logic_error("no read can follow readPrefix");
  }
  if (n > mObjLen - mProduced) {
    throw CorruptData(mFile, mFileOffset, "read of " + std::to_string(n) + " bytes past the object length " + std::to_string(mObjLen));
  }
  while (n > 0) {
    if (mChunkDone == mChunkOut) {
      nextChunk();
    }
    auto k = std::min(n, mChunkOut - mChunkDone);
    produceFromChunk(dst, k);
    dst += k;
    n -= k;
    mChunkDone += k;
    mProduced += k;
  }
}

size_t decompressBounded(std::span<const uint8_t> src, uint32_t objLen, uint8_t* dst, size_t maxOut,
                         std::string const& file, uint64_t fileOffset)
{
  auto n = std::min<size_t>(objLen, maxOut);
  PayloadDecompressor d{src, objLen, file, fileOffset};
  d.readPrefix(dst, n);
  return n;
}

uint64_t xxh64(std::span<const uint8_t> data)
{
  constexpr uint64_t P1 = 11400714785074694791ULL;
  constexpr uint64_t P2 = 14029467366897019727ULL;
  constexpr uint64_t P3 = 1609587929392839161ULL;
  constexpr uint64_t P4 = 9650029242287828579ULL;
  constexpr uint64_t P5 = 2870177450012600261ULL;
  auto read64 = [](uint8_t const* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return std::endian::native == std::endian::little ? v : bswap(v);
  };
  auto read32 = [](uint8_t const* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return std::endian::native == std::endian::little ? v : bswap(v);
  };
  auto round = [](uint64_t acc, uint64_t input) {
    acc += input * P2;
    return std::rotl(acc, 31) * P1;
  };
  auto merge = [&round](uint64_t acc, uint64_t v) {
    acc ^= round(0, v);
    return acc * P1 + P4;
  };

  auto const* p = data.data();
  auto const* end = p + data.size();
  uint64_t h;
  if (data.size() >= 32) {
    uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = -P1;
    for (; end - p >= 32; p += 32) {
      v1 = round(v1, read64(p));
      v2 = round(v2, read64(p + 8));
      v3 = round(v3, read64(p + 16));
      v4 = round(v4, read64(p + 24));
    }
    h = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
    h = merge(merge(merge(merge(h, v1), v2), v3), v4);
  } else {
    h = P5;
  }
  h += data.size();
  for (; end - p >= 8; p += 8) {
    h ^= round(0, read64(p));
    h = std::rotl(h, 27) * P1 + P4;
  }
  if (end - p >= 4) {
    h ^= (uint64_t)read32(p) * P1;
    h = std::rotl(h, 23) * P2 + P3;
    p += 4;
  }
  for (; p < end; ++p) {
    h ^= *p * P5;
    h = std::rotl(h, 11) * P1;
  }
  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

} // namespace o2::aod::native
