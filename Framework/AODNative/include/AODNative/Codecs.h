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

#ifndef O2_FRAMEWORK_AODNATIVE_CODECS_H_
#define O2_FRAMEWORK_AODNATIVE_CODECS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace o2::aod::native
{

/// Decompresses the chunks of a ROOT compressed payload incrementally into
/// caller-provided memory. Each read() continues where the previous one
/// stopped, so a caller can stop at the end of the value region without
/// producing (or holding) the rest. Only LZ4 needs scratch memory, and only
/// when a read ends in the middle of an LZ4 chunk.
class PayloadDecompressor
{
 public:
  PayloadDecompressor(std::span<const uint8_t> src, uint32_t objLen, std::string const& file, uint64_t fileOffset);
  ~PayloadDecompressor();
  PayloadDecompressor(PayloadDecompressor const&) = delete;
  PayloadDecompressor& operator=(PayloadDecompressor const&) = delete;

  /// Writes exactly @a n bytes to @a dst. Throws CorruptData if the payload
  /// ends first or contradicts its own headers.
  void read(uint8_t* dst, size_t n);
  /// Writes the first @a n bytes of the payload to @a dst, as the only read.
  /// Since the output is contiguous and stays in place, zstd decodes into it
  /// without a window buffer of its own.
  void readPrefix(uint8_t* dst, size_t n);
  /// Bytes produced so far.
  size_t produced() const { return mProduced; }

 private:
  struct Codec;
  void nextChunk();
  size_t produceFromChunk(uint8_t* dst, size_t n);
  size_t zstdPrefix(uint8_t* dst, size_t n);

  std::span<const uint8_t> mSrc;
  uint32_t mObjLen;
  std::string const& mFile;
  uint64_t mFileOffset;
  size_t mSrcPos = 0;
  size_t mProduced = 0;
  // Current chunk
  char mTag[2] = {0, 0};
  std::span<const uint8_t> mChunkIn;
  size_t mChunkOut = 0;  // uncompressed size of the current chunk
  size_t mChunkDone = 0; // produced from the current chunk
  std::unique_ptr<Codec> mCodec;
  std::vector<uint8_t> mScratch; // LZ4 chunk decoded ahead of a partial read, or the last zstd blocks of a prefix
  bool mPrefixOnly = false;
};

/// Decompresses all chunks of a payload, writing at most maxOut bytes to dst.
/// Returns the number of bytes written, which is min(objLen, maxOut).
size_t decompressBounded(std::span<const uint8_t> src, uint32_t objLen, uint8_t* dst, size_t maxOut,
                         std::string const& file = "<memory>", uint64_t fileOffset = 0);

/// XXH64 with seed 0, as used for the LZ4 chunk checksum.
uint64_t xxh64(std::span<const uint8_t> data);

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_CODECS_H_
