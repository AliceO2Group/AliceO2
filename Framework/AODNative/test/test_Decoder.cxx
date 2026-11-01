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

#include "AODNative/Decoder.h"
#include "AODNative/Errors.h"
#include "RootFraming.h"
#include <catch_amalgamated.hpp>
#include <bit>

using namespace o2::aod::native;
using namespace o2::aod::native::test;

namespace
{
// Serializes values big-endian, as ROOT stores them.
template <typename T>
std::vector<uint8_t> bigEndian(std::vector<T> const& values)
{
  BigEndianWriter w;
  for (auto v : values) {
    w.put(std::bit_cast<std::conditional_t<sizeof(T) == 4, uint32_t, std::conditional_t<sizeof(T) == 8, uint64_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint8_t>>>>(v));
  }
  return w.bytes;
}

/// Lays out baskets one after the other in a file, and describes them in a BranchInfo.
struct BranchBuilder {
  std::vector<uint8_t> file = std::vector<uint8_t>(100, 0); // some leading bytes, so seeks are non-zero
  BranchInfo branch;

  BranchBuilder(char leafClass, int fixedLen = 1)
  {
    branch.name = "fX";
    branch.leafClass = leafClass;
    branch.isUnsigned = false;
    branch.fixedLen = fixedLen;
    branch.entries = branch.totBytes = branch.zipBytes = 0;
  }

  void add(Algo algo, std::vector<uint8_t> const& values, int32_t nEntries, size_t tail = 0, size_t chunkSize = 0xffffff)
  {
    auto basket = makeBasket(algo, values, nEntries, tail, chunkSize);
    branch.baskets.push_back(BasketRef{file.size(), (uint32_t)basket.size(), branch.entries, nEntries, false, {}});
    file.insert(file.end(), basket.begin(), basket.end());
    branch.entries += nEntries;
  }

  void addEmbedded(std::vector<uint8_t> const& values, int32_t nEntries)
  {
    branch.baskets.push_back(BasketRef{0, 0, branch.entries, nEntries, true, values});
    branch.entries += nEntries;
  }
};
} // namespace

TEST_CASE("byteswapInPlace")
{
  std::vector<uint8_t> v{1, 2, 3, 4, 5, 6, 7, 8};
  byteswapInPlace(v.data(), 2, 4);
  CHECK(v == std::vector<uint8_t>{4, 3, 2, 1, 8, 7, 6, 5});
  byteswapInPlace(v.data(), 1, 8);
  CHECK(v == std::vector<uint8_t>{5, 6, 7, 8, 1, 2, 3, 4});
  byteswapInPlace(v.data(), 4, 2);
  CHECK(v == std::vector<uint8_t>{6, 5, 8, 7, 2, 1, 4, 3});
}

TEST_CASE("float values across codecs, uncompressed and embedded baskets")
{
  BranchBuilder b{'F'};
  std::vector<float> expected;
  int n = 0;
  for (auto algo : {Algo::ZSTD, Algo::None, Algo::ZLIB, Algo::LZMA, Algo::LZ4}) {
    std::vector<float> values;
    for (int i = 0; i < 1000; ++i) {
      values.push_back(n++ * 0.5f);
    }
    // Entry-offset tail after the values must never reach the destination.
    b.add(algo, bigEndian(values), values.size(), 4004);
    expected.insert(expected.end(), values.begin(), values.end());
  }
  std::vector<float> tail{-1.f, -2.f, -3.f};
  b.addEmbedded(bigEndian(tail), tail.size());
  expected.insert(expected.end(), tail.begin(), tail.end());

  for (bool mappable : {true, false}) {
    for (int threads : {1, 4}) {
      MemoryByteSource source{b.file, mappable};
      Options options;
      options.threads = threads;
      BranchDecoder decoder{source, options};
      std::vector<float> out(expected.size(), 42.f);
      decoder.decodeValues(b.branch, (uint8_t*)out.data(), out.size() * sizeof(float));
      CHECK(out == expected);
    }
  }
}

TEST_CASE("decoding stops exactly at the value region for every codec")
{
  // A poisoned entry-offset tail after the values: nothing past fLast - fKeylen
  // may reach the destination, which is followed by guard bytes. Sizes span
  // several zstd blocks and compression chunks.
  for (auto algo : {Algo::ZSTD, Algo::ZLIB, Algo::LZMA, Algo::LZ4, Algo::None}) {
    for (size_t chunkSize : {size_t(100000), size_t(0xffffff)}) {
      INFO("algorithm " << (int)algo << ", chunk size " << chunkSize);
      BranchBuilder b{'D'};
      std::vector<double> values(70001);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = i * 0.25 - 7;
      }
      b.add(algo, bigEndian(values), values.size(), 8 * values.size() + 4, chunkSize);
      MemoryByteSource source{b.file};
      BranchDecoder decoder{source, Options{}};
      std::vector<double> out(values.size() + 16, 1e300);
      decoder.decodeValues(b.branch, (uint8_t*)out.data(), values.size() * sizeof(double));
      CHECK(std::equal(values.begin(), values.end(), out.begin()));
      CHECK(std::all_of(out.begin() + values.size(), out.end(), [](double v) { return v == 1e300; }));
    }
  }
}

TEST_CASE("fixed-size arrays and wrong destination sizes")
{
  BranchBuilder b{'S', 3};
  std::vector<int16_t> values{1, 2, 3, -4, -5, -6};
  b.add(Algo::ZSTD, bigEndian(values), 2);
  MemoryByteSource source{b.file};
  BranchDecoder decoder{source, Options{}};
  std::vector<int16_t> out(6);
  decoder.decodeValues(b.branch, (uint8_t*)out.data(), 12);
  CHECK(out == values);
  CHECK_THROWS_AS(decoder.decodeValues(b.branch, (uint8_t*)out.data(), 10), FormatError);
}

TEST_CASE("booleans are packed across basket and window boundaries")
{
  // Basket sizes chosen so that boundaries fall inside bitmap bytes, and one
  // basket is larger than the 64 KiB window.
  BranchBuilder b{'O'};
  std::vector<bool> expected;
  for (int size : {3, 13, (int)BranchDecoder::kBooleanWindow + 5, 1, 7}) {
    std::vector<uint8_t> bytes;
    for (int i = 0; i < size; ++i) {
      bool v = ((expected.size() * 7) % 3) == 0;
      expected.push_back(v);
      // Any non-zero byte is true.
      bytes.push_back(v ? (i % 2 ? 1 : 0x80) : 0);
    }
    b.add(Algo::ZSTD, bytes, size, 8, 30000);
  }
  MemoryByteSource source{b.file};
  BranchDecoder decoder{source, Options{}};
  std::vector<uint8_t> bitmap((expected.size() + 7) / 8, 0xFF);
  decoder.decodeBooleans(b.branch, bitmap.data(), bitmap.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    INFO("value " << i);
    CHECK(((bitmap[i / 8] >> (i % 8)) & 1) == expected[i]);
  }
  // Padding bits are zero.
  for (size_t i = expected.size(); i < bitmap.size() * 8; ++i) {
    CHECK(((bitmap[i / 8] >> (i % 8)) & 1) == 0);
  }
}

TEST_CASE("variable-length arrays")
{
  BranchBuilder sizes{'I'};
  sizes.branch.name = "fY_size";
  std::vector<int32_t> s1{2, 0, 3}, s2{1};
  sizes.add(Algo::ZLIB, bigEndian(s1), 3);
  sizes.add(Algo::None, bigEndian(s2), 1);

  // Values baskets do not need to align with size baskets.
  BranchBuilder values{'D'};
  values.branch.name = "fY";
  values.branch.countBranch = "fY_size";
  std::vector<double> v1{1.5, 2.5}, v2{3.5, 4.5, 5.5, 6.5};
  values.add(Algo::LZ4, bigEndian(v1), 2);
  values.add(Algo::ZSTD, bigEndian(v2), 2);

  std::vector<uint8_t> file = sizes.file;
  // Place the values baskets after the size baskets.
  auto shift = file.size() - 100;
  file.insert(file.end(), values.file.begin() + 100, values.file.end());
  for (auto& basket : values.branch.baskets) {
    basket.seek += shift;
  }

  MemoryByteSource source{file};
  BranchDecoder decoder{source, Options{}};
  auto offsets = decoder.decodeOffsets(sizes.branch);
  CHECK(offsets == std::vector<int32_t>{0, 2, 2, 5, 6});
  std::vector<double> out(6);
  decoder.decodeVLAValues(values.branch, offsets, (uint8_t*)out.data(), out.size() * sizeof(double));
  CHECK(out == std::vector<double>{1.5, 2.5, 3.5, 4.5, 5.5, 6.5});
}

TEST_CASE("inconsistent baskets raise CorruptData")
{
  BranchBuilder b{'I'};
  std::vector<int32_t> values{1, 2, 3};
  b.add(Algo::ZSTD, bigEndian(values), 3);
  MemoryByteSource source{b.file};
  BranchDecoder decoder{source, Options{}};
  std::vector<int32_t> out(3);

  SECTION("entry count in the header differs from the TBranch")
  {
    auto branch = b.branch;
    branch.baskets[0].nEntries = 2;
    branch.entries = 2;
    CHECK_THROWS_AS(decoder.decodeValues(branch, (uint8_t*)out.data(), 8), CorruptData);
  }
  SECTION("baskets do not cover the entries")
  {
    auto branch = b.branch;
    branch.entries = 4;
    std::vector<int32_t> big(4);
    CHECK_THROWS_AS(decoder.decodeValues(branch, (uint8_t*)big.data(), 16), CorruptData);
  }
  SECTION("basket size differs from the TBranch")
  {
    auto branch = b.branch;
    branch.baskets[0].nbytes -= 1;
    CHECK_THROWS_AS(decoder.decodeValues(branch, (uint8_t*)out.data(), 12), CorruptData);
  }
  SECTION("basket location past the end of the file")
  {
    auto branch = b.branch;
    branch.baskets[0].seek = b.file.size();
    MemoryByteSource unmapped{b.file, false};
    BranchDecoder d2{unmapped, Options{}};
    CHECK_THROWS_AS(d2.decodeValues(branch, (uint8_t*)out.data(), 12), CorruptData);
  }
}
