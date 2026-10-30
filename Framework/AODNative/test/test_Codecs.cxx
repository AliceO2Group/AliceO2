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
#include "RootFraming.h"
#include <catch_amalgamated.hpp>
#include <numeric>
#include <random>

using namespace o2::aod::native;
using namespace o2::aod::native::test;

namespace
{
std::vector<uint8_t> sampleData(size_t n)
{
  // Compressible, but not trivially: a ramp with some noise.
  std::vector<uint8_t> data(n);
  std::mt19937 rng(42);
  for (size_t i = 0; i < n; ++i) {
    data[i] = (uint8_t)(i / 7 + (rng() % 4));
  }
  return data;
}

auto const allAlgos = {Algo::ZSTD, Algo::ZLIB, Algo::LZMA, Algo::LZ4};
} // namespace

TEST_CASE("xxh64 matches the reference implementation")
{
  std::vector<uint8_t> big(768);
  for (size_t i = 0; i < big.size(); ++i) {
    big[i] = i % 256;
  }
  CHECK(xxh64({}) == 0xef46db3751d8e999ULL);
  CHECK(xxh64(std::vector<uint8_t>{'a'}) == 0xd24ec4f1a98c6e5bULL);
  CHECK(xxh64(std::vector<uint8_t>{'a', 'b', 'c'}) == 0x44bc2cf5ad770999ULL);
  CHECK(xxh64(big) == 0x8e03c838c596036fULL);
}

TEST_CASE("each codec round-trips a single chunk")
{
  auto data = sampleData(100000);
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data);
    std::vector<uint8_t> out(data.size());
    REQUIRE(decompressBounded(payload, data.size(), out.data(), out.size()) == data.size());
    CHECK(out == data);
  }
}

TEST_CASE("each codec handles several chunks")
{
  auto data = sampleData(50000);
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data, 12345);
    std::vector<uint8_t> out(data.size());
    decompressBounded(payload, data.size(), out.data(), out.size());
    CHECK(out == data);
  }
}

TEST_CASE("bounded decompression stops exactly at maxOut")
{
  auto data = sampleData(40000);
  for (auto algo : allAlgos) {
    for (size_t maxOut : {size_t(1), size_t(9999), size_t(10000), size_t(25001), size_t(39999)}) {
      auto payload = rootCompress(algo, data, 10000);
      // One guard byte after the bound must stay untouched.
      std::vector<uint8_t> out(maxOut + 1, 0xAB);
      REQUIRE(decompressBounded(payload, data.size(), out.data(), maxOut) == maxOut);
      CHECK(std::equal(out.begin(), out.begin() + maxOut, data.begin()));
      CHECK(out[maxOut] == 0xAB);
    }
  }
}

TEST_CASE("prefix reads of large chunks stop exactly at the bound")
{
  // Several zstd blocks (at most 128 KiB each) per chunk, so that the bound
  // falls before, at and after block boundaries, and in the last block.
  auto data = sampleData(1 << 20);
  constexpr size_t block = 128 * 1024;
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data, 600000);
    for (size_t maxOut : {size_t(1), block - 1, block, block + 1, 2 * block + 17, size_t(599999), size_t(600000), size_t(600001), data.size() - 1}) {
      INFO("algorithm " << (int)algo << ", bound " << maxOut);
      std::vector<uint8_t> out(maxOut + 64, 0xAB);
      PayloadDecompressor d{payload, (uint32_t)data.size(), "memory", 0};
      d.readPrefix(out.data(), maxOut);
      CHECK(std::equal(out.begin(), out.begin() + maxOut, data.begin()));
      CHECK(std::all_of(out.begin() + maxOut, out.end(), [](uint8_t b) { return b == 0xAB; }));
    }
  }
}

TEST_CASE("no read can follow a prefix read")
{
  auto data = sampleData(1000);
  auto payload = rootCompress(Algo::ZSTD, data);
  std::vector<uint8_t> out(data.size());
  PayloadDecompressor d{payload, (uint32_t)data.size(), "memory", 0};
  d.readPrefix(out.data(), 10);
  CHECK_THROWS_AS(d.read(out.data(), 10), std::logic_error);
}

TEST_CASE("incremental reads of odd sizes produce the same bytes")
{
  auto data = sampleData(70000);
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data, 16384);
    PayloadDecompressor d{payload, (uint32_t)data.size(), "memory", 0};
    std::vector<uint8_t> out(data.size());
    size_t pos = 0;
    for (size_t step = 1; pos < data.size(); step = step * 3 + 1) {
      auto n = std::min(step, data.size() - pos);
      d.read(out.data() + pos, n);
      pos += n;
    }
    CHECK(out == data);
    CHECK(d.produced() == data.size());
  }
}

TEST_CASE("truncated payloads raise CorruptData")
{
  auto data = sampleData(30000);
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data, 10000);
    std::vector<uint8_t> out(data.size());
    for (size_t keep : {size_t(0), size_t(5), payload.size() / 2, payload.size() - 1}) {
      std::vector<uint8_t> truncated(payload.begin(), payload.begin() + keep);
      CHECK_THROWS_AS(decompressBounded(truncated, data.size(), out.data(), out.size()), CorruptData);
    }
  }
}

TEST_CASE("corrupt payloads raise CorruptData")
{
  auto data = sampleData(30000);
  for (auto algo : allAlgos) {
    auto payload = rootCompress(algo, data);
    std::vector<uint8_t> out(data.size());
    // Flip bits in the middle of the compressed stream.
    auto corrupt = payload;
    for (size_t i = corrupt.size() / 3; i < corrupt.size() / 3 + 16; ++i) {
      corrupt[i] ^= 0x5A;
    }
    // zlib and zstd may decode garbage of the right size without noticing; what
    // matters is that nothing crashes and that detected errors are CorruptData.
    try {
      decompressBounded(corrupt, data.size(), out.data(), out.size());
    } catch (CorruptData const&) {
    }
    // A wrong uncompressed size in the header is always detected.
    auto wrongSize = payload;
    wrongSize[6] ^= 0x01;
    CHECK_THROWS_AS(decompressBounded(wrongSize, data.size(), out.data(), out.size()), CorruptData);
  }
}

TEST_CASE("LZ4 checksum mismatch is detected")
{
  auto data = sampleData(5000);
  auto payload = rootCompress(Algo::LZ4, data);
  payload[9] ^= 0xFF; // first checksum byte
  std::vector<uint8_t> out(data.size());
  CHECK_THROWS_AS(decompressBounded(payload, data.size(), out.data(), out.size()), CorruptData);
}

TEST_CASE("legacy and unknown algorithms are unsupported")
{
  auto data = sampleData(100);
  auto payload = rootCompress(Algo::ZLIB, data);
  std::vector<uint8_t> out(data.size());
  payload[0] = 'C';
  payload[1] = 'S';
  CHECK_THROWS_AS(decompressBounded(payload, data.size(), out.data(), out.size()), UnsupportedLayout);
  payload[0] = 'Q';
  payload[1] = 'Q';
  CHECK_THROWS_AS(decompressBounded(payload, data.size(), out.data(), out.size()), UnsupportedLayout);
}

TEST_CASE("chunks larger than the object length are rejected")
{
  auto data = sampleData(1000);
  auto payload = rootCompress(Algo::ZSTD, data);
  std::vector<uint8_t> out(data.size());
  CHECK_THROWS_AS(decompressBounded(payload, 500, out.data(), out.size()), CorruptData);
}
