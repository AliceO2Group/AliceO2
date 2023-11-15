// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransPack.h
/// @author Michael Lettrich
/// @brief  test pack/unpack features

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>
#include <fmt/core.h>

#include "rANS/internal/pack/pack.h"
#include "rANS/internal/pack/eliasDelta.h"
#include "rANS/internal/containers/BitPtr.h"

using namespace o2::rans;
using namespace o2::rans::internal;
using namespace o2::rans::utils;

inline constexpr size_t BufferSize = 257;

template <typename source_T>
std::vector<source_T>
  makeRandomUniformVector(size_t nelems, source_T min = std::numeric_limits<source_T>::min(), source_T max = std::numeric_limits<source_T>::max())
{
  std::vector<source_T> result(nelems, 0);
  std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
  std::uniform_int_distribution<source_T> dist(min, max);
  std::generate(result.begin(), result.end(), [&dist, &mt]() { return dist(mt); });
  return result;
};

using buffer_types = boost::mp11::mp_list<int8_t, int16_t, int32_t, int64_t>;

using source_types = boost::mp11::mp_list<uint8_t, uint16_t, uint32_t, uint64_t>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_computePackingBufferSize, buffer_T, buffer_types)
{
  size_t packingWidth = utils::toBits<packing_type>() - 3;

  BOOST_CHECK_EQUAL(0, (computePackingBufferSize<buffer_T>(0, packingWidth)));
  BOOST_CHECK_EQUAL(sizeof(packing_type) / sizeof(buffer_T), (computePackingBufferSize<buffer_T>(1, packingWidth)));       // space it takes to pack 1 element takes 4 B
  BOOST_CHECK_EQUAL(sizeof(packing_type) * 62 / sizeof(buffer_T), (computePackingBufferSize<buffer_T>(64, packingWidth))); // space it takes to pack 64 elements takes 62 * packingType = 496 B
  BOOST_CHECK_EQUAL(sizeof(packing_type) * 22 / sizeof(buffer_T), (computePackingBufferSize<buffer_T>(23, packingWidth))); // space it takes to pack 23 elements takes 22 * packingType = 176 B
}

BOOST_AUTO_TEST_CASE(test_packUnpack)
{
  using source_type = uint64_t;

  for (size_t packingWidth = 1; packingWidth <= 58; ++packingWidth) {
    BOOST_TEST_MESSAGE(fmt::format("Checking {} Bit Packing", packingWidth));

    auto source = makeRandomUniformVector<source_type>(BufferSize, 0, utils::pow2(packingWidth) - 1);
    std::vector<uint8_t> packingBuffer(source.size() * sizeof(source_type), 0);

    BitPtr packIter{packingBuffer.data()};
    for (auto i : source) {
      packIter = pack(packIter, i, packingWidth);
    }

    BitPtr unpackIter{packingBuffer.data()};
    std::vector<source_type> result(source.size(), 0);
    for (size_t i = 0; i < source.size(); ++i) {
      result[i] = unpack<uint64_t>(unpackIter, packingWidth);
      unpackIter += packingWidth;
    }

    // check if results are equal;
    BOOST_CHECK_EQUAL(packIter, unpackIter);
    BOOST_CHECK_EQUAL_COLLECTIONS(source.begin(), source.end(), result.begin(), result.end());
  }
};

BOOST_AUTO_TEST_CASE(test_packUnpackLong)
{
  using source_type = uint64_t;

  for (size_t packingWidth = 1; packingWidth <= 63; ++packingWidth) {
    BOOST_TEST_MESSAGE(fmt::format("Checking {} Bit Packing", packingWidth));

    auto source = makeRandomUniformVector<source_type>(BufferSize, 0, utils::pow2(packingWidth) - 1);
    std::vector<uint8_t> packingBuffer(source.size() * sizeof(source_type), 0);

    BitPtr packIter{packingBuffer.data()};
    for (auto i : source) {
      packIter = packLong(packIter, i, packingWidth);
    }

    BitPtr unpackIter{packingBuffer.data()};
    std::vector<source_type> result(source.size(), 0);
    for (size_t i = 0; i < source.size(); ++i) {
      result[i] = unpackLong(unpackIter, packingWidth);
      unpackIter += packingWidth;
    }

    // check if results are equal;
    BOOST_CHECK_EQUAL(packIter, unpackIter);
    BOOST_CHECK_EQUAL_COLLECTIONS(source.begin(), source.end(), result.begin(), result.end());
  }
};

BOOST_AUTO_TEST_CASE(test_packUnpackStream)
{
  using source_type = uint64_t;

  for (size_t packingWidth = 1; packingWidth <= 63; ++packingWidth) {
    BOOST_TEST_MESSAGE(fmt::format("Checking {} Bit Packing", packingWidth));
    auto source = makeRandomUniformVector<source_type>(BufferSize, 0, utils::pow2(packingWidth) - 1);
    std::vector<uint8_t> packingBuffer(source.size() * sizeof(source_type), 0);

    source_type min = *std::min_element(source.begin(), source.end());

    [[maybe_unused]] auto packingBufferEnd = pack(source.data(), source.size(), packingBuffer.data(), packingWidth, min);

    std::vector<source_type> unpackBuffer(source.size(), 0);
    unpack(packingBuffer.data(), source.size(), unpackBuffer.data(), packingWidth, min);

    // compare if both yield the correct result
    BOOST_CHECK_EQUAL_COLLECTIONS(source.begin(), source.end(), unpackBuffer.begin(), unpackBuffer.end());
  }
};

BOOST_AUTO_TEST_CASE(test_packRUnpackEliasDelta)
{
  using source_type = uint32_t;

  for (size_t packingWidth = 1; packingWidth <= 32; ++packingWidth) {
    BOOST_TEST_MESSAGE(fmt::format("Checking {} Bit Elias Delta Coder", packingWidth));
    auto source = makeRandomUniformVector<source_type>(BufferSize, 1ull, utils::pow2(packingWidth) - 1);
    std::vector<uint64_t> packingBuffer(source.size(), 0);

    BitPtr iter{packingBuffer.data()};
    for (auto i : source) {
      iter = eliasDeltaEncode(iter, i);
    };

    BitPtr iterBegin{packingBuffer.data()};
    std::vector<source_type> unpackBuffer(source.size(), 0);
    for (size_t i = 0; i < source.size(); ++i) {
      const size_t maxOffset = std::min(static_cast<size_t>(iter - iterBegin), EliasDeltaDecodeMaxBits);
      unpackBuffer[i] = eliasDeltaDecode<source_type>(iter, maxOffset);
    };

    // compare if both yield the correct result
    BOOST_CHECK_EQUAL_COLLECTIONS(source.begin(), source.end(), unpackBuffer.rbegin(), unpackBuffer.rend());
  }
};