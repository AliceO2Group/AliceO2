// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DecoderSymbol.h
/// @author Michael Lettrich
/// @since  2020-04-15
/// @brief  Test rANS encoder/ decoder

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <cstring>
#include <random>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/vector.hpp>

#include "rANS/SimpleEncoder.h"
#include "rANS/SimpleDecoder.h"
#include "rANS/rans.h"

using source_t = uint8_t;
using stream_t = uint32_t;
using coder_t = uint64_t;

inline constexpr size_t SymbolTablePrecision = 24;

struct Fixture {
  Fixture()
  {
    std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
    const source_t trials = std::numeric_limits<source_t>::max();
    const double probability = 0.3;
    std::binomial_distribution<source_t> dist(trials, probability);

    const size_t sourceSize = 1ul << 8;
    sourceSymbols.reserve(sourceSize);

    for (size_t i = 0; i < sourceSize; ++i) {
      sourceSymbols.push_back(dist(mt));
    }
  };

  std::vector<source_t> sourceSymbols;
};

BOOST_FIXTURE_TEST_CASE(test_largeScaleEncodeDecode, Fixture)
{
  auto frequencies = o2::rans::renorm(o2::rans::makeFrequencyTableFromSamples(std::begin(sourceSymbols), std::end(sourceSymbols)), SymbolTablePrecision);

  o2::rans::SimpleEncoder<coder_t, stream_t, source_t> encoder{frequencies};
  o2::rans::SimpleDecoder<coder_t, stream_t, source_t> decoder{frequencies};

  std::vector<stream_t> encodeBuffer{};
  std::vector<source_t> decodeBuffer{};
  BOOST_CHECK_NO_THROW(encoder.process(std::begin(sourceSymbols), std::end(sourceSymbols), std::back_inserter(encodeBuffer)));
  BOOST_CHECK_NO_THROW(decoder.process(encodeBuffer.end(), std::back_inserter(decodeBuffer), sourceSymbols.size()));

  BOOST_CHECK_EQUAL_COLLECTIONS(sourceSymbols.begin(), sourceSymbols.end(), decodeBuffer.begin(), decodeBuffer.end());
};