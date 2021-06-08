// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransSymbolStatistics.cxx
/// @author Michael Lettrich
/// @since  2021-06-02
/// @brief

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/mpl/vector.hpp>

#include "rANS/rans.h"

template <typename T>
size_t getNUniqueSymbols(const T& container)
{
  return std::count_if(container.begin(), container.end(), [](uint32_t value) { return value != 0; });
};

BOOST_AUTO_TEST_CASE(test_empty)
{
  using namespace o2::rans;

  const std::vector<uint32_t> A{};
  const internal::SymbolStatistics symbolStats{A.begin(), A.end(), 0, 0u, 0u};
  const internal::SymbolTable<internal::DecoderSymbol> symbolTable{symbolStats};

  BOOST_CHECK_EQUAL(symbolTable.getMinSymbol(), 0);
  BOOST_CHECK_EQUAL(symbolTable.getMaxSymbol(), 0);
  BOOST_CHECK_EQUAL(symbolTable.size(), 1);
  BOOST_CHECK_EQUAL(symbolTable.getAlphabetRangeBits(), 1);
  BOOST_CHECK_EQUAL(symbolTable.getNUsedAlphabetSymbols(), 1);

  const auto escapeSymbol = symbolTable.getEscapeSymbol();
  BOOST_CHECK_EQUAL(escapeSymbol.getFrequency(), symbolTable[0].getFrequency());
  BOOST_CHECK_EQUAL(escapeSymbol.getCumulative(), symbolTable[0].getCumulative());
  BOOST_CHECK_EQUAL(escapeSymbol.getFrequency(), symbolTable.at(0).getFrequency());
  BOOST_CHECK_EQUAL(escapeSymbol.getCumulative(), symbolTable.at(0).getCumulative());
  BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(0), true);

  // out of range checks:
  const int outOfRangeSymbols[] = {-100, 100};
  for (auto symbol : outOfRangeSymbols) {
    const auto outOfRange = symbolTable[symbol];
    BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(symbol), true);
    BOOST_CHECK_EQUAL(outOfRange.getFrequency(), escapeSymbol.getFrequency());
    BOOST_CHECK_EQUAL(outOfRange.getCumulative(), escapeSymbol.getCumulative());
  }
}

BOOST_AUTO_TEST_CASE(test_symbolTable)
{
  using namespace o2::rans;

  const std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  const std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5, 1};
  const size_t scaleBits = 17;
  const int32_t min = *std::min_element(A.begin(), A.end());

  FrequencyTable f;
  f.addSamples(A.begin(), A.end());
  const internal::SymbolStatistics symbolStats{std::move(f), scaleBits};
  const internal::SymbolTable<internal::DecoderSymbol> symbolTable{symbolStats};

  BOOST_CHECK_EQUAL(symbolTable.getMinSymbol(), min);
  BOOST_CHECK_EQUAL(symbolTable.getMaxSymbol(), *std::max_element(A.begin(), A.end()) + 1);
  BOOST_CHECK_EQUAL(symbolTable.size(), histA.size());
  BOOST_CHECK_EQUAL(symbolTable.getAlphabetRangeBits(), std::ceil(std::log2(histA.size())));
  BOOST_CHECK_EQUAL(symbolTable.getNUsedAlphabetSymbols(), getNUniqueSymbols(histA));

  const auto escapeSymbol = symbolTable.getEscapeSymbol();
  const std::vector<uint32_t> frequencies{8738, 0, 0, 0, 8738, 0, 0, 8738, 8738, 0, 17476, 17477, 8738, 43690, 8739};
  const std::vector<uint32_t> cumulative{0, 8738, 8738, 8738, 8738, 17476, 17476, 17476, 26214, 34952, 34952, 52428, 69905, 78643, 122333};
  BOOST_CHECK_EQUAL(symbolTable.size(), frequencies.size());
  BOOST_CHECK_EQUAL(symbolTable.size(), cumulative.size());

  // all but last since this is the escape symbol
  for (size_t i = 0; i < frequencies.size() - 1; ++i) {
    const uint32_t symbol = min + i;

    const auto decodeSymbol = symbolTable[symbol];
    const auto decodeSymbolAt = symbolTable.at(i);
    BOOST_CHECK_EQUAL(decodeSymbol.getFrequency(), decodeSymbolAt.getFrequency());
    BOOST_CHECK_EQUAL(decodeSymbol.getCumulative(), decodeSymbolAt.getCumulative());

    const auto escapeSymbol = symbolTable.getEscapeSymbol();

    if (frequencies[i] == 0) {
      BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(symbol), true);
      BOOST_CHECK_EQUAL(decodeSymbol.getFrequency(), escapeSymbol.getFrequency());
      BOOST_CHECK_EQUAL(decodeSymbol.getCumulative(), escapeSymbol.getCumulative());
    } else {
      BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(symbol), false);
      BOOST_CHECK_EQUAL(decodeSymbol.getFrequency(), frequencies[i]);
      BOOST_CHECK_EQUAL(decodeSymbol.getCumulative(), cumulative[i]);
    }
  }
  //escape symbol:
  BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(0), true);
  BOOST_CHECK_EQUAL(escapeSymbol.getFrequency(), symbolTable.at(frequencies.size() - 1).getFrequency());
  BOOST_CHECK_EQUAL(escapeSymbol.getCumulative(), symbolTable.at(frequencies.size() - 1).getCumulative());

  // out of range checks:
  const int outOfRangeSymbols[] = {-100, 100};
  for (auto symbol : outOfRangeSymbols) {
    const auto escapeSymbol = symbolTable.getEscapeSymbol();
    const auto outOfRange = symbolTable[symbol];
    BOOST_CHECK_EQUAL(symbolTable.isEscapeSymbol(symbol), true);
    BOOST_CHECK_EQUAL(outOfRange.getFrequency(), escapeSymbol.getFrequency());
    BOOST_CHECK_EQUAL(outOfRange.getCumulative(), escapeSymbol.getCumulative());
  }
}
