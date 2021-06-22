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
  const std::vector<uint32_t> A{};
  o2::rans::internal::SymbolStatistics symbolStats{A.begin(), A.end(), 0, 0u, 0u};

  BOOST_CHECK_EQUAL(symbolStats.getMinSymbol(), 0);
  BOOST_CHECK_EQUAL(symbolStats.getMaxSymbol(), 0);
  BOOST_CHECK_EQUAL(symbolStats.size(), 1);
  BOOST_CHECK_EQUAL(symbolStats.getAlphabetRangeBits(), 1);
  BOOST_CHECK_EQUAL(symbolStats.getNUsedAlphabetSymbols(), 1);
  BOOST_CHECK_EQUAL(symbolStats.getSymbolTablePrecision(), o2::rans::internal::MIN_SCALE);

  BOOST_CHECK(symbolStats.begin() != symbolStats.end());

  const auto [freq, cumul] = symbolStats.getEscapeSymbol();
  BOOST_CHECK_EQUAL(freq, std::get<0>(symbolStats[0]));
  BOOST_CHECK_EQUAL(cumul, std::get<1>(symbolStats[0]));
  BOOST_CHECK_EQUAL(freq, std::get<0>(symbolStats.at(0)));
  BOOST_CHECK_EQUAL(cumul, std::get<1>(symbolStats.at(0)));
}

struct SymbolStatsFromFrequencyTable {
  static auto makeStats(const std::vector<int>& samples, size_t scaleBits)
  {
    o2::rans::FrequencyTable f;
    f.addSamples(std::begin(samples), std::end(samples));
    return o2::rans::internal::SymbolStatistics{f, scaleBits};
  };
};

struct SymbolStatsFromFrequencyTableRvalue {
  static auto makeStats(const std::vector<int>& samples, size_t scaleBits)
  {
    o2::rans::FrequencyTable f;
    f.addSamples(std::begin(samples), std::end(samples));
    return o2::rans::internal::SymbolStatistics{std::move(f), scaleBits};
  };
};

struct SymbolStatsFromIterator {
  static auto makeStats(const std::vector<int>& samples, size_t scaleBits)
  {
    o2::rans::FrequencyTable f;
    f.addSamples(std::begin(samples), std::end(samples));
    return o2::rans::internal::SymbolStatistics{f.begin(),
                                                f.end(),
                                                f.getMinSymbol(),
                                                scaleBits,
                                                f.getNUsedAlphabetSymbols()};
  };
};

using SymbolStats_t = boost::mpl::vector<SymbolStatsFromFrequencyTable,
                                         SymbolStatsFromFrequencyTableRvalue,
                                         SymbolStatsFromIterator>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_buildSymbolStats, Stats_T, SymbolStats_t)
{
  const std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  const std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5, 1};
  const size_t scaleBits = 17;

  auto symbolStats = Stats_T::makeStats(A, scaleBits);

  BOOST_CHECK_EQUAL(symbolStats.getMinSymbol(), *std::min_element(A.begin(), A.end()));
  BOOST_CHECK_EQUAL(symbolStats.getMaxSymbol(), *std::max_element(A.begin(), A.end()) + 1);
  BOOST_CHECK_EQUAL(symbolStats.size(), histA.size());
  BOOST_CHECK_EQUAL(symbolStats.getAlphabetRangeBits(), std::ceil(std::log2(histA.size())));
  BOOST_CHECK_EQUAL(symbolStats.getNUsedAlphabetSymbols(), getNUniqueSymbols(histA));
  BOOST_CHECK_EQUAL(symbolStats.getSymbolTablePrecision(), scaleBits);

  BOOST_CHECK(symbolStats.begin() != symbolStats.end());

  const std::vector<uint32_t> frequencies{8738, 0, 0, 0, 8738, 0, 0, 8738, 8738, 0, 17476, 17477, 8738, 43690, 8739};
  const std::vector<uint32_t> cumulative{0, 8738, 8738, 8738, 8738, 17476, 17476, 17476, 26214, 34952, 34952, 52428, 69905, 78643, 122333};
  BOOST_CHECK_EQUAL(symbolStats.size(), frequencies.size());
  BOOST_CHECK_EQUAL(symbolStats.size(), cumulative.size());

  for (size_t i = 0; i < symbolStats.size(); ++i) {
    const auto [freq, cumul] = symbolStats.at(i);
    BOOST_CHECK_EQUAL(freq, frequencies[i]);
    BOOST_CHECK_EQUAL(cumul, cumulative[i]);
  }
}
