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

/// @file   test_ransHistograms.cxx
/// @author Michael Lettrich
/// @brief test class that allows to build histogram of symbols from a source message

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>
#include <gsl/span>
#include <fmt/format.h>

#include "rANS/histogram.h"
#include "rANS/internal/transform/algorithm.h"
#include "rANS/internal/common/typetraits.h"
#include "rANS/compat.h"

using namespace o2::rans;

namespace mp = boost::mp11;

using small_dense_histogram_types = mp::mp_list<
  DenseHistogram<char>,
  DenseHistogram<uint8_t>,
  DenseHistogram<int8_t>,
  DenseHistogram<uint16_t>,
  DenseHistogram<int16_t>>;

using large_dense_histogram_types = mp::mp_list<DenseHistogram<int32_t>>;

using adaptive_histogram_types = mp::mp_list<AdaptiveHistogram<uint32_t>,
                                             AdaptiveHistogram<int32_t>>;

using sparse_histograms = mp::mp_list<SparseHistogram<uint32_t>,
                                      SparseHistogram<int32_t>>;

namespace boost
{
namespace test_tools
{
namespace tt_detail
{

// teach boost how to print std::pair

template <class F, class S>
struct print_log_value<::std::pair<F, S>> {
  void operator()(::std::ostream& os, ::std::pair<F, S> const& p)
  {
    os << "([" << p.first << "], [" << p.second << "])";
  }
};

} // namespace tt_detail
} // namespace test_tools
} // namespace boost

using histogram_types = mp::mp_flatten<mp::mp_list<small_dense_histogram_types, large_dense_histogram_types, adaptive_histogram_types, sparse_histograms>>;

using variable_histograms_types = mp::mp_flatten<mp::mp_list<large_dense_histogram_types, adaptive_histogram_types, sparse_histograms>>;

template <typename histogram_T>
void checkEquivalent(const histogram_T& a, const histogram_T& b)
{
  for (auto iter = a.begin(); iter != a.end(); ++iter) {
    auto index = internal::getIndex(a, iter);
    auto value = internal::getValue(iter);
    BOOST_CHECK_EQUAL(b[index], value);
  }
};

template <class histogram_T, typename map_T>
size_t getTableSize(const map_T& resultsMap)
{
  using namespace o2::rans::internal;
  using source_type = typename histogram_T::source_type;
  if constexpr (isDenseContainer_v<histogram_T>) {
    if constexpr (sizeof(source_type) < 4) {
      return static_cast<size_t>(std::numeric_limits<std::make_unsigned_t<source_type>>::max()) + 1;
    } else {
      const auto [minIter, maxIter] = std::minmax_element(std::begin(resultsMap), std::end(resultsMap), [](const auto& a, const auto& b) { return a.first < b.first; });
      return maxIter->first - minIter->first + std::is_signed_v<source_type>;
    }
  } else if constexpr (isAdaptiveContainer_v<histogram_T>) {
    std::vector<int32_t> buckets;
    for (const auto [key, value] : resultsMap) {
      buckets.push_back(key / histogram_T::container_type::getBucketSize());
    }
    std::sort(buckets.begin(), buckets.end());
    auto end = std::unique(buckets.begin(), buckets.end());
    return histogram_T::container_type::getBucketSize() * std::distance(buckets.begin(), end);
  } else {
    static_assert(isHashContainer_v<histogram_T> || isSetContainer_v<histogram_T>);
    return std::count_if(resultsMap.begin(), resultsMap.end(), [](const auto& val) { return val.second > 0; });
  }
};

template <class histogram_T, typename map_T>
auto getOffset(const map_T& resultsMap) -> typename map_T::key_type
{
  using namespace o2::rans::internal;
  using source_type = typename histogram_T::source_type;
  if constexpr (isDenseContainer_v<histogram_T>) {
    if constexpr (sizeof(source_type) < 4) {
      return std::numeric_limits<source_type>::min();
    } else {
      const auto [minIter, maxIter] = std::minmax_element(std::begin(resultsMap), std::end(resultsMap), [](const auto& a, const auto& b) { return a.first < b.first; });
      return minIter->first;
    }
  } else if constexpr (isAdaptiveContainer_v<histogram_T>) {
    return std::numeric_limits<source_type>::min();
  } else if constexpr (isHashContainer_v<histogram_T>) {
    return 0;
  } else {
    static_assert(isSetContainer_v<histogram_T>);

    source_type min = resultsMap.begin()->first;
    for (const auto& [index, value] : resultsMap) {
      if (value > 0) {
        min = std::min(min, index);
      }
    }
    return min;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_emptyTablesSmall, histogram_T, small_dense_histogram_types)
{
  using source_type = typename histogram_T::source_type;
  histogram_T histogram{};
  const size_t tableSize = 1ul << (sizeof(source_type) * 8);

  BOOST_CHECK_EQUAL(histogram.empty(), true);
  BOOST_CHECK_EQUAL(histogram.size(), tableSize);
  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_emptyTablesLarge, histogram_T, variable_histograms_types)
{
  using source_type = typename histogram_T::source_type;
  histogram_T histogram{};

  BOOST_CHECK_EQUAL(histogram.empty(), true);

  BOOST_CHECK_EQUAL(histogram.size(), 0);
  BOOST_CHECK(histogram.begin() == histogram.end());
  BOOST_CHECK(histogram.cbegin() == histogram.cend());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_addSamples, histogram_T, histogram_types)
{
  using source_type = typename histogram_T::source_type;

  std::vector<source_type> samples{
    static_cast<source_type>(-5),
    static_cast<source_type>(-2),
    static_cast<source_type>(1),
    static_cast<source_type>(3),
    static_cast<source_type>(5),
    static_cast<source_type>(8),
    static_cast<source_type>(-5),
    static_cast<source_type>(5),
    static_cast<source_type>(1),
    static_cast<source_type>(1),
    static_cast<source_type>(1),
    static_cast<source_type>(14),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
    static_cast<source_type>(8),
  };

  std::unordered_map<source_type, uint32_t> results{{static_cast<source_type>(-5), 2},
                                                    {static_cast<source_type>(-2), 1},
                                                    {static_cast<source_type>(-1), 0},
                                                    {static_cast<source_type>(0), 0},
                                                    {static_cast<source_type>(1), 4},
                                                    {static_cast<source_type>(2), 0},
                                                    {static_cast<source_type>(3), 1},
                                                    {static_cast<source_type>(4), 0},
                                                    {static_cast<source_type>(5), 2},
                                                    {static_cast<source_type>(8), 8},
                                                    {static_cast<source_type>(14), 1}};

  histogram_T histogram{};
  histogram.addSamples(samples.begin(), samples.end());

  histogram_T histogram2{};
  histogram2.addSamples(samples);

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
  BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));

  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());

  // lets add more frequencies;
  std::vector<source_type> samples2{
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
    static_cast<source_type>(-10),
    static_cast<source_type>(0),
    static_cast<source_type>(50),
  };

  results[static_cast<source_type>(-10)] = 6;
  results[static_cast<source_type>(0)] = 6;
  results[static_cast<source_type>(50)] = 6;

  histogram.addSamples(samples2.begin(), samples2.end());

  histogram2.addSamples(samples2);

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
  BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));

  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());

  BOOST_CHECK(countNUsedAlphabetSymbols(histogram) == 10);
  BOOST_CHECK(histogram.getNumSamples() == samples.size() + samples2.size());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_addFrequencies, histogram_T, histogram_types)
{
  using source_type = typename histogram_T::source_type;
  using value_type = typename histogram_T::value_type;
  std::vector<value_type> frequencies{0, 1, 2, 3, 4, 5};

  std::unordered_map<source_type, uint32_t> results{
    {static_cast<source_type>(1), 1},
    {static_cast<source_type>(2), 2},
    {static_cast<source_type>(3), 3},
    {static_cast<source_type>(4), 4},
    {static_cast<source_type>(5), 5},
  };

  histogram_T histogram{};
  histogram.addFrequencies(frequencies.begin(), frequencies.end(), 0);

  histogram_T histogram2{};
  histogram2.addFrequencies(gsl::make_span(frequencies), 0);

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
  BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));

  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());
  BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 5);
  BOOST_CHECK_EQUAL(histogram.getNumSamples(), 15);

  // lets add more frequencies;
  std::vector<value_type> frequencies2{3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0};

  if constexpr (std::is_signed_v<source_type>) {
    histogram.addFrequencies(frequencies2.begin(), frequencies2.end(), -1);
    histogram2.addFrequencies(gsl::make_span(frequencies2), -1);

    results[static_cast<source_type>(0 + -1)] += 3;
    results[static_cast<source_type>(2 + -1)] += 4;
    results[static_cast<source_type>(11 + -1)] += 5;

    BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
    BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));
    BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 7);
  } else {
    histogram.addFrequencies(frequencies2.begin(), frequencies2.end(), 3);
    histogram2.addFrequencies(gsl::make_span(frequencies2), 3);

    results[static_cast<source_type>(0 + 3)] += 3;
    results[static_cast<source_type>(2 + 3)] += 4;
    results[static_cast<source_type>(11 + 3)] += 5;

    BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
    BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));
    BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 6);
  }
  BOOST_CHECK_EQUAL(histogram.getNumSamples(), 27);

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_addFrequenciesSignChange, histogram_T, histogram_types)
{
  using source_type = typename histogram_T::source_type;
  using value_type = typename histogram_T::value_type;
  std::vector<value_type> frequencies{0, 1, 2, 3, 4, 5};

  std::unordered_map<source_type, uint32_t> results{
    {static_cast<source_type>(1), 1},
    {static_cast<source_type>(2), 2},
    {static_cast<source_type>(3), 3},
    {static_cast<source_type>(4), 4},
    {static_cast<source_type>(5), 5},
  };

  histogram_T histogram{};
  histogram.addFrequencies(frequencies.begin(), frequencies.end(), 0);

  histogram_T histogram2{};
  histogram2.addFrequencies(gsl::make_span(frequencies), 0);

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
  BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));
  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());
  BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 5);
  BOOST_CHECK_EQUAL(histogram.getNumSamples(), 15);

  // lets add more frequencies;
  std::vector<value_type> frequencies2{3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0};

  if constexpr (std::is_signed_v<source_type>) {
    const std::ptrdiff_t offset = utils::pow2(utils::toBits<source_type>() - 1);

    if constexpr (std::is_same_v<histogram_T, DenseHistogram<int32_t>>) {
      BOOST_CHECK_THROW(histogram.addFrequencies(frequencies2.begin(), frequencies2.end(), offset), HistogramError);
      BOOST_CHECK_THROW(histogram2.addFrequencies(gsl::make_span(frequencies2), offset), HistogramError);
    } else {

      histogram.addFrequencies(frequencies2.begin(), frequencies2.end(), offset);
      histogram2.addFrequencies(gsl::make_span(frequencies2), offset);

      results[static_cast<source_type>(0 + offset)] += 3;
      results[static_cast<source_type>(2 + offset)] += 4;
      results[static_cast<source_type>(11 + offset)] += 5;

      BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
      BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));
      BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 8);
    }
  } else {
    const std::ptrdiff_t offset = -1;
    histogram.addFrequencies(frequencies2.begin(), frequencies2.end(), offset);
    histogram2.addFrequencies(gsl::make_span(frequencies2), offset);

    results[static_cast<source_type>(0 + offset)] += 3;
    results[static_cast<source_type>(2 + offset)] += 4;
    results[static_cast<source_type>(11 + offset)] += 5;

    BOOST_CHECK_EQUAL(histogram.size(), getTableSize<histogram_T>(results));
    BOOST_CHECK_EQUAL(histogram.getOffset(), getOffset<histogram_T>(results));
    BOOST_CHECK_EQUAL(countNUsedAlphabetSymbols(histogram), 7);
  }

  if constexpr (std::is_same_v<histogram_T, DenseHistogram<int32_t>>) {
    // for the int32_t case we couldn't add samples, so no changes
    BOOST_CHECK_EQUAL(histogram.getNumSamples(), 15);
  } else {
    BOOST_CHECK_EQUAL(histogram.getNumSamples(), 27);
  }

  checkEquivalent(histogram, histogram2);

  for (const auto [symbol, value] : results) {
    BOOST_TEST_MESSAGE(fmt::format("testing symbol {}", static_cast<int64_t>(symbol)));
    BOOST_CHECK_EQUAL(histogram[symbol], value);
  }

  BOOST_CHECK_EQUAL(histogram.empty(), false);
  BOOST_CHECK(histogram.begin() != histogram.end());
  BOOST_CHECK(histogram.cbegin() != histogram.cend());
};

using renorm_types = mp::mp_list<DenseHistogram<uint8_t>, DenseHistogram<uint32_t>, AdaptiveHistogram<int32_t>, SparseHistogram<int32_t>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_renorm, histogram_T, renorm_types)
{
  std::vector<uint32_t> frequencies{1, 1, 2, 2, 2, 2, 6, 8, 4, 10, 8, 14, 10, 19, 26, 30, 31, 35, 41, 45, 51, 44, 47, 39, 58, 52, 42, 53, 50, 34, 50, 30, 32, 24, 30, 20, 17, 12, 16, 6, 8, 5, 6, 4, 4, 2, 2, 2, 1};
  histogram_T histogram{frequencies.begin(), frequencies.end(), static_cast<uint8_t>(0)};

  const size_t scaleBits = 8;

  auto renormedHistogram = renorm(std::move(histogram), scaleBits, RenormingPolicy::ForceIncompressible, 1);

  const std::vector<uint32_t> rescaledFrequencies{1, 2, 1, 3, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 13, 11, 12, 10, 14, 13, 10, 13, 12, 8, 12, 7, 8, 6, 7, 5, 4, 3, 4, 2, 2, 1, 2, 1, 1};
  auto rescaledFrequenciesView = makeHistogramView(rescaledFrequencies, 6);
  BOOST_CHECK_EQUAL(renormedHistogram.isRenormedTo(scaleBits), true);
  BOOST_CHECK_EQUAL(renormedHistogram.getNumSamples(), 1 << scaleBits);
  BOOST_CHECK_EQUAL(renormedHistogram.getIncompressibleSymbolFrequency(), 4);

  for (std::ptrdiff_t i = rescaledFrequenciesView.getMin(); i <= rescaledFrequenciesView.getMax(); ++i) {
    BOOST_CHECK_EQUAL(renormedHistogram[i], rescaledFrequenciesView[i]);
  }
}

using legacy_renorm_types = mp::mp_list<DenseHistogram<uint8_t>, DenseHistogram<uint32_t>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_renormLegacy, histogram_T, legacy_renorm_types)
{
  std::vector<uint32_t> frequencies{1, 1, 2, 2, 2, 2, 6, 8, 4, 10, 8, 14, 10, 19, 26, 30, 31, 35, 41, 45, 51, 44, 47, 39, 58, 52, 42, 53, 50, 34, 50, 30, 32, 24, 30, 20, 17, 12, 16, 6, 8, 5, 6, 4, 4, 2, 2, 2, 1};
  histogram_T histogram{frequencies.begin(), frequencies.end(), static_cast<uint8_t>(0)};

  const size_t scaleBits = 8;

  auto renormedHistogram = compat::renorm(std::move(histogram), scaleBits);
  const std::vector<uint32_t> rescaledFrequencies{1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 3, 3, 4, 6, 7, 7, 9, 9, 11, 12, 10, 11, 9, 13, 12, 10, 13, 11, 8, 12, 7, 7, 6, 7, 4, 4, 3, 4, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1};
  BOOST_CHECK_EQUAL(renormedHistogram.isRenormedTo(scaleBits), true);
  BOOST_CHECK_EQUAL(renormedHistogram.getNumSamples(), 1 << scaleBits);
  BOOST_CHECK_EQUAL(renormedHistogram.getIncompressibleSymbolFrequency(), 2);
  BOOST_CHECK_EQUAL_COLLECTIONS(renormedHistogram.begin(), renormedHistogram.begin() + rescaledFrequencies.size(), rescaledFrequencies.begin(), rescaledFrequencies.end());
}

BOOST_AUTO_TEST_CASE(test_ExpectedCodewordLength)
{
  using namespace internal;
  using namespace utils;
  using source_type = uint32_t;
  constexpr double_t eps = 1e-2;

  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  DenseHistogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  Metrics<source_type> metrics{histogram};
  const auto renormedHistogram = renorm(histogram, metrics);

  const double_t expectedCodewordLength = computeExpectedCodewordLength(histogram, renormedHistogram);
  BOOST_CHECK_CLOSE(expectedCodewordLength, 2.9573820061153833, eps);
  BOOST_CHECK_GE(expectedCodewordLength, metrics.getDatasetProperties().entropy);
}