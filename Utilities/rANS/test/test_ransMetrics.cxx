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

/// @file   test_ransMetrics.cxx
/// @author Michael Lettrich
/// @brief test the calculation of metrics required for renorming and encode/ packing decisions

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>
#include <gsl/span>

#include "rANS/histogram.h"
#include "rANS/metrics.h"

using namespace o2::rans;

using source_type = uint32_t;
using histogram_types = boost::mp11::mp_list<DenseHistogram<source_type>, AdaptiveHistogram<source_type>, SparseHistogram<source_type>>;

BOOST_AUTO_TEST_SUITE(test_DictSizeEstimate)
BOOST_AUTO_TEST_CASE(test_initDictSizeEstimate)
{
  using namespace internal;
  using namespace utils;

  DictSizeEstimate estimate{};
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_emptyDictSizeEstimate, histogram_T, histogram_types)
{
  using namespace internal;
  using namespace utils;

  std::vector<uint32_t> frequencies{};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 0};
  const auto [trimmedBegin, trimmedEnd] = internal::trim(histogram);
  const auto [min, max] = internal::getMinMax(histogram, trimmedBegin, trimmedEnd);

  DictSizeEstimate estimate{histogram.getNumSamples()};

  source_type lastIndex = min;
  forEachIndexValue(histogram, trimmedBegin, trimmedEnd, [&](const source_type& index, const uint32_t& frequency) {
    if (frequency) {
      BOOST_CHECK(lastIndex <= index);
      source_type delta = index - lastIndex;
      estimate.updateIndexSize(delta + (delta == 0));
      lastIndex = index;
      estimate.updateFreqSize(frequency);
    }
  });

  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_defaultDictSizeEstimate, histogram_T, histogram_types)
{
  using namespace internal;
  using namespace utils;

  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 0};

  const auto [trimmedBegin, trimmedEnd] = internal::trim(histogram);
  const auto [min, max] = internal::getMinMax(histogram, trimmedBegin, trimmedEnd);

  DictSizeEstimate estimate{histogram.getNumSamples()};

  source_type lastIndex = min;
  forEachIndexValue(histogram, trimmedBegin, trimmedEnd, [&](const source_type& index, const uint32_t& frequency) {
    if (frequency) {
      BOOST_CHECK(lastIndex <= index);
      source_type delta = index - lastIndex;
      estimate.updateIndexSize(delta + (delta == 0));
      lastIndex = index;
      estimate.updateFreqSize(frequency);
    }
  });

  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 33);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 5);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 224);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 28);
  BOOST_CHECK_EQUAL(estimate.getSizeB(countNUsedAlphabetSymbols(histogram), defaults::MinRenormPrecisionBits), 21);
};
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_RenormingPrecision)

class MetricsTester : public Metrics<source_type>
{
 public:
  inline MetricsTester(const DenseHistogram<source_type>& histogram, float_t cutoffPrecision = 0.999) : Metrics(histogram, cutoffPrecision){};
  inline MetricsTester(const AdaptiveHistogram<source_type>& histogram, float_t cutoffPrecision = 0.999) : Metrics(histogram, cutoffPrecision){};
  inline MetricsTester(const SparseHistogram<source_type>& histogram, float_t cutoffPrecision = 0.999) : Metrics(histogram, cutoffPrecision){};
  inline size_t testComputeRenormingPrecision(float_t cutoffPrecision = 0.999) noexcept { return computeRenormingPrecision(cutoffPrecision); };
  inline size_t testComputeIncompressibleCount(gsl::span<source_type> distribution, source_type renormingPrecision) noexcept { return computeIncompressibleCount(distribution, renormingPrecision); };
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_EmptyRenormingPrecision, histogram_T, histogram_types)
{
  std::array<uint32_t, 32> symbolLengthDistribution;
  std::array<uint32_t, 32> weightedSymbolLengthDistribution;
  const size_t nSamples = 0;
  const uint32_t renormingPrecision = 0;

  MetricsTester tester{histogram_T{}};
  tester.getDatasetProperties().symbolLengthDistribution = symbolLengthDistribution;
  tester.getDatasetProperties().weightedSymbolLengthDistribution = weightedSymbolLengthDistribution;
  tester.getDatasetProperties().numSamples = nSamples;

  BOOST_CHECK_EQUAL(tester.testComputeRenormingPrecision(), renormingPrecision);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(symbolLengthDistribution, renormingPrecision), 1);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(weightedSymbolLengthDistribution, renormingPrecision), 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_cutoffRenormingPrecision, histogram_T, histogram_types)
{
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};
  weightedSymbolLengthDistribution[31] = 44;
  symbolLengthDistribution[31] = 42;
  const size_t nSamples = 44;
  const uint32_t renormingPrecision = defaults::MaxRenormPrecisionBits;

  MetricsTester tester{histogram_T{}};
  tester.getDatasetProperties().symbolLengthDistribution = symbolLengthDistribution;
  tester.getDatasetProperties().weightedSymbolLengthDistribution = weightedSymbolLengthDistribution;
  tester.getDatasetProperties().numSamples = nSamples;

  BOOST_CHECK_EQUAL(tester.testComputeRenormingPrecision(), renormingPrecision);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(symbolLengthDistribution, renormingPrecision), 42);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(weightedSymbolLengthDistribution, renormingPrecision), nSamples);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_noCutoffRenormingPrecision, histogram_T, histogram_types)
{
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};
  weightedSymbolLengthDistribution[1] = 20;
  weightedSymbolLengthDistribution[5] = 20;
  weightedSymbolLengthDistribution[9] = 40;
  weightedSymbolLengthDistribution[12] = 10;
  weightedSymbolLengthDistribution[15] = 10;

  symbolLengthDistribution[1] = 2;
  symbolLengthDistribution[5] = 2;
  symbolLengthDistribution[9] = 4;
  symbolLengthDistribution[12] = 1;
  symbolLengthDistribution[15] = 1;

  const size_t nSamples = 100;
  const uint32_t renormingPrecision = 17;

  MetricsTester tester{histogram_T{}};
  tester.getDatasetProperties().symbolLengthDistribution = symbolLengthDistribution;
  tester.getDatasetProperties().weightedSymbolLengthDistribution = weightedSymbolLengthDistribution;
  tester.getDatasetProperties().numSamples = nSamples;

  BOOST_CHECK_EQUAL(tester.testComputeRenormingPrecision(), renormingPrecision);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(symbolLengthDistribution, renormingPrecision), 0);
  BOOST_CHECK_EQUAL(tester.testComputeIncompressibleCount(weightedSymbolLengthDistribution, renormingPrecision), 0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_Metrics)
BOOST_AUTO_TEST_CASE_TEMPLATE(test_emptyMetrics, histogram_T, histogram_types)
{
  std::vector<uint32_t> frequencies{};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 0};
  const float eps = 1e-2;
  const size_t nUsedAlphabetSymbols = 0;
  const auto [min, max] = getMinMax(histogram);

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, min);
  BOOST_CHECK_EQUAL(dataProperies.max, max);
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, 0);
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_SMALL(dataProperies.entropy, eps);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};

  uint32_t sumUnweighted = 0;
  uint32_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    // BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_EQUAL(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i]);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(*coderProperties.renormingPrecisionBits, 0);
  BOOST_CHECK_EQUAL(sumUnweighted, nUsedAlphabetSymbols);
  BOOST_CHECK_EQUAL(sumWeighted, 0);
  BOOST_CHECK_EQUAL(*coderProperties.nIncompressibleSymbols, 1);

  const auto& estimate = coderProperties.dictSizeEstimate;
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_singleElementMetrics, histogram_T, histogram_types)
{
  std::vector<uint32_t> frequencies{5};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 2};
  const auto [min, max] = getMinMax(histogram);
  const size_t nUsedAlphabetSymbols = countNUsedAlphabetSymbols(histogram);

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, min);
  BOOST_CHECK_EQUAL(dataProperies.max, max);
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, 0);
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_SMALL(dataProperies.entropy, 1e-5f);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};

  symbolLengthDistribution[0] = 1;
  weightedSymbolLengthDistribution[0] = 5;

  uint32_t sumUnweighted = 0;
  uint32_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    // BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_EQUAL(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i]);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(*coderProperties.renormingPrecisionBits, defaults::MinRenormPrecisionBits);
  BOOST_CHECK_EQUAL(sumUnweighted, nUsedAlphabetSymbols);
  BOOST_CHECK_EQUAL(sumWeighted, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(*coderProperties.nIncompressibleSymbols, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_computeMetrics, histogram_T, histogram_types)
{
  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 0};
  const auto [min, max] = getMinMax(histogram);
  const float eps = 1e-2;
  const size_t nUsedAlphabetSymbols = countNUsedAlphabetSymbols(histogram);

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, min);
  BOOST_CHECK_EQUAL(dataProperies.max, max);
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, internal::numBitsForNSymbols(max - min + 1));
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_CLOSE(dataProperies.entropy, 2.957295041922758, eps);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};

  weightedSymbolLengthDistribution[2] = 30;
  weightedSymbolLengthDistribution[3] = 12;
  weightedSymbolLengthDistribution[4] = 2;
  weightedSymbolLengthDistribution[5] = 1;

  symbolLengthDistribution[2] = 4;
  symbolLengthDistribution[3] = 3;
  symbolLengthDistribution[4] = 1;
  symbolLengthDistribution[5] = 1;

  uint32_t sumUnweighted = 0;
  uint32_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_EQUAL(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i]);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(sumUnweighted, nUsedAlphabetSymbols);
  BOOST_CHECK_EQUAL(sumWeighted, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(*coderProperties.renormingPrecisionBits, defaults::MinRenormPrecisionBits);
  BOOST_CHECK_EQUAL(*coderProperties.nIncompressibleSymbols, 0);

  const auto& estimate = coderProperties.dictSizeEstimate;
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 33);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 5);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 224);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 28);
  BOOST_CHECK_EQUAL(estimate.getSizeB(nUsedAlphabetSymbols, *coderProperties.renormingPrecisionBits), 21);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_SizeEstimate)
BOOST_AUTO_TEST_CASE_TEMPLATE(test_emptySizeEstimate, histogram_T, histogram_types)
{
  histogram_T histogram{};
  Metrics<source_type> metrics{histogram};
  SizeEstimate estimate{metrics};
  BOOST_CHECK_EQUAL(estimate.getEntropySizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getCompressedDatasetSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getCompressedDictionarySize<>(1.0), 8);
  BOOST_CHECK_EQUAL(estimate.getIncompressibleSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getPackedDatasetSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.preferPacking(1.0), true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_normalSizeEstimate, histogram_T, histogram_types)
{
  constexpr size_t entropySizeB = 17;

  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  histogram_T histogram{frequencies.begin(), frequencies.end(), 0};
  Metrics<source_type> metrics{histogram};
  SizeEstimate estimate{metrics};
  BOOST_CHECK_EQUAL(estimate.getEntropySizeB(), entropySizeB);
  BOOST_CHECK_EQUAL(estimate.getCompressedDatasetSize<>(1.0), addEncoderOverheadEstimateB<>(entropySizeB));
  BOOST_CHECK_EQUAL(estimate.getCompressedDictionarySize<>(1.0), 29);
  BOOST_CHECK_EQUAL(estimate.getIncompressibleSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getPackedDatasetSize<>(1.0), 29);
  BOOST_CHECK_EQUAL(estimate.preferPacking(1.0), true);
}
BOOST_AUTO_TEST_SUITE_END()