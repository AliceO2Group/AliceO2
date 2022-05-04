// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
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
/// @since  Aug 1, 2020
/// @brief

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <gsl/span>

#include "rANS/histogram.h"
#include "rANS/metrics.h"

using namespace o2::rans;

BOOST_AUTO_TEST_SUITE(test_DictSizeEstimate)
BOOST_AUTO_TEST_CASE(test_initDictSizeEstimate)
{
  using source_type = uint32_t;
  using namespace internal;

  DictSizeEstimate estimate{};
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
};

BOOST_AUTO_TEST_CASE(test_emptyDictSizeEstimate)
{
  using source_type = uint32_t;
  using namespace internal;

  std::vector<uint32_t> frequencies{};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  const auto view = internal::trim(makeHistogramView(histogram));

  DictSizeEstimate estimate{histogram.getNumSamples()};
  DictSizeEstimateCounter counter{&estimate};

  for (auto elem : view) {
    counter.update();
    if (elem > 0) {
      counter.update(elem);
    }
  }

  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
};

BOOST_AUTO_TEST_CASE(test_defaultDictSizeEstimate)
{
  using source_type = uint32_t;
  using namespace internal;

  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  const auto view = internal::trim(makeHistogramView(histogram));

  DictSizeEstimate estimate{histogram.getNumSamples()};
  DictSizeEstimateCounter counter{&estimate};

  for (auto elem : view) {
    counter.update();
    if (elem > 0) {
      counter.update(elem);
    }
  }

  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 33);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 5);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 224);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 28);
  BOOST_CHECK_EQUAL(estimate.getSizeB(histogram.countNUsedAlphabetSymbols(), defaults::MinRenormPrecisionBits), 21);
};
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_RenormingPrecision)
BOOST_AUTO_TEST_CASE(test_EmptyRenormingPrecision)
{
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};

  const uint32_t renormingPrecision = 0;

  BOOST_CHECK_EQUAL(internal::computeRenormingPrecision(weightedSymbolLengthDistribution.begin(),
                                                        weightedSymbolLengthDistribution.end()),
                    renormingPrecision);
  BOOST_CHECK_EQUAL(internal::computeNIncompressibleSymbols(symbolLengthDistribution.begin(),
                                                            symbolLengthDistribution.end(), renormingPrecision),
                    1);
}

BOOST_AUTO_TEST_CASE(test_cutoffRenormingPrecision)
{
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};
  weightedSymbolLengthDistribution[31] = 1.0f;
  symbolLengthDistribution[31] = 42;

  const uint32_t renormingPrecision = defaults::MaxRenormPrecisionBits;

  BOOST_CHECK_EQUAL(internal::computeRenormingPrecision(weightedSymbolLengthDistribution.begin(),
                                                        weightedSymbolLengthDistribution.end()),
                    renormingPrecision);
  BOOST_CHECK_EQUAL(internal::computeNIncompressibleSymbols(symbolLengthDistribution.begin(),
                                                            symbolLengthDistribution.end(), renormingPrecision),
                    42);
}

BOOST_AUTO_TEST_CASE(test_noCutoffRenormingPrecision)
{
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};
  weightedSymbolLengthDistribution[1] = 0.2f;
  weightedSymbolLengthDistribution[5] = 0.2f;
  weightedSymbolLengthDistribution[9] = 0.4f;
  weightedSymbolLengthDistribution[12] = 0.1f;
  weightedSymbolLengthDistribution[15] = 0.1f;

  symbolLengthDistribution[1] = 2;
  symbolLengthDistribution[5] = 2;
  symbolLengthDistribution[9] = 4;
  symbolLengthDistribution[12] = 1;
  symbolLengthDistribution[15] = 1;

  const uint32_t renormingPrecision = 17;

  BOOST_CHECK_EQUAL(internal::computeRenormingPrecision(weightedSymbolLengthDistribution.begin(),
                                                        weightedSymbolLengthDistribution.end()),
                    renormingPrecision);
  BOOST_CHECK_EQUAL(internal::computeNIncompressibleSymbols(symbolLengthDistribution.begin(),
                                                            symbolLengthDistribution.end(), renormingPrecision),
                    0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_Metrics)
BOOST_AUTO_TEST_CASE(test_emptyMetrics)
{
  using source_type = uint32_t;

  std::vector<uint32_t> frequencies{};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  const auto view = internal::trim(makeHistogramView(histogram));
  const float eps = 1e-2;
  const size_t nUsedAlphabetSymbols = 0;

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, view.getMin());
  BOOST_CHECK_EQUAL(dataProperies.max, view.getMax());
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, 0);
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_SMALL(dataProperies.entropy, eps);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};

  uint32_t sumUnweighted = 0;
  float_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    // BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_CLOSE(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i], eps);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(coderProperties.renormingPrecisionBits, 0);
  BOOST_CHECK_EQUAL(sumUnweighted, nUsedAlphabetSymbols);
  BOOST_CHECK_SMALL(sumWeighted, eps);
  BOOST_CHECK_EQUAL(coderProperties.nIncompressibleSymbols, 1);

  const auto& estimate = coderProperties.dictSizeEstimate;
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 0);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getSizeB(0, defaults::MinRenormPrecisionBits), 0);
}

BOOST_AUTO_TEST_CASE(test_singleElementMetrics)
{
  using source_type = uint32_t;

  std::vector<uint32_t> frequencies{5};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 2};
  const auto view = internal::trim(makeHistogramView(histogram));
  const float eps = 1e-2;
  const size_t nUsedAlphabetSymbols = 1;

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, view.getMin());
  BOOST_CHECK_EQUAL(dataProperies.max, view.getMax());
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, 0);
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_SMALL(dataProperies.entropy, 1e-5f);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};

  symbolLengthDistribution[0] = 5;
  weightedSymbolLengthDistribution[0] = 1;

  uint32_t sumUnweighted = 0;
  float_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    // BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_CLOSE(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i], eps);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(coderProperties.renormingPrecisionBits, defaults::MinRenormPrecisionBits);
  BOOST_CHECK_EQUAL(sumUnweighted, 5);
  BOOST_CHECK_CLOSE(sumWeighted, 1, eps);
  BOOST_CHECK_EQUAL(coderProperties.nIncompressibleSymbols, 0);
}

BOOST_AUTO_TEST_CASE(test_computeMetrics)
{
  using source_type = uint32_t;

  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  const auto view = internal::trim(makeHistogramView(histogram));
  const float eps = 1e-2;
  const size_t nUsedAlphabetSymbols = 9;

  const Metrics<source_type> metrics{histogram};
  const auto& dataProperies = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();

  BOOST_CHECK_EQUAL(dataProperies.min, view.getMin());
  BOOST_CHECK_EQUAL(dataProperies.max, view.getMax());
  BOOST_CHECK_EQUAL(dataProperies.numSamples, histogram.getNumSamples());
  BOOST_CHECK_EQUAL(dataProperies.alphabetRangeBits, internal::numBitsForNSymbols(view.size()));
  BOOST_CHECK_EQUAL(dataProperies.nUsedAlphabetSymbols, nUsedAlphabetSymbols);
  BOOST_CHECK_CLOSE(dataProperies.entropy, 2.957295041922758, eps);

  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<double_t, 32> weightedSymbolLengthDistribution{{}};

  symbolLengthDistribution[2] = 30;
  symbolLengthDistribution[3] = 12;
  symbolLengthDistribution[4] = 2;
  symbolLengthDistribution[5] = 1;
  weightedSymbolLengthDistribution[2] = 0.66666666;
  weightedSymbolLengthDistribution[3] = 0.26666666;
  weightedSymbolLengthDistribution[4] = 0.04444444;
  weightedSymbolLengthDistribution[5] = 0.02222222;

  uint32_t sumUnweighted = 0;
  float_t sumWeighted = 0;
  for (size_t i = 0; i < 32; ++i) {
    // BOOST_TEST_MESSAGE(fmt::format("checking length: {}", i));
    BOOST_CHECK_EQUAL(symbolLengthDistribution[i], dataProperies.symbolLengthDistribution[i]);
    BOOST_CHECK_CLOSE(weightedSymbolLengthDistribution[i], dataProperies.weightedSymbolLengthDistribution[i], eps);

    sumUnweighted += dataProperies.symbolLengthDistribution[i];
    sumWeighted += dataProperies.weightedSymbolLengthDistribution[i];
  }

  BOOST_CHECK_EQUAL(sumUnweighted, 45);
  BOOST_CHECK_CLOSE(sumWeighted, 1.0f, eps);
  BOOST_CHECK_EQUAL(coderProperties.renormingPrecisionBits, defaults::MinRenormPrecisionBits);
  BOOST_CHECK_EQUAL(coderProperties.nIncompressibleSymbols, 0);

  const auto& estimate = coderProperties.dictSizeEstimate;
  BOOST_CHECK_EQUAL(estimate.getIndexSize(), 33);
  BOOST_CHECK_EQUAL(estimate.getIndexSizeB(), 5);
  BOOST_CHECK_EQUAL(estimate.getFreqSize(), 224);
  BOOST_CHECK_EQUAL(estimate.getFreqSizeB(), 28);
  BOOST_CHECK_EQUAL(estimate.getSizeB(nUsedAlphabetSymbols, coderProperties.renormingPrecisionBits), 21);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(test_SizeEstimate)
BOOST_AUTO_TEST_CASE(test_emptySizeEstimate)
{
  using source_type = uint32_t;
  Histogram<source_type> histogram{};
  Metrics<uint32_t> metrics{histogram};
  SizeEstimate estimate{metrics};
  BOOST_CHECK_EQUAL(estimate.getEntropySizeB(), 0);
  BOOST_CHECK_EQUAL(estimate.getCompressedDatasetSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getCompressedDictionarySize<>(1.0), 8);
  BOOST_CHECK_EQUAL(estimate.getIncompressibleSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getPackedDatasetSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.preferPacking(1.0), true);
}

BOOST_AUTO_TEST_CASE(test_normalSizeEstimate)
{
  constexpr size_t entropySizeB = 17;

  using source_type = uint32_t;
  std::vector<uint32_t> frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  Histogram<source_type> histogram{frequencies.begin(), frequencies.end(), 0};
  Metrics<uint32_t> metrics{histogram};
  SizeEstimate estimate{metrics};
  BOOST_CHECK_EQUAL(estimate.getEntropySizeB(), entropySizeB);
  BOOST_CHECK_EQUAL(estimate.getCompressedDatasetSize<>(1.0), addEncoderOverheadEstimateB<>(entropySizeB));
  BOOST_CHECK_EQUAL(estimate.getCompressedDictionarySize<>(1.0), 29);
  BOOST_CHECK_EQUAL(estimate.getIncompressibleSize<>(1.0), 0);
  BOOST_CHECK_EQUAL(estimate.getPackedDatasetSize<>(1.0), 29);
  BOOST_CHECK_EQUAL(estimate.preferPacking(1.0), true);
}
BOOST_AUTO_TEST_SUITE_END()