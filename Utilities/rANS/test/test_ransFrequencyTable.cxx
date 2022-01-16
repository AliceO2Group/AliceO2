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

/// @file   test_ransFrequencyTable.cxx
/// @author Michael Lettrich
/// @since  Aug 1, 2020
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

struct ReferenceState {
  int32_t min{};
  int32_t max{};
  size_t size{};
  uint32_t alphabetBitRange{};
  uint32_t nSamples{};
  uint32_t nUsedSymbols{};
  uint32_t nIncompressibleSymbols{};
};

template <typename Fixture_T>
void stateChecker(Fixture_T f, o2::rans::FrequencyTable& frequencyTable)
{
  BOOST_CHECK_EQUAL(frequencyTable.getMinSymbol(), f.min);
  BOOST_CHECK_EQUAL(frequencyTable.getMaxSymbol(), f.max);
  BOOST_CHECK_EQUAL(frequencyTable.size(), f.size);
  BOOST_CHECK_EQUAL(frequencyTable.getAlphabetRangeBits(), f.alphabetBitRange);
  BOOST_CHECK_EQUAL(frequencyTable.getNumSamples(), f.nSamples);
  BOOST_CHECK_EQUAL(frequencyTable.getNUsedAlphabetSymbols(), f.nUsedSymbols);
  BOOST_CHECK_EQUAL(frequencyTable.getIncompressibleSymbolFrequency(), f.nIncompressibleSymbols);
}

struct FrequencyTableFixture {
  FrequencyTableFixture()
  {
    state.min = 0;
    state.max = 0;
    state.alphabetBitRange = 0;
    state.nSamples = 0;
    state.nUsedSymbols = 0;
    state.nIncompressibleSymbols = 0;
  }

  ReferenceState state{};
  o2::rans::FrequencyTable frequencyTable{};
};

struct EmptyFrequencyTable : FrequencyTableFixture {
  EmptyFrequencyTable() : FrequencyTableFixture()
  {
    std::vector<int32_t> A{};
    BOOST_CHECK_NO_THROW(frequencyTable.addSamples(std::begin(A), std::end(A)));
  }
};

struct EmptyFrequencyTableTrim : public FrequencyTableFixture {
  EmptyFrequencyTableTrim() : FrequencyTableFixture()
  {
    BOOST_CHECK_NO_THROW(frequencyTable.trim());
  }
};

struct EmptyFrequencyTableAddSamples : public FrequencyTableFixture {
  EmptyFrequencyTableAddSamples() : FrequencyTableFixture()
  {
    std::vector<int32_t> A{};
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), 0));
  }
};

using emptyFrequencyTables_t = boost::mpl::vector<EmptyFrequencyTable, EmptyFrequencyTableTrim, EmptyFrequencyTableAddSamples>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_empty, fTableContainer_T, emptyFrequencyTables_t)
{
  fTableContainer_T c{};
  auto& frequencyTable = c.frequencyTable;
  stateChecker(c.state, frequencyTable);
  BOOST_CHECK_EQUAL(frequencyTable.begin(), frequencyTable.end());
};

struct AddEmptyFrequencies : public FrequencyTableFixture {
  AddEmptyFrequencies() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    const int32_t min = -3;
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(frequencies.begin(), frequencies.end(), min));
    BOOST_CHECK_EQUAL(frequencyTable.begin(), frequencyTable.end());
  }

  std::vector<int32_t> frequencies{0, 0, 0, 0, 0, 0, 0};
};

struct AddFrequenciesTrim : public FrequencyTableFixture {
  AddFrequenciesTrim() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    const int32_t min = -3;
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(frequencies.begin(), frequencies.end(), min));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++frequencies.begin(), --frequencies.end());

    state.min = -2;
    state.max = 2;
    state.size = 5;
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), 0);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies);
  }

  std::vector<int32_t> frequencies{0, 1, 2, 3, 4, 5, 0};
};

struct addFrequenciesExpandRightOverlap : public FrequencyTableFixture {
  addFrequenciesExpandRightOverlap() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    const int32_t min = -3;
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), min));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), 1));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -2;
    state.max = 6;
    state.size = frequencies.size();
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), 0);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies);
  }

  std::vector<int32_t> A{0, 1, 2, 3, 4, 5, 0};
  std::vector<int32_t> frequencies{1, 2, 3, 4, 6, 2, 3, 4, 5};
};

struct addFrequenciesExpandleftOverlap : public FrequencyTableFixture {
  addFrequenciesExpandleftOverlap() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), -2));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), 0));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -1;
    state.max = 3;
    state.size = frequencies.size();
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), 0);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies);
  }

  std::vector<int32_t> A{0, 1, 2, 3, 0};
  std::vector<int32_t> frequencies{1, 2, 4, 2, 3};
};

struct addFrequenciesIncompressibleRightOverlap : public FrequencyTableFixture {
  addFrequenciesIncompressibleRightOverlap() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    const int32_t min = -3;
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), min));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_EQUAL(frequencyTable.getIncompressibleSymbolFrequency(), 0);
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), 1, false));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -2;
    state.max = 2;
    state.size = frequencies.size();
    state.nIncompressibleSymbols = 14;
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), state.nIncompressibleSymbols);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies) + 1;
  }

  std::vector<int32_t> A{0, 1, 2, 3, 4, 5, 0};
  std::vector<int32_t> frequencies{1, 2, 3, 4, 6};
};

struct addFrequenciesIncompressibleLeftOverlap : public FrequencyTableFixture {
  addFrequenciesIncompressibleLeftOverlap() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), -2));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_EQUAL(frequencyTable.getIncompressibleSymbolFrequency(), 0);
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), 0, false));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -1;
    state.max = 1;
    state.size = frequencies.size();
    state.nIncompressibleSymbols = 5;
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), state.nIncompressibleSymbols);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies) + 1;
  }

  std::vector<int32_t> A{0, 1, 2, 3, 0};
  std::vector<int32_t> frequencies{1, 2, 4};
};

struct resize : public FrequencyTableFixture {
  resize() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), -3));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_EQUAL(frequencyTable.getIncompressibleSymbolFrequency(), 0);
    BOOST_CHECK_NO_THROW(frequencyTable.resize(-5, 9));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -5;
    state.max = 9;
    state.size = frequencies.size();
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), 0);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies);
  }

  std::vector<uint32_t> A{0, 1, 2, 3, 4, 5, 0};
  std::vector<uint32_t> frequencies{0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0};
};

struct resizeTruncate : public FrequencyTableFixture {
  resizeTruncate() : FrequencyTableFixture()
  {
    frequencyTable = o2::rans::FrequencyTable();
    BOOST_CHECK_NO_THROW(frequencyTable.addFrequencies(A.begin(), A.end(), -3));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), ++A.begin(), --A.end());
    BOOST_CHECK_EQUAL(frequencyTable.getIncompressibleSymbolFrequency(), 0);
    BOOST_CHECK_NO_THROW(frequencyTable.resize(-1, 1, true));
    BOOST_CHECK_EQUAL_COLLECTIONS(frequencyTable.begin(), frequencyTable.end(), frequencies.begin(), frequencies.end());

    state.min = -1;
    state.max = 1;
    state.size = frequencies.size();
    state.nIncompressibleSymbols = 6;
    state.nSamples = std::accumulate(frequencies.begin(), frequencies.end(), state.nIncompressibleSymbols);
    state.alphabetBitRange = std::ceil(std::log2(state.max - state.min + 1 + 1));
    state.nUsedSymbols = getNUniqueSymbols(frequencies) + 1;
  }

  std::vector<uint32_t> A{0, 1, 2, 3, 4, 5, 0};
  std::vector<uint32_t> frequencies{2, 3, 4};
};

using frequencyFixtures_t = boost::mpl::vector<AddEmptyFrequencies,
                                               AddFrequenciesTrim,
                                               addFrequenciesExpandRightOverlap,
                                               addFrequenciesExpandleftOverlap,
                                               addFrequenciesIncompressibleRightOverlap,
                                               addFrequenciesIncompressibleLeftOverlap,
                                               resize,
                                               resizeTruncate>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_addFrequencies, fixture_T, frequencyFixtures_t)
{
  fixture_T f;
  auto& frequencyTable = f.frequencyTable;
  stateChecker(f.state, frequencyTable);
};

BOOST_AUTO_TEST_CASE(test_addFrequencies1)
{
  std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5};

  o2::rans::FrequencyTable ftmp;
  ftmp.addSamples(std::begin(A), std::end(A));

  o2::rans::FrequencyTable fA;
  fA.addFrequencies(std::begin(ftmp), std::end(ftmp), ftmp.getMinSymbol());

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), -5);
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), 8);
  BOOST_CHECK_EQUAL(fA.size(), histA.size());
  BOOST_CHECK_EQUAL(fA.getAlphabetRangeBits(), std::ceil(std::log2(histA.size())));
  BOOST_CHECK_EQUAL(fA.getNumSamples(), A.size());
  BOOST_CHECK_EQUAL(fA.getNUsedAlphabetSymbols(), getNUniqueSymbols(histA));

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histA), std::end(histA));

  std::vector<int> B{10, 8, -10};
  o2::rans::FrequencyTable fB;
  fB.addSamples(std::begin(B), std::end(B));

  fA = fA + fB;

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), -10);
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), 10);

  std::vector<uint32_t> histAandB{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 6, 0, 1};
  BOOST_CHECK_EQUAL(fA.size(), histAandB.size());
  BOOST_CHECK_EQUAL(fA.getAlphabetRangeBits(), std::ceil(std::log2(histAandB.size())));
  BOOST_CHECK_EQUAL(fA.getNumSamples(), A.size() + B.size());
  BOOST_CHECK_EQUAL(fA.getNUsedAlphabetSymbols(), getNUniqueSymbols(histAandB));

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histAandB), std::end(histAandB));
}

struct negativeOffset {
  std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5};
  std::vector<int> B{10, -10};
  std::vector<uint32_t> histAandB{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5, 0, 1};
};

struct positiveOffset {
  std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, 2, 7, 3};
  std::vector<uint32_t> histA{1, 1, 0, 2, 2, 1, 5};
  std::vector<int> B{10, -10};
  std::vector<uint32_t> histAandB{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 1, 5, 0, 1};
};

using samples_t = boost::mpl::vector<negativeOffset, positiveOffset>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_addSamples, samples_T, samples_t)
{
  samples_T s{};

  o2::rans::FrequencyTable fA;
  fA.addSamples(std::begin(s.A), std::end(s.A));

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), *std::min_element(s.A.begin(), s.A.end()));
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), *std::max_element(s.A.begin(), s.A.end()));
  BOOST_CHECK_EQUAL(fA.size(), s.histA.size());
  BOOST_CHECK_EQUAL(fA.getAlphabetRangeBits(), std::ceil(std::log2(s.histA.size())));
  BOOST_CHECK_EQUAL(fA.getNumSamples(), s.A.size());
  BOOST_CHECK_EQUAL(fA.getNUsedAlphabetSymbols(), getNUniqueSymbols(s.histA));

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(s.histA), std::end(s.histA));

  fA.addSamples(std::begin(s.B), std::end(s.B));

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), *std::min_element(s.B.begin(), s.B.end()));
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), *std::max_element(s.B.begin(), s.B.end()));

  BOOST_CHECK_EQUAL(fA.size(), s.histAandB.size());
  BOOST_CHECK_EQUAL(fA.getAlphabetRangeBits(), std::ceil(std::log2(s.histAandB.size())));
  BOOST_CHECK_EQUAL(fA.getNumSamples(), std::accumulate(s.histAandB.begin(), s.histAandB.end(), 0));
  BOOST_CHECK_EQUAL(fA.getNUsedAlphabetSymbols(), getNUniqueSymbols(s.histAandB));

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(s.histAandB), std::end(s.histAandB));
}

BOOST_AUTO_TEST_CASE(test_renorm)
{
  o2::rans::histogram_t frequencies{1, 1, 2, 2, 2, 2, 6, 8, 4, 10, 8, 14, 10, 19, 26, 30, 31, 35, 41, 45, 51, 44, 47, 39, 58, 52, 42, 53, 50, 34, 50, 30, 32, 24, 30, 20, 17, 12, 16, 6, 8, 5, 6, 4, 4, 2, 2, 2, 1};
  o2::rans::FrequencyTable frequencyTable{frequencies.begin(), frequencies.end(), 0};

  const size_t scaleBits = 8;

  auto renormedFrequencyTable = o2::rans::renorm(std::move(frequencyTable), scaleBits);
  const o2::rans::histogram_t rescaledFrequencies{1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 3, 3, 4, 6, 7, 7, 9, 9, 11, 12, 10, 11, 9, 13, 12, 10, 13, 11, 8, 12, 7, 7, 6, 7, 4, 4, 3, 4, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1};
  BOOST_CHECK_EQUAL(renormedFrequencyTable.isRenormedTo(scaleBits), true);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getNumSamples(), 1 << scaleBits);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getMinSymbol(), 0);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getMaxSymbol(), 48);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getIncompressibleSymbolFrequency(), 2);
  BOOST_CHECK_EQUAL_COLLECTIONS(renormedFrequencyTable.begin(), renormedFrequencyTable.end(), rescaledFrequencies.begin(), rescaledFrequencies.end());
}

BOOST_AUTO_TEST_CASE(test_renormIncompressible)
{
  o2::rans::histogram_t frequencies{1, 1, 2, 2, 2, 2, 6, 8, 4, 10, 8, 14, 10, 19, 26, 30, 31, 35, 41, 45, 51, 44, 47, 39, 58, 52, 42, 53, 50, 34, 50, 30, 32, 24, 30, 20, 17, 12, 16, 6, 8, 5, 6, 4, 4, 2, 2, 2, 1};
  o2::rans::FrequencyTable frequencyTable{frequencies.begin(), frequencies.end(), 0};

  const size_t scaleBits = 8;

  auto renormedFrequencyTable = o2::rans::renormCutoffIncompressible(std::move(frequencyTable), scaleBits, 1);

  const o2::rans::histogram_t rescaledFrequencies{1, 2, 1, 3, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 13, 11, 12, 10, 14, 13, 10, 13, 12, 8, 12, 7, 8, 6, 7, 5, 4, 3, 4, 2, 2, 1, 2, 1, 1};
  BOOST_CHECK_EQUAL(renormedFrequencyTable.isRenormedTo(scaleBits), true);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getNumSamples(), 1 << scaleBits);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getMinSymbol(), 6);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getMaxSymbol(), 44);
  BOOST_CHECK_EQUAL(renormedFrequencyTable.getIncompressibleSymbolFrequency(), 4);
  BOOST_CHECK_EQUAL_COLLECTIONS(renormedFrequencyTable.begin(), renormedFrequencyTable.end(), rescaledFrequencies.begin(), rescaledFrequencies.end());
}

BOOST_AUTO_TEST_CASE(test_computeEntropy)
{
  o2::rans::histogram_t frequencies{9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1};
  o2::rans::FrequencyTable frequencyTable{frequencies.begin(), frequencies.end(), 0};

  constexpr double entropy = 2.957295041922758;
  const double computedEntropy = o2::rans::computeEntropy(frequencyTable);
  BOOST_CHECK_CLOSE(computedEntropy, entropy, 1e-5);
}