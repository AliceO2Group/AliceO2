// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

BOOST_AUTO_TEST_CASE(test_empty)
{
  std::vector<int> A{};

  o2::rans::FrequencyTable frequencyTable;
  frequencyTable.addSamples(std::begin(A), std::end(A));

  BOOST_CHECK_EQUAL(frequencyTable.getMinSymbol(), 0);
  BOOST_CHECK_EQUAL(frequencyTable.getMaxSymbol(), 0);
  BOOST_CHECK_EQUAL(frequencyTable.size(), 0);
  BOOST_CHECK_EQUAL(frequencyTable.getAlphabetRangeBits(), 0);
  BOOST_CHECK_EQUAL(frequencyTable.getNumSamples(), 0);
  BOOST_CHECK_EQUAL(frequencyTable.getNUsedAlphabetSymbols(), 0);

  BOOST_CHECK_EQUAL(frequencyTable.begin(), frequencyTable.end());
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

BOOST_AUTO_TEST_CASE(test_addFrequencies)
{
  std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5};

  o2::rans::FrequencyTable ftmp;
  ftmp.addSamples(std::begin(A), std::end(A));

  o2::rans::FrequencyTable fA;
  fA.addFrequencies(std::begin(ftmp), std::end(ftmp), ftmp.getMinSymbol(), ftmp.getMaxSymbol());

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
