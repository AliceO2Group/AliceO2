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

#include "rANS/rans.h"

BOOST_AUTO_TEST_CASE(test_addSamples)
{
  std::vector<int> A{5, 5, 6, 6, 8, 8, 8, 8, 8, -1, -5, 2, 7, 3};
  std::vector<uint32_t> histA{1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5};

  o2::rans::FrequencyTable fA;
  fA.addSamples(std::begin(A), std::end(A));

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), -5);
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), 8);
  BOOST_CHECK_EQUAL(fA.size(), histA.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histA), std::end(histA));

  std::vector<int> B{10, -10};
  fA.addSamples(std::begin(B), std::end(B));

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), -10);
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), 10);

  std::vector<uint32_t> histAandB{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 5, 0, 1};
  BOOST_CHECK_EQUAL(fA.size(), histAandB.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histAandB), std::end(histAandB));
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

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histA), std::end(histA));

  std::vector<int> B{10, 8, -10};
  o2::rans::FrequencyTable fB;
  fB.addSamples(std::begin(B), std::end(B));

  fA = fA + fB;

  BOOST_CHECK_EQUAL(fA.getMinSymbol(), -10);
  BOOST_CHECK_EQUAL(fA.getMaxSymbol(), 10);

  std::vector<uint32_t> histAandB{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 6, 0, 1};
  BOOST_CHECK_EQUAL(fA.size(), histAandB.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(fA), std::end(fA), std::begin(histAandB), std::end(histAandB));
}
