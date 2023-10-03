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
#include <boost/mpl/vector.hpp>
#include <gsl/span>

#include "rANS/internal/containers/SparseVector.h"

using namespace o2::rans;
using namespace o2::rans::internal;

using source_type = int32_t;
using sparseVector_type = SparseVector<source_type, uint32_t>;

BOOST_AUTO_TEST_CASE(test_empty)
{

  sparseVector_type vec{};

  BOOST_CHECK_EQUAL(vec.empty(), true);
  BOOST_CHECK_EQUAL(vec.size(), 0);
  BOOST_CHECK(vec.begin() == vec.end());
  BOOST_CHECK(vec.cbegin() == vec.cend());
};

BOOST_AUTO_TEST_CASE(test_write)
{
  sparseVector_type vec{};

  std::vector<source_type> samples{-5, -2, 1, 3, 5, 8, -5, 5, 1, 1, 1, 14, 8, 8, 8, 8, 8, 8, 8, 8, utils::pow2(18) + 1};

  std::unordered_map<source_type, uint32_t> results{{-5, 2}, {-2, 1}, {-1, 0}, {0, 0}, {1, 4}, {2, 0}, {3, 1}, {4, 0}, {5, 2}, {8, 9}, {14, 1}, {utils::pow2(18) + 1, 1}};

  std::for_each(samples.begin(), samples.end(), [&vec](const source_type& s) { ++vec[s]; });

  BOOST_CHECK_EQUAL(vec.empty(), false);
  BOOST_CHECK_EQUAL(vec.getOffset(), std::numeric_limits<source_type>::min());
  BOOST_CHECK_EQUAL(vec.getOffset(), std::numeric_limits<source_type>::min());
  BOOST_CHECK_EQUAL(vec.size(), 3 * vec.getBucketSize());

  for (auto [key, value] : results) {
    BOOST_CHECK_EQUAL(vec[key], value);
  }

  const auto& vecRef = vec;

  for (auto [key, value] : results) {
    BOOST_CHECK_EQUAL(vecRef[key], value);
  }

  // iterate
  BOOST_CHECK(vec.begin() != vec.end());
  BOOST_CHECK(vec.cbegin() != vec.cend());

  BOOST_TEST_MESSAGE("testing forward iterators");
  for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
    auto value = iter->second;
    if (value > 0) {
      BOOST_TEST_MESSAGE(fmt::format("checking symbol [{}]:{}", iter->first, value));
      BOOST_CHECK_EQUAL(results[iter->first], value);
    }
  }

  BOOST_TEST_MESSAGE("testing const forward iterators");
  for (auto iter = vec.cbegin(); iter != vec.cend(); ++iter) {
    auto value = iter->second;
    if (value > 0) {
      BOOST_TEST_MESSAGE(fmt::format("checking symbol [{}]:{}", iter->first, value));
      BOOST_CHECK_EQUAL(results[iter->first], value);
    }
  }
}