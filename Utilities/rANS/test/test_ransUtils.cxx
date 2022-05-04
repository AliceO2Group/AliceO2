// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransSIMDEncoder.h
/// @author Michael Lettrich
/// @since  2020-04-15
/// @brief  Test rANS SIMD encoder/ decoder

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <vector>
#include <type_traits>
#include <array>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"

BOOST_AUTO_TEST_CASE(test_checkBounds)
{
  std::vector<size_t> A(2);
  BOOST_CHECK_THROW(o2::rans::checkBounds(std::end(A), std::begin(A)), o2::rans::OutOfBoundsError);
  BOOST_CHECK_NO_THROW(o2::rans::checkBounds(std::begin(A), std::end(A)));
};

BOOST_AUTO_TEST_CASE(test_checkAlphabetBitRange)
{
  using namespace o2::rans::internal;
  BOOST_CHECK_EQUAL(getRangeBits(-1, -1), 0ul); // empty or single value -> 2**0 = 1
  BOOST_CHECK_EQUAL(getRangeBits(-1, 0), 1ul);  // 2 unique values -> 1 Bit
  BOOST_CHECK_EQUAL(getRangeBits(-1, 1), 2ul);  // 3 unique values -> 2 Bits
  BOOST_CHECK_EQUAL(getRangeBits(-1, 2), 2ul);  // 4 unique values -> 2 Bits
  BOOST_CHECK_EQUAL(getRangeBits(-1, 3), 3ul);  // 5 unique values -> 3 Bits
};