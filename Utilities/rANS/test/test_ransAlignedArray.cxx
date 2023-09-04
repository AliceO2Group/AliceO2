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

/// @file   test_ransAlignedArray.h
/// @author Michael Lettrich
/// @brief  Test class that encapsulates SIMD Vectors

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <type_traits>
#include <array>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/defines.h"

#ifdef RANS_SIMD

#include "rANS/internal/containers/AlignedArray.h"

BOOST_AUTO_TEST_CASE(test_AlignedArray)
{
  using namespace o2::rans::internal::simd;

  using array_t = AlignedArray<uint32_t, SIMDWidth::SSE, 1>;

  const array_t a{1u, 2u, 3u, 4u};
  const std::array<uint32_t, 4> reference{1u, 2u, 3u, 4u};
  BOOST_CHECK_EQUAL(a.nElements(), 4);
  BOOST_CHECK_EQUAL(a.size(), 1);
  BOOST_CHECK_EQUAL(a.data(), &a(0));

  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(a).begin(), gsl::make_span(a).end(), reference.begin(), reference.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(a).rbegin(), gsl::make_span(a).rend(), reference.rbegin(), reference.rend());

  BOOST_CHECK_EQUAL(a[0], gsl::make_span(a.data(), 4));

  const array_t a2{1};
  const std::array<uint32_t, 4> reference2{1u, 1u, 1u, 1u};
  BOOST_CHECK_EQUAL(a2.size(), 1);
  BOOST_CHECK_EQUAL(a2.nElements(), 4);
  BOOST_CHECK_EQUAL(a2.data(), &a2(0));

  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(a2).begin(), gsl::make_span(a2).end(), reference2.begin(), reference2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(a2).rbegin(), gsl::make_span(a2).rend(), reference2.rbegin(), reference2.rend());
};

#else /* !defined(RANS_SIMD) */

BOOST_AUTO_TEST_CASE(test_NoSIMD)
{
  BOOST_TEST_WARN("Tests were not Compiled for SIMD, cannot run all tests");
}

#endif
