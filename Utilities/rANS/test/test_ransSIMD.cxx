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

/// @file   test_ransSIMD.h
/// @author Michael Lettrich
/// @brief  Test rANS SIMD features

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

#if defined(RANS_SIMD)

#include "rANS/internal/common/simdtypes.h"
#include "rANS/internal/common/simdops.h"

using namespace o2::rans::internal::simd;

// clang-format off
using pd_types = boost::mpl::list<pd_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                      , pd_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                      >;

using epi64_types = boost::mpl::list<epi64_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                          , epi64_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                          >;

using epi32_types = boost::mpl::list<epi32_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                          , epi32_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                          >;
// clang-format on

BOOST_AUTO_TEST_CASE(test_getLaneWidthBits)
{
  BOOST_CHECK_EQUAL(getLaneWidthBits(SIMDWidth::SSE), 128);
  BOOST_CHECK_EQUAL(getLaneWidthBits(SIMDWidth::AVX), 256);
};

BOOST_AUTO_TEST_CASE(test_getLaneWidthBytes)
{
  BOOST_CHECK_EQUAL(getLaneWidthBytes(SIMDWidth::SSE), 128 / 8);
  BOOST_CHECK_EQUAL(getLaneWidthBytes(SIMDWidth::AVX), 256 / 8);
};

BOOST_AUTO_TEST_CASE(test_getAlignment)
{
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::SSE), 16);
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::AVX), 32);
};

BOOST_AUTO_TEST_CASE(test_getElementCount)
{
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::SSE), 16);
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::AVX), 32);
};

struct ConvertingFixture64 {
  std::vector<uint64_t> uint64Data = {0x0, 0x1, 0xFFFFFFFFFFFFE, 0xFFFFFFFFFFFFF};
  std::vector<double> doubleData{};

  ConvertingFixture64()
  {
    for (auto i : uint64Data) {
      doubleData.push_back(static_cast<double>(i));
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(test_SIMDconvert64, ConvertingFixture64)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_uint64ToDouble, epi64_T, epi64_types)
{
  for (size_t i = 0; i < uint64Data.size(); ++i) {
    const epi64_T src{uint64Data[i]};
    const auto dest = store(uint64ToDouble(load(src)));

    for (auto elem : gsl::make_span(dest)) {
      BOOST_CHECK_EQUAL(elem, doubleData[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_doubleToUint64, pd_T, pd_types)
{
  for (size_t i = 0; i < doubleData.size(); ++i) {
    const pd_T src{doubleData[i]};
    const auto dest = store<uint64_t>(doubleToUint64(load(src)));

    for (auto elem : gsl::make_span(dest)) {
      BOOST_CHECK_EQUAL(elem, uint64Data[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

struct ConvertingFixture32 {
  epi32_t<SIMDWidth::SSE> uint32Data = {0x0, 0x1, 0x7FFFFFFE, 0x7FFFFFFF};
  std::vector<double> doubleData;

  ConvertingFixture32()
  {
    for (auto i : gsl::make_span(uint32Data)) {
      doubleData.push_back(static_cast<double>(i));
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(test_SIMDconvert32, ConvertingFixture32)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_int32ToDouble, epi32_T, epi32_types)
{
  constexpr SIMDWidth simdWidth_V = simdWidth_v<epi32_T>;

  for (size_t i = 0; i < uint32Data.size(); ++i) {
    const epi32_t<SIMDWidth::SSE> src{uint32Data(i)};
    auto dest = store(int32ToDouble<simdWidth_V>(load(src)));

    for (auto elem : gsl::make_span(dest)) {
      BOOST_CHECK_EQUAL(elem, doubleData[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

struct ModDivFixture {
  std::vector<uint32_t> numerator = {1, 6, 17};
  std::vector<uint32_t> denominator = {1, 3, 4};
  // test 1: mod = 0, div correctly rounded
  // test 2: div = 0, mod correclty rounded
  // test 3: mod, div nonzero and correctly rounded
  std::array<uint32_t, 3> mod;
  std::array<uint32_t, 3> div;

  ModDivFixture()
  {
    for (size_t i = 0; i < numerator.size(); ++i) {
      div[i] = numerator[i] / denominator[i];
      mod[i] = numerator[i] % denominator[i];
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(testModDiv, ModDivFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(modDiv, pd_T, pd_types)
{
  for (size_t i = 0; i < numerator.size(); ++i) {
    const pd_T numeratorPD{static_cast<double>(numerator[i])};
    const pd_T denominatorPD{static_cast<double>(denominator[i])};

    auto [divPDVec, modPDVec] = divMod(load(numeratorPD), load(denominatorPD));

    pd_T divPD = store(divPDVec);
    pd_T modPD = store(modPDVec);

    pd_T modResult{static_cast<double>(mod[i])};
    pd_T divResult{static_cast<double>(div[i])};

    BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(divResult).begin(), gsl::make_span(divResult).end(), gsl::make_span(divPD).begin(), gsl::make_span(divPD).end());
    BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(modResult).begin(), gsl::make_span(modResult).end(), gsl::make_span(modPD).begin(), gsl::make_span(modPD).end());
  }
}
BOOST_AUTO_TEST_SUITE_END()

#ifndef RANS_AVX2
BOOST_AUTO_TEST_CASE(test_NoAVX2)
{
  BOOST_TEST_WARN("Tests were not Compiled for AVX2, cannot run all tests");
}
#endif /* RANS_AVX2 */

#else /* !defined(RANS_SIMD) */

BOOST_AUTO_TEST_CASE(test_NoSIMD)
{
  BOOST_TEST_WARN("Tests were not Compiled for SIMD, cannot run all tests");
}

#endif /* RANS_SIMD */