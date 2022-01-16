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

#include <vector>
#include <type_traits>
#include <array>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <fairlogger/Logger.h>

#include "rANS/rans.h"
#include "rANS/internal/backend/simd/types.h"

using namespace o2::rans::internal::simd;

BOOST_AUTO_TEST_CASE(test_getLaneWidthBits)
{
  BOOST_CHECK_EQUAL(getLaneWidthBits(SIMDWidth::SSE), 128);
  BOOST_CHECK_EQUAL(getLaneWidthBits(SIMDWidth::AVX), 256);
}

BOOST_AUTO_TEST_CASE(test_getLaneWidthBytes)
{
  BOOST_CHECK_EQUAL(getLaneWidthBytes(SIMDWidth::SSE), 128 / 8);
  BOOST_CHECK_EQUAL(getLaneWidthBytes(SIMDWidth::AVX), 256 / 8);
}

BOOST_AUTO_TEST_CASE(test_getAlignment)
{
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::SSE), 16);
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::AVX), 32);
}

BOOST_AUTO_TEST_CASE(test_getElementCount)
{
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::SSE), 16);
  BOOST_CHECK_EQUAL(getAlignment(SIMDWidth::AVX), 32);
}

BOOST_AUTO_TEST_CASE(test_AlignedArray)
{
  using array_t = AlignedArray<uint32_t, SIMDWidth::SSE, 4>;

  const array_t a{1u, 2u, 3u, 4u};
  const std::array<uint32_t, 4> reference{1u, 2u, 3u, 4u};
  BOOST_CHECK_EQUAL(a.size(), 4);
  BOOST_CHECK_EQUAL(a.data(), &(a[0]));
  BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), reference.begin(), reference.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a.rbegin(), a.rend(), reference.rbegin(), reference.rend());

  const array_t a2{1};
  const std::array<uint32_t, 4> reference2{1u, 1u, 1u, 1u};
  BOOST_CHECK_EQUAL(a2.size(), 4);
  BOOST_CHECK_EQUAL(a2.data(), &(a2[0]));
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), reference2.begin(), reference2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.rbegin(), a2.rend(), reference2.rbegin(), reference2.rend());
}

BOOST_AUTO_TEST_CASE(test_ArrayView_array)
{
  const std::array<uint32_t, 16> a{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};

  ArrayView av{a};
  BOOST_CHECK_EQUAL(av.size(), a.size());
  BOOST_CHECK_EQUAL(av.data(), a.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.begin(), av.end(), a.begin(), a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.rbegin(), av.rend(), a.rbegin(), a.rend());
  auto sub = av.subView<8, 8>();
  BOOST_CHECK_EQUAL(sub.size(), 8);
  BOOST_CHECK_EQUAL(sub.data(), a.data() + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), a.begin() + 8, a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.rbegin(), sub.rend(), a.rbegin(), a.rend() - 8);
}

BOOST_AUTO_TEST_CASE(test_ArrayView_alignedArray)
{
  const AlignedArray<uint32_t, SIMDWidth::SSE, 16> a{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};

  ArrayView av{a};
  BOOST_CHECK_EQUAL(av.size(), a.size());
  BOOST_CHECK_EQUAL(av.data(), a.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.begin(), av.end(), a.begin(), a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.rbegin(), av.rend(), a.rbegin(), a.rend());
  auto sub = av.subView<8, 8>();
  BOOST_CHECK_EQUAL(sub.size(), 8);
  BOOST_CHECK_EQUAL(sub.data(), a.data() + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), a.begin() + 8, a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.rbegin(), sub.rend(), a.rbegin(), a.rend() - 8);
}

BOOST_AUTO_TEST_CASE(test_ArrayView_carray)
{
  constexpr size_t SizeA = 16;
  const uint32_t a[SizeA] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};

  ArrayView av{a};
  BOOST_CHECK_EQUAL(av.size(), SizeA);
  BOOST_CHECK_EQUAL(av.data(), a);
  BOOST_CHECK_EQUAL_COLLECTIONS(av.begin(), av.end(), a, a + SizeA);
  auto sub = av.subView<8, 8>();
  BOOST_CHECK_EQUAL(sub.size(), 8);
  BOOST_CHECK_EQUAL(sub.data(), a + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), a + 8, a + SizeA);
}

BOOST_AUTO_TEST_CASE(test_SIMDView_array)
{
  const std::array<uint64_t, 16> a{1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull};
  const std::array<uint64_t, 8> check{1ull, 3ull, 5ull, 7ull, 9ull, 11ull, 13ull, 15ull};

  SIMDView<const uint64_t, SIMDWidth::SSE, 8, false> sv{a};
  BOOST_CHECK_EQUAL(sv.size(), 8);
  BOOST_CHECK_EQUAL(sv.data(), a.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(sv.begin(), sv.end(), check.begin(), check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sv.rbegin(), sv.rend(), check.rbegin(), check.rend());
  auto sub = sv.subView<4, 4>();
  BOOST_CHECK_EQUAL(sub.size(), 4);
  BOOST_CHECK_EQUAL(sub.data(), a.data() + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), check.begin() + 4, check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.rbegin(), sub.rend(), check.rbegin(), check.rend() - 4);
}

BOOST_AUTO_TEST_CASE(test_SIMDView_alignedArray)
{
  AlignedArray<uint64_t, SIMDWidth::SSE, 16> a{1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull};
  const std::array<uint64_t, 8> check{1ull, 3ull, 5ull, 7ull, 9ull, 11ull, 13ull, 15ull};

  SIMDView<const uint64_t, SIMDWidth::SSE, 8, true> sv{a};
  BOOST_CHECK_EQUAL(sv.size(), 8);
  BOOST_CHECK_EQUAL(sv.data(), a.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(sv.begin(), sv.end(), check.begin(), check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sv.rbegin(), sv.rend(), check.rbegin(), check.rend());

  auto sub = sv.subView<4, 4>();
  BOOST_CHECK_EQUAL(sub.size(), 4);
  BOOST_CHECK_EQUAL(sub.data(), a.data() + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), check.begin() + 4, check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.rbegin(), sub.rend(), check.rbegin(), check.rend() - 4);

  for (size_t i = 0; i < sv.size(); ++i) {
    BOOST_CHECK((isAligned<const uint64_t, SIMDWidth::SSE>(&sv[i])));
  }

  auto av = static_cast<ArrayView<const uint64_t, 16>>(sv);
  BOOST_CHECK_EQUAL(av.size(), a.size());
  BOOST_CHECK_EQUAL(av.data(), a.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.begin(), av.end(), a.begin(), a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(av.rbegin(), av.rend(), a.rbegin(), a.rend());

  auto unaligned = static_cast<SIMDView<const uint64_t, SIMDWidth::SSE, 8, false>>(sv);
  BOOST_CHECK_EQUAL(unaligned.size(), 8);
  BOOST_CHECK_EQUAL(unaligned.data(), a.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(unaligned.begin(), unaligned.end(), check.begin(), check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(unaligned.rbegin(), unaligned.rend(), check.rbegin(), check.rend());

  auto aligned = static_cast<SIMDView<const uint64_t, SIMDWidth::SSE, 8, true>>(unaligned);
  BOOST_CHECK_EQUAL(aligned.size(), 8);
  BOOST_CHECK_EQUAL(aligned.data(), a.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(aligned.begin(), aligned.end(), check.begin(), check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(aligned.rbegin(), aligned.rend(), check.rbegin(), check.rend());
}

BOOST_AUTO_TEST_CASE(test_SIMDView_carray)
{
  constexpr size_t SizeA = 16;
  const uint64_t a[SizeA] = {1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull, 10ull, 11ull, 12ull, 13ull, 14ull, 15ull, 16ull};
  const std::array<uint64_t, 8> check{1ull, 3ull, 5ull, 7ull, 9ull, 11ull, 13ull, 15ull};

  SIMDView<const uint64_t, SIMDWidth::SSE, 8, false> sv{a};
  BOOST_CHECK_EQUAL(sv.size(), 8);
  BOOST_CHECK_EQUAL(sv.data(), a);
  BOOST_CHECK_EQUAL_COLLECTIONS(sv.begin(), sv.end(), check.begin(), check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sv.rbegin(), sv.rend(), check.rbegin(), check.rend());
  auto sub = sv.subView<4, 4>();
  BOOST_CHECK_EQUAL(sub.size(), 4);
  BOOST_CHECK_EQUAL(sub.data(), a + 8);
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.begin(), sub.end(), check.begin() + 4, check.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(sub.rbegin(), sub.rend(), check.rbegin(), check.rend() - 4);
}