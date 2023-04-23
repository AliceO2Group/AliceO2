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

/// @file   test_BitPtr.cxx
/// @author michael.lettrich@cern.ch
/// @brief test helper class that allows to point to a Bit in memory

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>

#include "rANS/internal/containers/BitPtr.h"

using namespace o2::rans;
using namespace o2::rans::internal;
using namespace o2::rans::utils;

using source_types = boost::mp11::mp_list<int8_t, int16_t, int32_t>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_BitPtrConstructors, source_type, source_types)
{
  BitPtr bitPtr{};
  BOOST_CHECK_EQUAL(bitPtr.getBitAddress(), 0);
  BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), nullptr);
  BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), 0);

  source_type* ptr = nullptr;
  bitPtr = BitPtr{ptr};
  BOOST_CHECK_EQUAL(bitPtr.getBitAddress(), 0);
  BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), nullptr);
  BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), 0);

  source_type v{};
  ptr = &v;
  bitPtr = BitPtr{ptr};
  BOOST_CHECK_EQUAL(bitPtr.getBitAddress(), adr2Bits(ptr));
  BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), ptr);
  BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), 0);

  intptr_t offset = 6;
  bitPtr = BitPtr{ptr, offset};
  BOOST_CHECK_EQUAL(bitPtr.getBitAddress(), adr2Bits(ptr) + offset);
  BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), ptr);
  BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), offset);

  offset = 13;
  bitPtr = BitPtr{ptr, offset};
  BOOST_CHECK_EQUAL(bitPtr.getBitAddress(), adr2Bits(ptr) + offset);

  if constexpr (sizeof(source_type) == 1) {
    BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), ptr + 1);
    BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), offset - 8);
  } else {
    BOOST_CHECK_EQUAL(bitPtr.toPtr<source_type>(), ptr);
    BOOST_CHECK_EQUAL(bitPtr.getOffset<source_type>(), offset);
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(test_BitPtr, source_type, source_types)
{
  source_type v[5]{};
  source_type* smallPtr = &v[0];
  source_type* bigPtr = &v[4];
  BitPtr smallBitPtr{smallPtr};
  BitPtr bigBitPtr{bigPtr};

  BitPtr smallBitPtrPlusOne{smallPtr, 1};

  // test equal
  BOOST_CHECK_EQUAL(smallBitPtr, smallBitPtr);
  // test not equal
  BOOST_CHECK_NE(smallBitPtr, bigBitPtr);
  // test smaller
  BOOST_CHECK_LT(smallBitPtr, bigBitPtr);
  // test greater
  BOOST_CHECK_GT(bigBitPtr, smallBitPtr);
  // test greater-equals
  BOOST_CHECK_GE(bigBitPtr, smallBitPtr);
  BOOST_CHECK_GE(smallBitPtr, smallBitPtr);
  // test smaller-equals
  BOOST_CHECK_LE(smallBitPtr, bigBitPtr);
  BOOST_CHECK_LE(smallBitPtr, smallBitPtr);

  // test pre-increment
  BitPtr ptr = smallBitPtr;
  ++ptr;
  BOOST_CHECK_EQUAL(ptr, smallBitPtrPlusOne);
  // test post-increment
  ptr = smallBitPtr;
  BOOST_CHECK_EQUAL(ptr++, smallBitPtr);
  BOOST_CHECK_EQUAL(ptr, smallBitPtrPlusOne);
  // test pre-decrement
  ptr = smallBitPtrPlusOne;
  --ptr;
  BOOST_CHECK_EQUAL(ptr, smallBitPtr);
  // test post-decrement
  ptr = smallBitPtrPlusOne;
  BOOST_CHECK_EQUAL(ptr--, smallBitPtrPlusOne);
  BOOST_CHECK_EQUAL(ptr, smallBitPtr);

  intptr_t increment = toBits((bigPtr - smallPtr) * sizeof(source_type));

  // test +=
  ptr = smallBitPtr;
  ptr += increment;
  BOOST_CHECK_EQUAL(ptr, bigBitPtr);
  // test +
  ptr = smallBitPtr;
  BOOST_CHECK_EQUAL(smallBitPtr + increment, bigBitPtr);
  BOOST_CHECK_EQUAL(increment + smallBitPtr, bigBitPtr);

  // check -=
  ptr = bigBitPtr;
  ptr -= increment;
  BOOST_CHECK_EQUAL(ptr, smallBitPtr);

  // check -
  BOOST_CHECK_EQUAL(bigBitPtr - increment, smallBitPtr);

  // check -
  BOOST_CHECK_EQUAL(bigBitPtr - smallBitPtr, increment);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_BitPtrCasts, source_type, source_types)
{
  source_type v[5]{};

  intptr_t distance = 3;
  intptr_t bitDistance = toBits<source_type>();

  const BitPtr begin{&v[0]};
  const BitPtr belowMinus{&v[1], -distance};
  const BitPtr below{&v[1]};
  BOOST_CHECK_EQUAL(begin.toPtr<source_type>(), belowMinus.toPtr<source_type>());
  const BitPtr ptr{&v[2]};
  const BitPtr above{&v[3]};
  const BitPtr abovePlus{&v[3], distance};
  BOOST_CHECK_EQUAL(abovePlus.toPtr<source_type>(), above.toPtr<source_type>());

  BOOST_CHECK_EQUAL(ptr - (bitDistance + distance), belowMinus);
  BOOST_CHECK_EQUAL(ptr - (bitDistance), below);
  BOOST_CHECK_EQUAL(ptr + (bitDistance), above);
  BOOST_CHECK_EQUAL(ptr + (bitDistance + distance), abovePlus);
}