// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransCombinedIterator.cxx
/// @author michael.lettrich@cern.ch
/// @since  2020-10-28
/// @brief

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "rANS/utils.h"

struct test_CombninedIteratorFixture {
  std::vector<uint16_t> a{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
  std::vector<uint16_t> b{a.rbegin(), a.rend()};
  size_t shift = 16;
  std::vector<uint32_t> aAndB{0x0001000f, 0x0002000e, 0x0003000d, 0x0004000c, 0x0005000b,
                              0x0006000a, 0x00070009, 0x00080008, 0x00090007, 0x000a0006,
                              0x000b0005, 0x000c0004, 0x000d0003, 0x000e0002, 0x000f0001};
};

class ReadShiftFunctor
{
 public:
  ReadShiftFunctor(size_t shift) : mShift(shift){};

  template <typename iterA_T, typename iterB_T>
  inline uint32_t operator()(iterA_T iterA, iterB_T iterB) const
  {
    return *iterB + (static_cast<uint32_t>(*iterA) << mShift);
  };

 private:
  size_t mShift;
};

class WriteShiftFunctor
{
 public:
  WriteShiftFunctor(size_t shift) : mShift(shift){};

  template <typename iterA_T, typename iterB_T>
  inline void operator()(iterA_T iterA, iterB_T iterB, uint32_t value) const
  {
    *iterA = value >> mShift;
    *iterB = value & ((1 << mShift) - 1);
  };

 private:
  size_t mShift;
};

BOOST_FIXTURE_TEST_CASE(test_CombinedInputIteratorBase, test_CombninedIteratorFixture)
{

  //  auto readOP = [](auto iterA, auto iterB) -> uint32_t {
  //    return *iterB + (static_cast<uint32_t>(*iterA) << 16);
  //  };

  ReadShiftFunctor f(shift);

  o2::rans::utils::CombinedInputIterator iter(a.begin(), b.begin(), f);
  // test equal
  const o2::rans::utils::CombinedInputIterator first(a.begin(), b.begin(), f);
  BOOST_CHECK_EQUAL(iter, first);
  // test not equal
  const o2::rans::utils::CombinedInputIterator second(++(a.begin()), ++(b.begin()), f);
  BOOST_CHECK_NE(iter, second);
  // test pre increment
  ++iter;
  BOOST_CHECK_EQUAL(iter, second);
  //test postIncrement
  iter = first;
  BOOST_CHECK_EQUAL(iter++, first);
  BOOST_CHECK_EQUAL(iter, second);
  //test deref
  const uint32_t val = first.operator*();
  BOOST_CHECK_EQUAL(val, aAndB.front());
}

BOOST_FIXTURE_TEST_CASE(test_CombinedOutputIteratorBase, test_CombninedIteratorFixture)
{
  std::vector<uint16_t> aOut(2, 0x0);
  std::vector<uint16_t> bOut(2, 0x0);

  //  auto writeOP = [](auto iterA, auto iterB, uint32_t value) -> void {
  //    const uint32_t shift = 16;
  //    *iterA = value >> shift;
  //    *iterB = value & ((1 << shift) - 1);
  //  };

  WriteShiftFunctor f(shift);

  o2::rans::utils::CombinedOutputIterator iter(aOut.begin(), bOut.begin(), f);

  // test deref:
  *iter = aAndB[0];
  BOOST_CHECK_EQUAL(aOut[0], a[0]);
  BOOST_CHECK_EQUAL(bOut[0], b[0]);
  aOut[0] = 0x0;
  bOut[0] = 0x0;

  // test pre increment
  *(++iter) = aAndB[1];
  BOOST_CHECK_EQUAL(aOut[0], 0);
  BOOST_CHECK_EQUAL(bOut[0], 0);
  BOOST_CHECK_EQUAL(aOut[1], a[1]);
  BOOST_CHECK_EQUAL(bOut[1], b[1]);
  aOut.assign(2, 0x0);
  bOut.assign(2, 0x0);
  iter = o2::rans::utils::CombinedOutputIterator(aOut.begin(), bOut.begin(), f);

  // test post increment
  auto preInc = iter++;
  *preInc = aAndB[0];
  BOOST_CHECK_EQUAL(aOut[0], a[0]);
  BOOST_CHECK_EQUAL(bOut[0], b[0]);
  BOOST_CHECK_EQUAL(aOut[1], 0x0);
  BOOST_CHECK_EQUAL(bOut[1], 0x0);
  aOut.assign(2, 0x0);
  bOut.assign(2, 0x0);
  *iter = aAndB[1];
  BOOST_CHECK_EQUAL(aOut[0], 0);
  BOOST_CHECK_EQUAL(bOut[0], 0);
  BOOST_CHECK_EQUAL(aOut[1], a[1]);
  BOOST_CHECK_EQUAL(bOut[1], b[1]);
}

BOOST_FIXTURE_TEST_CASE(test_CombinedInputIteratorReadArray, test_CombninedIteratorFixture)
{
  //  auto readOP = [](auto iterA, auto iterB) -> uint32_t {
  //    return *iterB + (static_cast<uint32_t>(*iterA) << 16);
  //  };
  ReadShiftFunctor f(shift);

  const o2::rans::utils::CombinedInputIterator begin(a.begin(), b.begin(), f);
  const o2::rans::utils::CombinedInputIterator end(a.end(), b.end(), f);
  BOOST_CHECK_EQUAL_COLLECTIONS(begin, end, aAndB.begin(), aAndB.end());
}

BOOST_FIXTURE_TEST_CASE(test_CombinedOutputIteratorWriteArray, test_CombninedIteratorFixture)
{
  std::vector<uint16_t> aRes(a.size(), 0);
  std::vector<uint16_t> bRes(b.size(), 0);

  o2::rans::utils::CombinedOutputIterator iter(aRes.begin(), bRes.begin(), WriteShiftFunctor(shift));
  for (auto input : aAndB) {
    *iter++ = input;
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(aRes.begin(), aRes.end(), a.begin(), a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(bRes.begin(), bRes.end(), b.begin(), b.end());
}
