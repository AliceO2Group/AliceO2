// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#define BOOST_TEST_MODULE Test MCHRaw bitset
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "BitSet.h"
#include <vector>
#include <fmt/format.h>
#include <ctime>

using namespace o2::mch::raw;

uint64_t allones = 0x3FFFFFFFFFFFF;
BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(bitset)

// Most of the tests (and their names) are adapted from github.com/mrrtf/sampa/pkg/bitset/bitset_test.go

BOOST_AUTO_TEST_CASE(TestCount)
{
  BitSet bs;
  BOOST_CHECK_NO_THROW(bs.set(1, true));
  BOOST_CHECK_NO_THROW(bs.set(2, true));
  BOOST_CHECK_NO_THROW(bs.set(3, false));
  BOOST_CHECK_NO_THROW(bs.set(9, true));
  BOOST_CHECK_EQUAL(bs.count(), 3);
}

BOOST_AUTO_TEST_CASE(TestAppend)
{
  BitSet bs;
  bs.append(true);
  bs.append(true);
  bs.append(false);
  bs.append(false);
  bs.append(true);
  BOOST_CHECK_EQUAL(bs.uint8(0, 5), 0x13);
  BitSet bs2;
  BOOST_CHECK_EQUAL(bs2.len(), 0);
  bs2.append(static_cast<uint64_t>(0x1555540F00113), 50);
  BOOST_CHECK_EQUAL(bs2.len(), 50);
}

BOOST_AUTO_TEST_CASE(TestPruneFirst)
{
  BitSet bs;
  BOOST_CHECK_NO_THROW(bs.setRangeFromString(0, 6, "1101011"));
  bs.pruneFirst(2);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "01011");
}

BOOST_AUTO_TEST_CASE(TestAny)
{
  BitSet bs;
  BOOST_CHECK_EQUAL(bs.any(), false);
  bs.set(2, true);
  BOOST_CHECK_EQUAL(bs.any(), true);
}

BOOST_AUTO_TEST_CASE(TestNew)
{
  BOOST_CHECK_NO_THROW(BitSet a(static_cast<uint8_t>(100)));
}

BOOST_AUTO_TEST_CASE(TestSet)
{
  BitSet bs;
  BOOST_CHECK_NO_THROW(bs.set(0, true));
  BOOST_CHECK_NO_THROW(bs.set(2, true));
  BOOST_CHECK_NO_THROW(bs.set(20, true));
  BOOST_CHECK_EQUAL(bs.size(), 32);
  BOOST_CHECK_EQUAL(bs.len(), 21);
  BOOST_CHECK_THROW(bs.set(-1, true), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(TestGet)
{
  BitSet bs;
  BOOST_CHECK_NO_THROW(bs.set(0, true));
  BOOST_CHECK_NO_THROW(bs.set(2, true));
  BOOST_CHECK_EQUAL(bs.get(0), true);
  BOOST_CHECK_EQUAL(bs.get(2), true);
  BOOST_CHECK_EQUAL(bs.get(1), false);
  BOOST_CHECK_THROW(bs.get(100), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(TestClear)
{
  BitSet bs;
  bs.set(24, true);
  BOOST_CHECK_EQUAL(bs.len(), 25);
  bs.clear();
  BOOST_CHECK_EQUAL(bs.len(), 0);
}

BOOST_AUTO_TEST_CASE(TestString)
{
  BitSet bs;
  bs.set(1, true);
  bs.set(3, true);
  bs.set(5, true);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "010101");
}

BOOST_AUTO_TEST_CASE(TestFromString)
{
  BOOST_CHECK_THROW(BitSet("00011x"), std::invalid_argument);
  BitSet bs("01011011");
  BOOST_CHECK_EQUAL(bs.len(), 8);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "01011011");
}

BOOST_AUTO_TEST_CASE(TestRangeFromString)
{
  BitSet bs("110011");
  BOOST_CHECK_NO_THROW(bs.setRangeFromString(2, 3, "11"));
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "111111");
  BOOST_CHECK_THROW(bs.setRangeFromString(2, 3, "x-"), std::invalid_argument);
  BOOST_CHECK_THROW(bs.setRangeFromString(4, 1, "101"), std::invalid_argument);
  BOOST_CHECK_THROW(bs.setRangeFromString(32, 38, "1100"), std::invalid_argument);
  BOOST_CHECK_NO_THROW(bs.setRangeFromString(32, 38, "1100110"));
  BOOST_CHECK_THROW(BitSet("abcd"), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(TestFromIntegers)
{
  uint64_t v = 0xF0F8FCFEFF3F3F1F;
  BitSet bs(v);
  std::string s = "1111100011111100111111001111111101111111001111110001111100001111";
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), s);
  uint64_t x = bs.uint64(0, 63);
  BOOST_CHECK_EQUAL(x, v);

  bs = BitSet(static_cast<uint8_t>(0x13));
  BOOST_CHECK_EQUAL(bs.uint8(0, 7), 0x13);

  bs = BitSet(static_cast<uint16_t>(0x8000));
  BOOST_CHECK_EQUAL(bs.uint16(0, 15), 0x8000);

  bs = BitSet(static_cast<uint32_t>(0xF0008000));
  BOOST_CHECK_EQUAL(bs.uint32(0, 31), 0xF0008000);

  bs = BitSet{};
  BOOST_CHECK_THROW(bs.setRangeFromUint(0, 9, static_cast<uint8_t>(0)), std::invalid_argument);
  BOOST_CHECK_THROW(bs.setRangeFromUint(9, 32, static_cast<uint16_t>(0)), std::invalid_argument);
  BOOST_CHECK_THROW(bs.setRangeFromUint(24, 57, static_cast<uint32_t>(0)), std::invalid_argument);
  BOOST_CHECK_THROW(bs.setRangeFromUint(56, 122, static_cast<uint64_t>(0)), std::invalid_argument);
  BOOST_CHECK_NO_THROW(bs.setRangeFromUint(0, 7, static_cast<uint8_t>(0xFF)));
  BOOST_CHECK_EQUAL(bs.len(), 8);
}

BOOST_AUTO_TEST_CASE(TestRangeFromUint16)
{
  uint16_t v = 0xF0F8;

  auto bs = BitSet(v);
  bs.setRangeFromUint(12, 14, static_cast<uint16_t>(0));
  BOOST_CHECK_EQUAL(bs.uint16(0, 15), 0x80F8);
}

BOOST_AUTO_TEST_CASE(TestRangeFromUint32)
{
  uint32_t v = 0xF0F8FCFE;

  auto bs = BitSet(v);
  bs.setRangeFromUint(28, 30, static_cast<uint32_t>(0));
  BOOST_CHECK_EQUAL(bs.uint32(0, 31), 0x80F8FCFE);
}

BOOST_AUTO_TEST_CASE(TestRangeFromUint64)
{
  uint64_t v = 0xF0F8FCFEFF3F3F1F;

  auto bs = BitSet(v);
  bs.setRangeFromUint(60, 62, static_cast<uint64_t>(0));
  uint64_t expected = 0x80F8FCFEFF3F3F1F;
  BOOST_CHECK_EQUAL(bs.uint64(0, 63), expected);
}

BOOST_AUTO_TEST_CASE(TestRangeFromIntegers)
{
  BitSet bs(static_cast<uint64_t>(0));
  BOOST_CHECK_NO_THROW(bs.setRangeFromUint(0, 5, static_cast<uint8_t>(0x13)));
  bs.set(8, true);
  BOOST_CHECK_NO_THROW(bs.setRangeFromUint(20, 23, static_cast<uint8_t>(0xF)));
  BOOST_CHECK_NO_THROW(bs.setRangeFromUint(29, 48, static_cast<uint32_t>(0xAAAAA)));
  BOOST_CHECK_EQUAL(bs.uint64(0, 63), UINT64_C(0x1555540F00113));
}

BOOST_AUTO_TEST_CASE(TestFromBytes)
{
  BitSet bs;
  std::vector<uint8_t> bytes = {0xfe, 0x5a, 0x1e, 0xda};
  bs.setFromBytes(bytes);
  BOOST_CHECK_EQUAL(bs.uint32(0, 31), 0XDA1E5AFE);
  BOOST_CHECK_EQUAL(bs.size(), 32);
  BOOST_CHECK_EQUAL(bs.len(), 32);
}

BOOST_AUTO_TEST_CASE(TestIsEqual)
{
  BitSet b1("110011");
  BitSet b2("110011");
  BOOST_CHECK(b1 == b2);
  b2 = BitSet("1010");
  BOOST_CHECK(b1 != b2);
}

BOOST_AUTO_TEST_CASE(TestSub)
{
  BitSet bs("110011");
  auto b = bs.subset(2, 4);
  BOOST_CHECK_EQUAL(b.stringLSBLeft(), "001");
  b.set(1, true);
  BOOST_CHECK_EQUAL(b.stringLSBLeft(), "011");
}

BOOST_AUTO_TEST_CASE(TestUint64)
{
  BitSet bs("110011");
  auto v2 = bs.uint64(0, 5);
  BOOST_CHECK_EQUAL(v2, 51);
  v2 = bs.uint64(0, 1);
  BOOST_CHECK_EQUAL(v2, 3);

  uint64_t v = UINT64_C(0xFFFFFFFFFFFFFFFF);
  bs = BitSet(v);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "1111111111111111111111111111111111111111111111111111111111111111");
  auto x = bs.uint64(0, 63);
  BOOST_CHECK_EQUAL(x, v);
}

BOOST_AUTO_TEST_CASE(TestGrow)
{
  BitSet bs(static_cast<uint16_t>(0xFFFF));
  BOOST_CHECK_EQUAL(bs.grow(15), false);
  BOOST_CHECK_THROW(bs.grow((BitSet::maxSize() + 1)), std::length_error);
  BOOST_CHECK_EQUAL(bs.grow(34), true);
}

BOOST_AUTO_TEST_CASE(TestUint16)
{
  uint16_t v = 0xFFFF;
  BitSet bs = BitSet(v);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "1111111111111111");
  auto x = bs.uint16(0, 15);
  BOOST_CHECK_EQUAL(x, v);
}

BOOST_AUTO_TEST_CASE(TestUint32)
{
  uint32_t v = 0xFFFFFFFF;
  BitSet bs = BitSet(v);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "11111111111111111111111111111111");
  auto x = bs.uint32(0, 31);
  BOOST_CHECK_EQUAL(x, v);
}

BOOST_AUTO_TEST_CASE(TestLast)
{
  BitSet bs("1010110101111");
  BOOST_CHECK(bs.last(4) == BitSet("1111"));
  BOOST_CHECK(bs.last(6) == BitSet("101111"));
}

BOOST_AUTO_TEST_CASE(TestEmptyBitSet)
{
  BitSet bs;
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "");
  BOOST_CHECK_EQUAL(bs.stringLSBRight(), "");
  BOOST_CHECK_EQUAL(bs.isEmpty(), true);
  BOOST_CHECK_EQUAL(bs.len(), 0);
  BOOST_CHECK(bs.size() > 0);
}

std::string bitNumberScale(int n, int nspaces, bool right2left)
{
  std::string line1;
  std::string line2;
  for (int i = 0; i < n; i++) {
    if (i > 0 && i % 10 == 0) {
      line1 += std::to_string(i / 10);
    } else {
      line1 += " ";
    }
    line2 += std::to_string(i % 10);
  }

  if (right2left) {
    std::reverse(begin(line1), end(line1));
    std::reverse(begin(line2), end(line2));
  }
  std::string spaces(nspaces, ' ');
  std::string rv;

  if (n > 10) {
    rv = spaces + line1 + "\n";
  }
  rv += spaces + line2;
  return rv;
}

BOOST_AUTO_TEST_CASE(TestAppendUint32)
{
  BitSet bs;
  auto C = UINT32_C(0XDA1E5AFE);

  bs.append(C);

  BOOST_CHECK_EQUAL(bs.len(), 32);
  BOOST_CHECK_EQUAL(bs.uint32(0, 31), C);

  // std::cout << fmt::format("BS -> {0}\n", bs.stringLSBLeft());
  // std::cout << bitNumberScale(32, 6, false) << "\n\n";
  // std::cout << fmt::format("BS <- {0}\n", bs.stringLSBRight());
  // std::cout << bitNumberScale(32, 6, true) << "\n\n";
}

BOOST_AUTO_TEST_CASE(TestAppendUint64)
{
  BitSet bs;

  auto C = UINT64_C(0x1555540f00113);
  bs.append(C, 64);

  BOOST_CHECK_EQUAL(bs.len(), 64);
  BOOST_CHECK_EQUAL(bs.uint64(0, 63), C);

  // std::cout << fmt::format("BS -> {0}\n", bs.stringLSBLeft());
  // std::cout << bitNumberScale(64, 6, false) << "\n\n";
  // std::cout << fmt::format("BS <- {0}\n", bs.stringLSBRight());
  // std::cout << bitNumberScale(64, 6, true) << "\n\n";
}

BOOST_AUTO_TEST_CASE(TestAppendUint64Bis)
{
  BitSet bs;

  for (int i = 0; i < 50; i++) {
    bs.set(i, true);
  }
  BOOST_CHECK_EQUAL(bs.len(), 50);
  BOOST_CHECK_EQUAL(bs.uint64(0, 49), allones);
}

BOOST_AUTO_TEST_CASE(TestAppendUint8)
{
  BitSet bs;

  uint8_t a(42);

  BOOST_CHECK_THROW(bs.append(a, 9), std::invalid_argument);

  // append bits, letting appendUint8 compute the number
  // of actual bits to add
  BOOST_CHECK_NO_THROW(bs.append(a));

  BOOST_CHECK_EQUAL(bs.len(), 6);
  BOOST_CHECK_EQUAL(bs.uint8(0, 5), a);

  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "010101");
  BOOST_CHECK_EQUAL(bs.stringLSBRight(), "101010");

  bs.clear();
  // append bits, forcing 8 bits to be added
  bs.append(a, 8);
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "01010100");
  BOOST_CHECK_EQUAL(bs.uint8(0, 7), a);

  bs = BitSet("111");
  bs.append(static_cast<uint8_t>(128), 8);

  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), "11100000001");
}

void compare(std::string_view s1, std::string_view s2)
{
  if (s1.size() != s2.size()) {
    std::cout << "sizes differ\n";
    return;
  }
  std::vector<int> dif;

  for (auto i = 0; i < s1.size(); i++) {
    if (s1[i] != s2[i]) {
      dif.push_back(i);
    }
  }
  if (dif.size()) {
    std::cout << "indices of " << dif.size() << " differences:\n";
    for (auto d : dif) {
      std::cout << d << " ";
    }
    std::cout << "\n";
  }
}

BOOST_AUTO_TEST_CASE(TestLongAppend)
{
  std::string expected = "11010101010110010101011011100011000011101000010110010100101100010100001001100100110010110100000000111100110110101101101110001111010110010010000101101111101110101011011111100101101010111011011111000010011010000100111111001000011010100111101101011110110001010";
  BitSet bs;

  for (int i = 0; i < expected.size(); i++) {
    if (expected[i] == '1') {
      bs.append(true);
    } else {
      bs.append(false);
    }
    if (bs.stringLSBLeft() != expected.substr(0, bs.len())) {
      std::cout << "diff\n";
    }
  }
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), expected);
}

BOOST_AUTO_TEST_CASE(TestLoopAppend)
{
  std::string expected;
  BitSet bs;

  std::srand(std::time(nullptr));
  for (int i = 0; i < BitSet::maxSize(); i++) {
    bool bit = static_cast<bool>(rand() % 2);
    if (bit) {
      expected += "1";
    } else {
      expected += "0";
    }
    bs.append(bit);
  }
  BOOST_CHECK_EQUAL(bs.stringLSBLeft(), expected);
}

BOOST_AUTO_TEST_CASE(TestLimitedCtor)
{
  BOOST_CHECK_THROW(BitSet bs(static_cast<uint8_t>(123), 9), std::invalid_argument);
  BOOST_CHECK_NO_THROW(BitSet bs(static_cast<uint8_t>(123), 4));
  BitSet bs(static_cast<uint8_t>(123), 4);
  BOOST_CHECK_EQUAL(bs.uint8(0, 3), 11);
}

BOOST_AUTO_TEST_CASE(TestCircularAppend)
{
  uint64_t syncValue = 0x1555540F00113;
  BitSet bs;
  BitSet sync(syncValue, 50);

  int next = circularAppend(bs, sync, 0, 10);
  BOOST_CHECK_EQUAL(bs.len(), 10);
  BOOST_CHECK_EQUAL(bs, sync.subset(0, 9));
  BOOST_CHECK_EQUAL(next, 10);

  next = circularAppend(bs, sync, next, 40);
  BOOST_CHECK_EQUAL(bs.len(), 50);
  BOOST_CHECK_EQUAL(bs, sync);
  BOOST_CHECK_EQUAL(next, 0);

  next = circularAppend(bs, sync, next, 145);
  BitSet expected("110010001000000000001111000000101010101010101010101100100010000000000011110000001010101010101010101011001000100000000000111100000010101010101010101010110010001000000000001111000000101010101010101");

  BOOST_CHECK_EQUAL(bs, expected);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
