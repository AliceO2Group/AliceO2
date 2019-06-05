// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_BitstreamReader.cxx
/// @author Matthias Richter
/// @since  2019-06-05
/// @brief  Test program for BitstreamReader utility

#define BOOST_TEST_MODULE Algorithm BitstreamReader unit test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <bitset>
#include "../include/Algorithm/BitstreamReader.h"

namespace o2
{
namespace algorithm
{

BOOST_AUTO_TEST_CASE(test_BitstreamReader_basic)
{
  std::array<uint8_t, 8> data = { 'd', 'e', 'a', 'd', 'b', 'e', 'e', 'f' };
  std::array<uint8_t, 10> expected7bit = { 0x32, 0x19, 0x2c, 0x16, 0x23, 0x09, 0x4a, 0x65, 0x33, 0x0 };
  auto reference = expected7bit.begin();
  constexpr size_t totalBits = data.size() * sizeof(decltype(data)::value_type) * 8;
  size_t bitsRead = 0;

  BitstreamReader<uint8_t> reader(data.data(), data.data() + data.size());
  while (bitsRead < totalBits) {
    BOOST_REQUIRE(reference != expected7bit.end());
    BOOST_CHECK(reader.eof() == false);
    uint8_t value;
    reader.peek(value);
    // we use 7 bits of the data
    value >>= 1;
    reader.seek(7);
    bitsRead += 7;
    // in the last call should there is not enough data
    BOOST_CHECK(reader.good() == (bitsRead <= totalBits));
    BOOST_REQUIRE(reference != expected7bit.end());
    //std::cout << "value " << (int)value << "  expected " << (int)*reference << std::endl;
    BOOST_CHECK(value == *reference);
    ++reference;
  }
}

BOOST_AUTO_TEST_CASE(test_BitstreamReader_operator)
{
  std::array<uint8_t, 8> data = { 'd', 'e', 'a', 'd', 'b', 'e', 'e', 'f' };
  std::array<uint8_t, 10> expected7bit = { 0x32, 0x19, 0x2c, 0x16, 0x23, 0x09, 0x4a, 0x65, 0x33, 0x0 };
  auto reference = expected7bit.begin();
  constexpr size_t totalBits = data.size() * sizeof(decltype(data)::value_type) * 8;
  size_t bitsRead = 0;

  BitstreamReader<uint8_t> reader(data.data(), data.data() + data.size());
  while (bitsRead < totalBits) {
    BOOST_REQUIRE(reference != expected7bit.end());
    BOOST_CHECK(reader.eof() == false);
    {
      decltype(reader)::Bits<uint8_t> value;
      reader >> value;
      // we use 7 bits of the data
      *value >>= 1;
      value.markUsed(7);
      //std::cout << "value " << (int)*value << "  expected " << (int)*reference << std::endl;
      BOOST_CHECK(*value == *reference);
    }
    bitsRead += 7;
    // in the last call should there is not enough data
    BOOST_CHECK(reader.good() == (bitsRead <= totalBits));
    BOOST_REQUIRE(reference != expected7bit.end());
    ++reference;
  }
}

BOOST_AUTO_TEST_CASE(test_BitstreamReader_bitset)
{
  std::array<uint8_t, 8> data = { 'd', 'e', 'a', 'd', 'b', 'e', 'e', 'f' };
  std::array<uint8_t, 10> expected7bit = { 0x32, 0x19, 0x2c, 0x16, 0x23, 0x09, 0x4a, 0x65, 0x33, 0x0 };
  auto reference = expected7bit.begin();
  constexpr size_t totalBits = data.size() * sizeof(decltype(data)::value_type) * 8;
  size_t bitsRead = 0;

  BitstreamReader<uint8_t> reader(data.data(), data.data() + data.size());
  while (bitsRead < totalBits) {
    BOOST_REQUIRE(reference != expected7bit.end());
    BOOST_CHECK(reader.eof() == false);
    std::bitset<13> value;
    reader.peek(value, value.size());
    // we use 7 bits of the data
    value >>= value.size() - 7;
    reader.seek(7);
    bitsRead += 7;
    // in the last call should there is not enough data
    BOOST_CHECK(reader.good() == (bitsRead <= totalBits));
    BOOST_REQUIRE(reference != expected7bit.end());
    BOOST_CHECK_MESSAGE(value.to_ulong() == *reference, std::string("mismatch: value ") << value.to_ulong() << ",  expected " << (int)*reference);
    ++reference;
  }

  reader.reset();
  std::bitset<16> aBitset;
  reader >> aBitset;
  BOOST_CHECK_MESSAGE(aBitset.to_ulong() == 0x6465, std::string("mismatch: value 0x") << std::hex << aBitset.to_ulong() << ",  expected 0x6465");
}

} // namespace algorithm
} // namespace o2
