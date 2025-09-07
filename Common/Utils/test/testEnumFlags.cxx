// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Flags
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include <string>

#include "CommonUtils/EnumFlags.h"

// Example enum to use with EnumFlags
enum class TestEnum : uint8_t {
  Bit1,
  Bit2,
  Bit3,
  Bit4,
  Bit5VeryLongName,
};

// Very long enum
// to test that it works beyond 32 bits
enum class TestEnumLong : uint64_t {
  Bit1,
  Bit2,
  Bit3,
  Bit4,
  Bit5,
  Bit6,
  Bit7,
  Bit8,
  Bit9,
  Bit10,
  Bit11,
  Bit12,
  Bit13,
  Bit14,
  Bit15,
  Bit16,
  Bit17,
  Bit18,
  Bit19,
  Bit20,
  Bit21,
  Bit22,
  Bit23,
  Bit24,
  Bit25,
  Bit26,
  Bit27,
  Bit28,
  Bit29,
  Bit30,
  Bit31,
  Bit32,
  Bit33,
  Bit34,
  // ...
};

BOOST_AUTO_TEST_CASE(Flags_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test default initialization
  EFlags flags;
  BOOST_TEST(flags.None == 0);
  BOOST_TEST(flags.All == 31);
  BOOST_TEST(flags.value() == 0);
  BOOST_TEST(!flags.any());

  // Test initialization with a single flag
  EFlags flag1(TestEnum::Bit1);
  BOOST_TEST(flag1.test(TestEnum::Bit1));
  BOOST_TEST(!flag1.test(TestEnum::Bit2));
  BOOST_TEST(flag1.value() == (1 << static_cast<unsigned int>(TestEnum::Bit1)));

  // Test initialization with initializer list
  EFlags multipleFlags({TestEnum::Bit1, TestEnum::Bit3});
  BOOST_TEST(multipleFlags.test(TestEnum::Bit1));
  BOOST_TEST(multipleFlags.test(TestEnum::Bit3));
  BOOST_TEST(!multipleFlags.test(TestEnum::Bit2));
  BOOST_TEST(multipleFlags.any());

  // Test reset
  multipleFlags.reset(TestEnum::Bit1);
  BOOST_TEST(!multipleFlags.test(TestEnum::Bit1));
  BOOST_TEST(multipleFlags.test(TestEnum::Bit3));
  multipleFlags.reset();
  BOOST_TEST(!multipleFlags.any());

  // Test operator|
  EFlags combinedFlags = flag1 | EFlags(TestEnum::Bit2);
  BOOST_TEST(combinedFlags.test(TestEnum::Bit1));
  BOOST_TEST(combinedFlags.test(TestEnum::Bit2));
  BOOST_TEST(!combinedFlags.test(TestEnum::Bit3));

  // Test operator[]
  BOOST_TEST(combinedFlags[TestEnum::Bit1]);
  BOOST_TEST(combinedFlags[TestEnum::Bit2]);
  BOOST_TEST(!combinedFlags[TestEnum::Bit3]);

  // Test operator|=
  combinedFlags |= TestEnum::Bit3;
  BOOST_TEST(combinedFlags.test(TestEnum::Bit3));

  // Test operator&
  EFlags intersection = combinedFlags & TestEnum::Bit1;
  BOOST_TEST(intersection.test(TestEnum::Bit1));
  BOOST_TEST(!intersection.test(TestEnum::Bit2));
  BOOST_TEST(intersection.value() == (1 << static_cast<unsigned int>(TestEnum::Bit1)));

  // Test operator&=
  combinedFlags &= TestEnum::Bit1;
  BOOST_TEST(combinedFlags.test(TestEnum::Bit1));
  BOOST_TEST(!combinedFlags.test(TestEnum::Bit2));
  BOOST_TEST(!combinedFlags.test(TestEnum::Bit3));

  // Test operator~ (complement)
  EFlags complement = ~EFlags(TestEnum::Bit1);
  BOOST_TEST(!complement.test(TestEnum::Bit1));
  BOOST_TEST(complement.test(TestEnum::Bit2));
  BOOST_TEST(complement.test(TestEnum::Bit3));

  // Test string() method
  {
    std::string flagString = flag1.string();
    BOOST_TEST(flagString.back() == '1'); // Ensure the least significant bit is set for flag1
  }

  // Test set with binary string
  {
    std::string binaryStr = "101";
    flags.set(binaryStr, 2);
    BOOST_TEST(flags.test(TestEnum::Bit1));
    BOOST_TEST(!flags.test(TestEnum::Bit2));
    BOOST_TEST(flags.test(TestEnum::Bit3));
  }

  // Test invalid binary string in set
  BOOST_CHECK_THROW(flags.set(std::string("invalid"), 2), std::invalid_argument);

  // Test range validation in set
  BOOST_CHECK_THROW(flags.set(std::string("100000000"), 2), std::out_of_range);

  { // Test that return lists are sensible
    const auto n = flags.getNames();
    const auto v = flags.getValues();
    BOOST_CHECK(n.size() == v.size());
  }

  { // print test
    std::cout << flags;
  }

  // Test flag tokenization and parsing
  {
    { // only one scoped flag
      std::string str = "TestEnum::Bit2";
      flags.set(str);
      BOOST_TEST(flags.test(TestEnum::Bit2));
      BOOST_TEST(flags.none_of(TestEnum::Bit1, TestEnum::Bit3, TestEnum::Bit4));
    }

    { // test with ws-triming and scope mixing
      std::string str = "Bit4|TestEnum::Bit2 | Bit1 ";
      flags.set(str);
      BOOST_TEST(flags.test(TestEnum::Bit1));
      BOOST_TEST(flags.test(TestEnum::Bit2));
      BOOST_TEST(!flags.test(TestEnum::Bit3));
      BOOST_TEST(flags.test(TestEnum::Bit4));
    }

    { // test with different delimiter
      std::string str = "Bit4,TestEnum::Bit2 , Bit1 ";
      flags.set(str);
      BOOST_TEST(flags.test(TestEnum::Bit1));
      BOOST_TEST(flags.test(TestEnum::Bit2));
      BOOST_TEST(!flags.test(TestEnum::Bit3));
      BOOST_TEST(flags.test(TestEnum::Bit4));
    }

    { // throw test with mixed delimiter
      std::string str = "Bit4|TestEnum::Bit2 , Bit1 ";
      BOOST_CHECK_THROW(flags.set(str), std::invalid_argument);
    }

    { // test throw
      std::string str = "Invalid";
      BOOST_CHECK_THROW(flags.set(str), std::invalid_argument);
    }
  }

  // Test all_of and none_of
  {
    EFlags allFlags({TestEnum::Bit1, TestEnum::Bit2, TestEnum::Bit3});
    BOOST_TEST(allFlags.all_of(TestEnum::Bit1, TestEnum::Bit2));
    BOOST_TEST(!allFlags.all_of(TestEnum::Bit4));
    BOOST_TEST(allFlags.none_of(TestEnum::Bit4));
  }

  // Test toggle
  {
    EFlags toggleFlags;
    toggleFlags.toggle(TestEnum::Bit4);
    BOOST_TEST(toggleFlags.test(TestEnum::Bit4));
    toggleFlags.toggle(TestEnum::Bit4);
    BOOST_TEST(!toggleFlags.test(TestEnum::Bit4));
  }

  // Create a flag set and serialize it
  {
    EFlags serializedFlags{TestEnum::Bit1, TestEnum::Bit3};
    std::string serialized = serializedFlags.serialize();
    BOOST_CHECK_EQUAL(serialized, "5"); // 5 in binary is 0101, meaning Bit1 and Bit3 are set.

    // Deserialize back into a flag set
    EFlags deserializedFlags;
    deserializedFlags.deserialize(serialized);
    BOOST_CHECK(deserializedFlags == serializedFlags); // Ensure the deserialized flags match the original
  }

  // Test with an empty flag set
  {
    EFlags emptyFlags;
    std::string serialized = emptyFlags.serialize();
    BOOST_CHECK_EQUAL(serialized, "0");

    EFlags deserialized;
    deserialized.deserialize(serialized);
    BOOST_CHECK(deserialized == emptyFlags);

    // Test with all flags set
    EFlags allFlags(EFlags::All);
    serialized = allFlags.serialize();
    BOOST_CHECK_EQUAL(serialized, std::to_string(EFlags::All));

    deserialized.deserialize(serialized);
    BOOST_CHECK(deserialized == allFlags);
  }

  // check throw deserializng out of range
  {
    EFlags flag;
    std::string str = "999999";
    BOOST_CHECK_THROW(flag.deserialize(str), std::out_of_range);
  }

  // Create two flag sets
  {
    EFlags flags1{TestEnum::Bit1, TestEnum::Bit2};
    EFlags flags2{TestEnum::Bit3, TestEnum::Bit4};

    // Perform a union operation
    EFlags unionFlags = flags1.union_with(flags2);
    BOOST_CHECK(unionFlags.test(TestEnum::Bit1));
    BOOST_CHECK(unionFlags.test(TestEnum::Bit2));
    BOOST_CHECK(unionFlags.test(TestEnum::Bit3));
    BOOST_CHECK(unionFlags.test(TestEnum::Bit4));
    BOOST_CHECK_EQUAL(unionFlags.value(), 15); // 1111 in binary
  }

  // Create two overlapping flag sets
  {
    EFlags flags3{TestEnum::Bit1, TestEnum::Bit2, TestEnum::Bit3};
    EFlags flags4{TestEnum::Bit2, TestEnum::Bit3, TestEnum::Bit4};

    // Perform an intersection operation
    EFlags intersectionFlags = flags3.intersection_with(flags4);
    BOOST_CHECK(intersectionFlags.test(TestEnum::Bit2));
    BOOST_CHECK(intersectionFlags.test(TestEnum::Bit3));
    BOOST_CHECK(!intersectionFlags.test(TestEnum::Bit1));
    BOOST_CHECK(!intersectionFlags.test(TestEnum::Bit4));
    BOOST_CHECK_EQUAL(intersectionFlags.value(), 6); // 0110 in binary
  }

  {
    // Create two flag sets
    EFlags flags1{TestEnum::Bit1, TestEnum::Bit2, TestEnum::Bit3};
    EFlags flags2{TestEnum::Bit2, TestEnum::Bit3};

    // Check containment
    BOOST_CHECK(flags1.contains(flags2));  // flags1 contains all flags in flags2
    BOOST_CHECK(!flags2.contains(flags1)); // flags2 does not contain all flags in flags1

    // Test with disjoint sets
    EFlags flags3{TestEnum::Bit4};
    BOOST_CHECK(!flags1.contains(flags3)); // flags1 does not contain flags3
  }

  {
    // Test compilation using an enum with more than 32 bits
    o2::utils::EnumFlags<TestEnumLong> test;
    test.set("Bit32");
    BOOST_CHECK(test.test(TestEnumLong::Bit32));
  }
}
