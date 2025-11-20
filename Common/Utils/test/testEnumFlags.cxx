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
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <stdexcept>
#include <string>

#include "CommonUtils/EnumFlags.h"

// Example enum to use with EnumFlags
enum class TestEnum : uint8_t {
  Bit1 = 0,
  Bit2,
  Bit3,
  Bit4,
  Bit5VeryLongName,
};

// Very long enum
// to test that it works beyond 32 bits upto 64 bits
#define ENUM_BIT_NAME(n) Bit##n
#define ENUM_BIT_NAME_EXPAND(n) ENUM_BIT_NAME(n)
#define ENUM_BIT(z, n, _) ENUM_BIT_NAME_EXPAND(BOOST_PP_INC(n)) = (n),
enum class TestEnumLong : uint64_t {
  BOOST_PP_REPEAT(64, ENUM_BIT, _)
};
#undef ENUM_BIT
#undef ENUM_BIT_NAME
#undef ENUM_BIT_NAME_EXPAND

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

  // Test multiset
  multipleFlags.reset();
  multipleFlags.set(TestEnum::Bit2, TestEnum::Bit4);
  BOOST_TEST(!multipleFlags.test(TestEnum::Bit1));
  BOOST_TEST(multipleFlags.test(TestEnum::Bit2));
  BOOST_TEST(!multipleFlags.test(TestEnum::Bit3));
  BOOST_TEST(multipleFlags.test(TestEnum::Bit4));
  BOOST_TEST(!multipleFlags.test(TestEnum::Bit5VeryLongName));

  // Test operator|
  EFlags combinedFlags = flag1 | EFlags(TestEnum::Bit2);
  BOOST_TEST(combinedFlags.test(TestEnum::Bit1));
  BOOST_TEST(combinedFlags.test(TestEnum::Bit2));
  BOOST_TEST(!combinedFlags.test(TestEnum::Bit3));
  combinedFlags |= TestEnum::Bit5VeryLongName;
  BOOST_TEST(combinedFlags.test(TestEnum::Bit5VeryLongName));

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

    { // test with , delimiter
      std::string str = "Bit4,TestEnum::Bit2 , Bit1 ";
      flags.set(str);
      BOOST_TEST(flags.test(TestEnum::Bit1));
      BOOST_TEST(flags.test(TestEnum::Bit2));
      BOOST_TEST(!flags.test(TestEnum::Bit3));
      BOOST_TEST(flags.test(TestEnum::Bit4));
    }

    { // test with ; delimiter
      std::string str = "Bit4;TestEnum::Bit2 ; Bit1 ";
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

    // test xor
    auto flagsXOR = flags3 ^ flags4;
    BOOST_CHECK(flagsXOR.test(TestEnum::Bit1, TestEnum::Bit4));

    // test and
    auto flagsAND = flags3 & flags4;
    BOOST_CHECK(flagsAND.test(TestEnum::Bit2, TestEnum::Bit3));

    // Perform an intersection operation
    EFlags intersectionFlags = flags3.intersection_with(flags4);
    BOOST_CHECK(intersectionFlags.test(TestEnum::Bit2));
    BOOST_CHECK(intersectionFlags.test(TestEnum::Bit3));
    BOOST_CHECK(!intersectionFlags.test(TestEnum::Bit1));
    BOOST_CHECK(!intersectionFlags.test(TestEnum::Bit4));
    BOOST_CHECK_EQUAL(intersectionFlags.value(), 6); // 0110 in binary
  }

  {
    // Check special flag names.
    EFlags flag("all");
    BOOST_CHECK(flag.all());
    flag.set("none");
    BOOST_CHECK(!flag.any());
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
    // Also tests space delimiter and construction from string.
    o2::utils::EnumFlags<TestEnumLong> test("Bit32 Bit34");
    BOOST_CHECK(test.test(TestEnumLong::Bit32, TestEnumLong::Bit34));
    BOOST_CHECK(!test.test(TestEnumLong::Bit1, TestEnumLong::Bit23));
  }
}

BOOST_AUTO_TEST_CASE(Flags_case_insensitive_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test case-insensitive flag names
  {
    EFlags flags("bit1"); // lowercase
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(!flags.test(TestEnum::Bit2));
  }

  {
    EFlags flags("BIT2"); // uppercase
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(!flags.test(TestEnum::Bit1));
  }

  {
    EFlags flags("BiT3"); // mixed case
    BOOST_CHECK(flags.test(TestEnum::Bit3));
  }

  {
    EFlags flags("bit1|BIT2|BiT3"); // mixed case with delimiter
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(flags.test(TestEnum::Bit3));
  }

  // Test special keywords case-insensitive
  {
    EFlags flags("ALL");
    BOOST_CHECK(flags.all());
  }

  {
    EFlags flags("None");
    BOOST_CHECK(!flags.any());
  }
}

BOOST_AUTO_TEST_CASE(Flags_error_recovery_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test that previous state is restored on exception
  {
    EFlags flags({TestEnum::Bit1, TestEnum::Bit2});
    auto previousValue = flags.value();

    // Try to set with invalid string
    BOOST_CHECK_THROW(flags.set("InvalidFlag"), std::invalid_argument);

    // Verify state was restored
    BOOST_CHECK_EQUAL(flags.value(), previousValue);
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
  }

  {
    EFlags flags({TestEnum::Bit3, TestEnum::Bit4});
    auto previousValue = flags.value();

    // Try to set with out-of-range value
    BOOST_CHECK_THROW(flags.set("999999", 10), std::out_of_range);

    // Verify state was restored
    BOOST_CHECK_EQUAL(flags.value(), previousValue);
    BOOST_CHECK(flags.test(TestEnum::Bit3));
    BOOST_CHECK(flags.test(TestEnum::Bit4));
  }

  {
    EFlags flags(TestEnum::Bit5VeryLongName);
    auto previousValue = flags.value();

    // Try to set with invalid binary string
    BOOST_CHECK_THROW(flags.set("10102", 2), std::invalid_argument);

    // Verify state was restored
    BOOST_CHECK_EQUAL(flags.value(), previousValue);
    BOOST_CHECK(flags.test(TestEnum::Bit5VeryLongName));
  }
}

BOOST_AUTO_TEST_CASE(Flags_whitespace_handling_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test leading/trailing whitespace
  {
    EFlags flags("  Bit1  ");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
  }

  {
    EFlags flags("  Bit1 | Bit2  ");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
  }

  // Test excessive whitespace between flags
  {
    EFlags flags("Bit1    |    Bit3");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit3));
    BOOST_CHECK(!flags.test(TestEnum::Bit2));
  }

  // Test tabs and other whitespace (should work with space delimiter)
  {
    EFlags flags("Bit1 Bit2 Bit3");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(flags.test(TestEnum::Bit3));
  }
}

BOOST_AUTO_TEST_CASE(Flags_count_bits_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test counting set bits
  {
    EFlags flags;
    BOOST_CHECK_EQUAL(flags.count(), 0);
  }

  {
    EFlags flags(TestEnum::Bit1);
    BOOST_CHECK_EQUAL(flags.count(), 1);
  }

  {
    EFlags flags({TestEnum::Bit1, TestEnum::Bit2});
    BOOST_CHECK_EQUAL(flags.count(), 2);
  }

  {
    EFlags flags({TestEnum::Bit1, TestEnum::Bit2, TestEnum::Bit3, TestEnum::Bit4});
    BOOST_CHECK_EQUAL(flags.count(), 4);
  }

  {
    EFlags flags(EFlags::All);
    BOOST_CHECK_EQUAL(flags.count(), 5); // TestEnum has 5 members
  }

  // Test count after operations
  {
    EFlags flags({TestEnum::Bit1, TestEnum::Bit2, TestEnum::Bit3});
    BOOST_CHECK_EQUAL(flags.count(), 3);

    flags.reset(TestEnum::Bit2);
    BOOST_CHECK_EQUAL(flags.count(), 2);

    flags.set(TestEnum::Bit4);
    BOOST_CHECK_EQUAL(flags.count(), 3);

    flags.toggle(TestEnum::Bit1);
    BOOST_CHECK_EQUAL(flags.count(), 2);
  }
}

BOOST_AUTO_TEST_CASE(Flags_mixed_delimiter_validation_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test that mixed delimiters throw an error
  {
    BOOST_CHECK_THROW(EFlags("Bit1|Bit2,Bit3"), std::invalid_argument);
  }

  {
    BOOST_CHECK_THROW(EFlags("Bit1;Bit2|Bit3"), std::invalid_argument);
  }

  {
    BOOST_CHECK_THROW(EFlags("Bit1,Bit2;Bit3"), std::invalid_argument);
  }

  {
    BOOST_CHECK_THROW(EFlags("Bit1|Bit2,Bit3;Bit4"), std::invalid_argument);
  }

  // Test that single delimiter types work
  {
    EFlags flags1("Bit1|Bit2|Bit3");
    BOOST_CHECK_EQUAL(flags1.count(), 3);
  }

  {
    EFlags flags2("Bit1,Bit2,Bit3");
    BOOST_CHECK_EQUAL(flags2.count(), 3);
  }

  {
    EFlags flags3("Bit1;Bit2;Bit3");
    BOOST_CHECK_EQUAL(flags3.count(), 3);
  }
}

BOOST_AUTO_TEST_CASE(Flags_empty_and_edge_cases_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test empty string
  {
    EFlags flags({TestEnum::Bit1, TestEnum::Bit2});
    flags.set(""); // Should be no-op
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
  }

  // Test with only whitespace
  {
    EFlags flags({TestEnum::Bit1});
    flags.set("   "); // Should result in empty after tokenization
    // Depending on implementation, this might clear or throw
    // Adjust expectation based on actual behavior
  }

  // Test duplicate flags (should work, setting same bit twice is idempotent)
  {
    EFlags flags("Bit1|Bit1|Bit1");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK_EQUAL(flags.count(), 1);
  }

  // Test scoped and unscoped mixed
  {
    EFlags flags("Bit1|TestEnum::Bit2");
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
  }
}

BOOST_AUTO_TEST_CASE(Flags_binary_decimal_parsing_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test binary parsing
  {
    EFlags flags("101", 2);
    BOOST_CHECK(flags.test(TestEnum::Bit1));  // bit 0
    BOOST_CHECK(!flags.test(TestEnum::Bit2)); // bit 1
    BOOST_CHECK(flags.test(TestEnum::Bit3));  // bit 2
  }

  // Test decimal parsing
  {
    EFlags flags("7", 10); // 7 = 0b111
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(flags.test(TestEnum::Bit3));
    BOOST_CHECK(!flags.test(TestEnum::Bit4));
  }

  // Test hexadecimal parsing
  {
    EFlags flags("F", 16); // 15 = 0b1111
    BOOST_CHECK(flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(flags.test(TestEnum::Bit3));
    BOOST_CHECK(flags.test(TestEnum::Bit4));
    BOOST_CHECK(!flags.test(TestEnum::Bit5VeryLongName));
  }

  // Test hexadecimal with 0x prefix
  {
    EFlags flags("0xA", 16); // 10 = 0b1010
    BOOST_CHECK(!flags.test(TestEnum::Bit1));
    BOOST_CHECK(flags.test(TestEnum::Bit2));
    BOOST_CHECK(!flags.test(TestEnum::Bit3));
    BOOST_CHECK(flags.test(TestEnum::Bit4));
  }

  // Test hexadecimal with 0X prefix (uppercase)
  {
    EFlags flags("0X1F", 16); // 31 = all 5 bits
    BOOST_CHECK(flags.all());
  }

  // Test lowercase hex digits
  {
    EFlags flags("0xa", 16);
    BOOST_CHECK_EQUAL(flags.value(), 10);
  }

  // Test thros
  {
    BOOST_CHECK_THROW(EFlags("0xAbCd", 16), std::out_of_range);
  }

  // Test invalid binary string (contains 2)
  {
    BOOST_CHECK_THROW(EFlags("1012", 2), std::invalid_argument);
  }

  // Test out of range for base
  {
    BOOST_CHECK_THROW(EFlags("100000", 2), std::out_of_range);
  }
}

BOOST_AUTO_TEST_CASE(Flags_operator_bool_test)
{
  using EFlags = o2::utils::EnumFlags<TestEnum>;

  // Test explicit bool conversion
  {
    EFlags empty;
    BOOST_CHECK(!static_cast<bool>(empty));
  }

  {
    EFlags withFlag(TestEnum::Bit1);
    BOOST_CHECK(static_cast<bool>(withFlag));
  }

  // Test in conditional
  {
    EFlags flags;
    if (flags) {
      BOOST_FAIL("Empty flags should be false");
    }

    flags.set(TestEnum::Bit1);
    if (!flags) {
      BOOST_FAIL("Non-empty flags should be true");
    }
  }
}
