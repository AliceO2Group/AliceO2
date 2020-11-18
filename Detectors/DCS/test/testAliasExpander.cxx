// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DCS AliasExpander
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "DetectorsDCS/AliasExpander.h"

BOOST_AUTO_TEST_CASE(ExpandAliasesIsNoopWhenNoPatternGiven)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"ab"});

  std::vector<std::string> expected = {"ab"};

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandAliasesReturnsEmptyVectorWhenPatternIsIncorrect)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"ab[c"});

  std::vector<std::string> expected = {};

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());

  aliases = o2::dcs::expandAliases({"ab]c"});

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());

  aliases = o2::dcs::expandAliases({"ab[1.2]c"});

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandAliasesWithIntegerRange)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"a[1..2]bcde[99..101]toto"});

  std::vector<std::string> expected = {
    "a1bcde099toto",
    "a1bcde100toto",
    "a1bcde101toto",
    "a2bcde099toto",
    "a2bcde100toto",
    "a2bcde101toto"};

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandAliasesWithIntegerRangeWithCustomFormat)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"a[1..3{:03d}]"});

  std::vector<std::string> expected = {
    "a001",
    "a002",
    "a003"};

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandAliasesWithIntegerRangeWithCustomFormatBis)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"a[1..3{:d}]"});

  std::vector<std::string> expected = {
    "a1",
    "a2",
    "a3"};

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandAliasesWithStringList)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases({"a[1..2]bcde[99..101][toto,titi,tata]"});

  std::vector<std::string> expected = {
    "a1bcde099tata",
    "a1bcde099titi",
    "a1bcde099toto",
    "a1bcde100tata",
    "a1bcde100titi",
    "a1bcde100toto",
    "a1bcde101tata",
    "a1bcde101titi",
    "a1bcde101toto",
    "a2bcde099tata",
    "a2bcde099titi",
    "a2bcde099toto",
    "a2bcde100tata",
    "a2bcde100titi",
    "a2bcde100toto",
    "a2bcde101tata",
    "a2bcde101titi",
    "a2bcde101toto",
  };

  BOOST_TEST(aliases == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ExpandMch)
{
  std::vector<std::string> aliases = o2::dcs::expandAliases(
    {"MchHvLvLeft/Chamber[00..03]Left/Quad1Sect[0..2].actual.[vMon,iMon]",
     "MchHvLvLeft/Chamber[00..03]Left/Quad2Sect[0..2].actual.[vMon,iMon]",
     "MchHvLvLeft/Chamber[04..09]Left/Slat[00..08].actual.[vMon,iMon]",
     "MchHvLvLeft/Chamber[06..09]Left/Slat[09..12].actual.[vMon,iMon]",
     "MchHvLvRight/Chamber[00..03]Right/Quad0Sect[0..2].actual.[vMon,iMon]",
     "MchHvLvRight/Chamber[00..03]Right/Quad3Sect[0..2].actual.[vMon,iMon]",
     "MchHvLvRight/Chamber[04..09]Right/Slat[00..08].actual.[vMon,iMon]",
     "MchHvLvRight/Chamber[06..09]Right/Slat[09..12].actual.[vMon,iMon]"});

  BOOST_TEST(aliases.size(), 376);
}
