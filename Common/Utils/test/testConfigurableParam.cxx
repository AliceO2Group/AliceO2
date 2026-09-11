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

#include <boost/test/tools/old/interface.hpp>
#include <stdexcept>
#define BOOST_TEST_MODULE Test ConfigurableParams
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <list>
#include <set>
#include <unordered_set>
#include <vector>

#include "CommonUtils/ConfigurableParamTest.h"

using namespace o2::conf;
using namespace o2::conf::test;

BOOST_AUTO_TEST_CASE(ConfigurableParam_Basic)
{
  // Tests the default parameters and also getter helpers.
  auto& param = TestParam::Instance();
  BOOST_CHECK_EQUAL(param.iValue, 42);
  BOOST_CHECK_EQUAL(param.dValue, 3.14);
  BOOST_CHECK_EQUAL(param.bValue, true);
  BOOST_CHECK_EQUAL(param.sValue, "default");
  BOOST_CHECK_EQUAL(static_cast<int>(param.eValue), 2);

  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<int>("TestParam.iValue"), 42);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<double>("TestParam.dValue"), 3.14);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<bool>("TestParam.bValue"), true);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<std::string>("TestParam.sValue"), "default");
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_SG_Fundamental)
{
  // tests runtime setting and getting for fundamental types
  ConfigurableParam::setValue("TestParam.iValue", "100");
  ConfigurableParam::setValue("TestParam.dValue", "2.718");
  ConfigurableParam::setValue("TestParam.bValue", "0");
  ConfigurableParam::setValue("TestParam.sValue", "modified");
  ConfigurableParam::setValue("TestParam.eValue", "0");

  auto& param = TestParam::Instance();
  BOOST_CHECK_EQUAL(param.iValue, 100);
  BOOST_CHECK_EQUAL(param.dValue, 2.718);
  BOOST_CHECK_EQUAL(param.bValue, false);
  BOOST_CHECK_EQUAL(param.sValue, "modified");
  BOOST_CHECK_EQUAL(static_cast<int>(param.eValue), 0);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_SG_CArray)
{
  // tests setting and getting for a c-style array type
  auto& param = TestParam::Instance();
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<int>("TestParam.caValue[0]"), 0);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<int>("TestParam.caValue[1]"), 1);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<int>("TestParam.caValue[2]"), 2);

  ConfigurableParam::setValue("TestParam.caValue[1]", "99");
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<int>("TestParam.caValue[1]"), 99);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_SG_STD)
{
  // tests setting and getting for a std type
  ConfigurableParam::setValue("TestParam.vec", "[10,20,30,40]");
  auto& param = TestParam::Instance();
  BOOST_CHECK_EQUAL(param.vec.size(), 4);
  BOOST_CHECK_EQUAL(param.vec[0], 10);
  BOOST_CHECK_EQUAL(param.vec[1], 20);
  BOOST_CHECK_EQUAL(param.vec[2], 30);
  BOOST_CHECK_EQUAL(param.vec[3], 40);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<std::string>("TestParam.vec"), "[10,20,30,40]");

  ConfigurableParam::setValue("TestParam.u8vec", "[1,2,255]");
  BOOST_CHECK_EQUAL(param.u8vec.size(), 3);
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(param.u8vec[0]), 1);
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(param.u8vec[1]), 2);
  BOOST_CHECK_EQUAL(static_cast<unsigned int>(param.u8vec[2]), 255);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<std::string>("TestParam.u8vec"), "[1,2,255]");

  ConfigurableParam::setValues({{"TestParam.map", "{0:1,10:42}"}});
  BOOST_CHECK_EQUAL(param.map.size(), 2);
  BOOST_CHECK(param.map.contains(0));
  BOOST_CHECK_EQUAL(param.map.at(0), 1);
  BOOST_CHECK(param.map.contains(10));
  BOOST_CHECK_EQUAL(param.map.at(10), 42);
  BOOST_CHECK_THROW(param.map.at(33), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_Provenance)
{
  // tests correct setting of provenance
  BOOST_CHECK_EQUAL(ConfigurableParam::getProvenance("TestParam.iValueProvenanceTest"), ConfigurableParam::EParamProvenance::kCODE);
  ConfigurableParam::setValue("TestParam.iValueProvenanceTest", "123");
  BOOST_CHECK_EQUAL(ConfigurableParam::getProvenance("TestParam.iValueProvenanceTest"), ConfigurableParam::EParamProvenance::kRT);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_FileIO_Ini)
{
  // test for ini file serialization
  const std::string testFileName = "test_config.ini";
  auto iValueBefore = TestParam::Instance().iValue;
  auto sValueBefore = TestParam::Instance().sValue;
  ConfigurableParam::setValue("TestParam.vec", "[7,8]");
  const std::vector<int> vecBefore = TestParam::Instance().vec;
  ConfigurableParam::writeINI(testFileName);
  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", testFileName);
  ConfigurableParam::setValue("TestParam.vec", "[1]");
  ConfigurableParam::updateFromFile(testFileName);
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, iValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, sValueBefore);
  BOOST_CHECK_EQUAL_COLLECTIONS(TestParam::Instance().vec.begin(), TestParam::Instance().vec.end(), vecBefore.begin(), vecBefore.end());
  std::remove(testFileName.c_str());
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_FileIO_Json)
{
  // test for json file serialization
  const std::string testFileName = "test_config.json";
  auto iValueBefore = TestParam::Instance().iValue;
  auto sValueBefore = TestParam::Instance().sValue;
  ConfigurableParam::setValues({{"TestParam.map", "{3:4,5:6}"}});
  const std::map<int, unsigned int> mapBefore = TestParam::Instance().map;
  ConfigurableParam::writeJSON(testFileName);
  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", testFileName);
  ConfigurableParam::setValues({{"TestParam.map", "{1:2}"}});
  ConfigurableParam::updateFromFile(testFileName);
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, iValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, sValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().map.size(), mapBefore.size());
  BOOST_CHECK_EQUAL(TestParam::Instance().map.at(3), mapBefore.at(3));
  BOOST_CHECK_EQUAL(TestParam::Instance().map.at(5), mapBefore.at(5));
  std::remove(testFileName.c_str());
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_JSONString_UnchangedOnly)
{
  ConfigurableParam::setValue("TestParam.ulValue", "2");
  ConfigurableParam::setProvenance("TestParam", "lValue", ConfigurableParam::kCODE);
  ConfigurableParam::setProvenance("TestParam", "ulValue", ConfigurableParam::kRT);
  ConfigurableParam::updateFromJSONString(R"json({"TestParam":{"lValue":"77","ulValue":"88"}})json", "TestParam", true);

  BOOST_CHECK_EQUAL(TestParam::Instance().lValue, 77);
  BOOST_CHECK_EQUAL(TestParam::Instance().ulValue, 2);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_JSONString_FromAsJSON)
{
  ConfigurableParam::setValue("TestParam.iValue", "321");
  ConfigurableParam::setValue("TestParam.sValue", "json-source");
  ConfigurableParam::setValues({{"TestParam.map", "{7:8,9:10}"}});
  const auto mapBefore = TestParam::Instance().map;
  const auto json = ConfigurableParam::asJSON("TestParam");

  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", "json-modified");
  ConfigurableParam::setValues({{"TestParam.map", "{1:2}"}});
  ConfigurableParam::updateFromJSONString(json, "TestParam");

  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, 321);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, "json-source");
  BOOST_CHECK_EQUAL(TestParam::Instance().map.size(), mapBefore.size());
  BOOST_CHECK_EQUAL(TestParam::Instance().map.at(7), mapBefore.at(7));
  BOOST_CHECK_EQUAL(TestParam::Instance().map.at(9), mapBefore.at(9));
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_JSONString_ParamsListMissing)
{
  BOOST_CHECK_THROW(ConfigurableParam::updateFromJSONString(ConfigurableParam::asJSON("TestParam"), "MissingParam"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_FileIO_ROOT)
{
  // test for root file serialization
  const std::string testFileName = "test_config.root";
  auto iValueBefore = TestParam::Instance().iValue;
  auto sValueBefore = TestParam::Instance().sValue;
  TFile* testFile = TFile::Open(testFileName.c_str(), "RECREATE");
  TestParam::Instance().serializeTo(testFile);
  testFile->Close();
  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", testFileName);
  ConfigurableParam::fromCCDB(testFileName);
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, iValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, sValueBefore);
  std::remove(testFileName.c_str());
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_Cli)
{
  // test setting values from as a cli arg string
  ConfigurableParam::updateFromString("TestParam.iValue=55;TestParam.sValue=cli");
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, 55);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, "cli");
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_LiteralSuffix)
{
  // test setting values with the correct literal suffix
  ConfigurableParam::updateFromString("TestParam.fValue=42.f");
  BOOST_CHECK_EQUAL(TestParam::Instance().fValue, 42.f);

  ConfigurableParam::setValue("TestParam.ullValue", "999ull");
  BOOST_CHECK_EQUAL(TestParam::Instance().ullValue, 999ULL);
  // check using wrong literal suffix fails, prints error to std
  ConfigurableParam::setValue("TestParam.ullValue", "888u");
  BOOST_CHECK_NE(TestParam::Instance().ullValue, 888);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_ContainerParserVector)
{
  auto v = ContainerParser::parse<std::vector<int>>("[1,2,3,4,5]");
  BOOST_CHECK_EQUAL(v.size(), 5);
  BOOST_CHECK_EQUAL(v[0], 1);
  BOOST_CHECK_EQUAL(v[4], 5);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_ContainerParserMap)
{
  auto m = ContainerParser::parse<std::map<std::string, double>>("{alpha:0.5,beta:0.3,gamma:0.2}");
  BOOST_CHECK_EQUAL(m.size(), 3);
  BOOST_CHECK_EQUAL(m["alpha"], 0.5);
  BOOST_CHECK_EQUAL(m["beta"], 0.3);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_ContainerParserSetAndSequenceTypes)
{
  // All non-map containers go through the single parseSequence path. Verify the
  // set containers (which previously took a separate, doubly-parsing parseSet
  // branch) still parse and de-duplicate correctly.
  auto se = ContainerParser::parse<std::set<int>>("[2,3,2,3,2,5]");
  BOOST_CHECK_EQUAL(se.size(), 3);
  BOOST_CHECK(se == (std::set<int>{2, 3, 5}));

  auto us = ContainerParser::parse<std::unordered_set<int>>("[1,2,3,3]");
  BOOST_CHECK_EQUAL(us.size(), 3);
  BOOST_CHECK(us.count(1) && us.count(2) && us.count(3));

  auto li = ContainerParser::parse<std::list<int>>("[7,8]");
  BOOST_CHECK(li == (std::list<int>{7, 8}));

  auto dq = ContainerParser::parse<std::deque<int>>("[9,10]");
  BOOST_CHECK(dq == (std::deque<int>{9, 10}));
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_ContainerParserRejectsEmptyToken)
{
  // A stray delimiter must be reported, not silently swallowed: "[1,,3]" used to
  // parse to a 2-element vector with no error, masking malformed configuration.
  BOOST_CHECK_THROW(ContainerParser::parse<std::vector<int>>("[1,,3]"), std::exception);
  BOOST_CHECK_THROW(ContainerParser::parse<std::vector<int>>("[1,2,]"), std::exception);
  // ... and on the map side, an empty value/entry is rejected too.
  BOOST_CHECK_THROW((ContainerParser::parse<std::map<int, int>>("{1:2,,3:4}")), std::exception);
  BOOST_CHECK_THROW((ContainerParser::parse<std::map<int, int>>("{1:}")), std::exception);
  // Well-formed input is unaffected.
  auto v = ContainerParser::parse<std::vector<int>>("[1,2,3]");
  BOOST_CHECK_EQUAL(v.size(), 3);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_DamerauLevenshteinDistance)
{
  BOOST_CHECK_EQUAL(damerauLevenshteinDistance("TestParam.iValue", "TestParam.iValue"), 0);
  BOOST_CHECK_EQUAL(damerauLevenshteinDistance("TestParam.iValu", "TestParam.iValue"), 1);
  BOOST_CHECK_EQUAL(damerauLevenshteinDistance("TestParam.jValue", "TestParam.iValue"), 1);
  BOOST_CHECK_EQUAL(damerauLevenshteinDistance("TestParam.iVaule", "TestParam.iValue"), 1);
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_UnknownKeySuggestionsNonFatal)
{
  setenv("ALICEO2_CONFIGURABLEPARAM_WRONGKEYISNONFATAL", "1", 1);
  ConfigurableParam::setValue("TestParam.iValue", "222");
  ConfigurableParam::setValue("TestParam.caValue[1]", "33");
  ConfigurableParam::setValues({{"TestParam.iValu", "777"},
                                {"TstParam.iValue", "888"},
                                {"TestParam.caValue[", "999"}});
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, 222);
  BOOST_CHECK_EQUAL(TestParam::Instance().caValue[1], 33);
  unsetenv("ALICEO2_CONFIGURABLEPARAM_WRONGKEYISNONFATAL");
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_Container_FileIO_ROOT)
{
  // test for root file serialization
  const std::string testFileName = "test_config.root";
  TFile* testFile = TFile::Open(testFileName.c_str(), "RECREATE");
  ConfigurableParam::setValue("TestParam.vec", "[1,2,3]");
  ConfigurableParam::setValue("TestParam.u8vec", "[4,5,6]");
  ConfigurableParam::setValue("TestParam.map", "{1:16,2:9,3:23456}");
  ConfigurableParam::setValue("TestParam.smap", "{a:16,b:9,c:23456}");
  ConfigurableParam::setValue("TestParam.set", "[2,3,2,3,2,5]");
  TestParam::Instance().serializeTo(testFile);
  testFile->Close();
  ConfigurableParam::setValue("TestParam.vec", "[9]");
  ConfigurableParam::fromCCDB(testFileName);
  const auto& tp = TestParam::Instance();
  const std::vector<int> v = {1, 2, 3};
  BOOST_CHECK_EQUAL_COLLECTIONS(tp.vec.begin(), tp.vec.end(), v.begin(), v.end());
  const std::vector<uint8_t> v8 = {4, 5, 6};
  BOOST_CHECK_EQUAL_COLLECTIONS(tp.u8vec.begin(), tp.u8vec.end(), v8.begin(), v8.end());
  std::map<int, uint32_t> map{{1, 16}, {2, 9}, {3, 23456}};
  auto testMapEqual = [](const auto& m1, const auto& m2) {
    for (const auto& [k, v] : m1) {
      BOOST_CHECK(m2.find(k) != m2.end());
      BOOST_CHECK_EQUAL(v, m2.at(k));
    }
  };
  testMapEqual(map, tp.map);
  std::map<std::string, uint32_t> smap{{"a", 16}, {"b", 9}, {"c", 23456}};
  testMapEqual(smap, tp.smap);
  std::set<uint16_t> set{2, 3, 5};
  BOOST_CHECK_EQUAL(set.size(), tp.set.size());
  for (const auto& s : set) {
    BOOST_CHECK(tp.set.contains(s));
  }
  std::remove(testFileName.c_str());
}
