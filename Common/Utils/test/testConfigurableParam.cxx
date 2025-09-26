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

#define BOOST_TEST_MODULE Test ConfigurableParams
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>

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
  param.printKeyValues();
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
  ConfigurableParam::writeINI(testFileName);
  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", testFileName);
  ConfigurableParam::updateFromFile(testFileName);
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, iValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, sValueBefore);
  std::remove(testFileName.c_str());
}

BOOST_AUTO_TEST_CASE(ConfigurableParam_FileIO_Json)
{
  // test for json file serialization
  const std::string testFileName = "test_config.json";
  auto iValueBefore = TestParam::Instance().iValue;
  auto sValueBefore = TestParam::Instance().sValue;
  ConfigurableParam::writeJSON(testFileName);
  ConfigurableParam::setValue("TestParam.iValue", "999");
  ConfigurableParam::setValue("TestParam.sValue", testFileName);
  ConfigurableParam::updateFromFile(testFileName);
  BOOST_CHECK_EQUAL(TestParam::Instance().iValue, iValueBefore);
  BOOST_CHECK_EQUAL(TestParam::Instance().sValue, sValueBefore);
  std::remove(testFileName.c_str());
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
