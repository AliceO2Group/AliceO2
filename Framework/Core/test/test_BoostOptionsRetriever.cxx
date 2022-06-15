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
#define BOOST_TEST_MODULE Test Framework BoostOptionsRetriever
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TrivialBoostOptionsRetrieverTest)
{
  using namespace o2::framework;
  namespace bpo = boost::program_options;

  auto specs = std::vector<ConfigParamSpec>{
    {"someInt", VariantType::Int, 2, {"some int option"}},
    {"someInt8", VariantType::Int8, static_cast<int8_t>(2), {"some int8 option"}},
    {"someInt16", VariantType::Int16, static_cast<int16_t>(2), {"some int16 option"}},
    {"someUInt8", VariantType::UInt8, static_cast<uint8_t>(2u), {"some uint8 option"}},
    {"someUInt16", VariantType::UInt16, static_cast<uint16_t>(2u), {"some uint16 option"}},
    {"someUInt32", VariantType::UInt32, 2u, {"some uint32 option"}},
    {"someUInt64", VariantType::UInt64, static_cast<uint64_t>(2ul), {"some uint64 option"}},
    {"someInt64", VariantType::Int64, 4ll, {"some int64 option"}},
    {"someBool", VariantType::Bool, false, {"some bool option"}},
    {"someFloat", VariantType::Float, 2.0f, {"some float option"}},
    {"someDouble", VariantType::Double, 2.0, {"some double option"}},
    {"someString", VariantType::String, strdup("barfoo"), {"some string option"}}};
  const char* args[] = {
    "test",
    "--someBool",
    "--someInt", "1",
    "--someInt8", "1",
    "--someInt16", "1",
    "--someUInt8", "1",
    "--someUInt16", "1",
    "--someUInt32", "1",
    "--someUInt64", "1",
    "--someInt64", "50000000000000",
    "--someFloat", "0.5",
    "--someDouble", "0.5",
    "--someString", "foobar"};
  bpo::variables_map vm;
  bpo::options_description opts;

  ConfigParamsHelper::populateBoostProgramOptions(opts, specs);

  bpo::store(parse_command_line(sizeof(args) / sizeof(char*), args, opts), vm);
  bpo::notify(vm);
  BOOST_CHECK(vm["someInt"].as<int>() == 1);
  BOOST_CHECK(vm["someInt8"].as<int8_t>() == '1');
  BOOST_CHECK(vm["someInt16"].as<int16_t>() == 1);
  BOOST_CHECK(vm["someUInt8"].as<uint8_t>() == '1');
  BOOST_CHECK(vm["someUInt16"].as<uint16_t>() == 1);
  BOOST_CHECK(vm["someUInt32"].as<uint32_t>() == 1);
  BOOST_CHECK(vm["someUInt64"].as<uint64_t>() == 1);
  BOOST_CHECK(vm["someInt64"].as<int64_t>() == 50000000000000ll);
  BOOST_CHECK(vm["someBool"].as<bool>() == true);
  BOOST_CHECK(vm["someString"].as<std::string>() == "foobar");
  BOOST_CHECK(vm["someFloat"].as<float>() == 0.5);
  BOOST_CHECK(vm["someDouble"].as<double>() == 0.5);
}
