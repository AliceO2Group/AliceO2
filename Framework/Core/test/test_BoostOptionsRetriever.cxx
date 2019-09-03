// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    {"someBool", VariantType::Bool, false, {"some bool option"}},
    {"someFloat", VariantType::Float, 2.0f, {"some float option"}},
    {"someDouble", VariantType::Double, 2.0, {"some double option"}},
    {"someString", VariantType::String, strdup("barfoo"), {"some string option"}}};
  const char* args[] = {
    "test",
    "--someBool",
    "--someInt", "1",
    "--someFloat", "0.5",
    "--someDouble", "0.5",
    "--someString", "foobar"};
  bpo::variables_map vm;
  bpo::options_description opts;

  ConfigParamsHelper::populateBoostProgramOptions(opts, specs);

  bpo::store(parse_command_line(sizeof(args) / sizeof(char*), args, opts), vm);
  bpo::notify(vm);
  BOOST_CHECK(vm["someInt"].as<int>() == 1);
  BOOST_CHECK(vm["someBool"].as<bool>() == true);
  BOOST_CHECK(vm["someString"].as<std::string>() == "foobar");
  BOOST_CHECK(vm["someFloat"].as<float>() == 0.5);
  BOOST_CHECK(vm["someDouble"].as<double>() == 0.5);
}
