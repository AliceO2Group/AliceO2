// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework OptionsRetriever
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/program_options.hpp>
#include "Framework/FairOptionsRetriever.h"
#include <fairmq/options/FairMQProgOptions.h>

namespace bpo = boost::program_options;
using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestOptionsRetriever)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"))    //
    ("aNested.int", bpo::value<int>()->default_value(2))                  //
    ("aNested.float", bpo::value<float>()->default_value(3.f));           //

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd", "--aFloat", "1.0",
                     "--aDouble", "2.0",
                     "--anInt", "10",
                     "--aBoolean",
                     "--aString", "somethingelse",
                     "--aNested.int", "1",
                     "--aNested.float", "2"},
                    false);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.int", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.float", VariantType::Float, 3.f, {"a float option, nested in an object"}},
  };
  FairOptionsRetriever retriever(specs, options);
  BOOST_CHECK_EQUAL(retriever.getFloat("aFloat"), 1.0);
  BOOST_CHECK_EQUAL(retriever.getDouble("aDouble"), 2.0);
  BOOST_CHECK_EQUAL(retriever.getInt("anInt"), 10);
  BOOST_CHECK_EQUAL(retriever.getBool("aBoolean"), true);
  BOOST_CHECK_EQUAL(retriever.getString("aString"), "somethingelse");
  BOOST_CHECK_EQUAL(retriever.getInt("aNested.int"), 1);
  BOOST_CHECK_EQUAL(retriever.getInt("aNested.float"), 2.f);
  // We can get nested objects also via their top-level ptree.
  auto pt = retriever.getPTree("aNested");
  BOOST_CHECK_EQUAL(pt.get<int>("int"), 1);
  BOOST_CHECK_EQUAL(pt.get<float>("float"), 2.f);
}

BOOST_AUTO_TEST_CASE(TestOptionsDefaults)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"))    //
    ("aNested.int", bpo::value<int>()->default_value(2))                  //
    ("aNested.float", bpo::value<float>()->default_value(3.f));           //

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd"}, false);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.int", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.float", VariantType::Float, 3.f, {"a float option, nested in an object"}},
  };
  FairOptionsRetriever retriever(specs, options);
  BOOST_CHECK_EQUAL(retriever.getFloat("aFloat"), 10.f);
  BOOST_CHECK_EQUAL(retriever.getDouble("aDouble"), 20.);
  BOOST_CHECK_EQUAL(retriever.getInt("anInt"), 1);
  BOOST_CHECK_EQUAL(retriever.getBool("aBoolean"), false);
  BOOST_CHECK_EQUAL(retriever.getString("aString"), "something");
  BOOST_CHECK_EQUAL(retriever.getInt("aNested.int"), 2);
  BOOST_CHECK_EQUAL(retriever.getInt("aNested.float"), 3.f);
}
