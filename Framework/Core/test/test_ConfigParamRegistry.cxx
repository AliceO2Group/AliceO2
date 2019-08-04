// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework ConfigParamRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamRegistry.h"

#include <fairmq/options/FairMQProgOptions.h>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;
using namespace o2::framework;

struct Foo {
  // Providing a class with a constructor which takes a ptree
  // allows for getting the object
  explicit Foo(boost::property_tree::ptree in)
    : x{in.get<int>("x")},
      y{in.get<float>("y")}
  {
  }
  int x;
  float y;
};

BOOST_AUTO_TEST_CASE(TestConfigParamRegistry)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString,s", bpo::value<std::string>()->default_value("something"))  //
    ("aNested.x", bpo::value<int>()->default_value(2))                    //
    ("aNested.y", bpo::value<float>()->default_value(3.f));               //

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd", "--aFloat", "1.0",
                     "--aDouble", "2.0",
                     "--anInt", "10",
                     "--aBoolean",
                     "-s", "somethingelse",
                     "--aNested.x", "1",
                     "--aNested.y", "2"},
                    false);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString,s", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.x", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.y", VariantType::Float, 3.f, {"a float option, nested in an object"}},
  };

  auto retriever = std::make_unique<FairOptionsRetriever>(specs, options);
  ConfigParamRegistry registry(std::move(retriever));

  BOOST_CHECK_EQUAL(registry.get<float>("aFloat"), 1.0);
  BOOST_CHECK_EQUAL(registry.get<double>("aDouble"), 2.0);
  BOOST_CHECK_EQUAL(registry.get<int>("anInt"), 10);
  BOOST_CHECK_EQUAL(registry.get<bool>("aBoolean"), true);
  BOOST_CHECK_EQUAL(registry.get<std::string>("aString"), "somethingelse");
  BOOST_CHECK_EQUAL(registry.get<int>("aNested.x"), 1);
  BOOST_CHECK_EQUAL(registry.get<int>("aNested.y"), 2.f);
  // We can get nested objects also via their top-level ptree.
  auto pt = registry.get<boost::property_tree::ptree>("aNested");
  BOOST_CHECK_EQUAL(pt.get<int>("x"), 1);
  BOOST_CHECK_EQUAL(pt.get<float>("y"), 2.f);
  // And we can get it as a generic object as well.
  Foo obj = registry.get<Foo>("aNested");
  BOOST_CHECK_EQUAL(obj.x, 1);
  BOOST_CHECK_EQUAL(obj.y, 2.f);
}
