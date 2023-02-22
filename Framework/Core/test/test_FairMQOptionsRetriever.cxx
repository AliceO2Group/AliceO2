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

#include <catch_amalgamated.hpp>
#include <boost/program_options.hpp>
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamStore.h"
#include <fairmq/ProgOptions.h>
#include <boost/property_tree/ptree.hpp>

namespace bpo = boost::program_options;
using namespace o2::framework;

TEST_CASE("TestFairMQOptionsRetriever")
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("anInt64", bpo::value<int64_t>()->default_value(1ll))                //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"))    //
    ("aNested.int", bpo::value<int>()->default_value(2))                  //
    ("aNested.float", bpo::value<float>()->default_value(3.f));           //

  fair::mq::ProgOptions* options = new fair::mq::ProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd", "--aFloat", "1.0",
                     "--aDouble", "2.0",
                     "--anInt", "10",
                     "--anInt64", "50000000000000",
                     "--aBoolean",
                     "--aString", "somethingelse",
                     "--aNested.int", "1",
                     "--aNested.float", "2"},
                    true);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"anInt64", VariantType::Int64, 1ll, {"an int64_t option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.int", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.float", VariantType::Float, 3.f, {"a float option, nested in an object"}},
  };
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new FairOptionsRetriever(options)};
  retrievers.emplace_back(std::move(fairmqRetriver));

  ConfigParamStore store{specs, std::move(retrievers)};
  store.preload();
  store.activate();

  REQUIRE(store.store().get<float>("aFloat") == 1.0);
  REQUIRE(store.store().get<double>("aDouble") == 2.0);
  REQUIRE(store.store().get<int>("anInt") == 10);
  REQUIRE(store.store().get<int64_t>("anInt64") == 50000000000000ll);
  REQUIRE(store.store().get<bool>("aBoolean") == true);
  REQUIRE(store.store().get<std::string>("aString") == "somethingelse");
  REQUIRE(store.store().get<int>("aNested.int") == 1);
  REQUIRE(store.store().get<float>("aNested.float") == 2.f);
  // We can get nested objects also via their top-level ptree.
  auto pt = store.store().get_child("aNested");
  REQUIRE(pt.get<int>("int") == 1);
  REQUIRE(pt.get<float>("float") == 2.f);
}

TEST_CASE("TestOptionsDefaults")
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("anInt64", bpo::value<int64_t>()->default_value(-50000000000000ll))  //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"))    //
    ("aNested.int", bpo::value<int>()->default_value(2))                  //
    ("aNested.float", bpo::value<float>()->default_value(3.f));           //

  fair::mq::ProgOptions* options = new fair::mq::ProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd"}, true);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"anInt64", VariantType::Int64, -50000000000000ll, {"an int64_t option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.int", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.float", VariantType::Float, 3.f, {"a float option, nested in an object"}},
    ConfigParamSpec{"aNested.double", VariantType::Double, 4., {"a float option, nested in an object"}},
  };
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new FairOptionsRetriever(options)};
  retrievers.emplace_back(std::move(fairmqRetriver));

  ConfigParamStore store{specs, std::move(retrievers)};
  store.preload();
  store.activate();

  REQUIRE(store.store().get<float>("aFloat") == 10.f);
  REQUIRE(store.store().get<double>("aDouble") == 20.);
  REQUIRE(store.store().get<int>("anInt") == 1);
  REQUIRE(store.store().get<int64_t>("anInt64") == -50000000000000ll);
  REQUIRE(store.store().get<bool>("aBoolean") == false);
  REQUIRE(store.store().get<std::string>("aString") == "something");
  REQUIRE(store.store().get<int>("aNested.int") == 2);
  REQUIRE(store.store().get<float>("aNested.float") == 3.f);

  /// They come from FairMQ, not default in any case...
  REQUIRE(store.provenance("aFloat") == "fairmq");
  REQUIRE(store.provenance("aDouble") == "fairmq");
  REQUIRE(store.provenance("anInt") == "fairmq");
  REQUIRE(store.provenance("anInt64") == "fairmq");
  REQUIRE(store.provenance("aBoolean") == "fairmq");
  REQUIRE(store.provenance("aString") == "fairmq");
  REQUIRE(store.provenance("aNested.int") == "fairmq");
  REQUIRE(store.provenance("aNested.float") == "fairmq");
  REQUIRE(store.provenance("aNested.double") == "default");
}
