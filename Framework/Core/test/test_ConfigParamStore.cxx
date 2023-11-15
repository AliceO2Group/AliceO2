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
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamStore.h"

#include <fairmq/ProgOptions.h>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace bpo = boost::program_options;
using namespace o2::framework;

TEST_CASE("TestConfigParamStore")
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("anInt8", bpo::value<int8_t>()->default_value(1))                    //
    ("anInt16", bpo::value<int16_t>()->default_value(1))                  //
    ("anUInt8", bpo::value<uint8_t>()->default_value(1))                  //
    ("anUInt16", bpo::value<uint16_t>()->default_value(1))                //
    ("anUInt32", bpo::value<uint32_t>()->default_value(1))                //
    ("anUInt64", bpo::value<uint64_t>()->default_value(1))                //
    ("anInt64", bpo::value<int64_t>()->default_value(1ll))                //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString,s", bpo::value<std::string>()->default_value("something"))  //
    ("aNested.x", bpo::value<int>()->default_value(2))                    //
    ("aNested.y", bpo::value<float>()->default_value(3.f));               //

  fair::mq::ProgOptions* options = new fair::mq::ProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd", "--aFloat", "1.0",
                     "--aDouble", "2.0",
                     "--anInt", "10",
                     "--anInt8", "2",
                     "--anInt16", "10",
                     "--anUInt8", "2",
                     "--anUInt16", "10",
                     "--anUInt32", "10",
                     "--anUInt64", "10",
                     "--anInt64", "50000000000000",
                     "--aBoolean",
                     "-s", "somethingelse",
                     "--aNested.x", "1",
                     "--aNested.y", "2"},
                    true);
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"anInt8", VariantType::Int8, static_cast<int8_t>(1), {"an int8 option"}},
    ConfigParamSpec{"anInt16", VariantType::Int16, static_cast<int16_t>(1), {"an int16 option"}},
    ConfigParamSpec{"anUInt8", VariantType::UInt8, static_cast<uint8_t>(1u), {"an int option"}},
    ConfigParamSpec{"anUInt16", VariantType::UInt16, static_cast<uint16_t>(1u), {"an int option"}},
    ConfigParamSpec{"anUInt32", VariantType::UInt32, 1u, {"an int option"}},
    ConfigParamSpec{"anUInt64", VariantType::UInt64, static_cast<uint64_t>(1ul), {"an int option"}},
    ConfigParamSpec{"anInt64", VariantType::Int64, 1ll, {"an int64_t option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString,s", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.x", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.y", VariantType::Float, 3.f, {"a float option, nested in an object"}},
    ConfigParamSpec{"aNested.z", VariantType::Float, 4.f, {"a float option, nested in an object"}},
  };

  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new FairOptionsRetriever(options)};
  retrievers.emplace_back(std::move(fairmqRetriver));

  ConfigParamStore store{specs, std::move(retrievers)};
  store.preload();
  store.activate();

  std::stringstream ss;
  boost::property_tree::json_parser::write_json(ss, store.store());
  INFO(ss.str());
  REQUIRE(store.store().get<float>("aFloat") == 1.0);
  REQUIRE(store.store().get<double>("aDouble") == 2.0);
  REQUIRE(store.store().get<int>("anInt") == 10);
  REQUIRE(store.store().get<int8_t>("anInt8") == '2');
  REQUIRE(store.store().get<int16_t>("anInt16") == 10);
  REQUIRE(store.store().get<uint8_t>("anUInt8") == '2');
  REQUIRE(store.store().get<uint16_t>("anUInt16") == 10);
  REQUIRE(store.store().get<uint32_t>("anUInt32") == 10);
  REQUIRE(store.store().get<uint64_t>("anUInt64") == 10);
  REQUIRE(store.store().get<int64_t>("anInt64") == 50000000000000ll);
  REQUIRE(store.store().get<bool>("aBoolean") == true);
  REQUIRE(store.store().get<std::string>("aString") == "somethingelse");
  REQUIRE(store.store().get<int>("aNested.x") == 1);
  REQUIRE(store.store().get<int>("aNested.y") == 2.f);
  // We can get nested objects also via their top-level ptree.
  auto pt = store.store().get_child("aNested");
  REQUIRE(pt.get<int>("x") == 1);
  REQUIRE(pt.get<float>("y") == 2.f);
  //
  std::stringstream ss2;
  boost::property_tree::json_parser::write_json(ss2, store.provenanceTree());
  INFO(ss2.str());
  REQUIRE(store.provenance("aFloat") == "fairmq");
  REQUIRE(store.provenance("aDouble") == "fairmq");
  REQUIRE(store.provenance("anInt") == "fairmq");
  REQUIRE(store.provenance("anInt8") == "fairmq");
  REQUIRE(store.provenance("anInt16") == "fairmq");
  REQUIRE(store.provenance("anUInt8") == "fairmq");
  REQUIRE(store.provenance("anUInt16") == "fairmq");
  REQUIRE(store.provenance("anUInt32") == "fairmq");
  REQUIRE(store.provenance("anUInt64") == "fairmq");
  REQUIRE(store.provenance("anInt64") == "fairmq");
  REQUIRE(store.provenance("aBoolean") == "fairmq");
  REQUIRE(store.provenance("aString") == "fairmq");
  REQUIRE(store.provenance("aNested.x") == "fairmq");
  REQUIRE(store.provenance("aNested.y") == "fairmq");
  REQUIRE(store.provenance("aNested.z") == "default");
}
