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
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/Array2D.h"

#include <fairmq/ProgOptions.h>
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

TEST_CASE("TestConfigParamRegistry")
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
    ConfigParamSpec{"anUInt8", VariantType::UInt8, static_cast<uint8_t>(1u), {"an uint8 option"}},
    ConfigParamSpec{"anUInt16", VariantType::UInt16, static_cast<uint16_t>(1u), {"an uint16 option"}},
    ConfigParamSpec{"anUInt32", VariantType::UInt32, 1u, {"an uint32 option"}},
    ConfigParamSpec{"anUInt64", VariantType::UInt64, static_cast<uint64_t>(1ul), {"an uint64 option"}},
    ConfigParamSpec{"anInt64", VariantType::Int64, 1ll, {"an int64_t option"}},
    ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}},
    ConfigParamSpec{"aString,s", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aBoolean", VariantType::Bool, true, {"a boolean option"}},
    ConfigParamSpec{"aNested.x", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.y", VariantType::Float, 3.f, {"a float option, nested in an object"}},
    ConfigParamSpec{"aNested.z", VariantType::Float, 4.f, {"a float option, nested in an object"}},
    ConfigParamSpec{"aDict", VariantType::Dict, emptyDict(), {"A dictionary"}}};

  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new FairOptionsRetriever(options)};
  retrievers.emplace_back(std::move(fairmqRetriver));

  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  ConfigParamRegistry registry(std::move(store));

  REQUIRE(registry.get<float>("aFloat") == 1.0);
  REQUIRE(registry.get<double>("aDouble") == 2.0);
  REQUIRE(registry.get<int>("anInt") == 10);
  REQUIRE(registry.get<int8_t>("anInt8") == '2');
  REQUIRE(registry.get<int16_t>("anInt16") == 10);
  REQUIRE(registry.get<uint8_t>("anUInt8") == '2');
  REQUIRE(registry.get<uint16_t>("anUInt16") == 10);
  REQUIRE(registry.get<uint32_t>("anUInt32") == 10);
  REQUIRE(registry.get<uint64_t>("anUInt64") == 10);
  REQUIRE(registry.get<int64_t>("anInt64") == 50000000000000ll);
  REQUIRE(registry.get<bool>("aBoolean") == true);
  REQUIRE(registry.get<std::string>("aString") == "somethingelse");
  REQUIRE(registry.get<int>("aNested.x") == 1);
  REQUIRE(registry.get<float>("aNested.y") == 2.f);
  REQUIRE(registry.get<float>("aNested.z") == 4.f);
  // We can get nested objects also via their top-level ptree.
  auto pt = registry.get<boost::property_tree::ptree>("aNested");
  auto pt2 = registry.get<boost::property_tree::ptree>("aDict");
  REQUIRE(pt.get<int>("x") == 1);
  REQUIRE(pt.get<float>("y") == 2.f);
  // And we can get it as a generic object as well.
  Foo obj = registry.get<Foo>("aNested");
  REQUIRE(obj.x == 1);
  REQUIRE(obj.y == 2.f);
}

TEST_CASE("TestConfigParamRegistryOverride")
{
  float m[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  int l[2][2] = {{1, 2}, {3, 4}};
  // a Variant cannot be created from a temporary vector, so the defaults are named
  std::vector<int> defaultVector{1, 2, 3, 4, 5};
  std::vector<std::string> defaultStringVector{"a", "b", "c"};
  std::vector<double> defaultNestedVector{0.5, 1.5};
  // the store keeps a reference to the specs, which have to outlive it
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}},
    ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}},
    ConfigParamSpec{"aVector", VariantType::ArrayInt, defaultVector, {"an int array option"}},
    ConfigParamSpec{"aStringVector", VariantType::ArrayString, defaultStringVector, {"a string array option"}},
    ConfigParamSpec{"anArray2D", VariantType::Array2DFloat, Array2D<float>{&m[0][0], 3, 4}, {"a 2D array option"}},
    ConfigParamSpec{"aLabeledArray", VariantType::LabeledArrayInt, LabeledArray<int>{&l[0][0], 2, 2, {"r1", "r2"}, {"c1", "c2"}}, {"a labeled array option"}},
    ConfigParamSpec{"aNested.x", VariantType::Int, 2, {"an int option, nested in an object"}},
    ConfigParamSpec{"aNested.vec", VariantType::ArrayDouble, defaultNestedVector, {"a double array option, nested in an object"}},
  };
  auto store = std::make_unique<ConfigParamStore>(specs, std::vector<std::unique_ptr<ParamRetriever>>{});
  store->preload();
  store->activate();
  ConfigParamRegistry registry(std::move(store));
  REQUIRE(registry.get<std::vector<int>>("aVector").size() == 5);

  // leaf values keep working
  registry.override("anInt", 10);
  REQUIRE(registry.get<int>("anInt") == 10);

  // a shorter vector replaces the whole subtree, no stale elements are left
  registry.override("aVector", std::vector<int>{7, 8});
  REQUIRE(registry.get<std::vector<int>>("aVector") == std::vector<int>{7, 8});
  registry.override("aStringVector", std::vector<std::string>{"x"});
  REQUIRE(registry.get<std::vector<std::string>>("aStringVector") == std::vector<std::string>{"x"});

  // a smaller 2D array replaces the whole subtree
  float m2[1][2] = {{0.25, 0.75}};
  registry.override("anArray2D", Array2D<float>{&m2[0][0], 1, 2});
  auto a2d = registry.get<Array2D<float>>("anArray2D");
  REQUIRE(a2d.rows == 1);
  REQUIRE(a2d.cols == 2);
  REQUIRE(a2d(0, 1) == 0.75f);

  // labels are replaced together with the values
  int l2[1][3] = {{5, 6, 7}};
  registry.override("aLabeledArray", LabeledArray<int>{&l2[0][0], 1, 3, {"row"}, {"x", "y", "z"}});
  auto la = registry.get<LabeledArray<int>>("aLabeledArray");
  REQUIRE(la.getLabelsRows() == std::vector<std::string>{"row"});
  REQUIRE(la.getLabelsCols() == std::vector<std::string>{"x", "y", "z"});
  REQUIRE(la.get("row", "z") == 7);
  // an unlabeled value is stored as it is, without merging the previous labels
  registry.override("aLabeledArray", LabeledArray<int>{&l[0][0], 2, 2});
  la = registry.get<LabeledArray<int>>("aLabeledArray");
  REQUIRE(la.getLabelsRows().empty());
  REQUIRE(la.getLabelsCols().empty());
  REQUIRE(la.get(1U, 0U) == 3);

  // nested subtrees are replaced without touching their siblings
  registry.override("aNested.vec", std::vector<double>{2.5});
  REQUIRE(registry.get<std::vector<double>>("aNested.vec") == std::vector<double>{2.5});
  REQUIRE(registry.get<int>("aNested.x") == 2);
  REQUIRE(registry.get<std::string>("aString") == "foo");
  REQUIRE(registry.get<std::vector<int>>("aVector") == std::vector<int>{7, 8});

  // a key which does not exist yet is created
  REQUIRE(registry.hasOption("aNewVector") == false);
  registry.override("aNewVector", std::vector<float>{1.f, 2.f});
  REQUIRE(registry.hasOption("aNewVector"));
  REQUIRE(registry.get<std::vector<float>>("aNewVector") == std::vector<float>{1.f, 2.f});

  // a leaf value can be replaced by a subtree
  registry.override("aString", std::vector<std::string>{"now", "an", "array"});
  REQUIRE(registry.get<std::vector<std::string>>("aString") == std::vector<std::string>{"now", "an", "array"});
}
