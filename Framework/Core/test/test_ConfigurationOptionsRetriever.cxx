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

#include "../src/ConfigurationOptionsRetriever.h"
#include "Framework/ConfigParamStore.h"
#include <catch_amalgamated.hpp>
#include <Configuration/ConfigurationFactory.h>
#include <Configuration/ConfigurationInterface.h>

#include <fstream>

using namespace o2::framework;
using namespace o2::configuration;

TEST_CASE("TestOptionsRetriever")
{
  const std::string TEMP_FILE = "/tmp/alice_o2_configuration_test_file.ini";
  // Put stuff in temp file
  {
    std::ofstream stream(TEMP_FILE);
    stream << "key=value\n"
              "[section]\n"
              "key_int=123\n"
              "key_float=4.56\n"
              "key_string=hello\n";
  }
  // Get file configuration interface from factory
  auto conf = ConfigurationFactory::getConfiguration("ini:/" + TEMP_FILE);

  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"key", VariantType::String, "someDifferentValue", {"a string option"}},
    ConfigParamSpec{"section.key_int", VariantType::Int64, 1ll, {"an int64_t option"}},
    ConfigParamSpec{"section.key_float", VariantType::Float, 2.0f, {"a float option"}},
    ConfigParamSpec{"section.key_string", VariantType::String, "foo", {"a string option"}},
  };

  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new ConfigurationOptionsRetriever(conf.get(), "")};
  retrievers.emplace_back(std::move(fairmqRetriver));

  ConfigParamStore store{specs, std::move(retrievers)};
  store.preload();
  store.activate();

  REQUIRE(store.store().get<std::string>("key") == "value");
  REQUIRE(store.store().get<int>("section.key_int") == 123);
  REQUIRE(store.store().get<float>("section.key_float") == 4.56f);
  REQUIRE(store.store().get<std::string>("section.key_string") == "hello");
  // We can get nested objects also via their top-level ptree.
  auto pt = store.store().get_child("section");
  REQUIRE(pt.get<int>("key_int") == 123);
  REQUIRE(pt.get<float>("key_float") == 4.56f);
}
