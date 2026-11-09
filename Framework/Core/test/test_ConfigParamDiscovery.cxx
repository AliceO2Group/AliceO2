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
#include "Framework/ConfigParamDiscovery.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/Array2D.h"

#include <stdexcept>

using namespace o2::framework;

TEST_CASE("TestParseOriginLevelMapping")
{
  auto map = ConfigParamDiscovery::parseOriginLevelMapping("EMB:0,AOD:1");
  REQUIRE(map.getLabelsRows() == std::vector<std::string>{"EMB", "AOD"});
  REQUIRE(map.getLabelsCols() == std::vector<std::string>{"level"});
  REQUIRE(map.rows() == 2);
  REQUIRE(map.cols() == 1);
  REQUIRE(map.get(0U, 0U) == 0);
  REQUIRE(map.get(1U, 0U) == 1);
  REQUIRE(map.getRow("AOD")[0] == 1);
  REQUIRE(map.getRow("DYN") == nullptr);

  auto single = ConfigParamDiscovery::parseOriginLevelMapping("DYN:12");
  REQUIRE(single.getLabelsRows() == std::vector<std::string>{"DYN"});
  REQUIRE(single.get(0U, 0U) == 12);

  for (auto bad : {"", "EMB", "EMB:", ":1", "EMB:x", "EMB:1x", "EMB:-1", "TOOLONG:1", "EMB:0,EMB:1", "EMB:0,"}) {
    INFO("mapping: \"" << bad << "\"");
    REQUIRE_THROWS_AS(ConfigParamDiscovery::parseOriginLevelMapping(bad), std::runtime_error);
  }
}

namespace
{
// The plain options as discovered from level 0 (EMB), plus the metadata of each origin:
// AOD reads level 1, XYZ reads a level without metadata.
std::vector<ConfigParamSpec> makeSpecs(std::string const& replace, bool withOrigins)
{
  static std::vector<std::string> plainTables{"O2bc", "O2track"};
  static std::vector<std::string> embTables{"O2bc", "O2track"};
  static std::vector<std::string> aodTables{"O2bc"};
  static std::string values[3][6] = {
    {"MC", "LHC24a2", "x", "", "/path/L0.root", ""},
    {"RAW", "LHC23zzm", "", "y", "/path/L1.root", ""},
    {"", "", "", "", "/path/L2.root", "1"}};
  static LabeledArray<std::string> origins{&values[0][0], 3, 6, {"EMB", "AOD", "XYZ"}, {"DataType", "LPMProductionTag", "OnlyL0", "OnlyL1", "source", "disable"}};
  std::vector<ConfigParamSpec> specs{
    ConfigParamSpec{"aod-origin-replace", VariantType::String, replace, {"origin"}},
    ConfigParamSpec{"aod-metadata-DataType", VariantType::String, "MC", {"metadata"}},
    ConfigParamSpec{"aod-metadata-LPMProductionTag", VariantType::String, "LHC24a2", {"metadata"}},
    ConfigParamSpec{"aod-metadata-OnlyL0", VariantType::String, "x", {"metadata"}},
    ConfigParamSpec{"aod-metadata-source", VariantType::String, "/path/L0.root", {"metadata"}},
    ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, plainTables, {"metadata"}},
  };
  if (withOrigins) {
    specs.push_back(ConfigParamSpec{"aod-metadata-origins", VariantType::LabeledArrayString, origins, {"metadata per origin"}});
    specs.push_back(ConfigParamSpec{"aod-metadata-origin-EMB-tables", VariantType::ArrayString, embTables, {"tables"}});
    specs.push_back(ConfigParamSpec{"aod-metadata-origin-AOD-tables", VariantType::ArrayString, aodTables, {"tables"}});
  }
  return specs;
}

std::unique_ptr<ConfigParamRegistry> makeRegistry(std::vector<ConfigParamSpec> const& specs)
{
  auto store = std::make_unique<ConfigParamStore>(specs, std::vector<std::unique_ptr<ParamRetriever>>{});
  store->preload();
  store->activate();
  auto registry = std::make_unique<ConfigParamRegistry>(std::move(store));
  ConfigParamDiscovery::applyOriginMetadata(*registry);
  return registry;
}

void requireLevel0(ConfigParamRegistry const& registry)
{
  REQUIRE(registry.get<std::string>("aod-metadata-DataType") == "MC");
  REQUIRE(registry.get<std::string>("aod-metadata-LPMProductionTag") == "LHC24a2");
  REQUIRE(registry.get<std::string>("aod-metadata-OnlyL0") == "x");
  REQUIRE(registry.hasOption("aod-metadata-OnlyL1") == false);
  REQUIRE(registry.get<std::string>("aod-metadata-source") == "/path/L0.root");
  REQUIRE(registry.get<std::vector<std::string>>("aod-metadata-tables") == std::vector<std::string>{"O2bc", "O2track"});
  REQUIRE(registry.hasOption("aod-metadata-disable") == false);
}
} // namespace

TEST_CASE("TestApplyOriginMetadata")
{
  SECTION("replaced origin reading level 0")
  {
    auto specs = makeSpecs("EMB", true);
    auto registry = makeRegistry(specs);
    requireLevel0(*registry);
  }

  SECTION("AOD reading level 1")
  {
    auto specs = makeSpecs("", true);
    auto registry = makeRegistry(specs);
    REQUIRE(registry->get<std::string>("aod-metadata-DataType") == "RAW");
    REQUIRE(registry->get<std::string>("aod-metadata-LPMProductionTag") == "LHC23zzm");
    // a key which exists only at level 0 is removed
    REQUIRE(registry->hasOption("aod-metadata-OnlyL0") == false);
    REQUIRE(registry->get<std::string>("aod-metadata-OnlyL1") == "y");
    REQUIRE(registry->get<std::string>("aod-metadata-source") == "/path/L1.root");
    REQUIRE(registry->get<std::vector<std::string>>("aod-metadata-tables") == std::vector<std::string>{"O2bc"});
    REQUIRE(registry->hasOption("aod-metadata-disable") == false);
  }

  SECTION("origin without a row")
  {
    auto specs = makeSpecs("FOO", true);
    auto registry = makeRegistry(specs);
    requireLevel0(*registry);
  }

  SECTION("no metadata per origin")
  {
    auto specs = makeSpecs("", false);
    auto registry = makeRegistry(specs);
    requireLevel0(*registry);
  }

  SECTION("origin reading a level without metadata")
  {
    auto specs = makeSpecs("XYZ", true);
    auto registry = makeRegistry(specs);
    for (auto key : {"aod-metadata-DataType", "aod-metadata-LPMProductionTag", "aod-metadata-OnlyL0", "aod-metadata-OnlyL1", "aod-metadata-source", "aod-metadata-tables"}) {
      INFO("key: " << key);
      REQUIRE(registry->hasOption(key) == false);
    }
    REQUIRE(registry->get<std::string>("aod-metadata-disable") == "1");
  }
}
