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

#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/BoostOptionsRetriever.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/DataProcessorSpec.h"
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <catch_amalgamated.hpp>
#include <sstream>
#include <vector>

using namespace o2::framework;

TEST_CASE("TrivialBoostOptionsRetrieverTest")
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
  REQUIRE(vm["someInt"].as<int>() == 1);
  REQUIRE(vm["someInt8"].as<int8_t>() == '1');
  REQUIRE(vm["someInt16"].as<int16_t>() == 1);
  REQUIRE(vm["someUInt8"].as<uint8_t>() == '1');
  REQUIRE(vm["someUInt16"].as<uint16_t>() == 1);
  REQUIRE(vm["someUInt32"].as<uint32_t>() == 1);
  REQUIRE(vm["someUInt64"].as<uint64_t>() == 1);
  REQUIRE(vm["someInt64"].as<int64_t>() == 50000000000000ll);
  REQUIRE(vm["someBool"].as<bool>() == true);
  REQUIRE(vm["someString"].as<std::string>() == "foobar");
  REQUIRE(vm["someFloat"].as<float>() == 0.5);
  REQUIRE(vm["someDouble"].as<double>() == 0.5);
}

namespace
{
constexpr char const* fullLabeledJSON = R"({"labels_rows":["a","b"],"labels_cols":["x","y"],"values":[[5,6],[7,8]]})";

LabeledArray<int> defaultLabeledArray()
{
  int m[2][2] = {{1, 2}, {3, 4}};
  return LabeledArray<int>{&m[0][0], 2, 2, {"r1", "r2"}, {"c1", "c2"}};
}

std::unique_ptr<ConfigParamRegistry> registryFromCommandLine(std::vector<ConfigParamSpec> const& specs, std::vector<char const*> args)
{
  args.insert(args.begin(), "test");
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  retrievers.emplace_back(std::make_unique<BoostOptionsRetriever>(true, args.size(), const_cast<char**>(args.data())));
  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  return std::make_unique<ConfigParamRegistry>(std::move(store));
}
} // namespace

TEST_CASE("LabeledArrayBoostOptionsTest")
{
  namespace bpo = boost::program_options;
  auto specs = std::vector<ConfigParamSpec>{
    {"someLabeledArray", VariantType::LabeledArrayInt, defaultLabeledArray(), {"some labeled array option"}}};

  // the option is registered as a string holding the JSON representation
  bpo::options_description opts;
  ConfigParamsHelper::populateBoostProgramOptions(opts, specs);
  const char* args[] = {"test", "--someLabeledArray", fullLabeledJSON};
  bpo::variables_map vm;
  bpo::store(parse_command_line(sizeof(args) / sizeof(char*), args, opts), vm);
  bpo::notify(vm);
  REQUIRE(vm["someLabeledArray"].as<std::string>() == fullLabeledJSON);

  // the default is the JSON representation of the default value
  bpo::variables_map vmDefault;
  const char* noArgs[] = {"test"};
  bpo::store(parse_command_line(1, noArgs, opts), vmDefault);
  REQUIRE(vmDefault["someLabeledArray"].as<std::string>() == specs[0].defaultValue.asString());
}

TEST_CASE("LabeledArrayBoostOptionsRetrieverTest")
{
  auto specs = std::vector<ConfigParamSpec>{
    {"someLabeledArray", VariantType::LabeledArrayInt, defaultLabeledArray(), {"some labeled array option"}}};

  // full JSON on the command line, its labels are kept
  auto registry = registryFromCommandLine(specs, {"--someLabeledArray", fullLabeledJSON});
  auto la = registry->get<LabeledArray<int>>("someLabeledArray");
  REQUIRE(la.getLabelsRows() == std::vector<std::string>{"a", "b"});
  REQUIRE(la.getLabelsCols() == std::vector<std::string>{"x", "y"});
  REQUIRE(la.get("b", "y") == 8);

  // absent from the command line: the default value
  registry = registryFromCommandLine(specs, {});
  la = registry->get<LabeledArray<int>>("someLabeledArray");
  REQUIRE(la.get("r2", "c2") == 4);

  // only values on the command line: the default labels are used
  registry = registryFromCommandLine(specs, {"--someLabeledArray", R"({"values":[[5,6],[7,8]]})"});
  la = registry->get<LabeledArray<int>>("someLabeledArray");
  REQUIRE(la.getLabelsRows() == std::vector<std::string>{"r1", "r2"});
  REQUIRE(la.getLabelsCols() == std::vector<std::string>{"c1", "c2"});
  REQUIRE(la.get("r2", "c1") == 7);

  // values which do not match the default labels
  REQUIRE_THROWS(registryFromCommandLine(specs, {"--someLabeledArray", R"({"values":[[5,6,7]]})"}));
}

TEST_CASE("DefaultValueForHelpTest")
{
  auto const max = ConfigParamsHelper::MaxHelpDefaultLength;
  REQUIRE(ConfigParamsHelper::defaultValueForHelp("") == "");
  REQUIRE(ConfigParamsHelper::defaultValueForHelp("short") == "short");
  std::string atLimit(max, 'x');
  REQUIRE(ConfigParamsHelper::defaultValueForHelp(atLimit) == atLimit);

  std::string longValue = std::string("BEGIN") + std::string(100, '-') + "END";
  auto shown = ConfigParamsHelper::defaultValueForHelp(longValue);
  REQUIRE(shown.size() == max);
  REQUIRE(shown.starts_with("BEGIN"));
  REQUIRE(shown.ends_with("END"));
  REQUIRE(shown.find("...") != std::string::npos);
  REQUIRE(ConfigParamsHelper::defaultValueForHelp(std::string(max + 1, 'x')).size() == max);
}

TEST_CASE("LongDefaultInHelpTest")
{
  namespace bpo = boost::program_options;
  std::vector<int> values(30, 12345);
  std::vector<std::string> rows{"row"};
  std::vector<std::string> cols;
  for (auto i = 0U; i < values.size(); ++i) {
    cols.push_back("column" + std::to_string(i));
  }
  LabeledArray<int> la{values.data(), 1, static_cast<uint32_t>(values.size()), rows, cols};
  auto workflowOptions = std::vector<ConfigParamSpec>{
    {"longLabeledArray", VariantType::LabeledArrayInt, la, {"a labeled array option with a long default"}}};
  auto full = workflowOptions[0].defaultValue.asString();
  REQUIRE(full.size() > ConfigParamsHelper::MaxHelpDefaultLength);

  // the help shows the abbreviated default only
  auto description = ConfigParamsHelper::prepareOptionDescriptions(std::vector<DataProcessorSpec>{}, workflowOptions);
  std::stringstream help;
  help << description;
  REQUIRE(help.str().find(ConfigParamsHelper::defaultValueForHelp(full)) != std::string::npos);
  REQUIRE(help.str().find(full) == std::string::npos);

  // the default value used when parsing is still the full one
  bpo::options_description forwarded;
  ConfigParamsHelper::dpl2BoostOptions(workflowOptions, forwarded);
  const char* noArgs[] = {"test"};
  bpo::variables_map vm;
  bpo::store(parse_command_line(1, noArgs, forwarded), vm);
  REQUIRE(vm["longLabeledArray"].as<std::string>() == full);

  // also for the options registered for the devices
  bpo::options_description deviceOptions;
  ConfigParamsHelper::populateBoostProgramOptions(deviceOptions, workflowOptions);
  std::stringstream deviceHelp;
  deviceHelp << deviceOptions;
  REQUIRE(deviceHelp.str().find(full) == std::string::npos);
  bpo::variables_map deviceVm;
  bpo::store(parse_command_line(1, noArgs, deviceOptions), deviceVm);
  REQUIRE(deviceVm["longLabeledArray"].as<std::string>() == full);
}
