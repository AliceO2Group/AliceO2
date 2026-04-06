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

#include "Framework/WorkflowSpec.h"
#include "../src/WorkflowSerializationHelpers.h"
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("TestVerifyWorkflowSerialization")
{
  using namespace o2::framework;
  WorkflowSpec w0{                       //
                  DataProcessorSpec{"A", //
                                    {InputSpec{"foo", "A", "COLLISIONCONTEXT", 1, Lifetime::Condition, {
                                                                                                         ConfigParamSpec{"aUrl", VariantType::String, "foo/bar", {"A InputSpec option"}},       //
                                                                                                         ConfigParamSpec{"bUrl", VariantType::String, "foo/foo", {"Another InputSpec option"}}, //
                                                                                                       }}},                                                                                     //
                                    {OutputSpec{{"bar"}, "C", "D", 2, Lifetime::Timeframe}},                                                                                                    //
                                    AlgorithmSpec{[](ProcessingContext& ctx) {}},                                                                                                               //
                                    {                                                                                                                                                           //
                                     ConfigParamSpec{"aInt", VariantType::Int, 0, {"An Int"}},                                                                                                  //
                                     ConfigParamSpec{"aFloat", VariantType::Float, 1.3, {"A Float"}},                                                                                           //
                                     ConfigParamSpec{"aBool", VariantType::Bool, true, {"A Bool"}},                                                                                             //
                                     ConfigParamSpec{"aString", VariantType::String, "some string", {"A String"}}}},                                                                            //                                                                                                    //
                  DataProcessorSpec{"B",                                                                                                                                                        //
                                    {InputSpec{"foo", "C", "D"}},                                                                                                                               //
                                    {                                                                                                                                                           //
                                     OutputSpec{{"bar1"}, "E", "F", 0},                                                                                                                         //
                                     OutputSpec{{"bar2"}, "E", "F", 1}},                                                                                                                        //
                                    AlgorithmSpec{[](ProcessingContext& ctx) {}},                                                                                                               //
                                    {}},                                                                                                                                                        //
                  DataProcessorSpec{"C", {},                                                                                                                                                    //
                                    {                                                                                                                                                           //
                                     OutputSpec{{"bar"}, "G", "H"}},                                                                                                                            //
                                    AlgorithmSpec{[](ProcessingContext& ctx) {}},                                                                                                               //
                                    {}},                                                                                                                                                        //
                  DataProcessorSpec{"D", {InputSpec{"foo", {"C", "D"}}},                                                                                                                        //
                                    {OutputSpec{{"bar"}, {"I", "L"}}},                                                                                                                          //
                                    AlgorithmSpec{[](ProcessingContext& ctx) {}},                                                                                                               //
                                    {},                                                                                                                                                         //
                                    CommonServices::defaultServices(),                                                                                                                          //
                                    {{"label a"}, {"label \"b\""}},
                                    {{"key1", "v\"al'1"}, {"", "val2"}, {"key3", ""}, {"", ""}}}};

  std::vector<DataProcessorInfo> dataProcessorInfoOut{
    {.name = "A", .executable = "test_Framework_test_SerializationWorkflow", .cmdLineArgs = {"foo"}, .workflowOptions = {ConfigParamSpec{"aBool", VariantType::Bool, true, {"A Bool"}}}},
    {.name = "B", .executable = "test_Framework_test_SerializationWorkflow", .cmdLineArgs = {"b-bar", "bfoof", "fbdbfaso"}},
    {.name = "C", .executable = "test_Framework_test_SerializationWorkflow"},
    {.name = "D", .executable = "test_Framework_test_SerializationWorkflow"},
  };

  CommandInfo commandInfoOut{"o2-dpl-workflow -b --option 1 --option 2"};

  std::vector<DataProcessorInfo> dataProcessorInfoIn{};
  CommandInfo commandInfoIn;

  std::ostringstream firstDump;
  WorkflowSerializationHelpers::dump(firstDump, w0, dataProcessorInfoOut, commandInfoOut);
  std::istringstream is;
  is.str(firstDump.str());
  WorkflowSpec w1;
  WorkflowSerializationHelpers::import(is, w1, dataProcessorInfoIn, commandInfoIn);

  std::ostringstream secondDump;
  WorkflowSerializationHelpers::dump(secondDump, w1, dataProcessorInfoIn, commandInfoIn);

  REQUIRE(w0.size() == 4);
  REQUIRE(w0.size() == w1.size());
  REQUIRE(firstDump.str() == secondDump.str());
  REQUIRE(commandInfoIn.command == commandInfoOut.command);

  // also check if the conversion to ConcreteDataMatcher is working at import
  REQUIRE(std::get_if<ConcreteDataMatcher>(&w1[0].inputs[0].matcher) != nullptr);
}

/// Test a workflow with a single data processor with a single input
/// which has a wildcard on subspec.
TEST_CASE("TestVerifyWildcard")
{
  using namespace o2::framework;
  WorkflowSpec w0{
    DataProcessorSpec{
      .name = "A",
      .inputs = {{"clbPayload", "CLP"}, {"clbWrapper", "CLW"}},
    }};

  std::vector<DataProcessorInfo> dataProcessorInfoOut{
    {.name = "A", .executable = "test_Framework_test_SerializationWorkflow"},
  };

  CommandInfo commandInfoOut{"o2-dpl-workflow -b --option 1 --option 2"};

  std::vector<DataProcessorInfo> dataProcessorInfoIn{};
  CommandInfo commandInfoIn;

  std::ostringstream firstDump;
  WorkflowSerializationHelpers::dump(firstDump, w0, dataProcessorInfoOut, commandInfoOut);
  std::istringstream is;
  is.str(firstDump.str());
  WorkflowSpec w1;
  WorkflowSerializationHelpers::import(is, w1, dataProcessorInfoIn, commandInfoIn);

  std::ostringstream secondDump;
  WorkflowSerializationHelpers::dump(secondDump, w1, dataProcessorInfoIn, commandInfoIn);

  REQUIRE(w0.size() == 1);
  REQUIRE(w0.size() == w1.size());
  REQUIRE(firstDump.str() == secondDump.str());
  REQUIRE(commandInfoIn.command == commandInfoOut.command);

  // also check if the conversion to ConcreteDataMatcher is working at import
  // REQUIRE(std::get_if<ConcreteDataTypeMatcher>(&w1[0].inputs[0].matcher) != nullptr);;
}

TEST_CASE("TestInputOutputSpecMetadata")
{
  WorkflowSpec wso{
    DataProcessorSpec{
      .name = "S1",
      .outputs = {OutputSpec{OutputLabel{"o1"}, o2::header::DataOrigin{"TST"}, "OUTPUT1", 0, Lifetime::Timeframe, {{"param1", VariantType::Bool, true, ConfigParamSpec::HelpString{"\"\""}}, {"param2", VariantType::Bool, true, ConfigParamSpec::HelpString{"\"\""}}}},
                  OutputSpec{OutputLabel{"o2"}, o2::header::DataOrigin{"TST"}, "OUTPUT2"}}}};

  std::vector<DataProcessorInfo> dataProcessorInfoOut{
    {.name = "S1", .executable = "test_Framework_test_SerializationWorkflow"},
  };

  CommandInfo commandInfoOut{"o2-dpl-workflow -b"};

  std::vector<DataProcessorInfo> dataProcessorInfoIn{};
  CommandInfo commandInfoIn;

  std::ostringstream firstDump;
  WorkflowSerializationHelpers::dump(firstDump, wso, dataProcessorInfoOut, commandInfoOut);
  std::istringstream is;
  is.str(firstDump.str());

  WorkflowSpec wsi;
  WorkflowSerializationHelpers::import(is, wsi, dataProcessorInfoIn, commandInfoIn);

  REQUIRE(wsi[0].outputs[0].metadata.size() == 2);
  REQUIRE(wsi[0].outputs[1].metadata.size() == 0);
  REQUIRE(wso[0].outputs[0].metadata.size() == wsi[0].outputs[0].metadata.size());
  REQUIRE(wso[0].outputs[1].metadata.size() == wsi[0].outputs[1].metadata.size());
  REQUIRE(wso[0].outputs[0].metadata[0] == wsi[0].outputs[0].metadata[0]);
  REQUIRE(wso[0].outputs[0].metadata[1] == wsi[0].outputs[0].metadata[1]);
}
