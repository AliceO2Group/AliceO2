// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowDefinitionContext.h"
#include "Framework/CompletionPolicyHelpers.h"
#include <catch_amalgamated.hpp>
#include <algorithm>

using namespace o2::framework;

namespace
{
// A plugin with a processor called @a processor, which declares the
// workflow options @a options and records the value of "shared" it sees.
WorkflowDefinition makeDefinition(std::string processor, std::vector<ConfigParamSpec> options, int& shared)
{
  return WorkflowDefinition{
    .customize = [processor, options](WorkflowDefinitionContext& context) {
      context.workflowOptions = options;
      context.completionPolicies.push_back(CompletionPolicyHelpers::consumeWhenAll(processor.c_str(), [](auto const&) { return false; })); },
    .defineWorkflow = [processor, &shared](WorkflowDefinitionContext& context, ConfigContext& configContext) {
      shared = configContext.options().get<int>("shared");
      context.specs.push_back(DataProcessorSpec{.name = processor}); }};
}

size_t countOption(WorkflowDefinitionContext const& context, std::string const& name)
{
  return std::count_if(context.workflowOptions.begin(), context.workflowOptions.end(), [&name](auto const& option) { return option.name == name; });
}
} // namespace

TEST_CASE("TestMergeWorkflowDefinitions")
{
  int sharedA = 0;
  int sharedB = 0;
  std::vector<WorkflowDefinition> definitions{
    makeDefinition("a", {{"shared", VariantType::Int, 1, {"shared option"}}, {"a-only", VariantType::Int, 2, {"option of a"}}}, sharedA),
    makeDefinition("b", {{"shared", VariantType::Int, 1, {"shared option"}}, {"b-only", VariantType::String, "x", {"option of b"}}}, sharedB)};
  char const* args[] = {"test", "--shared", "3", "--b-only", "y"};
  auto context = defineWorkflow(definitions, {"libA:a", "libB:b"}, 5, const_cast<char**>(args));

  // Options are parsed once and each plugin sees the value of the shared one.
  REQUIRE(countOption(context, "shared") == 1);
  REQUIRE(countOption(context, "a-only") == 1);
  REQUIRE(countOption(context, "b-only") == 1);
  REQUIRE(sharedA == 3);
  REQUIRE(sharedB == 3);
  REQUIRE(context.configContext->options().get<std::string>("b-only") == "y");
  // Options of the plugins come first, then those required by DPL.
  REQUIRE(context.workflowOptions[0].name == "shared");
  REQUIRE(context.workflowOptions[1].name == "a-only");
  REQUIRE(context.workflowOptions[2].name == "b-only");
  REQUIRE(countOption(context, "pipeline") == 1);

  // Data processors and policies are merged in plugin order, the default policies last.
  REQUIRE(context.specs.size() == 2);
  REQUIRE(context.specs[0].name == "a");
  REQUIRE(context.specs[1].name == "b");
  REQUIRE(context.completionPolicies.size() == 2 + CompletionPolicy::createDefaultPolicies().size());
  REQUIRE(context.completionPolicies[0].name == "a");
  REQUIRE(context.completionPolicies[1].name == "b");
  REQUIRE(context.completionPolicies[2].name == CompletionPolicy::createDefaultPolicies()[0].name);
}

TEST_CASE("TestConflictingWorkflowOptions")
{
  int shared = 0;
  char const* args[] = {"test"};
  // Same option with a different default.
  std::vector<WorkflowDefinition> definitions{
    makeDefinition("a", {{"shared", VariantType::Int, 1, {"shared option"}}}, shared),
    makeDefinition("b", {{"shared", VariantType::Int, 2, {"shared option"}}}, shared)};
  REQUIRE_THROWS(defineWorkflow(definitions, {"libA:a", "libB:b"}, 1, const_cast<char**>(args)));
  // Same option with a different type.
  definitions[1] = makeDefinition("b", {{"shared", VariantType::Float, 1.f, {"shared option"}}}, shared);
  REQUIRE_THROWS(defineWorkflow(definitions, {"libA:a", "libB:b"}, 1, const_cast<char**>(args)));
}

TEST_CASE("TestDuplicateDataProcessor")
{
  int shared = 0;
  char const* args[] = {"test"};
  std::vector<WorkflowDefinition> definitions{
    makeDefinition("same", {{"shared", VariantType::Int, 1, {"shared option"}}}, shared),
    makeDefinition("same", {{"shared", VariantType::Int, 1, {"shared option"}}}, shared)};
  REQUIRE_THROWS(defineWorkflow(definitions, {"libA:a", "libB:b"}, 1, const_cast<char**>(args)));
}
