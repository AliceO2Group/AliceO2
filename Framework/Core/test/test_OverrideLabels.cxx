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

// We prevent runDataProcessing from starting a workflow
#define main anything_else_than_main
#include "Framework/runDataProcessing.h"
#undef main
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const&) { return {}; }

using namespace o2::framework;

// Mockup for a workflow with labels as args. It will behave as expected only in single-threaded code!!!
static std::vector<ConfigParamSpec> specs;
static ConfigParamRegistry registry{nullptr};
std::unique_ptr<o2::framework::ConfigContext> mockupLabels(std::string labelArg)
{
  // FIXME: Ugly... We need to fix ownership and make sure the ConfigContext
  //        either owns or shares ownership of the registry.
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  specs = WorkflowCustomizationHelpers::requiredWorkflowOptions();
  specs.push_back(ConfigParamSpec{"labels", VariantType::String, std::move(labelArg), {"labels specification"}});
  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  registry = ConfigParamRegistry(std::move(store));
  auto context = std::make_unique<ConfigContext>(registry, 0, nullptr);
  return context;
}

#include <catch_amalgamated.hpp>

TEST_CASE("OverrideLabels")
{
  {
    // invalid format
    WorkflowSpec workflow{{"A"}};
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A:"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels(":A"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A:asdf,:"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A:asdf,:A"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A:asdf,B:"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A:asdf,B"), workflow), std::runtime_error);
    REQUIRE_THROWS_AS(overrideLabels(*mockupLabels("A,B:asdf"), workflow), std::runtime_error);
  }
  {
    // one processor, one label
    WorkflowSpec workflow{{"A"}};
    auto ctx = mockupLabels("A:abc");
    overrideLabels(*ctx, workflow);
    REQUIRE(workflow[0].labels[0].value == "abc");
  }
  {
    // many processors, many labels
    WorkflowSpec workflow{{"A"}, {"B"}, {"C"}};
    auto ctx = mockupLabels("A:a1:a2,B:b1,C:c1:c2:c3");
    overrideLabels(*ctx, workflow);
    REQUIRE(workflow[0].labels[0].value == "a1");
    REQUIRE(workflow[0].labels[1].value == "a2");
    REQUIRE(workflow[1].labels[0].value == "b1");
    REQUIRE(workflow[2].labels[0].value == "c1");
    REQUIRE(workflow[2].labels[1].value == "c2");
    REQUIRE(workflow[2].labels[2].value == "c3");
  }
  {
    // duplicate labels in arg
    WorkflowSpec workflow{{"A"}};
    auto ctx = mockupLabels("A:a1:a1");
    overrideLabels(*ctx, workflow);
    REQUIRE(workflow[0].labels.size() == 1);
    REQUIRE(workflow[0].labels[0].value == "a1");
  }
  {
    // duplicate labels - one in WF, one in arg
    WorkflowSpec workflow{{"A"}};
    workflow[0].labels.push_back({"a1"});
    auto ctx = mockupLabels("A:a1");
    overrideLabels(*ctx, workflow);
    REQUIRE(workflow[0].labels.size() == 1);
    REQUIRE(workflow[0].labels[0].value == "a1");
  }
}
