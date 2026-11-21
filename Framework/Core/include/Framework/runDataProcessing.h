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
#ifndef FRAMEWORK_RUN_DATA_PROCESSING_H
#define FRAMEWORK_RUN_DATA_PROCESSING_H

#include <fmt/format.h>
#include "Framework/ConfigParamSpec.h"
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigurableHelpers.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/SendingPolicy.h"
#include "Framework/WorkflowSpec.h" // IWYU pragma: export
#include "Framework/ConfigContext.h"
#include "Framework/CustomWorkflowTerminationHook.h"
#include "Framework/CommonServices.h"
#include "Framework/WorkflowCustomizationHelpers.h"
#include "Framework/Logger.h" // IWYU pragma: export
#include "Framework/WorkflowDefinitionContext.h"
#include "Framework/Plugins.h"
#include "Framework/CheckTypes.h"
#include "Framework/StructToTuple.h"
#include "ResourcePolicy.h"
#include <vector>

namespace o2::framework
{
using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;
} // namespace o2::framework

/// To be implemented by the user to specify one or more DataProcessorSpec.
///
/// Use the ConfigContext @a context in input to get the value of global configuration
/// properties like command line options, number of available CPUs or whatever
/// can affect the creation of the actual workflow.
///
/// @returns a std::vector of DataProcessorSpec which represents the actual workflow
///         to be executed
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& context);

// This template magic allow users to customize the behavior of the process
// by (optionally) implementing a `configure` method which modifies one of the
// objects in question.
//
// For example it can be optionally implemented by the user to specify the
// channel policies for your setup. Use this if you want to customize the way
// your devices communicate between themself, e.g. if you want to use REQ/REP
// in place of PUB/SUB.
//
// The advantage of this approach is that we do not need to expose the
// configurability / configuration object to the user, unless he really wants to
// modify it. The drawback is that we need to declare the `customize` method
// before include this file.

// By default we leave the channel policies unchanged. Notice that the default still include
// a "match all" policy which uses pub / sub

void defaultConfiguration(std::vector<o2::framework::ConfigParamSpec>& globalWorkflowOptions)
{
  o2::framework::call_if_defined<struct WorkflowOptions>([&](auto* ptr) {
    ptr = new std::decay_t<decltype(*ptr)>;
    o2::framework::homogeneous_apply_refs([&globalWorkflowOptions](auto what) {
      return o2::framework::ConfigurableHelpers::appendOption(globalWorkflowOptions, what);
    },
                                          *ptr);
  });
}

void defaultConfiguration(std::vector<o2::framework::ServiceSpec>& services)
{
  if (services.empty()) {
    services = o2::framework::CommonServices::defaultServices();
  }
}

/// Workflow options which are required by DPL in order to work.
std::vector<o2::framework::ConfigParamSpec> requiredWorkflowOptions();

template <typename T>
concept WithUserOverride = requires(T& something) { customize(something); };

template <typename T>
concept WithNonTrivialDefault = !WithUserOverride<T> && requires(T& something) { defaultConfiguration(something); };

struct UserCustomizationsHelper {
  static auto userDefinedCustomization(WithUserOverride auto& something) -> void
  {
    customize(something);
  }

  static auto userDefinedCustomization(WithNonTrivialDefault auto& something) -> void
  {
    defaultConfiguration(something);
  }

  static auto userDefinedCustomization(auto&) -> void
  {
  }
};

namespace o2::framework
{
class ConfigContext;
class ConfigParamRegistry;
class ConfigParamSpec;
} // namespace o2::framework
/// Helper used to customize a workflow pipelining options
void overridePipeline(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

/// Helper used to customize a workflow via a template data processor
void overrideCloning(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

/// Helper used to add labels to Data Processors
void overrideLabels(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

void doDefaultWorkflowTerminationHook();

template <typename T>
  requires requires(T& policy) { { T::createDefaultPolicies() } -> std::same_as<std::vector<T>>; }
std::vector<T> injectCustomizations()
{
  std::vector<T> policies;
  UserCustomizationsHelper::userDefinedCustomization(policies);
  auto defaultPolicies = T::createDefaultPolicies();
  policies.insert(std::end(policies), std::begin(defaultPolicies), std::end(defaultPolicies));
  return policies;
}

template <typename T>
  requires requires(T& hook) { customize(hook); }
void callWorkflowTermination(T& hook, char const* idstring)
{
  customize(hook);
  hook(idstring);
  doDefaultWorkflowTerminationHook();
}

// Do not call the user hook if it's not there.
template <typename T>
void callWorkflowTermination(T&, char const* idstring)
{
  doDefaultWorkflowTerminationHook();
}

void overrideAll(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

std::unique_ptr<o2::framework::ConfigContext> createConfigContext(std::unique_ptr<o2::framework::ConfigParamRegistry>& workflowOptionsRegistry,
                                                                  o2::framework::ServiceRegistry& configRegistry,
                                                                  std::vector<o2::framework::ConfigParamSpec>& workflowOptions,
                                                                  std::vector<o2::framework::ConfigParamSpec>& extraOptions, int argc, char** argv);

std::unique_ptr<o2::framework::ServiceRegistry> createRegistry();

char* getIdString(int argc, char** argv);

#define DPL_STRINGIZE_NX(A) #A
#define DPL_STRINGIZE(A) DPL_STRINGIZE_NX(A)

// The workflow definition is a plugin. By default it is embedded in the
// executable as DPL_WORKFLOW_PLUGIN_NAME, and main() runs it. A workflow built
// as a plugin library gets DPL_WORKFLOW_PLUGIN_NAME at build time: it has no
// main(), and the executable which links it calls dpl_workflow_main() instead.
#ifndef DPL_WORKFLOW_PLUGIN_NAME
#define DPL_WORKFLOW_PLUGIN_NAME InternalCustomWorkflow
#define DPL_WORKFLOW_PLUGIN_EMBEDDED
#endif

struct DPL_WORKFLOW_PLUGIN_NAME : o2::framework::WorkflowPlugin {
  o2::framework::WorkflowDefinition* create() override
  {
    return new o2::framework::WorkflowDefinition{
      .customize = [](o2::framework::WorkflowDefinitionContext& workflowContext) {
        using namespace o2::framework;
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.workflowOptions);
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.completionPolicies);
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.dispatchPolicies);
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.resourcePolicies);
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.callbacksPolicies);
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.sendingPolicies); },
      .defineWorkflow = [](o2::framework::WorkflowDefinitionContext& workflowContext, o2::framework::ConfigContext& configContext) {
        using namespace o2::framework;
        workflowContext.specs = defineDataProcessing(configContext);
        overrideAll(configContext, workflowContext.specs);
        for (auto& spec : workflowContext.specs) {
          UserCustomizationsHelper::userDefinedCustomization(spec.requiredServices);
        }
        UserCustomizationsHelper::userDefinedCustomization(workflowContext.channelPolicies); }};
  }
};

// This is like the plugin macros, we simply do it explicitly to avoid macro inside macro expansion
extern "C" {
DPLPluginHandle* dpl_plugin_callback(DPLPluginHandle* previous)
{
  return new DPLPluginHandle{new DPL_WORKFLOW_PLUGIN_NAME{}, strdup(DPL_STRINGIZE(DPL_WORKFLOW_PLUGIN_NAME)), o2::framework::DplPluginKind::Workflow, previous};
}
}

// Loads the workflow plugin @a pluginSpec, i.e. "<library>:<plugin>", and runs it.
// An empty library means the plugin embedded in the executable, found via @a embeddedPlugins.
int callMain(int argc, char** argv, char const* pluginSpec, DPLPluginHandle* (*embeddedPlugins)(DPLPluginHandle*));

#ifdef DPL_WORKFLOW_PLUGIN_EMBEDDED
int main(int argc, char** argv)
#else
extern "C" int dpl_workflow_main(int argc, char** argv)
#endif
{
  using namespace o2::framework;

  int result = callMain(argc, argv, ":" DPL_STRINGIZE(DPL_WORKFLOW_PLUGIN_NAME), dpl_plugin_callback);

  char* idstring = getIdString(argc, argv);
  o2::framework::OnWorkflowTerminationHook onWorkflowTerminationHook;
  callWorkflowTermination(onWorkflowTerminationHook, idstring);

  return result;
}
#endif
