// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_WORKFLOWDEFINITIONCONTEXT_H_
#define O2_FRAMEWORK_WORKFLOWDEFINITIONCONTEXT_H_

#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/ResourcePolicy.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/SendingPolicy.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ServiceRegistry.h"
#include <functional>
#include <memory>
#include <vector>

namespace o2::framework
{

struct WorkflowDefinitionContext {
  std::vector<ConfigParamSpec> workflowOptions;
  std::vector<CompletionPolicy> completionPolicies;
  std::vector<DispatchPolicy> dispatchPolicies;
  std::vector<ResourcePolicy> resourcePolicies;
  std::vector<CallbacksPolicy> callbacksPolicies;
  std::vector<SendingPolicy> sendingPolicies;
  std::vector<ConfigParamSpec> extraOptions;
  std::vector<ChannelConfigurationPolicy> channelPolicies;
  std::unique_ptr<ConfigContext> configContext;

  // For the moment, let's put them here. We should
  // probably move them to a different place, since these are not really part
  // of the workflow definition but will be there also at runtime.
  std::unique_ptr<ServiceRegistry> configRegistry{nullptr};
  std::unique_ptr<ConfigParamRegistry> workflowOptionsRegistry{nullptr};

  o2::framework::WorkflowSpec specs;
};

struct WorkflowDefinition {
  std::function<o2::framework::WorkflowDefinitionContext(int argc, char** argv)> defineWorkflow;
};

struct WorkflowPlugin {
  virtual o2::framework::WorkflowDefinition* create() = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_WORKFLOWDEFINITIONCONTEXT_H_
