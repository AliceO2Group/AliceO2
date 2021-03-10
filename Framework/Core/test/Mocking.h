// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "test_HelperMacros.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/SimpleOptionsRetriever.h"
#include "Framework/WorkflowCustomizationHelpers.h"

std::unique_ptr<o2::framework::ConfigContext> makeEmptyConfigContext()
{
  using namespace o2::framework;
  // FIXME: Ugly... We need to fix ownership and make sure the ConfigContext
  //        either owns or shares ownership of the registry.
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  static std::vector<ConfigParamSpec> specs = WorkflowCustomizationHelpers::requiredWorkflowOptions();
  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  static ConfigParamRegistry registry(std::move(store));
  auto context = std::make_unique<ConfigContext>(registry, 0, nullptr);
  return context;
}
