// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RUN_DATA_PROCESSING_H
#define FRAMEWORK_RUN_DATA_PROCESSING_H

#include <unistd.h>
#include <vector>
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace framework
{
using WorkflowSpec = std::vector<DataProcessorSpec>;
using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

} // namespace framework
} // namespace o2

/// To be implemented by the user to specify one or more DataProcessorSpec.
/// The reason why this passes a preallocated specs, rather than asking the
/// caller to allocate his / her own is that if we end up wrapping this in
/// some scripting language, we do not need to delegate the allocation to the
/// scripting language itself.
void defineDataProcessing(o2::framework::WorkflowSpec& specs);

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
// FIXME: add a debug statement saying that the default policy was used?
void defaultConfiguration(std::vector<o2::framework::ChannelConfigurationPolicy>& channelPolicies) {}

struct UserCustomizationsHelper {
  template <typename T>
  static auto userDefinedCustomization(T& something, int preferUser) -> decltype(customize(something), void())
  {
    customize(something);
  }

  template <typename T>
  static auto userDefinedCustomization(T& something, long preferUser)
    -> decltype(defaultConfiguration(something), void())
  {
    defaultConfiguration(something);
  }
};

// This comes from the framework itself. This way we avoid code duplication.
int doMain(int argc, char** argv, o2::framework::WorkflowSpec const& specs,
           std::vector<o2::framework::ChannelConfigurationPolicy> const& channelPolicies);

int main(int argc, char** argv)
{
  using namespace o2::framework;

  WorkflowSpec specs;
  defineDataProcessing(specs);

  // The default policy is a catch all pub/sub setup to be consistent with the past.
  std::vector<ChannelConfigurationPolicy> channelPolicies;
  auto defaultPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  channelPolicies.insert(std::end(channelPolicies), std::begin(defaultPolicies), std::end(defaultPolicies));

  // The 0 here is an int, therefore having the template matching in the
  // SFINAE expression above fit better the version which invokes user code over
  // the default one.
  UserCustomizationsHelper::userDefinedCustomization(channelPolicies, 0);
  auto result = doMain(argc, argv, specs, channelPolicies);
  LOG(INFO) << "Process " << getpid() << " is exiting.";
  return result;
}

#endif
