// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "proxy-name", VariantType::String, "dpl-output-proxy", {"name of the proxy processor, will be the default output channel name as well"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "dataspec", VariantType::String, "dpl-output-proxy:TST/CLUSTERS;dpl-output-proxy:TST/TRACKS", {"selection string for the data to be proxied"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "output-proxy-method", VariantType::String, "bind", {"proxy socket method: bind, connect"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "output-proxy-address", VariantType::String, "0.0.0.0", {"address to connect / bind to"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "default-transport", VariantType::String, "shmem", {"default transport: shmem, zeromq"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "default-port", VariantType::Int, 4200, {"default port number"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  std::string processorName = config.options().get<std::string>("proxy-name");
  std::string inputConfig = config.options().get<std::string>("dataspec");
  int defaultPort = config.options().get<int>("default-port");
  std::string defaultTransportConfig = config.options().get<std::string>("default-transport");
  if (defaultTransportConfig == "zeromq") {
    // nothing to do for the moment
  } else if (defaultTransportConfig == "shmem") {
    // nothing to do for the moment
  } else {
    throw std::runtime_error("invalid argument for option --default-transport : '" + defaultTransportConfig + "'");
  }

  std::vector<InputSpec> inputs = select(inputConfig.c_str());
  if (inputs.size() == 0) {
    throw std::runtime_error("invalid dataspec '" + inputConfig + "'");
  }

  // we build the default channel configuration from the binding of the first input
  // in order to have more than one we would need to possibility to have support for
  // vectored options
  // use the OutputChannelSpec as a tool to create the default configuration for the out-of-band channel
  OutputChannelSpec externalChannelSpec;
  externalChannelSpec.name = "downstream";
  externalChannelSpec.type = ChannelType::Push;
  if (config.options().get<std::string>("output-proxy-method") == "bind") {
    externalChannelSpec.method = ChannelMethod::Bind;
  } else if (config.options().get<std::string>("output-proxy-method") == "connect") {
    externalChannelSpec.method = ChannelMethod::Connect;
  }
  externalChannelSpec.hostname = config.options().get<std::string>("output-proxy-address");
  externalChannelSpec.port = defaultPort;
  externalChannelSpec.listeners = 0;
  // in principle, protocol and transport are two different things but fur simplicity
  // we use ipc when shared memory is selected and the normal tcp url whith zeromq,
  // this is for building the default configuration which can be simply changed from the
  // command line
  if (!defaultTransportConfig.empty()) {
    if (defaultTransportConfig == "zeromq") {
      externalChannelSpec.protocol = ChannelProtocol::Network;
    } else if (defaultTransportConfig == "shmem") {
      externalChannelSpec.protocol = ChannelProtocol::IPC;
    }
  }
  std::string defaultChannelConfig = formatExternalChannelConfiguration(externalChannelSpec);
  // at some point the formatting tool might add the transport as well so we have to check
  if (!defaultTransportConfig.empty() && defaultTransportConfig.find("transport=") == std::string::npos) {
    defaultChannelConfig += ",transport=" + defaultTransportConfig;
  }

  std::vector<DataProcessorSpec> workflow;
  workflow.emplace_back(std::move(specifyFairMQDeviceOutputProxy(processorName.c_str(), inputs, defaultChannelConfig.c_str())));
  return workflow;
}
