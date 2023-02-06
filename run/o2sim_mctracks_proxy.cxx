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

#include <boost/program_options.hpp>

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/Task.h"
#include "Framework/DataRef.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "Framework/DataProcessingHeader.h"
#include <CommonUtils/FileSystemUtils.h>
#include <unistd.h>

using namespace o2::framework;
using namespace o2::header;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"enable-test-consumer", o2::framework::VariantType::Bool, false, {"enable a simple test consumer for injected MC tracks"}});
  workflowOptions.push_back(ConfigParamSpec{"o2sim-pid", o2::framework::VariantType::Int, -1, {"The process id of the source o2-sim"}});
}

#include "Framework/runDataProcessing.h"

// a simple (test) consumer task for MCTracks and MCEventHeaders injected from
// the proxy
class ConsumerTask
{
 public:
  void init(o2::framework::InitContext& ic) {}
  void run(o2::framework::ProcessingContext& pc)
  {
    LOG(debug) << "Running simple kinematics consumer client";
    for (const DataRef& ref : InputRecordWalker(pc.inputs())) {
      auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      LOG(debug) << "Payload size " << dh->payloadSize << " method " << dh->payloadSerializationMethod.as<std::string>();
    }
    auto tracks = pc.inputs().get<std::vector<o2::MCTrack>>("mctracks");
    auto eventheader = pc.inputs().get<o2::dataformats::MCEventHeader*>("mcheader");
    LOG(info) << "Got " << tracks.size() << " tracks";
    LOG(info) << "Got " << eventheader->GetB() << " as impact parameter in the event header";
  }
};

/// Function converting raw input data to DPL data format. Uses knowledge of how MCTracks and MCEventHeaders
/// are sent from the o2sim side.
InjectorFunction o2simKinematicsConverter(std::vector<OutputSpec> const& specs, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);

  return [timesliceId, specs, step](TimingInfo&, fair::mq::Device& device, fair::mq::Parts& parts, ChannelRetriever channelRetriever) {
    // We iterate on all the parts and we send them two by two,
    // adding the appropriate O2 header.
    for (int i = 0; i < parts.Size(); ++i) {
      DataHeader dh;
      ConcreteDataMatcher matcher = DataSpecUtils::asConcreteDataMatcher(specs[i]);
      dh.dataOrigin = matcher.origin;
      dh.dataDescription = matcher.description;
      dh.subSpecification = matcher.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();
      if (i == 0) {
        dh.payloadSerializationMethod = gSerializationMethodROOT;
      } else if (i == 1) {
        dh.payloadSerializationMethod = gSerializationMethodROOT;
      }
      DataProcessingHeader dph{*timesliceId, 0};
      // we have to move the incoming data
      o2::header::Stack headerStack{dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), specs[i], channelRetriever);
    }
    *timesliceId += step;
  };
}

/// Describe the DPL workflow
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  // make a proxy (connecting to an external channel) and forwarding in DPL speak
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MC", "MCHEADER", 0, Lifetime::Timeframe);
  outputs.emplace_back("MC", "MCTRACKS", 0, Lifetime::Timeframe);
  o2::framework::InjectorFunction f = o2simKinematicsConverter(outputs, 0, 1);

  // construct the input channel to listen on
  // use given pid
  // TODO: this could go away with a proper pipeline implementation
  std::string channelspec;
  std::string channelbase = "type=sub,method=connect,address=ipc://";
  if (configcontext.options().get<int>("o2sim-pid") != -1) {
    std::stringstream channelstr;
    channelstr << channelbase << "/tmp/o2sim-hitmerger-kineforward-" << configcontext.options().get<int>("o2sim-pid") << ",rateLogging=100";
    channelspec = channelstr.str();
  } else {
    // we try to detect an existing channel by name ... as long as it's unique ... else we fail
    sleep(2); // give time for sim to startup
    LOG(info) << "Looking for simulation MC-tracks socket";
    auto socketlist = o2::utils::listFiles("/tmp", "o2sim-hitmerger-kineforward-.*");
    if (socketlist.size() != 1) {
      for (auto s : socketlist) {
        LOG(info) << s;
      }
      LOG(fatal) << "Too many or no socket found " << socketlist.size() << "; Please pass sim pid via --o2sim-pid";
    }
    LOG(info) << "Found socket " << socketlist[0];
    channelspec = channelbase + socketlist[0] + ",rateLogging=100";
  }

  specs.emplace_back(specifyExternalFairMQDeviceProxy("o2sim-mctrack-proxy",
                                                      outputs,
                                                      channelspec.c_str(), f));

  if (configcontext.options().get<bool>("enable-test-consumer")) {
    // connect a test consumer
    std::vector<InputSpec> inputs;
    inputs.emplace_back("mctracks", "MC", "MCTRACKS", 0., Lifetime::Timeframe);
    inputs.emplace_back("mcheader", "MC", "MCHEADER", 0., Lifetime::Timeframe);
    specs.emplace_back(DataProcessorSpec{"sample-MCTrack-consumer",
                                         inputs,
                                         {},
                                         AlgorithmSpec{adaptFromTask<ConsumerTask>()},
                                         {}});
  }

  return specs;
}
