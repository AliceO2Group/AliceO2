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

#include "../Framework/Core/src/ArrowSupport.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
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
  workflowOptions.push_back(ConfigParamSpec{"nevents", o2::framework::VariantType::Int, -1, {"The number of events expected to arrive on the proxy"}});
  workflowOptions.push_back(ConfigParamSpec{"aggregate-timeframe", o2::framework::VariantType::Int, -1, {"The number of events to aggregate per timeframe"}});
}

#include "Framework/runDataProcessing.h"

// a simple (test) consumer task for MCTracks and MCEventHeaders injected from
// the proxy
class ConsumerTask
{
 public:
  void init(o2::framework::InitContext& /*ic*/) {}
  void run(o2::framework::ProcessingContext& pc)
  {
    LOG(debug) << "Running simple kinematics consumer client";
    for (const DataRef& ref : InputRecordWalker(pc.inputs())) {
      auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      LOG(debug) << "Payload size " << dh->payloadSize << " method " << dh->payloadSerializationMethod.as<std::string>();
    }
    try {
      auto tracks = pc.inputs().get<std::vector<o2::MCTrack>>("mctracks");
      auto eventheader = pc.inputs().get<o2::dataformats::MCEventHeader*>("mcheader");
      LOG(info) << "Got " << tracks.size() << " tracks";
      LOG(info) << "Got " << eventheader->GetB() << " as impact parameter in the event header";
    } catch (...) {
    }
  }
};

static DataHeader headerFromSpec(OutputSpec const& spec, size_t size, o2::header::SerializationMethod method, int splitParts = 0, int partIndex = 0)
{
  DataHeader dh;
  ConcreteDataMatcher matcher = DataSpecUtils::asConcreteDataMatcher(spec);
  dh.dataOrigin = matcher.origin;
  dh.dataDescription = matcher.description;
  dh.subSpecification = matcher.subSpec;
  dh.payloadSize = size;
  dh.payloadSerializationMethod = method;
  if (splitParts > 0) {
    dh.splitPayloadParts = splitParts;
    dh.splitPayloadIndex = partIndex;
  }
  return dh;
}

/// Function converting raw input data to DPL data format. Uses knowledge of how MCTracks and MCEventHeaders
/// are sent from the o2sim side.
/// If aggregate-timeframe is set to non-negative value N, this number of events is accumulated and then sent
/// as a multipart message, which is useful for AOD creation
InjectorFunction o2simKinematicsConverter(std::vector<OutputSpec> const& specs, uint64_t startTime, uint64_t step, int nevents, int nPerTF)
{
  auto timesliceId = std::make_shared<size_t>(startTime);
  auto totalEventCounter = std::make_shared<int>(0);
  auto eventCounter = std::make_shared<int>(0);
  auto TFcounter = std::make_shared<size_t>(startTime);
  auto MCHeadersMessageCache = std::make_shared<fair::mq::Parts>();
  auto MCTracksMessageCache = std::make_shared<fair::mq::Parts>();
  auto Nparts = std::make_shared<int>(nPerTF);

  return [timesliceId, specs, step, nevents, nPerTF, totalEventCounter, eventCounter, TFcounter, Nparts, MCHeadersMessageCache = MCHeadersMessageCache, MCTracksMessageCache = MCTracksMessageCache](TimingInfo& ti, fair::mq::Device& device, fair::mq::Parts& parts, ChannelRetriever channelRetriever, size_t newTimesliceId, bool& stop) mutable {
    if (nPerTF < 0) {
      // if no aggregation requested, forward each message with the DPL header
      if (*timesliceId != newTimesliceId) {
        LOG(fatal) << "Time slice ID provided from oldestPossible mechanism " << newTimesliceId << " is out of sync with expected value " << *timesliceId;
      }
      // We iterate on all the parts and we send them two by two,
      // adding the appropriate O2 header.
      for (auto i = 0U; i < parts.Size(); ++i) {
        DataHeader dh = headerFromSpec(specs[i], parts.At(i)->GetSize(), gSerializationMethodROOT);
        DataProcessingHeader dph{newTimesliceId, 0};
        // we have to move the incoming data
        o2::header::Stack headerStack{dh, dph};
        sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), specs[i], channelRetriever);
      }
      *timesliceId += step;
    } else {
      if (*eventCounter == 0) {
        *Nparts = ((nevents - *totalEventCounter) < nPerTF) ? nevents - *totalEventCounter : nPerTF;
      }
      // if aggregation is requested, colelct the payloads into a multipart message
      ti.timeslice = *TFcounter;
      ti.tfCounter = *TFcounter;

      auto headerSize = parts.At(0)->GetSize();
      auto tracksSize = parts.At(1)->GetSize();

      DataProcessingHeader hdph{*TFcounter, 0};
      DataHeader headerDH = headerFromSpec(specs[0], headerSize, gSerializationMethodROOT, *Nparts, *eventCounter);
      o2::header::Stack hhs{headerDH, hdph};

      DataProcessingHeader tdph{*TFcounter, 0};
      DataHeader tracksDH = headerFromSpec(specs[1], tracksSize, gSerializationMethodROOT, *Nparts, *eventCounter);
      o2::header::Stack ths{tracksDH, tdph};

      appendForSending(device, std::move(hhs), *TFcounter, std::move(parts.At(0)), specs[0], *MCHeadersMessageCache.get(), channelRetriever);
      appendForSending(device, std::move(ths), *TFcounter, std::move(parts.At(1)), specs[1], *MCTracksMessageCache.get(), channelRetriever);
      ++(*eventCounter);
    }

    ++(*totalEventCounter);
    if (nPerTF > 0 && *eventCounter == nPerTF) {
      // if aggregation is requested, only send the accumulated vectors
      LOGP(info, ">> Events: {}; TF counter: {}", *eventCounter, *TFcounter);
      *eventCounter = 0;
      ++(*TFcounter);
      sendOnChannel(device, *MCHeadersMessageCache.get(), channelRetriever(specs[0], *TFcounter), *TFcounter);
      sendOnChannel(device, *MCTracksMessageCache.get(), channelRetriever(specs[1], *TFcounter), *TFcounter);
      MCHeadersMessageCache->Clear();
      MCTracksMessageCache->Clear();
    }

    if (*totalEventCounter == nevents) {
      if (nPerTF > 0) {
        // send accumulated messages if the limit is reached
        ++(*TFcounter);
        sendOnChannel(device, *MCHeadersMessageCache.get(), channelRetriever(specs[0], *TFcounter), *TFcounter);
        sendOnChannel(device, *MCTracksMessageCache.get(), channelRetriever(specs[1], *TFcounter), *TFcounter);
        MCHeadersMessageCache->Clear();
        MCTracksMessageCache->Clear();
      }
      // I am done (I don't expect more events to convert); so tell the proxy device to shut-down
      stop = true;
    }
    return;
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

  // fetch the number of events to expect
  auto nevents = configcontext.options().get<int>("nevents");
  auto nEventsPerTF = configcontext.options().get<int>("aggregate-timeframe");
  o2::framework::InjectorFunction f = o2simKinematicsConverter(outputs, 0, 1, nevents, nEventsPerTF);

  // construct the input channel to listen on
  // use given pid
  // TODO: this could go away with a proper pipeline implementation
  std::string channelspec;
  std::string channelbase = "type=pair,method=connect,address=ipc://";
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

  auto proxy = specifyExternalFairMQDeviceProxy("o2sim-mctrack-proxy",
                                                outputs,
                                                channelspec.c_str(), f, 0, true);
  // add monitoring service to be able to report number of timeframes sent for the rate limiting to work
  proxy.requiredServices.push_back(o2::framework::ArrowSupport::arrowBackendSpec());
  // if aggregation is requested, set the enumeration repetitions to aggregation size
  if (nEventsPerTF > 0) {
    proxy.inputs.emplace_back(InputSpec{"clock", "enum", "DPL", 0, Lifetime::Enumeration, {ConfigParamSpec{"repetitions", VariantType::Int64, static_cast<int64_t>(nEventsPerTF), {"merged events"}}}});
  }
  specs.push_back(proxy);

  if (configcontext.options().get<bool>("enable-test-consumer") && (nEventsPerTF < 0)) {
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
