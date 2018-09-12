// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Sandro Wenzel

#include "O2SimDevice.h"
#include <fairmq/DeviceRunner.h>
#include <boost/program_options.hpp>
#include <memory>
#include <string>
#include <FairMQChannel.h>
#include <FairMQLogger.h>
#include <FairMQParts.h>
#include <FairMQTransportFactory.h>
#include <TStopwatch.h>
#include <sys/wait.h>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
}

void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }

// this will initialize the simulation setup
// once before initializing the actual FairMQ device
bool initializeSim(std::string transport, std::string address, std::unique_ptr<FairRunSim>& simptr)
{
  // This needs an already running PrimaryServer
  auto factory = FairMQTransportFactory::CreateTransportFactory(transport);
  auto channel = FairMQChannel{ "primary-get", "req", factory };
  channel.Connect(address);
  channel.ValidateChannel();

  return o2::devices::O2SimDevice::initSim(channel, simptr);
}

o2::devices::O2SimDevice* getDevice()
{
  auto app = static_cast<o2::steer::O2MCApplication*>(TVirtualMCApplication::Instance());
  auto vmc = TVirtualMC::GetMC();

  if (app == nullptr) {
    LOG(WARNING) << "no vmc application found at this stage";
  }
  return new o2::devices::O2SimDevice(app, vmc);
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return getDevice();
}

int initAndRunDevice(int argc, char* argv[])
{
  using namespace fair::mq;
  using namespace fair::mq::hooks;

  try {
    fair::mq::DeviceRunner runner{ argc, argv };

    runner.AddHook<SetCustomCmdLineOptions>([](DeviceRunner& r) {
      boost::program_options::options_description customOptions("Custom options");
      addCustomOptions(customOptions);
      r.fConfig.AddToCmdLineOptions(customOptions);
    });

    runner.AddHook<InstantiateDevice>([](DeviceRunner& r) {
      r.fDevice = std::move(decltype(r.fDevice){ getDevice(r.fConfig) });
    });

    return runner.Run();
  } catch (std::exception& e) {
    LOG(error) << "Unhandled exception reached the top of main: " << e.what()
               << ", application will now exit";
    return 1;
  } catch (...) {
    LOG(error) << "Non-exception instance being thrown. Please make sure you use std::runtime_exception() instead. "
               << "Application will now exit.";
    return 1;
  }
}

int runSim(std::string transport, std::string primaddress, std::string mergeraddress)
{
  auto factory = FairMQTransportFactory::CreateTransportFactory(transport);
  auto primchannel = FairMQChannel{ "primary-get", "req", factory };
  primchannel.Connect(primaddress);
  primchannel.ValidateChannel();

  auto datachannel = FairMQChannel{ "simdata", "push", factory };
  datachannel.Connect(mergeraddress);
  datachannel.ValidateChannel();

  // the channels are setup

  // init the sim object
  auto sim = getDevice();

  // the simplified runloop
  while (sim->Kernel(primchannel, datachannel)) {
  }
  LOG(INFO) << "simulation is done";
  return 0;
}

int main(int argc, char* argv[])
{
  auto internalfork = getenv("ALICE_SIMFORKINTERNAL");
  if (internalfork) {

    std::string serveraddress("tcp://localhost:25005");
    std::string mergeraddress("tcp://localhost:25009");
    auto host = getenv("ALICE_SIMMAINHOST");
    if (host) {
      // argv[1] is supposed to be an IP address or hostname
      serveraddress = "tcp://" + std::string(host) + ":25005";
      mergeraddress = "tcp://" + std::string(host) + ":25009";
    }
    LOG(INFO) << serveraddress;
    LOG(INFO) << mergeraddress;

    // This is a solution based on initializing the simulation once
    // and then fork the process to share the simulation memory across
    // many processes. Here we are not using FairMQDevices and just setup
    // some channels manually and do our own runloop.

    // we init the simulation first
    std::unique_ptr<FairRunSim> simrun;
    // TODO: take the addresses from somewhere else
    if (!initializeSim("zeromq", serveraddress, simrun)) {
      LOG(ERROR) << "Could not initialize simulation";
      return 1;
    }

    int nworkers = std::thread::hardware_concurrency() / 2;
    auto f = getenv("ALICE_NSIMWORKERS");
    if (f) {
      nworkers = atoi(f);
    }
    LOG(INFO) << "Running with " << nworkers << " sim workers "
              << "(customize using the ALICE_NSIMWORKERS environment variable)\n";

    // then we fork and create a device in each fork
    for (int i = 0; i < nworkers; ++i) {
      auto pid = fork();
      if (pid == 0) {
        runSim("zeromq", serveraddress, mergeraddress);
        return 0;
      }
    }
    int status;
    wait(&status); /* only the parent waits */
    return 0;
  } else {
    // This the solution where we setup an ordinary FairMQDevice
    // (each if which will setup its own simulation). Parallelism
    // is achieved outside by instantiating multiple device processes.
    return initAndRunDevice(argc, argv);
  }
}
