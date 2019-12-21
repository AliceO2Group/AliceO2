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
#include <pthread.h> // to set cpu affinity
#include <cmath>
#include <csignal>
#include <unistd.h>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
namespace bpo = boost::program_options;

std::vector<int> gChildProcesses; // global vector of child pids
int gMasterProcess = -1;

// custom signal handler to ensure that we
// distribute signals to all detached forks
void sighandler(int signal)
{
  if (signal == SIGTERM || signal == SIGINT) {
    auto pid = getpid();
    if (pid == gMasterProcess) {
      // the master
      for (auto child : gChildProcesses) {
        kill(child, signal);
      }
    }
    _exit(0);
  }
}

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
  auto channel = FairMQChannel{"primary-get", "req", factory};
  channel.Connect(address);
  channel.Validate();

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
    fair::mq::DeviceRunner runner{argc, argv};

    runner.AddHook<SetCustomCmdLineOptions>([](DeviceRunner& r) {
      boost::program_options::options_description customOptions("Custom options");
      addCustomOptions(customOptions);
      r.fConfig.AddToCmdLineOptions(customOptions);
    });

    runner.AddHook<InstantiateDevice>([](DeviceRunner& r) {
      r.fDevice = std::unique_ptr<FairMQDevice>{getDevice(r.fConfig)};
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
  auto primchannel = FairMQChannel{"primary-get", "req", factory};
  primchannel.Connect(primaddress);
  primchannel.Validate();

  auto datachannel = FairMQChannel{"simdata", "push", factory};
  datachannel.Connect(mergeraddress);
  datachannel.Validate();
  // the channels are setup

  // init the sim object
  auto sim = getDevice();
  sim->lateInit();

  // the simplified runloop
  while (sim->Kernel(primchannel, datachannel)) {
  }
  LOG(INFO) << "simulation is done";
  return 0;
}

void pinToCPU(unsigned int cpuid)
{
// MacOS does not support this API so we add a protection
#ifndef __APPLE__
  auto affinity = getenv("ALICE_CPUAFFINITY");
  if (affinity) {
    pthread_t thread;

    thread = pthread_self();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);

    auto s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      LOG(WARNING) << "FAILED TO SET PTHREAD AFFINITY";
    }

    /* Check the actual affinity mask assigned to the thread */
    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      LOG(WARNING) << "FAILED TO GET PTHREAD AFFINITY";
    }

    for (int j = 0; j < CPU_SETSIZE; j++) {
      if (CPU_ISSET(j, &cpuset)) {
        LOG(INFO) << "ENABLED CPU " << j;
      }
    }
  }
#else
  LOG(WARN) << "CPU AFFINITY NOT IMPLEMENTED ON APPLE";
#endif
}

int main(int argc, char* argv[])
{
  // enable signal handler for termination signals
  if (signal(SIGTERM, sighandler) == SIG_IGN) {
    signal(SIGTERM, SIG_IGN);
  }
  if (signal(SIGINT, sighandler) == SIG_IGN) {
    signal(SIGINT, SIG_IGN);
  }
  signal(SIGCHLD, SIG_IGN); /* Silently reap children to avoid zombies. */

  // extract the path to FairMQ config
  bpo::options_description desc{"Options"};
  // clang-format off
  desc.add_options()
      ("control","control type")
      ("id","ID")
      ("config-key","config key")
      ("mq-config",bpo::value<std::string>(),"path to FairMQ config")
      ("severity","log severity");
  // clang-format on
  bpo::variables_map vm;
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  std::string FMQconfig;
  if (vm.count("mq-config")) {
    FMQconfig = vm["mq-config"].as<std::string>();
  }

  auto internalfork = getenv("ALICE_SIMFORKINTERNAL");
  if (internalfork) {
    if (FMQconfig.empty()) {
      throw std::runtime_error("This should never be called without FairMQ config.");
    }
    // read the JSON config
    FILE* fp = fopen(FMQconfig.c_str(), "r");
    constexpr unsigned short usmax = std::numeric_limits<unsigned short>::max() - 1;
    char readBuffer[usmax];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    // retrieve correct server and merger URLs

    std::string serveraddress;
    std::string mergeraddress;
    std::string s;

    auto& options = d["fairMQOptions"];
    assert(options.IsObject());
    for (auto option = options.MemberBegin(); option != options.MemberEnd();
         ++option) {
      s = option->name.GetString();
      if (s == "devices") {
        assert(option->value.IsArray());
        auto devices = option->value.GetArray();
        for (auto& device : devices) {
          s = device["id"].GetString();
          if (s == "primary-server") {
            auto channels = device["channels"].GetArray();
            auto sockets = (channels[0])["sockets"].GetArray();
            auto address = (sockets[0])["address"].GetString();
            serveraddress = address;
          }
          if (s == "hitmerger") {
            auto channels = device["channels"].GetArray();
            for (auto& channel : channels) {
              s = channel["name"].GetString();
              if (s == "simdata") {
                auto sockets = channel["sockets"].GetArray();
                auto address = (sockets[0])["address"].GetString();
                mergeraddress = address;
              }
            }
          }
        }
      }
    }

    LOG(INFO) << serveraddress << "\n";
    LOG(INFO) << mergeraddress << "\n";
    if (serveraddress.empty() || mergeraddress.empty()) {
      throw std::runtime_error("Could not determine server or merger URLs.");
    }
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

    // should be factored out?
    unsigned int nworkers = std::max(1u, std::thread::hardware_concurrency() / 2);
    auto f = getenv("ALICE_NSIMWORKERS");
    if (f) {
      nworkers = static_cast<unsigned int>(std::stoi(f));
    }
    LOG(INFO) << "Running with " << nworkers << " sim workers ";

    gMasterProcess = getpid();
    // then we fork and create a device in each fork
    for (auto i = 0u; i < nworkers; ++i) {
      // we use the current process as one of the workers as it has nothing else to do
      auto pid = (i == nworkers - 1) ? 0 : fork();
      if (pid == 0) {
        // we will try to pin each worker to a particular CPU
        // this can be made configurable via enviroment variables??
        pinToCPU(i);

        runSim("zeromq", serveraddress, mergeraddress);

        _exit(0);
      } else {
        gChildProcesses.push_back(pid);
      }
    }
    int status;
    wait(&status); /* only the parent waits */
    _exit(0);
  } else {
    // This the solution where we setup an ordinary FairMQDevice
    // (each if which will setup its own simulation). Parallelism
    // is achieved outside by instantiating multiple device processes.
    _exit(initAndRunDevice(argc, argv));
  }
}
