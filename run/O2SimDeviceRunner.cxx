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
#include "SimPublishChannelHelper.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
namespace bpo = boost::program_options;

std::vector<int> gChildProcesses; // global vector of child pids
int gMasterProcess = -1;
int gDriverProcess = -1;

void sigaction_handler(int signal, siginfo_t* signal_info, void*)
{
  auto pid = getpid();
  LOG(INFO) << pid << " caught signal " << signal << " from source " << signal_info->si_pid;
  auto groupid = getpgrp();
  if (pid == gMasterProcess) {
    killpg(pid, signal); // master kills whole process group
  } else {
    if (signal_info->si_pid != gDriverProcess) {
      // forward to master if coming internally
      kill(groupid, signal);
    }
  }
  if (signal_info->si_pid == gDriverProcess) {
    _exit(0); // external requests are not treated as error
  }
  if (signal == SIGTERM) {
    // normal termination is not error
    _exit(0);
  }
  // we treat internal signal interruption as an error
  // because only ordinary termination is good in the context of the distributed system
  _exit(128 + signal);
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
  auto channel = FairMQChannel{"o2sim-primserv-info", "req", factory};
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

struct KernelSetup {
  o2::devices::O2SimDevice* sim = nullptr;
  FairMQChannel* primchannel = nullptr;
  FairMQChannel* datachannel = nullptr;
  FairMQChannel* primstatuschannel = nullptr;
  int workerID = -1;
};

KernelSetup initSim(std::string transport, std::string primaddress, std::string primstatusaddress, std::string mergeraddress, int workerID)
{
  auto factory = FairMQTransportFactory::CreateTransportFactory(transport);
  auto primchannel = new FairMQChannel{"primary-get", "req", factory};
  primchannel->Connect(primaddress);
  primchannel->Validate();

  auto prim_status_channel = new FairMQChannel{"o2sim-primserv-info", "req", factory};
  prim_status_channel->Connect(primstatusaddress);
  prim_status_channel->Validate();

  auto datachannel = new FairMQChannel{"simdata", "push", factory};
  datachannel->Connect(mergeraddress);
  datachannel->Validate();
  // the channels are setup

  // init the sim object
  auto sim = getDevice();
  sim->lateInit();

  return KernelSetup{sim, primchannel, datachannel, prim_status_channel, workerID};
}

int runSim(KernelSetup setup)
{
  // the simplified runloop
  while (setup.sim->Kernel(setup.workerID, *setup.primchannel, *setup.datachannel, setup.primstatuschannel)) {
  }
  LOG(INFO) << "[W" << setup.workerID << "] simulation is done";
  return 0;
}

void pinToCPU(unsigned int cpuid)
{
  auto affinity = getenv("ALICE_CPUAFFINITY");
  if (affinity) {
    // MacOS does not support this API so we add a protection
#ifndef __APPLE__

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
#else
    LOG(WARN) << "CPU AFFINITY NOT IMPLEMENTED ON APPLE";
#endif
  }
}

bool waitForControlInput()
{
  auto factory = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto channel = FairMQChannel{"o2sim-control", "sub", factory};
  auto controlsocketname = getenv("ALICE_O2SIMCONTROL");
  LOG(DEBUG) << "AWAITING CONTROL ON SOCKETNAME " << controlsocketname;
  channel.Connect(std::string(controlsocketname));
  channel.Validate();
  std::unique_ptr<FairMQMessage> reply(channel.NewMessage());

  LOG(DEBUG) << "WAITING FOR INPUT";
  if (channel.Receive(reply) > 0) {
    auto data = reply->GetData();
    auto size = reply->GetSize();

    std::string command(reinterpret_cast<char const*>(data), size);
    LOG(INFO) << "message: " << command;

    o2::conf::SimReconfigData reconfig;
    o2::conf::parseSimReconfigFromString(command, reconfig);
    if (reconfig.stop) {
      LOG(INFO) << "Stop asked, shutting down";
      return false;
    }
    LOG(INFO) << "Processing " << reconfig.nEvents << " new events";
  } else {
    LOG(INFO) << "NOTHING RECEIVED";
  }
  return true;
}

int main(int argc, char* argv[])
{
  struct sigaction act;
  memset(&act, 0, sizeof act);
  sigemptyset(&act.sa_mask);
  act.sa_sigaction = &sigaction_handler;
  act.sa_flags = SA_SIGINFO; // <--- enable sigaction

  std::vector<int> handledsignals = {SIGTERM, SIGINT, SIGQUIT, SIGSEGV, SIGBUS, SIGFPE}; // <--- may need to be completed
  // remember that SIGKILL can't be handled
  for (auto s : handledsignals) {
    if (sigaction(s, &act, nullptr)) {
      LOG(ERROR) << "Could not install signal handler for " << s;
      exit(EXIT_FAILURE);
    }
  }

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
    int driverPID = getppid();
    auto pubchannel = o2::simpubsub::createPUBChannel(o2::simpubsub::getPublishAddress("o2sim-worker-notifications", driverPID));

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
    std::string serverstatus_address;
    std::string s;

    auto& options = d["fairMQOptions"];
    assert(options.IsObject());
    for (auto option = options.MemberBegin(); option != options.MemberEnd(); ++option) {
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
            sockets = (channels[1])["sockets"].GetArray();
            address = (sockets[0])["address"].GetString();
            serverstatus_address = address;
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

    LOG(INFO) << "Parsed primary server address " << serveraddress;
    LOG(INFO) << "Parsed primary server status address " << serverstatus_address;
    LOG(INFO) << "Parsed merger address " << mergeraddress;
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
    if (!initializeSim("zeromq", serverstatus_address, simrun)) {
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
    gDriverProcess = getppid();
    // then we fork and create a device in each fork
    for (auto i = 0u; i < nworkers; ++i) {
      // we use the current process as one of the workers as it has nothing else to do
      auto pid = (i == nworkers - 1) ? 0 : fork();
      if (pid == 0) {
        // Each worker can publish its progress/state on a ZMQ channel.
        // We actually use a push/pull mechanism to collect all messages in the
        // master worker which can then publish using PUB/SUB.
        // auto factory = FairMQTransportFactory::CreateTransportFactory("zeromq");
        auto collectAndPubThreadFunction = [driverPID, &pubchannel]() {
          auto collectorchannel = o2::simpubsub::createPUBChannel(o2::simpubsub::getPublishAddress("o2sim-workerinternal", driverPID), "pull");
          std::unique_ptr<FairMQMessage> msg(collectorchannel.NewMessage());

          while (true) {
            if (collectorchannel.Receive(msg) > 0) {
              auto data = msg->GetData();
              auto size = msg->GetSize();
              std::string text(reinterpret_cast<char const*>(data), size);
              // LOG(INFO) << "Collector message: " << text;
              o2::simpubsub::publishMessage(pubchannel, text);
            }
          }
        };
        if (i == nworkers - 1) { // <---- extremely important to take non-forked version since ZMQ sockets do not behave well on fork
          std::vector<std::thread> threads;
          threads.push_back(std::thread(collectAndPubThreadFunction));
          threads.back().detach();
        }

        // everyone else is getting a push socket for notifications
        auto pushchannel = o2::simpubsub::createPUBChannel(o2::simpubsub::getPublishAddress("o2sim-workerinternal", driverPID), "push");

        // we will try to pin each worker to a particular CPU
        // this can be made configurable via environment variables??
        pinToCPU(i);

        auto kernelSetup = initSim("zeromq", serveraddress, serverstatus_address, mergeraddress, i);

        std::stringstream worker;
        worker << "WORKER" << i;
        o2::simpubsub::publishMessage(pushchannel, o2::simpubsub::simStatusString(worker.str(), "STATUS", "SETUP COMPLETED"));

        auto& conf = o2::conf::SimConfig::Instance();

        bool more = true;
        while (more) {
          runSim(kernelSetup);

          if (conf.asService()) {
            LOG(INFO) << "IN SERVICE MODE WAITING";
            o2::simpubsub::publishMessage(pushchannel, o2::simpubsub::simStatusString(worker.str(), "STATUS", "AWAITING INPUT"));
            more = waitForControlInput();
            usleep(100); // --> why?
          } else {
            o2::simpubsub::publishMessage(pushchannel, o2::simpubsub::simStatusString(worker.str(), "STATUS", "TERMINATING"));

            LOG(INFO) << "FINISHING";
            more = false;
          }
        }
        sleep(10); // ---> give some time for message to be delivered to merger (destructing too early might affect the ZQM buffers)
                   // The process will in any case be terminated by the main o2-sim driver.

        // destruct setup (using _exit due to problems in ROOT shutdown (segmentation violations)
        // Clearly at some moment, a more robust solution would be appreciated
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
