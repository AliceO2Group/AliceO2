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

/// @author Sandro Wenzel

#include <fairmq/TransportFactory.h>
#include <fairmq/Channel.h>
#include <fairmq/Message.h>

#include <cstdlib>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <SimConfig/SimConfig.h>
#include <sys/wait.h>
#include <vector>
#include <functional>
#include <thread>
#include <csignal>
#include "TStopwatch.h"
#include <fairlogger/Logger.h>
#include "CommonUtils/ShmManager.h"
#include "TFile.h"
#include "TTree.h"
#include <sys/types.h>
#include "CommonUtils/NameConf.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "O2Version.h"
#include <cstdio>
#include <unordered_map>
#include <filesystem>

#include "SimPublishChannelHelper.h"
#include <CommonUtils/FileSystemUtils.h>

std::string getServerLogName()
{
  auto& conf = o2::conf::SimConfig::Instance();
  std::stringstream str;
  str << conf.getOutPrefix() << "_serverlog";
  return str.str();
}

std::string getWorkerLogName()
{
  auto& conf = o2::conf::SimConfig::Instance();
  std::stringstream str;
  str << conf.getOutPrefix() << "_workerlog";
  return str.str();
}

std::string getMergerLogName()
{
  auto& conf = o2::conf::SimConfig::Instance();
  std::stringstream str;
  str << conf.getOutPrefix() << "_mergerlog";
  return str.str();
}

void remove_tmp_files()
{
  // remove all (known) socket files in /tmp
  // using the naming convention /tmp/o2sim-.*PID
  std::stringstream searchstr;
  searchstr << "o2sim-.*-" << getpid() << "$";
  auto filenames = o2::utils::listFiles("/tmp/", searchstr.str());
  // remove those files
  for (auto& fn : filenames) {
    try {
      std::filesystem::remove(std::filesystem::path(fn));
    } catch (...) {
      LOG(warn) << "Couldn't remove tmp file " << fn;
    }
  }
}

void cleanup()
{
  remove_tmp_files();
  o2::utils::ShmManager::Instance().release();

  // special mode in which we dump the output from various
  // log files to terminal (mainly interesting for CI mode)
  if (getenv("ALICE_O2SIM_DUMPLOG")) {
    std::cerr << "------------- START OF EVENTSERVER LOG ----------" << std::endl;
    std::stringstream catcommand1;
    catcommand1 << "cat " << getServerLogName() << ";";
    if (system(catcommand1.str().c_str()) != 0) {
      LOG(warn) << "error executing system call";
    }

    std::cerr << "------------- START OF SIM WORKER(S) LOG --------" << std::endl;
    std::stringstream catcommand2;
    catcommand2 << "cat " << getWorkerLogName() << "*;";
    if (system(catcommand2.str().c_str()) != 0) {
      LOG(warn) << "error executing system call";
    }

    std::cerr << "------------- START OF MERGER LOG ---------------" << std::endl;
    std::stringstream catcommand3;
    catcommand3 << "cat " << getMergerLogName() << ";";
    if (system(catcommand3.str().c_str()) != 0) {
      LOG(warn) << "error executing system call";
    }
  }
}

// quick cross check of simulation output
int checkresult()
{
  int errors = 0;
  // We can put more or less complex things
  // here.
  auto& conf = o2::conf::SimConfig::Instance();
  if (!conf.writeToDisc()) {
    return 0;
  }
  // easy check: see if we have number of entries in output tree == number of events asked
  std::string filename = o2::base::NameConf::getMCKinematicsFileName(conf.getOutPrefix().c_str());
  TFile f(filename.c_str(), "OPEN");
  if (f.IsZombie()) {
    LOG(warn) << "Kinematics file corrupted or does not exist";
    return 1;
  }
  auto tr = static_cast<TTree*>(f.Get("o2sim"));
  if (!tr) {
    errors++;
  } else {
    if (!conf.isFilterOutNoHitEvents()) {
      if (tr->GetEntries() != conf.getNEvents()) {
        LOG(warn) << "There are fewer events in the output than asked";
      }
    }
  }
  // add more simple checks

  return errors;
}

// ---> THE FOLLOWING CAN BE PUT INTO A "STATE" STRUCT
std::vector<int> gChildProcesses; // global vector of child pids
// record distributed events in a container
std::vector<int> gDistributedEvents;
// record finished events in a container
std::vector<int> gFinishedEvents;
int gAskedEvents;

std::string getControlAddress()
{
  std::stringstream controlsocketname;
  controlsocketname << "ipc:///tmp/o2sim-control-" << getpid();
  return controlsocketname.str();
}
std::string getInternalControlAddress()
{
  // creates names for an internal-only socket
  // hashing to distinguish from more "public" sockets
  std::hash<std::string> hasher;
  std::stringstream str;
  str << "o2sim-internal_" << getpid();
  std::string tmp(std::to_string(hasher(str.str())));
  std::stringstream controlsocketname;
  controlsocketname << "ipc:///tmp/" << tmp.substr(0, 10) << "_" << getpid();
  return controlsocketname.str();
}

// signal handler for graceful exit
void sighandler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    LOG(info) << "o2-sim driver: Signal caught ... clean up and exit";
    // forward signal to all children
    for (auto& pid : gChildProcesses) {
      killpg(pid, signal);
    }
    cleanup();
    exit(0);
  }
}

bool isBusy()
{
  if (gFinishedEvents.size() != gAskedEvents) {
    return true;
  }
  return false;
}

// launches a thread that listens for control command from outside
// or that propagates control strings to all children
void launchControlThread()
{
  static std::vector<std::thread> threads;
  auto controladdress = getControlAddress();
  auto internalcontroladdress = getInternalControlAddress();
  LOG(info) << "Control address is: " << controladdress;
  setenv("ALICE_O2SIMCONTROL", internalcontroladdress.c_str(), 1);

  auto lambda = [controladdress, internalcontroladdress]() {
    auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

    auto internalchannel = fair::mq::Channel{"o2sim-control", "pub", factory};
    internalchannel.Bind(internalcontroladdress);
    internalchannel.Validate();
    std::unique_ptr<fair::mq::Message> message(internalchannel.NewMessage());

    auto outsidechannel = fair::mq::Channel{"o2sim-outside-exchange", "rep", factory};
    outsidechannel.Bind(controladdress);
    outsidechannel.Validate();
    std::unique_ptr<fair::mq::Message> request(outsidechannel.NewMessage());

    bool keepgoing = true;
    while (keepgoing) {
      outsidechannel.Init();
      outsidechannel.Bind(controladdress);
      outsidechannel.Validate();
      if (outsidechannel.Receive(request) > 0) {
        std::string command(reinterpret_cast<char const*>(request->GetData()), request->GetSize());
        LOG(info) << "Control message: " << command;
        int code = -1;
        if (isBusy()) {
          code = 1; // code = 1 --> busy
          std::unique_ptr<fair::mq::Message> reply(outsidechannel.NewSimpleMessage(code));
          outsidechannel.Send(reply);
        } else {
          code = 0; // code = 0 --> ok

          o2::conf::SimReconfigData reconfig;
          auto success = o2::conf::parseSimReconfigFromString(command, reconfig);
          if (!success) {
            LOG(warn) << "CONTROL REQUEST COULD NOT BE PARSED";
            code = 2; // code = 2 --> error with request data
          }
          std::unique_ptr<fair::mq::Message> reply(outsidechannel.NewSimpleMessage(code));
          outsidechannel.Send(reply);

          if (code == 0) {
            gAskedEvents = reconfig.nEvents;
            gDistributedEvents.clear();
            gFinishedEvents.clear();
            // forward request from outside to all internal processes
            internalchannel.Send(request);
            keepgoing = !reconfig.stop;
          }
        }
      }
    }
  };
  threads.push_back(std::thread(lambda));
  threads.back().detach();
}

// launches a thread that listens for control command from outside
// or that propagates control strings to all children
void launchWorkerListenerThread()
{
  static std::vector<std::thread> threads;
  auto lambda = []() {
    auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

    auto listenchannel = fair::mq::Channel{"channel0", "sub", factory};
    listenchannel.Init();
    std::stringstream address;
    address << "ipc:///tmp/o2sim-worker-notifications-" << getpid();
    listenchannel.Connect(address.str());
    listenchannel.Validate();
    std::unique_ptr<fair::mq::Message> message(listenchannel.NewMessage());

    while (true) {
      if (listenchannel.Receive(message) > 0) {
        std::string msg(reinterpret_cast<char const*>(message->GetData()), message->GetSize());
        LOG(info) << "Worker message: " << msg;
      }
    }
  };
  threads.push_back(std::thread(lambda));
  threads.back().detach();
}

// monitors a certain incoming event pipes and displays new information
// gives possibility to exec a callback at these events
void launchThreadMonitoringEvents(
  int pipefd, std::string text, std::vector<int>& eventcontainer,
  std::function<void(std::vector<int> const&)> callback = [](std::vector<int> const&) {})
{
  static std::vector<std::thread> threads;
  auto lambda = [pipefd, text, callback, &eventcontainer]() {
    int eventcounter;
    while (1) {
      ssize_t count = read(pipefd, &eventcounter, sizeof(eventcounter));
      if (count == -1) {
        LOG(info) << "ERROR READING";
        if (errno == EINTR) {
          continue;
        } else {
          return;
        }
      } else if (count == 0) {
        break;
      } else {
        LOG(info) << text.c_str() << eventcounter;
        eventcontainer.push_back(eventcounter);
        callback(eventcontainer);
      }
    };
  };
  threads.push_back(std::thread(lambda));
  threads.back().detach();
}

// helper executable to launch all the devices/processes
// for parallel simulation
int main(int argc, char* argv[])
{
  LOG(info) << "This is o2-sim version " << o2::fullVersion() << " (" << o2::gitRevision() << ")";
  LOG(info) << o2::getBuildInfo();

  signal(SIGINT, sighandler);
  signal(SIGTERM, sighandler);
  // we enable the forked version of the code by default
  setenv("ALICE_SIMFORKINTERNAL", "ON", 1);

  TStopwatch timer;
  timer.Start();
  auto o2env = getenv("O2_ROOT");
  if (!o2env) {
    LOG(fatal) << "O2_ROOT environment not defined";
  }
  std::string rootpath(o2env);
  std::string installpath = rootpath + "/bin";

  // copy topology file to working dir and update ports
  std::stringstream configss;
  configss << rootpath << "/share/config/o2simtopology_template.json";
  auto localconfig = std::string("o2simtopology_") + std::to_string(getpid()) + std::string(".json");

  // need to add pid to channel urls to allow simultaneous deploys!
  // we simply insert the PID into the topology template
  std::ifstream in(configss.str());
  std::ofstream out(localconfig);
  std::string wordToReplace("#PID#");
  std::string wordToReplaceWith = std::to_string(getpid());
  std::string line;
  size_t len = wordToReplace.length();
  while (std::getline(in, line)) {
    size_t pos = line.find(wordToReplace);
    if (pos != std::string::npos) {
      line.replace(pos, len, wordToReplaceWith);
    }
    out << line << '\n';
  }
  in.close();
  out.close();

  // create a channel for outside event notifications --> factor out into common function
  // auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto externalpublishchannel = o2::simpubsub::createPUBChannel(o2::simpubsub::getPublishAddress("o2sim-notifications"));

  auto& conf = o2::conf::SimConfig::Instance();
#ifdef SIM_RUN5
  conf.setRun5();
#endif
  if (!conf.resetFromArguments(argc, argv)) {
    return 1;
  }
  // in case of zero events asked (only setup geometry etc) we just call the non-distributed version
  // (otherwise we would need to add more synchronization between the actors)
  if (conf.getNEvents() <= 0 && !conf.asService()) {
    LOG(info) << "No events to be simulated; Switching to non-distributed mode";
    const int Nargs = argc + 1;
#ifdef SIM_RUN5
    std::string name("o2-sim-serial-run5");
#else
    std::string name("o2-sim-serial");
#endif
    const char* arguments[Nargs];
    arguments[0] = name.c_str();
    for (int i = 1; i < argc; ++i) {
      arguments[i] = argv[i];
    }
    arguments[argc] = nullptr;
    std::string path = installpath + "/" + name;
    auto r = execv(path.c_str(), (char* const*)arguments);
    if (r != 0) {
      perror(nullptr);
    }
    return r;
  }

  gAskedEvents = conf.getNEvents();
  if (conf.asService()) {
    launchControlThread();
    // launchWorkerListenerThread();
  }

  // we create the global shared mem pool; just enough to serve
  // n simulation workers
  int nworkers = conf.getNSimWorkers();
  setenv("ALICE_NSIMWORKERS", std::to_string(nworkers).c_str(), 1);
  LOG(info) << "Running with " << nworkers << " sim workers ";

  o2::utils::ShmManager::Instance().createGlobalSegment(nworkers);

  // we can try to disable it here
  if (getenv("ALICE_NOSIMSHM")) {
    o2::utils::ShmManager::Instance().disable();
  }

  int pipe_serverdriver_fd[2];
  if (pipe(pipe_serverdriver_fd) != 0) {
    perror("problem in creating pipe");
  }

  // the server
  int pid = fork();
  if (pid == 0) {
    int fd = open(getServerLogName().c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    setenv("ALICE_O2SIMSERVERTODRIVER_PIPE", std::to_string(pipe_serverdriver_fd[1]).c_str(), 1);

    dup2(fd, 1); // make stdout go to file
    dup2(fd, 2); // make stderr go to file - you may choose to not do this
                 // or perhaps send stderr to another file
    close(pipe_serverdriver_fd[0]);
    close(fd); // fd no longer needed - the dup'ed handles are sufficient

    const std::string name("o2-sim-primary-server-device-runner");
    const std::string path = installpath + "/" + name;
    const std::string config = localconfig;

    // copy all arguments into a common vector
#ifdef SIM_RUN5
    const int addNArgs = 10;
#else
    const int addNArgs = 9;
#endif
    const int Nargs = argc + addNArgs;
    const char* arguments[Nargs];
    arguments[0] = name.c_str();
    arguments[1] = "--control";
    arguments[2] = "static";
    arguments[3] = "--id";
    arguments[4] = "primary-server";
    arguments[5] = "--mq-config";
    arguments[6] = config.c_str();
    arguments[7] = "--severity";
    arguments[8] = "debug";
#ifdef SIM_RUN5
    arguments[9] = "--isRun5";
#endif
    for (int i = 1; i < argc; ++i) {
      arguments[addNArgs - 1 + i] = argv[i];
    }
    arguments[Nargs - 1] = nullptr;
    for (int i = 0; i < Nargs; ++i) {
      if (arguments[i]) {
        std::cerr << arguments[i] << "\n";
      }
    }
    std::cerr << "$$$$\n";
    auto r = execv(path.c_str(), (char* const*)arguments);
    LOG(info) << "Starting the server"
              << "\n";
    if (r != 0) {
      perror(nullptr);
    }
    return r;
  } else {
    gChildProcesses.push_back(pid);
    setpgid(pid, pid);
    close(pipe_serverdriver_fd[1]);
    std::cout << "Spawning particle server on PID " << pid << "; Redirect output to " << getServerLogName() << "\n";

    // A simple callback for distributed primary-chunk "events"
    auto distributionCallback = [&conf, &externalpublishchannel](std::vector<int> const& v) {
      std::stringstream str;
      str << "EVENT " << v.back() << " DISTRIBUTED";
      o2::simpubsub::publishMessage(externalpublishchannel, o2::simpubsub::simStatusString("O2SIM", "INFO", str.str()));
    };
    launchThreadMonitoringEvents(pipe_serverdriver_fd[0], "DISTRIBUTING EVENT : ", gDistributedEvents, distributionCallback);
  }

  auto internalfork = getenv("ALICE_SIMFORKINTERNAL");
  if (internalfork) {
    // forking will be done internally to profit from copy-on-write
    nworkers = 1;
  }
  for (int id = 0; id < nworkers; ++id) {
    // the workers
    std::stringstream workerlogss;
    workerlogss << getWorkerLogName() << id;

    // the workers
    std::stringstream workerss;
    workerss << "worker" << id;

    pid = fork();
    if (pid == 0) {
      int fd = open(workerlogss.str().c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
      dup2(fd, 1); // make stdout go to file
      dup2(fd, 2); // make stderr go to file - you may choose to not do this
                   // or perhaps send stderr to another file
      close(fd);   // fd no longer needed - the dup'ed handles are sufficient

      const std::string name("o2-sim-device-runner");
      const std::string path = installpath + "/" + name;
      execl(path.c_str(), name.c_str(), "--control", "static", "--id", workerss.str().c_str(), "--config-key",
            "worker", "--mq-config", localconfig.c_str(), "--severity", "info", (char*)nullptr);
      return 0;
    } else {
      gChildProcesses.push_back(pid);
      setpgid(pid, pid); // the worker processes will form their own group
      std::cout << "Spawning sim worker " << id << " on PID " << pid
                << "; Redirect output to " << workerlogss.str() << "\n";
    }
  }

  // the hit merger
  int pipe_mergerdriver_fd[2];
  if (pipe(pipe_mergerdriver_fd) != 0) {
    perror("problem in creating pipe");
  }

  pid = fork();
  if (pid == 0) {
    int fd = open(getMergerLogName().c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    dup2(fd, 1); // make stdout go to file
    dup2(fd, 2); // make stderr go to file - you may choose to not do this
                 // or perhaps send stderr to another file
    close(fd);   // fd no longer needed - the dup'ed handles are sufficient
    close(pipe_mergerdriver_fd[0]);
    setenv("ALICE_O2SIMMERGERTODRIVER_PIPE", std::to_string(pipe_mergerdriver_fd[1]).c_str(), 1);
    const std::string name("o2-sim-hit-merger-runner");
    const std::string path = installpath + "/" + name;
    execl(path.c_str(), name.c_str(), "--control", "static", "--catch-signals", "0", "--id", "hitmerger", "--mq-config", localconfig.c_str(),
          (char*)nullptr);
    return 0;
  } else {
    std::cout << "Spawning hit merger on PID " << pid << "; Redirect output to " << getMergerLogName() << "\n";
    setpgid(pid, pid);
    gChildProcesses.push_back(pid);
    close(pipe_mergerdriver_fd[1]);

    // A simple callback that determines if the simulation is complete and triggers
    // a shutdown of all child processes. This appears to be more robust than leaving
    // that decision upon the children (sometimes there are problems with that).
    auto finishCallback = [&conf, &externalpublishchannel](std::vector<int> const& v) {
      std::stringstream str;
      str << "EVENT " << v.back() << " FINISHED " << gAskedEvents << " " << v.size();
      o2::simpubsub::publishMessage(externalpublishchannel, o2::simpubsub::simStatusString("O2SIM", "INFO", str.str()));
      if (gAskedEvents == v.size()) {
        o2::simpubsub::publishMessage(externalpublishchannel, o2::simpubsub::simStatusString("O2SIM", "STATE", "DONE"));
        if (!conf.asService()) {
          LOG(info) << "SIMULATION IS DONE. INITIATING SHUTDOWN.";
          for (auto p : gChildProcesses) {
            killpg(p, SIGTERM);
          }
        } else {
          LOG(info) << "SIMULATION DONE. STAYING AS DAEMON.";
        }
      }
    };

    launchThreadMonitoringEvents(pipe_mergerdriver_fd[0], "EVENT FINISHED : ", gFinishedEvents, finishCallback);
  }

  // wait on merger (which when exiting completes the workflow)
  auto mergerpid = gChildProcesses.back();

  int status, cpid;
  // wait just blocks and waits until any child returns; but we make sure to wait until merger is here
  bool errored = false;
  while ((cpid = wait(&status)) != mergerpid) {
    if (WEXITSTATUS(status) || WIFSIGNALED(status)) {
      LOG(info) << "Process " << cpid << " EXITED WITH CODE " << WEXITSTATUS(status) << " SIGNALED "
                << WIFSIGNALED(status) << " SIGNAL " << WTERMSIG(status);

      // we bring down all processes if one of them had problems or got a termination signal
      // if (WTERMSIG(status) == SIGABRT || WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS || WTERMSIG(status) == SIGTERM) {
      LOG(info) << "Problem detected (or child received termination signal) ... shutting down whole system ";
      for (auto p : gChildProcesses) {
        LOG(info) << "TERMINATING " << p;
        killpg(p, SIGTERM); // <--- makes sure to shutdown "unknown" child pids via the group property
      }
      LOG(error) << "SHUTTING DOWN DUE TO SIGNALED EXIT IN COMPONENT " << cpid;
      errored = true;
    }
  }
  // This marks the actual end of the computation (since results are available)
  LOG(info) << "Merger process " << mergerpid << " returned";
  LOG(info) << "Simulation process took " << timer.RealTime() << " s";

  if (!errored) {
    // ordinary shutdown of the rest
    for (auto p : gChildProcesses) {
      if (p != mergerpid) {
        LOG(info) << "SHUTTING DOWN CHILD PROCESS " << p;
        killpg(p, SIGTERM);
      }
    }
  }
  // definitely wait on all children
  // otherwise this breaks accounting in the /usr/bin/time command
  while ((cpid = wait(&status))) {
    if (cpid == -1) {
      break;
    }
  }

  LOG(debug) << "ShmManager operation " << o2::utils::ShmManager::Instance().isOperational() << "\n";

  // do a quick check to see if simulation produced something reasonable
  // (mainly useful for continuous integration / automated testing suite)
  auto returncode = errored ? 1 : checkresult();
  if (returncode == 0) {
    // Extract a single file for MCEventHeaders
    // This file will be small and can quickly unblock start of signal transport (in embedding).
    // This is useful when we cache background events on the GRID. The headers file can be copied quickly
    // and the rest of kinematics + Hits may follow asyncronously since they are only needed at much
    // later stages (digitization).

    auto& conf = o2::conf::SimConfig::Instance();
    if (conf.writeToDisc()) {
      // easy check: see if we have number of entries in output tree == number of events asked
      std::string kinefilename = o2::base::NameConf::getMCKinematicsFileName(conf.getOutPrefix().c_str());
      std::string headerfilename = o2::base::NameConf::getMCHeadersFileName(conf.getOutPrefix().c_str());
      o2::dataformats::MCEventHeader::extractFileFromKinematics(kinefilename, headerfilename);
    }
    LOG(info) << "SIMULATION RETURNED SUCCESFULLY";
  }

  cleanup();
  return returncode;
}
