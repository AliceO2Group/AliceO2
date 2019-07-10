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

#include <cstdlib>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <SimConfig/SimConfig.h>
#include <sys/wait.h>
#include <vector>
#include <thread>
#include <csignal>
#include "TStopwatch.h"
#include "FairLogger.h"
#include "CommonUtils/ShmManager.h"
#include "TFile.h"
#include "TTree.h"
#include "Framework/FreePortFinder.h"
#include <sys/types.h>

const char* serverlogname = "serverlog";
const char* workerlogname = "workerlog";
const char* mergerlogname = "mergerlog";

void cleanup()
{
  o2::utils::ShmManager::Instance().release();

  // special mode in which we dump the output from various
  // log files to terminal (mainly interesting for CI mode)
  if (getenv("ALICE_O2SIM_DUMPLOG")) {
    std::cerr << "------------- START OF EVENTSERVER LOG ----------" << std::endl;
    std::stringstream catcommand1;
    catcommand1 << "cat " << serverlogname << ";";
    if (system(catcommand1.str().c_str()) != 0) {
      LOG(WARN) << "error executing system call";
    }

    std::cerr << "------------- START OF SIM WORKER(S) LOG --------" << std::endl;
    std::stringstream catcommand2;
    catcommand2 << "cat " << workerlogname << "*;";
    if (system(catcommand2.str().c_str()) != 0) {
      LOG(WARN) << "error executing system call";
    }

    std::cerr << "------------- START OF MERGER LOG ---------------" << std::endl;
    std::stringstream catcommand3;
    catcommand3 << "cat " << mergerlogname << ";";
    if (system(catcommand3.str().c_str()) != 0) {
      LOG(WARN) << "error executing system call";
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
  // easy check: see if we have number of entries in output tree == number of events asked
  std::string filename = std::string(conf.getOutPrefix()) + ".root";
  TFile f(filename.c_str(), "OPEN");
  auto tr = static_cast<TTree*>(f.Get("o2sim"));
  if (!tr) {
    errors++;
  } else {
    if (!conf.isFilterOutNoHitEvents()) {
      errors += tr->GetEntries() != conf.getNEvents();
    }
  }

  // add more simple checks

  return errors;
}

// signal handler for graceful exit
void sighandler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    LOG(INFO) << "signal caught ... clean up and exit";
    cleanup();
    exit(0);
  }
}

// updates/sets up the ports to be used in this simulation run
// this should enable parallel instances of o2sim ... to be generalized
// with DDS for example
bool updatePorts(std::string configfilename)
{
  // original ports
  const int SERVERPORT = 25005;
  const int MERGERPORT = 25009;

  auto pid = getpid();

  int portstart = SERVERPORT + pid % 64; // somewhat randomize the port start
  int portend = 50000;
  int step = 2; // we need 2 ports
  o2::framework::FreePortFinder finder(portstart, portend, step);
  finder.setVerbose(false); // disable verbose output
  finder.scan();
  auto newserverport = finder.port();
  auto newmergerport = newserverport + 1;

  LOG(INFO) << "SERVER PORT " << newserverport;
  LOG(INFO) << "MERGER PORT " << newmergerport;
  // publish these numbers for other processes
  setenv("ALICE_O2SIM_SERVERPORT", std::to_string(newserverport).c_str(), 1);
  setenv("ALICE_O2SIM_MERGERPORT", std::to_string(newmergerport).c_str(), 1);

  // fix ports in the configuration file template (there are for sure nicer ways of doing that
  std::stringstream sedcmd;
  sedcmd << "sed -i'.original' "
         << "-e 's/:" << SERVERPORT << "/:" << newserverport << "/' "
         << "-e 's/:" << MERGERPORT << "/:" << newmergerport << "/' "
         << configfilename;
  auto r = system(sedcmd.str().c_str());
  return true;
}

// monitores a certain incoming pipe and displays new information
void launchThreadMonitoringEvents(int pipefd, std::string text)
{
  static std::vector<std::thread> threads;
  auto lambda = [pipefd, text]() {
    int eventcounter;
    while (1) {
      ssize_t count = read(pipefd, &eventcounter, sizeof(eventcounter));
      if (count == -1) {
        LOG(INFO) << "ERROR READING";
        if (errno == EINTR) {
          continue;
        } else {
          return;
        }
      } else if (count == 0) {
        break;
      } else {
        LOG(INFO) << text.c_str() << eventcounter;
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
  signal(SIGINT, sighandler);
  signal(SIGTERM, sighandler);
  // we enable the forked version of the code by default
  setenv("ALICE_SIMFORKINTERNAL", "ON", 1);

  TStopwatch timer;
  timer.Start();
  auto o2env = getenv("O2_ROOT");
  if (!o2env) {
    LOG(FATAL) << "O2_ROOT environment not defined";
  }
  std::string rootpath(o2env);
  std::string installpath = rootpath + "/bin";

  // copy topology file to working dir and update ports
  std::stringstream configss;
  configss << rootpath << "/share/config/o2simtopology.json";
  std::string localconfig = std::string("o2simtopology_") + std::to_string(getpid()) + std::string(".json");
  std::stringstream cpcmd;
  cpcmd << "cp " << configss.str() << " " << localconfig;
  if (system(cpcmd.str().c_str()) != 0) {
    LOG(WARN) << "error executing system call";
  }
  updatePorts(localconfig.c_str());

  auto& conf = o2::conf::SimConfig::Instance();
  if (!conf.resetFromArguments(argc, argv)) {
    return 1;
  }
  // in case of zero events asked (only setup geometry etc) we just call the non-distributed version
  // (otherwise we would need to add more synchronization between the actors)
  if (conf.getNEvents() <= 0) {
    LOG(INFO) << "No events to be simulated; Switching to non-distributed mode";
    const int Nargs = argc + 1;
    std::string name("o2-sim-serial");
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

  // we create the global shared mem pool; just enough to serve
  // n simulation workers
  int nworkers = conf.getNSimWorkers();
  setenv("ALICE_NSIMWORKERS", std::to_string(nworkers).c_str(), 1);
  LOG(INFO) << "Running with " << nworkers << " sim workers ";

  o2::utils::ShmManager::Instance().createGlobalSegment(nworkers);

  // we can try to disable it here
  if (getenv("ALICE_NOSIMSHM")) {
    o2::utils::ShmManager::Instance().disable();
  }

  std::vector<int> childpids;

  int pipe_serverdriver_fd[2];
  if (pipe(pipe_serverdriver_fd) != 0) {
    perror("problem in creating pipe");
  }

  // the server
  int pid = fork();
  if (pid == 0) {
    int fd = open(serverlogname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
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
    const int Nargs = argc + 7;
    const char* arguments[Nargs];
    arguments[0] = name.c_str();
    arguments[1] = "--control";
    arguments[2] = "static";
    arguments[3] = "--id";
    arguments[4] = "primary-server";
    arguments[5] = "--mq-config";
    arguments[6] = config.c_str();
    for (int i = 1; i < argc; ++i) {
      arguments[6 + i] = argv[i];
    }
    arguments[Nargs - 1] = nullptr;
    for (int i = 0; i < Nargs; ++i) {
      if (arguments[i]) {
        std::cerr << arguments[i] << "\n";
      }
    }
    std::cerr << "$$$$\n";
    auto r = execv(path.c_str(), (char* const*)arguments);
    if (r != 0) {
      perror(nullptr);
    }
    return r;
  } else {
    childpids.push_back(pid);
    close(pipe_serverdriver_fd[1]);
    std::cout << "Spawning particle server on PID " << pid << "; Redirect output to " << serverlogname << "\n";
    launchThreadMonitoringEvents(pipe_serverdriver_fd[0], "EVENTS DISTRIBUTED : ");
  }

  auto internalfork = getenv("ALICE_SIMFORKINTERNAL");
  if (internalfork) {
    // forking will be done internally to profit from copy-on-write
    nworkers = 1;
  }
  for (int id = 0; id < nworkers; ++id) {
    // the workers
    std::stringstream workerlogss;
    workerlogss << workerlogname << id;

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
      childpids.push_back(pid);
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
    int fd = open(mergerlogname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
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
    std::cout << "Spawning hit merger on PID " << pid << "; Redirect output to " << mergerlogname << "\n";
    childpids.push_back(pid);
    close(pipe_mergerdriver_fd[1]);
    launchThreadMonitoringEvents(pipe_mergerdriver_fd[0], "EVENT FINISHED : ");
  }

  // wait on merger (which when exiting completes the workflow)
  auto mergerpid = childpids.back();

  int status, cpid;
  // wait just blocks and waits until any child returns; but we make sure to wait until merger is here
  while ((cpid = wait(&status)) != mergerpid) {
    if (WIFSIGNALED(status)) {
      LOG(INFO) << "Process " << cpid << " EXITED WITH CODE " << WEXITSTATUS(status) << " SIGNALED "
                << WIFSIGNALED(status) << " SIGNAL " << WTERMSIG(status);
    }
    // we bring down all processes if one of them aborts
    if (WTERMSIG(status) == SIGABRT) {
      for (auto p : childpids) {
        kill(p, SIGABRT);
      }
      cleanup();
      LOG(FATAL) << "ABORTING DUE TO ABORT IN COMPONENT";
    }
  }
  // This marks the actual end of the computation (since results are available)
  LOG(INFO) << "Merger process " << mergerpid << " returned";
  LOG(INFO) << "Simulation process took " << timer.RealTime() << " s";

  // make sure the rest shuts down
  for (auto p : childpids) {
    if (p != mergerpid) {
      kill(p, SIGKILL);
    }
  }

  LOG(DEBUG) << "ShmManager operation " << o2::utils::ShmManager::Instance().isOperational() << "\n";
  cleanup();

  // do a quick check to see if simulation produced something reasonable
  // (mainly useful for continuous integration / automated testing suite)
  auto returncode = checkresult();
  if (returncode == 0) {
    LOG(INFO) << "SIMULATION RETURNED SUCCESFULLY";
  }

  return returncode;
}
