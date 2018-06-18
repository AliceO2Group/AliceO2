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
#include <signal.h>
#include "TStopwatch.h"
#include "FairLogger.h"

const char* serverlogname = "serverlog";
const char* workerlogname = "workerlog";
const char* mergerlogname = "mergerlog";

// helper executable to launch all the devices/processes
// for parallel simulation
int main(int argc, char* argv[])
{
  TStopwatch timer;
  timer.Start();
  std::string rootpath(getenv("O2_ROOT"));
  std::string installpath = rootpath + "/bin";

  std::stringstream configss;
  configss << rootpath << "/share/config/o2simtopology.json";

  auto& conf = o2::conf::SimConfig::Instance();
  if (!conf.resetFromArguments(argc, argv)) {
    return 1;
  }

  std::vector<int> childpids;

  // the server
  int pid = fork();
  if (pid == 0) {
    int fd = open(serverlogname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    dup2(fd, 1); // make stdout go to file
    dup2(fd, 2); // make stderr go to file - you may choose to not do this
                 // or perhaps send stderr to another file
    close(fd);   // fd no longer needed - the dup'ed handles are sufficient

    const std::string name("O2PrimaryServerDeviceRunner");
    const std::string path = installpath + "/" + name;
    const std::string config = configss.str();

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
    std::cerr << "$$$$";
    execv(path.c_str(), (char* const*)arguments);
    return 0;
  } else {
    childpids.push_back(pid);
    std::cout << "Spawning particle server on PID " << pid << "; Redirect output to " << serverlogname << "\n";
  }

  int nworkers = std::thread::hardware_concurrency() / 2;
  auto internalfork = getenv("ALICE_SIMFORKINTERNAL");
  if (internalfork) {
    // forking will be done internally to profit from copy-on-write
    nworkers = 1;
  } else {
    auto f = getenv("ALICE_NSIMWORKERS");
    if (f) {
      nworkers = atoi(f);
    }
    std::cout << "Running with " << nworkers << " sim workers "
              << "(customize using the ALICE_NSIMWORKERS environment variable)\n";
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

      const std::string name("O2SimDeviceRunner");
      const std::string path = installpath + "/" + name;
      execl(path.c_str(), name.c_str(), "--control", "static", "--id", workerss.str().c_str(), "--config-key",
            "worker", "--mq-config", configss.str().c_str(), "--severity", "info", (char*)0);
      return 0;
    } else {
      childpids.push_back(pid);
      std::cout << "Spawning sim worker " << id << " on PID " << pid
                << "; Redirect output to " << workerlogss.str() << "\n";
    }
  }

  // the hit merger
  int status, cpid;
  pid = fork();
  if (pid == 0) {
    int fd = open(mergerlogname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    dup2(fd, 1); // make stdout go to file
    dup2(fd, 2); // make stderr go to file - you may choose to not do this
                 // or perhaps send stderr to another file
    close(fd);   // fd no longer needed - the dup'ed handles are sufficient

    const std::string name("O2HitMergerRunner");
    const std::string path = installpath + "/" + name;
    execl(path.c_str(), name.c_str(), "--control", "static", "--id", "hitmerger", "--mq-config", configss.str().c_str(),
          (char*)0);
    return 0;
  } else {
    std::cout << "Spawning hit merger on PID " << pid << "; Redirect output to " << mergerlogname << "\n";
    childpids.push_back(pid);
  }

  // wait on merger (which when exiting completes the workflow)
  auto mergerpid = childpids.back();

  // wait just blocks and waits until any child returns; make sure that we wait until merger is here
  while ((cpid = wait(&status)) != mergerpid) {
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
  return 0;
}
