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

#include <fairmq/shmem/Common.h>
#include <fairmq/shmem/UnmanagedRegion.h>
#include <fairmq/shmem/Segment.h>
#include <fairmq/shmem/Monitor.h>

#include <fairmq/tools/Unique.h>

#include <fairlogger/Logger.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <csignal>

#include <chrono>
#include <map>
#include <string>
#include <thread>

#if !defined(__MACH__) && !defined(__APPLE__)
#include <syscall.h>
#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3
#endif

using namespace std;
using namespace boost::program_options;

namespace
{
volatile sig_atomic_t gStopping = 0;
volatile sig_atomic_t gResetContent = 0;
}

void signalHandler(int /* signal */)
{
  gStopping = 1;
}

void resetContentHandler(int /* signal */)
{
  gResetContent = 1;
}

struct ShmManager {
  ShmManager(uint64_t _shmId, const vector<string>& _segments, const vector<string>& _regions, bool zero = true)
    : shmId(fair::mq::shmem::makeShmIdStr(_shmId))
  {
    LOG(info) << "Starting ShmManager for shmId: " << shmId;
    LOG(info) << "Performing full reset...";
    FullReset();
    LOG(info) << "Done.";
    LOG(info) << "Adding managed segments...";
    AddSegments(_segments, zero);
    LOG(info) << "Done.";
    LOG(info) << "Adding unmanaged regions...";
    AddRegions(_regions, zero);
    LOG(info) << "Done.";
    LOG(info) << "Shared memory is ready for use.";
  }

  void AddSegments(const vector<string>& _segments, bool zero)
  {
    for (const auto& s : _segments) {
      vector<string> conf;
      boost::algorithm::split(conf, s, boost::algorithm::is_any_of(","));
      if (conf.size() != 3) {
        LOG(error) << "incorrect format for --segments. Expecting pairs of <id>,<size><numaid>.";
        fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
        throw runtime_error("incorrect format for --segments. Expecting pairs of <id>,<size>,<numaid>.");
      }
      uint16_t id = stoi(conf.at(0));
      uint64_t size = stoull(conf.at(1));
      segmentCfgs.emplace_back(fair::mq::shmem::SegmentConfig{id, size, "rbtree_best_fit"});

#if !defined(__MACH__) && !defined(__APPLE__)
      int numaid = stoi(conf.at(2));
      if (numaid == -2) {
        LOG(info) << "Setting memory allocation to process default";
        syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
      } else if (numaid == -1) {
        LOG(info) << "Setting memory allocation to NUMA interleaving";
        unsigned long nodemask = 0xffffff;
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
      } else {
        LOG(info) << "Setting memory allocation to NUMA id " << numaid;
        unsigned long nodemask = 1 << numaid;
        syscall(SYS_set_mempolicy, MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
      }
#endif

      auto ret = segments.emplace(id, fair::mq::shmem::Segment(shmId, id, size, fair::mq::shmem::rbTreeBestFit));
      fair::mq::shmem::Segment& segment = ret.first->second;
      LOG(info) << "Created segment " << id << " of size " << segment.GetSize() << ", starting at " << segment.GetData() << ". Locking...";
      segment.Lock();
      LOG(info) << "Done.";
      if (zero) {
        LOG(info) << "Zeroing...";
        segment.Zero();
        LOG(info) << "Done.";
      }
    }
  }

  void AddRegions(const vector<string>& _regions, bool zero)
  {
    for (const auto& r : _regions) {
      vector<string> conf;
      boost::algorithm::split(conf, r, boost::algorithm::is_any_of(","));
      if (conf.size() != 3) {
        LOG(error) << "incorrect format for --regions. Expecting pairs of <id>,<size>,<numaid>.";
        fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
        throw runtime_error("incorrect format for --regions. Expecting pairs of <id>,<size>,<numaid>.");
      }
      uint16_t id = stoi(conf.at(0));
      uint64_t size = stoull(conf.at(1));
      fair::mq::RegionConfig cfg;
      cfg.id = id;
      cfg.size = size;
      regionCfgs.push_back(cfg);

#if !defined(__MACH__) && !defined(__APPLE__)
      int numaid = stoi(conf.at(2));
      if (numaid == -2) {
        LOG(info) << "Setting memory allocation to process default";
        syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0);
      } else if (numaid == -1) {
        LOG(info) << "Setting memory allocation to NUMA interleaving";
        unsigned long nodemask = 0xffffff;
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
      } else {
        LOG(info) << "Setting memory allocation to NUMA id " << numaid;
        unsigned long nodemask = 1 << numaid;
        syscall(SYS_set_mempolicy, MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
      }
#endif

      auto ret = regions.emplace(id, make_unique<fair::mq::shmem::UnmanagedRegion>(shmId, id, size));
      fair::mq::shmem::UnmanagedRegion& region = *(ret.first->second);
      LOG(info) << "Created unamanged region " << id << " of size " << region.GetSize() << ", starting at " << region.GetData() << ". Locking...";
      region.Lock();
      LOG(info) << "Done.";
      if (zero) {
        LOG(info) << "Zeroing...";
        region.Zero();
        LOG(info) << "Done.";
      }
    }
  }

  bool CheckPresence()
  {
    for (const auto& sc : segmentCfgs) {
      if (!(fair::mq::shmem::Monitor::SegmentIsPresent(fair::mq::shmem::ShmId{shmId}, sc.id))) {
        return false;
      }
    }
    for (const auto& rc : regionCfgs) {
      if (!(fair::mq::shmem::Monitor::RegionIsPresent(fair::mq::shmem::ShmId{shmId}, rc.id.value()))) {
        return false;
      }
    }
    return true;
  }

  void ResetContent()
  {
    fair::mq::shmem::Monitor::ResetContent(fair::mq::shmem::ShmId{shmId}, segmentCfgs, regionCfgs);
  }

  void FullReset()
  {
    segments.clear();
    regions.clear();
    fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
  }

  ~ShmManager()
  {
    // clean all segments, regions and any other shmem objects belonging to this shmId
    fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
  }

  std::string shmId;
  map<uint16_t, fair::mq::shmem::Segment> segments;
  map<uint16_t, unique_ptr<fair::mq::shmem::UnmanagedRegion>> regions;
  std::vector<fair::mq::shmem::SegmentConfig> segmentCfgs;
  std::vector<fair::mq::RegionConfig> regionCfgs;
};

int main(int argc, char** argv)
{
  fair::Logger::SetConsoleColor(true);

  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);
  signal(SIGUSR1, resetContentHandler);

  try {
    bool nozero = false;
    bool checkPresence = true;
    uint64_t shmId = 0;
    vector<string> segments;
    vector<string> regions;

    options_description desc("Options");
    desc.add_options()(
      "shmid", value<uint64_t>(&shmId)->required(), "Shm id")(
      "segments", value<vector<string>>(&segments)->multitoken()->composing(), "Segments, as <id>,<size>,<numaid> <id>,<size>,<numaid> <id>,<size>,<numaid> ... (numaid: -2 disabled, -1 interleave, >=0 node)")(
      "regions", value<vector<string>>(&regions)->multitoken()->composing(), "Regions, as <id>,<size> <id>,<size>,<numaid> <id>,<size>,<numaid> ...")(
      "nozero", value<bool>(&nozero)->default_value(false)->implicit_value(true), "Do not zero segments after initialization")(
      "check-presence", value<bool>(&checkPresence)->default_value(true)->implicit_value(true), "Check periodically if configured segments/regions are still present, and cleanup and leave if they are not")(
      "help,h", "Print help");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      LOG(info) << "ShmManager"
                << "\n"
                << desc;
      return 0;
    }

    notify(vm);

    ShmManager shmManager(shmId, segments, regions, !nozero);

    std::thread resetContentThread([&shmManager]() {
      while (!gStopping) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (gResetContent == 1) {
          LOG(info) << "Resetting content for shmId " << shmManager.shmId;
          shmManager.ResetContent();
          gResetContent = 0;
          LOG(info) << "Done resetting content for shmId " << shmManager.shmId;
        }
      }
    });

    if (checkPresence) {
      while (!gStopping) {
        if (shmManager.CheckPresence() == false) {
          LOG(error) << "Failed to find segments, exiting.";
          gStopping = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    if (resetContentThread.joinable()) {
      resetContentThread.join();
    }

    LOG(info) << "stopping.";
  } catch (exception& e) {
    LOG(error) << "Exception reached the top of main: " << e.what() << ", exiting";
    return 2;
  }

  return 0;
}
