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

using namespace std;
using namespace boost::program_options;

namespace
{
volatile sig_atomic_t gStopping = 0;
}

void signalHandler(int /* signal */)
{
  gStopping = 1;
}

struct ShmManager {
  ShmManager(uint64_t _shmId, const vector<string>& _segments, const vector<string>& _regions, bool zero = true)
    : shmId(fair::mq::shmem::makeShmIdStr(_shmId))
  {
    AddSegments(_segments, zero);
    AddRegions(_regions, zero);
  }

  void AddSegments(const vector<string>& _segments, bool zero)
  {
    for (const auto& s : _segments) {
      vector<string> conf;
      boost::algorithm::split(conf, s, boost::algorithm::is_any_of(","));
      if (conf.size() != 2) {
        LOG(error) << "incorrect format for --segments. Expecting pairs of <id>,<size>.";
        fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
        throw runtime_error("incorrect format for --segments. Expecting pairs of <id>,<size>.");
      }
      uint16_t id = stoi(conf.at(0));
      uint64_t size = stoull(conf.at(1));
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
      if (conf.size() != 2) {
        LOG(error) << "incorrect format for --regions. Expecting pairs of <id>,<size>.";
        fair::mq::shmem::Monitor::Cleanup(fair::mq::shmem::ShmId{shmId});
        throw runtime_error("incorrect format for --regions. Expecting pairs of <id>,<size>.");
      }
      uint16_t id = stoi(conf.at(0));
      uint64_t size = stoull(conf.at(1));
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

  void ResetContent()
  {
    fair::mq::shmem::Monitor::ResetContent(fair::mq::shmem::ShmId{shmId});
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
};

int main(int argc, char** argv)
{
  fair::Logger::SetConsoleColor(true);

  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  try {
    bool nozero = false;
    uint64_t shmId = 0;
    vector<string> segments;
    vector<string> regions;

    options_description desc("Options");
    desc.add_options()(
      "shmid", value<uint64_t>(&shmId)->required(), "Shm id")(
      "segments", value<vector<string>>(&segments)->multitoken()->composing(), "Segments, as <id>,<size> <id>,<size> <id>,<size> ...")(
      "regions", value<vector<string>>(&regions)->multitoken()->composing(), "Regions, as <id>,<size> <id>,<size> <id>,<size> ...")(
      "nozero", value<bool>(&nozero)->default_value(false)->implicit_value(true), "Do not zero segments after initialization")(
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

    while (!gStopping) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    LOG(info) << "stopping.";
  } catch (exception& e) {
    LOG(error) << "Exception reached the top of main: " << e.what() << ", exiting";
    return 2;
  }

  return 0;
}
