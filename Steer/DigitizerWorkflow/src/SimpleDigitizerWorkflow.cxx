// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/runDataProcessing.h"

#include "SimReaderSpec.h"
#include "CollisionTimePrinter.h"

// for TPC
#include "TPCDriftTimeDigitizerSpec.h"
#include "TPCDigitRootWriterSpec.h"
#include "TPCBase/Sector.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>

using namespace o2::framework;

// extract num TPC lanes, a lane is a streaming line of processors (digitizer-clusterizer-etc)
// by default this will be std::max(the number of physical cores, numberofsectors)
// as a temporary means to fully use a machine and as a way to play with different topologies
int getNumTPCLanes(std::vector<int> const& sectors)
{
  int lanes = std::thread::hardware_concurrency();
  if (const char* f = std::getenv("DPL_TPC_LANES")) {
    lanes = atoi(f);
  }
  // crosscheck with sectors
  return std::min(lanes, (int)sectors.size());
}

void extractTPCSectors(std::vector<int>& sectors)
{
  if (const char* f = std::getenv("DPL_TPC_SECTORS")) {
    // we expect them to be , separated
    std::stringstream ss(f);
    std::vector<std::string> stringtokens;
    while (ss.good()) {
      std::string substr;
      getline(ss, substr, ',');
      stringtokens.push_back(substr);
    }
    // now try to convert each token to int
    for (auto& token : stringtokens) {
      try {
        auto s = std::stoi(token);
        sectors.emplace_back(s);
      } catch (std::invalid_argument e) {
      }
    }
  } else {
    // all sectors otherwise by default
    for (int s = 0; s < o2::TPC::Sector::MAXSECTOR; ++s) {
      sectors.emplace_back(s);
    }
  }
}

bool wantCollisionTimePrinter()
{
  if (const char* f = std::getenv("DPL_COLLISION_TIME_PRINTER")) {
    return true;
  }
  return false;
}

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  int fanoutsize = 0;
  if (wantCollisionTimePrinter()) {
    specs.emplace_back(o2::steer::getCollisionTimePrinter(fanoutsize++));
  }

  // keeps track of which subchannels correspond to tpc channels
  auto tpclanes = std::make_shared<std::vector<int>>();
  // keeps track of which tpc sectors to process
  auto tpcsectors = std::make_shared<std::vector<int>>();
  extractTPCSectors(*tpcsectors.get());
  auto lanes = getNumTPCLanes(*tpcsectors.get());

  for (int l = 0; l < lanes; ++l) {
    specs.emplace_back(o2::steer::getTPCDriftTimeDigitizer(fanoutsize));
    tpclanes->emplace_back(fanoutsize); // this records that TPC is "listening under this subchannel"
    fanoutsize++;
  }

  // for writing digits to disc
  specs.emplace_back(o2::TPC::getTPCDigitRootWriterSpec(lanes));
  specs.emplace_back(o2::steer::getSimReaderSpec(fanoutsize, tpcsectors, tpclanes));
  return specs;
}
