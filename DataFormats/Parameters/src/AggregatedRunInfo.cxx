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

/// \file AggregatedRunInfo.cxx
/// \author sandro.wenzel@cern.ch

#include "DataFormatsParameters/AggregatedRunInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"

using namespace o2::parameters;

o2::parameters::AggregatedRunInfo AggregatedRunInfo::buildAggregatedRunInfo(o2::ccdb::CCDBManagerInstance& ccdb, int runnumber)
{
  // TODO: could think about caching results per runnumber to
  // avoid going to CCDB multiple times ---> but should be done inside the CCDBManagerInstance

  // we calculate the first orbit of a run based on sor (start-of-run) and eor
  // we obtain these by calling getRunDuration
  auto [sor, eor] = ccdb.getRunDuration(runnumber);

  // determine a good timestamp to query OrbitReset for this run
  // --> the middle of the run is very appropriate and safer than just sor
  auto run_mid_timestamp = sor + (eor - sor) / 2;

  // query the time of the orbit reset (when orbit is defined to be 0)
  auto ctpx = ccdb.getForTimeStamp<std::vector<Long64_t>>("CTP/Calib/OrbitReset", run_mid_timestamp);
  int64_t tsOrbitReset = (*ctpx)[0]; // us

  // get timeframe length from GRPECS
  std::map<std::string, std::string> metadata;
  metadata["runNumber"] = Form("%d", runnumber);
  auto grpecs = ccdb.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", run_mid_timestamp, metadata);
  auto nOrbitsPerTF = grpecs->getNHBFPerTF();

  // calculate SOR orbit
  int64_t orbitSOR = (sor * 1000 - tsOrbitReset) / o2::constants::lhc::LHCOrbitMUS;
  int64_t orbitEOR = (eor * 1000 - tsOrbitReset) / o2::constants::lhc::LHCOrbitMUS;

  // adjust to the nearest TF edge to satisfy condition (orbitSOR % nOrbitsPerTF == 0)
  orbitSOR = orbitSOR / nOrbitsPerTF * nOrbitsPerTF;

  return AggregatedRunInfo{runnumber, sor, eor, nOrbitsPerTF, tsOrbitReset, orbitSOR, orbitEOR};
}
