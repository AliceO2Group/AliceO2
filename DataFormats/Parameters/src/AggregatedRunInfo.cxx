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
#include <map>

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
  bool oldFatalState = ccdb.getFatalWhenNull();
  ccdb.setFatalWhenNull(false);
  auto ctp_first_run_orbit = ccdb.getForTimeStamp<std::vector<Long64_t>>("CTP/Calib/FirstRunOrbit", run_mid_timestamp);
  ccdb.setFatalWhenNull(oldFatalState);
  return buildAggregatedRunInfo(runnumber, sor, eor, tsOrbitReset, grpecs, ctp_first_run_orbit);
}

o2::parameters::AggregatedRunInfo AggregatedRunInfo::buildAggregatedRunInfo(int runnumber, long sorMS, long eorMS, long orbitResetMUS, const o2::parameters::GRPECSObject* grpecs, const std::vector<Long64_t>* ctfFirstRunOrbitVec)
{
  auto nOrbitsPerTF = grpecs->getNHBFPerTF();
  // calculate SOR/EOR orbits
  int64_t orbitSOR = -1;
  if (ctfFirstRunOrbitVec && ctfFirstRunOrbitVec->size() >= 3) { // if we have CTP first run orbit available, we should use it
    int64_t creation_timeIGNORED = (*ctfFirstRunOrbitVec)[0];    // do not use CTP start of run time!
    int64_t ctp_run_number = (*ctfFirstRunOrbitVec)[1];
    int64_t ctp_orbitSOR = (*ctfFirstRunOrbitVec)[2];
    if (creation_timeIGNORED == -1 && ctp_run_number == -1 && ctp_orbitSOR == -1) {
      LOGP(warn, "Default dummy CTP/Calib/FirstRunOrbit was provides, ignoring");
    } else if (ctp_run_number == runnumber) {
      orbitSOR = ctp_orbitSOR;
      auto sor_new = (int64_t)((orbitResetMUS + ctp_orbitSOR * o2::constants::lhc::LHCOrbitMUS) / 1000.);
      if (sor_new != sorMS) {
        LOGP(warn, "Adjusting SOR from {} to {}", sorMS, sor_new);
        sorMS = sor_new;
      }
    } else {
      LOGP(error, "AggregatedRunInfo: run number inconsistency found (asked: {} vs CTP found: {}, ignoring", runnumber, ctp_run_number);
    }
  }
  int64_t orbitEOR = (eorMS * 1000 - orbitResetMUS) / o2::constants::lhc::LHCOrbitMUS;
  if (runnumber > 523897) { // condition was introduced starting from LHC22o
    orbitEOR = orbitEOR / nOrbitsPerTF * nOrbitsPerTF;
  }
  if (orbitSOR < 0) { // extract from SOR
    orbitSOR = (sorMS * 1000 - orbitResetMUS) / o2::constants::lhc::LHCOrbitMUS;
    if (runnumber > 523897) {
      orbitSOR = (orbitSOR / nOrbitsPerTF + 1) * nOrbitsPerTF;
    }
  }
  return AggregatedRunInfo{runnumber, sorMS, eorMS, nOrbitsPerTF, orbitResetMUS, orbitSOR, orbitEOR, grpecs};
}
