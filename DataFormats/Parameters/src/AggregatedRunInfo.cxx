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
  int64_t orbitSOR = (sorMS * 1000 - orbitResetMUS) / o2::constants::lhc::LHCOrbitMUS;
  int64_t orbitEOR = (eorMS * 1000 - orbitResetMUS) / o2::constants::lhc::LHCOrbitMUS;
  // adjust to the nearest TF edge to satisfy condition (orbitSOR % nOrbitsPerTF == 0)
  orbitSOR = (orbitSOR / nOrbitsPerTF + 1) * nOrbitsPerTF; // +1 to choose the safe boundary ... towards run middle
  orbitEOR = orbitEOR / nOrbitsPerTF * nOrbitsPerTF;
  // temporary map of orbit shifts for runs <=LHC22m (while waiting for complete run list from CTP/Calib/FirstRunOrbit)
  std::map<int, int> mapOrbitShift;
  mapOrbitShift[517619] = 109;
  mapOrbitShift[517620] = 109;
  mapOrbitShift[517623] = 109;
  mapOrbitShift[517677] = 127;
  mapOrbitShift[517678] = 127;
  mapOrbitShift[517679] = 127;
  mapOrbitShift[517685] = 127;
  mapOrbitShift[517690] = 127;
  mapOrbitShift[517693] = 127;
  mapOrbitShift[517737] = 127;
  mapOrbitShift[517748] = 127;
  mapOrbitShift[517751] = 127;
  mapOrbitShift[517753] = 127;
  mapOrbitShift[517758] = 127;
  mapOrbitShift[517767] = 127;
  mapOrbitShift[518541] = 40;
  mapOrbitShift[518543] = 92;
  mapOrbitShift[518546] = 124;
  mapOrbitShift[518547] = 47;
  mapOrbitShift[519041] = 59;
  mapOrbitShift[519043] = 59;
  mapOrbitShift[519045] = 59;
  mapOrbitShift[519497] = 86;
  mapOrbitShift[519498] = 86;
  mapOrbitShift[519499] = 86;
  mapOrbitShift[519502] = 86;
  mapOrbitShift[519503] = 86;
  mapOrbitShift[519504] = 86;
  mapOrbitShift[519506] = 86;
  mapOrbitShift[519507] = 86;
  mapOrbitShift[519903] = 62;
  mapOrbitShift[519904] = 62;
  mapOrbitShift[519905] = 62;
  mapOrbitShift[519906] = 62;
  mapOrbitShift[520259] = 76;
  mapOrbitShift[520294] = 76;
  mapOrbitShift[520471] = 46;
  mapOrbitShift[520472] = 46;
  mapOrbitShift[520473] = 46;
  mapOrbitShift[523142] = 127;
  mapOrbitShift[523148] = 127;
  mapOrbitShift[523182] = 127;
  mapOrbitShift[523186] = 127;
  mapOrbitShift[523298] = 28;
  mapOrbitShift[523306] = 28;
  mapOrbitShift[523308] = 28;
  mapOrbitShift[523309] = 28;
  mapOrbitShift[523397] = 110;
  mapOrbitShift[523399] = 110;
  mapOrbitShift[523401] = 110;
  mapOrbitShift[523441] = 117;
  mapOrbitShift[523541] = 103;
  mapOrbitShift[523559] = 103;
  mapOrbitShift[523669] = 39;
  mapOrbitShift[523671] = 39;
  mapOrbitShift[523677] = 39;
  mapOrbitShift[523728] = 113;
  mapOrbitShift[523731] = 113;
  mapOrbitShift[523779] = 41;
  mapOrbitShift[523783] = 41;
  mapOrbitShift[523786] = 41;
  mapOrbitShift[523788] = 41;
  mapOrbitShift[523789] = 41;
  mapOrbitShift[523792] = 41;
  mapOrbitShift[523797] = 41;
  mapOrbitShift[523821] = 36;
  mapOrbitShift[523897] = 38;
  if (mapOrbitShift.find(runnumber) != mapOrbitShift.end()) {
    orbitSOR += mapOrbitShift[runnumber];
    orbitEOR += mapOrbitShift[runnumber];
  }

  if (ctfFirstRunOrbitVec && ctfFirstRunOrbitVec->size() >= 3) { // if we have CTP first run orbit available, we should use it
    int64_t creation_timeIGNORED = (*ctfFirstRunOrbitVec)[0];    // do not use CTP start of run time!
    int64_t ctp_run_number = (*ctfFirstRunOrbitVec)[1];
    int64_t ctp_orbitSOR = (*ctfFirstRunOrbitVec)[2];
    if (creation_timeIGNORED == -1 && ctp_run_number == -1 && ctp_orbitSOR == -1) {
      LOGP(warn, "Default dummy CTP/Calib/FirstRunOrbit was provide, ignoring");
    } else if (ctp_run_number == runnumber) { // overwrite orbitSOR
      if (ctp_orbitSOR != orbitSOR) {
        LOGP(warn, "The calculated orbitSOR {} differs from CTP orbitSOR {}", orbitSOR, ctp_orbitSOR);
        // reasons for this is different unit of time storage in RunInformation (ms) and orbitReset (us), etc.
        // so we need to adjust the SOR timings to be consistent
        auto sor_new = (int64_t)((orbitResetMUS + ctp_orbitSOR * o2::constants::lhc::LHCOrbitMUS) / 1000.);
        if (sor_new != sorMS) {
          LOGP(warn, "Adjusting SOR from {} to {}", sorMS, sor_new);
          sorMS = sor_new;
        }
      }
      orbitSOR = ctp_orbitSOR;
    } else {
      LOGP(error, "AggregatedRunInfo: run number inconsistency found (asked: {} vs CTP found: {}, ignoring", runnumber, ctp_run_number);
    }
  }
  return AggregatedRunInfo{runnumber, sorMS, eorMS, nOrbitsPerTF, orbitResetMUS, orbitSOR, orbitEOR, grpecs};
}
