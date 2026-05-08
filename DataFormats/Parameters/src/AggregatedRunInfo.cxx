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
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"
#include <map>

using namespace o2::parameters;

o2::parameters::AggregatedRunInfo AggregatedRunInfo::buildAggregatedRunInfo_DATA(o2::ccdb::CCDBManagerInstance& ccdb, int runnumber)
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
  auto grplhcif = ccdb.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", run_mid_timestamp); // no run metadata here
  bool oldFatalState = ccdb.getFatalWhenNull();
  ccdb.setFatalWhenNull(false);
  auto ctp_first_run_orbit = ccdb.getForTimeStamp<std::vector<Long64_t>>("CTP/Calib/FirstRunOrbit", run_mid_timestamp);
  ccdb.setFatalWhenNull(oldFatalState);
  return buildAggregatedRunInfo(runnumber, sor, eor, tsOrbitReset, grpecs, ctp_first_run_orbit, grplhcif);
}

o2::parameters::AggregatedRunInfo AggregatedRunInfo::buildAggregatedRunInfo(int runnumber, long sorMS, long eorMS, long orbitResetMUS, const o2::parameters::GRPECSObject* grpecs, const std::vector<Long64_t>* ctfFirstRunOrbitVec, const o2::parameters::GRPLHCIFData* grplhcif)
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
  return AggregatedRunInfo{runnumber, sorMS, eorMS, nOrbitsPerTF, orbitResetMUS, orbitSOR, orbitEOR, grpecs, grplhcif};
}

namespace
{

// get path where to find MC production info
std::string getFullPath_MC(std::string const& username, std::string const& lpm_prod_tag)
{
  // construct the path where to lookup
  std::string path = "/Users/" + std::string(1, username[0]) + "/" + username;
  std::string fullpath = path + "/" + "MCProdInfo/" + lpm_prod_tag;
  return fullpath;
}

} // namespace

std::map<std::string, std::string> AggregatedRunInfo::getMCProdInfo(o2::ccdb::CCDBManagerInstance& ccdb,
                                                                    int run_number,
                                                                    std::string const& lpm_prod_tag,
                                                                    std::string const& username)
{
  std::map<std::string, std::string> metaDataFilter;
  metaDataFilter["LPMProductionTag"] = lpm_prod_tag;

  // fetch the meta information for MC productions
  auto header_data = ccdb.getCCDBAccessor().retrieveHeaders(getFullPath_MC(username, lpm_prod_tag), metaDataFilter, run_number);
  return header_data;
}

void AggregatedRunInfo::adjust_from_MC(o2::ccdb::CCDBManagerInstance& ccdb,
                                       int run_number,
                                       std::string const& lpm_prod_tag,
                                       std::string const& username)
{
  auto header_data = AggregatedRunInfo::getMCProdInfo(ccdb, run_number, lpm_prod_tag, username);

  // adjust timeframe length if we find entry for MC production
  auto iter = header_data.find("OrbitsPerTF");
  if (iter != header_data.end()) {
    auto mc_orbitsPerTF = std::stoi(iter->second);
    if (mc_orbitsPerTF != orbitsPerTF) {
      LOG(info) << "Adjusting OrbitsPerTF from " << orbitsPerTF << " to " << mc_orbitsPerTF << " based on differing MC info";
      orbitsPerTF = mc_orbitsPerTF;
    }
  } else {
    LOG(warn) << "No OrbitsPerTF information found for MC production " << lpm_prod_tag << " and run number " << run_number;
  }
}

AggregatedRunInfo AggregatedRunInfo::buildAggregatedRunInfo(o2::ccdb::CCDBManagerInstance& ccdb, int run_number, std::string const& lpm_prod_tag, std::string const& username)
{
  // (a) lookup the AggregatedRunInfo for the data run
  // (b) modify/overwrite the info object with MC specific settings if lpm_prod_tag is given

  auto original_info = buildAggregatedRunInfo_DATA(ccdb, run_number);

  if (lpm_prod_tag.size() == 0) {
    return original_info;
  }

  // in this case we adjust the info from MC
  original_info.adjust_from_MC(ccdb, run_number, lpm_prod_tag, username);

  return original_info;
}
