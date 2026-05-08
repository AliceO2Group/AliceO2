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

/// \file GRPECSObject.h
/// \brief Header of the AggregatedRunInfo struct
/// \author ruben.shahoyan@cern.ch sandro.wenzel@cern.ch

#ifndef ALICEO2_DATA_AGGREGATEDRUNINFO_H_
#define ALICEO2_DATA_AGGREGATEDRUNINFO_H_

#include <cstdint>
#include "CCDB/BasicCCDBManager.h"

namespace o2::parameters
{

class GRPECSObject;
class GRPLHCIFData;

/// Composite struct where one may collect important global properties of data "runs"
/// aggregated from various sources (GRPECS, RunInformation CCDB entries, etc.).
/// Also offers the authoritative algorithms to collect these information for easy reuse
/// across various algorithms (anchoredMC, analysis, ...)
struct AggregatedRunInfo {
  int runNumber = 0;       // run number
  int64_t sor = 0;         // best known timestamp for the start of run
  int64_t eor = 0;         // best known timestamp for end of run
  int64_t orbitsPerTF = 0; // number of orbits per TF (takes precedence over that in GRPECS)
  int64_t orbitReset = 0;  // timestamp of orbit reset before run
  int64_t orbitSOR = 0;    // orbit when run starts after orbit reset
  int64_t orbitEOR = 0;    // orbit when run ends after orbit reset

  // we may have pointers to actual data source objects GRPECS, ...
  const o2::parameters::GRPECSObject* grpECS = nullptr; // pointer to GRPECSobject (fetched during struct building)
  const o2::parameters::GRPLHCIFData* grpLHC = nullptr;

  static AggregatedRunInfo buildAggregatedRunInfo(int runnumber, long sorMS, long eorMS, long orbitResetMUS, const o2::parameters::GRPECSObject* grpecs, const std::vector<Long64_t>* ctfFirstRunOrbitVec, const o2::parameters::GRPLHCIFData* grplhcif = nullptr);

  // fills and returns AggregatedRunInfo for a given data run number.
  static AggregatedRunInfo buildAggregatedRunInfo_DATA(o2::ccdb::CCDBManagerInstance& ccdb, int runnumber);

  // Returns the meta-data (MCProdInfo) associated to production lpm_prod_tag (performed by username)
  static std::map<std::string, std::string> getMCProdInfo(o2::ccdb::CCDBManagerInstance& ccdb, int runnumber,
                                                          std::string const& lpm_prod_tag, std::string const& username = "aliprod");

  // function that adjusts with values from MC
  void adjust_from_MC(o2::ccdb::CCDBManagerInstance& ccdb, int run_number, std::string const& lpm_prod_tag, std::string const& username = "aliprod");

  // Fills and returns AggregatedRunInfo for a given run number.
  // If a non-empty lpm_prod_tag is given, it will potentially override values with specifics from a
  // MC production identified by that tag and username.
  static AggregatedRunInfo buildAggregatedRunInfo(o2::ccdb::CCDBManagerInstance& ccdb,
                                                  int runnumber,
                                                  std::string const& lpm_prod_tag = "",
                                                  std::string const& username = "aliprod");

  ClassDefNV(AggregatedRunInfo, 1);
};

} // namespace o2::parameters

#endif
