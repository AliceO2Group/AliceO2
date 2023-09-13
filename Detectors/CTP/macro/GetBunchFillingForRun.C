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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsCTP/Configuration.h>
#endif
using namespace o2::ctp;

void GetBunchFillingForRun(bool test = 1)
{
  if (test == 0) {
    return;
  }
  o2::ccdb::CcdbApi api;
  api.init("http://alice-ccdb.cern.ch"); // alice-ccdb.cern.ch
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  int runNumber = 526926;
  std::string fillNumber = "8238";
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  std::string srun = std::to_string(runNumber);
  map<string, string> metadata; // can be empty
  // metadata["runNumber"] = srun;
  metadata["fillNumber"] = fillNumber;
  auto lhcifdata = ccdbMgr.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp, metadata);
  auto bfilling = lhcifdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  for (auto const& bc : bcs) {
    std::cout << bc << std::endl;
  }
}
