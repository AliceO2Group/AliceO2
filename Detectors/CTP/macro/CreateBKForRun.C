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
#include <iomanip>
#include <TMath.h>
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsCTP/Configuration.h>
#include <DataFormatsParameters/GRPLHCIFData.h>
#endif
using namespace o2::ctp;

void CreateBKForRun()
{
  std::vector<int> runs = {558124,558126,558215,558217,558221,558244,558247};
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";
  //
  std::string filename = "BKcounters.txt";
  std::ofstream outfile(filename);
  if (!outfile) {
    Error("", "Failed to open file %s", filename.c_str());
    return;
  }
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  for(auto const& runNumber : runs) {
    auto soreor = ccdbMgr.getRunDuration(runNumber);
    uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
    std::cout << runNumber << " Timestamp:" << timeStamp << std::endl;
    //
    std::string srun = std::to_string(runNumber);
    std::map<string, string> metadata;
    metadata["runNumber"] = srun;
    auto ctpscalers = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
    if (ctpscalers == nullptr) {
      LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    }
    auto ctpcfg = ccdbMgr.getSpecific<CTPConfiguration>(mCCDBPathCTPConfig, timeStamp, metadata);
    if (ctpcfg == nullptr) {
      LOG(info) << "CTPRunConfig not in database, timestamp:" << timeStamp;
    }
    //
    ctpscalers->convertRawToO2();
    std::vector<CTPClass>& ctpcls = ctpcfg->getCTPClasses();
    std::vector<int> clslist = ctpcfg->getTriggerClassList();
    auto times = ctpscalers->getTimeLimit();
    for (size_t i = 0; i < clslist.size(); i++) {
      // std::cout << i << " " << ctpcls[i].name ;
      std::array<uint64_t, 7> cnts = ctpscalers->getIntegralForClass(i);
      if (clslist[i] != (int)cnts[0]) {
        LOG(fatal) << "cls list incompatible with counters";
      }
      std::cout << std::setw(21) << ctpcls[cnts[0]].name;
      outfile << runNumber << ", " << ctpcls[i].name << ", " << std::get<1>(times)/1000;
      for (int j = 1; j < 7; j++) {
        // std::cout << std::setw(21) << " " << cnts[j];
        std::cout << ", " << cnts[j];
        outfile << ", " << cnts[j];
      }
      std::cout << std::endl;
      outfile << std::endl;
    }
  }
  // ctpscalers->printFromZero(std::cout);
  outfile.close();
}
