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
#include <DataFormatsCTP/CTPRateFetcher.h>
#endif
using namespace o2::ctp;

void TestFetcher(int runNumber = 557251)
{
  auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
  std::pair<int64_t, int64_t> pp = ccdb.getRunDuration(runNumber);
  long ts = pp.first + 60;
  std::cout << "Run duration:" << pp.first << " " << pp.second << std::endl;
  // Opening run
  CTPRateFetcher fetcher;
  fetcher.setupRun(runNumber, &ccdb, ts, 0);
  ccdb.setURL("http://ali-qcdb-gpn.cern.ch:8083/");
  std::string QCDBPathCTPScalers = "qc/CTP/Scalers";
  map<string, string> metadata; // can be empty
  std::string run = std::to_string(runNumber);
  metadata["runNumber"] = run;
  CTPRunScalers* ctpscalers = ccdb.getSpecific<CTPRunScalers>(QCDBPathCTPScalers, ts, metadata);
  auto tt = ctpscalers->getTimeLimitFromRaw();
  std::cout << "1st scalers duration:" << tt.first << " " << tt.second << std::endl;
  fetcher.updateScalers(*ctpscalers);
  auto rate = fetcher.fetchNoPuCorr(&ccdb, ts, runNumber, "T0VTX");
  std::cout << "1st rate:" << rate << std::endl;
  // Running on the same run
  ts = ts + 5 * 1000 * 3600;
  ctpscalers = ccdb.getSpecific<CTPRunScalers>(QCDBPathCTPScalers, ts, metadata);
  std::cout << "Later scalers duration:" << tt.first << " " << tt.second << std::endl;
  fetcher.updateScalers(*ctpscalers);
  rate = fetcher.fetchNoPuCorr(&ccdb, ts, runNumber, "T0VTX");
  std::cout << "Later rate:" << rate << std::endl;
}
