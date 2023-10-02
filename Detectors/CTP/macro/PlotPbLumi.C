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

/// \file TestCTPScalers.C
/// \brief create CTP scalers, test it and add to database
/// \author Roman Lietava
// root -b -q "GetScalers.C(\"519499\", 1656286373953)"
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Scalers.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
void PlotPbLumi(int runNumber, int fillN, std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  //
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) /2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  //
  std::string sfill = std::to_string(fillN);
  std::map<string,string> metadata;
  metadata["fillNumber"] = sfill;
  auto lhcifdata = ccdbMgr.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp, metadata);
  auto bfilling = lhcifdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  std::cout << "Number of interacting bc:" << bcs.size() << std::endl;
  //
  std::string srun = std::to_string(runNumber);
  metadata.clear(); // can be empty
  metadata["runNumber"] = srun;
  ccdbMgr.setURL("http://ccdb-test.cern.ch:8080");
  auto scl = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
  if (scl == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    return;
  }
  //
  //CTPConfiguration ctpcfg;
  scl->convertRawToO2();
  std::vector<CTPScalerRecordO2> recs = scl->getScalerRecordO2();
  std::cout << "TVX,TSC,TCE,ZNC:" << std::endl;
  scl->printInputRateAndIntegral(3);
  scl->printInputRateAndIntegral(4);
  scl->printInputRateAndIntegral(5);
  scl->printInputRateAndIntegral(26);
  std::cout << " TVX,TVX&TCE,TVX&TSC,TVX&VCH:" << std::endl;
  for(int i = 0; i<10; i++) {
    scl->printClassBRateAndIntegral(i+1);
  }
  //scl.printLMBRateVsT();
  int n = recs.size()-1;
  std::cout << "size n:" << n << std::endl;
  std::cout << "size:" << n << ":" << recs[0].scalersInps[26] << ":" << recs[n].scalersInps[26] << std::endl;
  int itsc =  4;
  Double_t x[n-1],znc[n-1];
  for(int i = 0; i < n; i++) {
    x[i] = (double_t)(recs[i+1].intRecord.orbit);
    double_t tt = (double_t)(recs[i+1].intRecord.orbit - recs[i].intRecord.orbit );
    tt = tt*88e-6;
    znc[i] = (double_t)(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])/28./tt;
    auto had = recs[i+1].scalers[itsc+1].lmBefore - recs[i].scalers[itsc+1].lmBefore;
    had += recs[i+1].scalers[itsc].lmBefore - recs[i].scalers[itsc].lmBefore;
    //auto had = recs[i+1].scalers[itsc].lmBefore - recs[i].scalers[itsc].lmBefore;
    double_t rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    //znc[i] = rat;
  }
  TGraph *gr1 = new TGraph(n,x,znc);
  gr1->SetTitle("R=ZNC/28 rate [Hz]; time[Orbit]; R");
  //gr1->SetTitle("R=(TSC+TCE)*TVTX*B*28/ZNC; time[Orbit]; R");
  //gr1->SetTitle("R=(TCE)*TVTX*B*28/ZNC; time[Orbit]; R");
  TCanvas *c1 = new TCanvas("c1",srun.c_str(),200,10,600,400);
  gr1->Draw("A*");
}
