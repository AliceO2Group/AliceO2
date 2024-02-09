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
//"http://ccdb-test.cern.ch:8080"
using namespace o2::ctp;
void PlotOrbit(int runNumber)
{ //
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  // Timestamp
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  // Scalers
  std::string srun = std::to_string(runNumber);
  std::map<string, string> metadata;
  metadata["runNumber"] = srun;
  // ccdbMgr.setURL("http://ccdb-test.cern.ch:8080");
  auto scl = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
  if (scl == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    return;
  }
  scl->convertRawToO2();
  std::vector<CTPScalerRecordO2> recs = scl->getScalerRecordO2();
  //
  //
  // Anal
  //
  // Times
  int64_t time0 = recs[0].epochTime;
  int64_t timeL = recs[recs.size() - 1].epochTime;
  double_t Trun = timeL - time0;
  // double_t orbit0 = recs[0].intRecord.orbit;
  int64_t orbit0 = scl->getOrbitLimit().first;
  int64_t orbitL = scl->getOrbitLimit().second;
  int n = recs.size() - 1;
  std::cout << " Run duration:" << Trun << " Scalers size:" << n + 1 << std::endl;
  Double_t x[n], orbit[n];
  // Double_t tcetsctoznc[n], tcetoznc[n], vchtoznc[n];
  for (int i = 0; i < n; i++) {
    // x[i] = i;
    x[i] = recs[i + 1].epochTime - time0;
    orbit[i] = recs[i + 1].intRecord.orbit - orbit0;
  }
  //
  gStyle->SetMarkerSize(0.5);
  TGraph* gr1 = new TGraph(n, x, orbit);
  // TGraph* gr2 = new TGraph(n, x, tcetsctoznc);
  // TGraph* gr3 = new TGraph(n, x, tcetoznc);
  // TGraph* gr4 = new TGraph(n, x, vchtoznc);
  gr1->SetMarkerStyle(20);
  // gr2->SetMarkerStyle(21);
  // gr3->SetMarkerStyle(23);
  // gr4->SetMarkerStyle(23);
  std::string title = "Orbit vs EpochTime for run " + srun + " ;EpochTime[s]; Orbit";
  gr1->SetTitle(title.c_str());
  // gr2->SetTitle("R=(TSC+TCE)*TVTX*B*28/ZNC; time[sec]; R");
  //  gr2->GetHistogram()->SetMaximum(1.1);
  //  gr2->GetHistogram()->SetMinimum(0.9);
  // gr3->SetTitle("R=(TCE)*TVTX*B*28/ZNC; time[sec]; R");
  //  gr3->GetHistogram()->SetMaximum(0.6);
  //  gr3->GetHistogram()->SetMinimum(0.4);
  // gr4->SetTitle("R=(VCH)*TVTX*B*28/ZNC; time[sec]; R");
  //  gr4->GetHistogram()->SetMaximum(0.6);
  //  gr4->GetHistogram()->SetMinimum(0.4);
  TCanvas* c1 = new TCanvas("c1", srun.c_str(), 200, 10, 800, 500);
  gr1->Draw("AP");
  std::cout << "epoch0:" << time0 << " epochL:" << timeL << " T:" << Trun << std::endl;
  std::cout << "orbit0:" << orbit0 << " orbitL:" << orbitL << " T:" << (orbitL - orbit0) * 88e-6 << std::endl;
}
