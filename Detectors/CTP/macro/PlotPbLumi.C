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
{ //
  // what = 1: znc rate
  // what = 2: (TCE+TSC)/ZNC
  // what = 3: TCE/ZNC
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  // Timestamp
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  // Filling
  std::string sfill = std::to_string(fillN);
  std::map<string, string> metadata;
  metadata["fillNumber"] = sfill;
  auto lhcifdata = ccdbMgr.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp, metadata);
  auto bfilling = lhcifdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  int nbc = bcs.size();
  std::cout << "Number of interacting bc:" << nbc << std::endl;
  // Scalers
  std::string srun = std::to_string(runNumber);
  metadata.clear(); // can be empty
  metadata["runNumber"] = srun;
  ccdbMgr.setURL("http://ccdb-test.cern.ch:8080");
  auto scl = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
  if (scl == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    return;
  }
  scl->convertRawToO2();
  std::vector<CTPScalerRecordO2> recs = scl->getScalerRecordO2();
  //
  // CTPConfiguration ctpcfg;
  auto ctpcfg = ccdbMgr.getSpecific<CTPConfiguration>(mCCDBPathCTPConfig, timeStamp, metadata);
  if (ctpcfg == nullptr) {
    LOG(info) << "CTPRunConfig not in database, timestamp:" << timeStamp;
    return;
  }
  std::vector<int> clslist = ctpcfg->getTriggerClassList();
  // std::vector<uint32_t> clslist = scl->getClassIndexes();
  std::map<int, int> clsIndexToScaler;
  std::cout << "Classes:";
  int i = 0;
  for (auto const& cls : clslist) {
    std::cout << cls << " ";
    clsIndexToScaler[cls] = i;
    i++;
  }
  std::cout << std::endl;
  std::vector<CTPClass> ctpcls = ctpcfg->getCTPClasses();
  int tsc = 255;
  int tce = 255;
  int vch = 255;
  for (auto const& cls : ctpcls) {
    if (cls.name.find("CMTVXTSC-B-NOPF") != std::string::npos && tsc == 255) {
      int itsc = cls.getIndex();
      tsc = clsIndexToScaler[itsc];
      // tsc = scl->getScalerIndexForClass(itsc);
      std::cout << cls.name << ":" << tsc << ":" << itsc << std::endl;
    }
    if (cls.name.find("CMTVXTCE-B-NOPF-CRU") != std::string::npos) {
      int itce = cls.getIndex();
      tce = clsIndexToScaler[itce];
      // tce = scl->getScalerIndexForClass(itce);
      std::cout << cls.name << ":" << tce << ":" << itce << std::endl;
    }
    if (cls.name.find("CMTVXVCH-B-NOPF-CRU") != std::string::npos) {
      int ivch = cls.getIndex();
      vch = clsIndexToScaler[ivch];
      // vch = scl->getScalerIndexForClass(ivch);
      std::cout << cls.name << ":" << vch << ":" << ivch << std::endl;
    }
  }
  if (tsc == 255 || tce == 255 || vch == 255) {
    std::cout << " One of dcalers not available, check config to find alternative)" << std::endl;
    return;
  }
  //
  // Anal
  //
  // Times
  double_t frev = 11245;
  double_t time0 = recs[0].epochTime;
  double_t timeL = recs[recs.size() - 1].epochTime;
  double_t Trun = timeL - time0;
  double_t orbit0 = recs[0].intRecord.orbit;
  int n = recs.size() - 1;
  std::cout << " Run duration:" << Trun << " Scalers size:" << n + 1 << std::endl;
  Double_t x[n], znc[n], zncpp[n];
  Double_t tcetsctoznc[n], tcetoznc[n], vchtoznc[n];
  for (int i = 0; i < n; i++) {
    x[i] = (double_t)(recs[i + 1].intRecord.orbit + recs[i].intRecord.orbit) / 2. - orbit0;
    x[i] *= 88e-6;
    // x[i] = (double_t)(recs[i+1].epochTime + recs[i].epochTime)/2.;
    double_t tt = (double_t)(recs[i + 1].intRecord.orbit - recs[i].intRecord.orbit);
    tt = tt * 88e-6;
    //
    // std::cout << recs[i+1].scalersInps[25] << std::endl;
    double_t znci = (double_t)(recs[i + 1].scalersInps[25] - recs[i].scalersInps[25]);
    double_t mu = -TMath::Log(1. - znci / tt / nbc / frev);
    double_t zncipp = mu * nbc * frev;
    zncpp[i] = zncipp / 28.;
    znc[i] = znci / 28. / tt;
    //
    auto had = recs[i + 1].scalers[tce].lmBefore - recs[i].scalers[tce].lmBefore;
    // std::cout << recs[i+1].scalers[tce].lmBefore << std::endl;
    had += recs[i + 1].scalers[tsc].lmBefore - recs[i].scalers[tsc].lmBefore;
    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    tcetsctoznc[i] = (double_t)(had) / zncpp[i] / tt;
    had = recs[i + 1].scalers[tce].lmBefore - recs[i].scalers[tce].lmBefore;
    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    tcetoznc[i] = (double_t)(had) / zncpp[i] / tt;
    had = recs[i + 1].scalers[vch].lmBefore - recs[i].scalers[vch].lmBefore;
    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    vchtoznc[i] = (double_t)(had) / zncpp[i] / tt;
  }
  //
  gStyle->SetMarkerSize(0.5);
  TGraph* gr1 = new TGraph(n, x, znc);
  TGraph* gr2 = new TGraph(n, x, tcetsctoznc);
  TGraph* gr3 = new TGraph(n, x, tcetoznc);
  TGraph* gr4 = new TGraph(n, x, vchtoznc);
  gr1->SetMarkerStyle(20);
  gr2->SetMarkerStyle(21);
  gr3->SetMarkerStyle(23);
  gr4->SetMarkerStyle(23);
  gr1->SetTitle("R=ZNC/28 rate [Hz]; time[sec]; R");
  gr2->SetTitle("R=(TSC+TCE)*TVTX*B*28/ZNC; time[sec]; R");
  // gr2->GetHistogram()->SetMaximum(1.1);
  // gr2->GetHistogram()->SetMinimum(0.9);
  gr3->SetTitle("R=(TCE)*TVTX*B*28/ZNC; time[sec]; R");
  // gr3->GetHistogram()->SetMaximum(0.6);
  // gr3->GetHistogram()->SetMinimum(0.4);
  gr4->SetTitle("R=(VCH)*TVTX*B*28/ZNC; time[sec]; R");
  // gr4->GetHistogram()->SetMaximum(0.6);
  // gr4->GetHistogram()->SetMinimum(0.4);
  TCanvas* c1 = new TCanvas("c1", srun.c_str(), 200, 10, 800, 500);
  c1->Divide(2, 2);
  c1->cd(1);
  gr1->Draw("AP");
  c1->cd(2);
  gr2->Draw("AP");
  c1->cd(3);
  gr3->Draw("AP");
  c1->cd(4);
  gr4->Draw("AP");
  // getRate test:
  double tt = timeStamp / 1000.;
  std::pair<double, double> r1 = scl->getRateGivenT(tt, 25, 7);
  std::cout << "ZDC input getRateGivetT:" << r1.first / 28. << " " << r1.second / 28. << std::endl;
  std::pair<double, double> r2 = scl->getRateGivenT(tt, tce, 1);
  std::cout << "LM before TCE class getRateGivetT:" << r2.first << " " << r2.second << std::endl;
}
