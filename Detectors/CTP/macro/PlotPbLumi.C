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

/// \file PlotPbLumi.C
/// \brief create CTP scalers, test it and add to database
/// \author Roman Lietava
// root "PLotPbLumi.C(519499)"
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Scalers.h"
#include "DataFormatsCTP/Configuration.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "TGraph.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
//
// sum = 0: TCE and TSC separatelly otherwise TCE and (TCE+TSC)
// qc = 0: takes scalers from CCDB (available only for finished runs) otherwise from QCCDB (available for active runs)
// t0-tlast: window in seconds counted from beginning of run
//
void PlotPbLumi(int runNumber = 572073, bool sum = 1, double cut = 0, bool qc = 0, Double_t t0 = 0., Double_t tlast = 0.)
{ //
  // PLots in one canvas
  // znc rate/28
  // R = (TCE+TSC)*TVX*B*/ZNC*28
  // R = TCE*TVX*B/ZNC*28
  // R = VCH*TVX*B/ZNC*28
  std::string ccdbHost = "http://alice-ccdb.cern.ch";
  std::string mCCDBPathCTPScalers = "/CTP/Calib/Scalers";
  std::string mCCDBPathCTPScalersQC = "qc/CTP/Scalers";
  std::string mCCDBPathCTPConfig = "CTP/Config/Config";
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  // Timestamp
  auto soreor = ccdbMgr.getRunDuration(runNumber);
  uint64_t timeStamp = (soreor.second - soreor.first) / 2 + soreor.first;
  std::cout << "Timestamp:" << timeStamp << std::endl;
  // Filling
  auto lhcifdata = ccdbMgr.getForRun<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", runNumber);
  // auto lhcifdata = ccdbMgr.getSpecific<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", timeStamp, metadata);
  if (!lhcifdata) {
    throw std::runtime_error("No GRPLHCIFData for run " + std::to_string(runNumber));
  }
  auto bfilling = lhcifdata->getBunchFilling();
  std::vector<int> bcs = bfilling.getFilledBCs();
  int nbc = bcs.size();
  std::cout << "Number of interacting bc:" << nbc << std::endl;
  // Scalers
  std::string srun = std::to_string(runNumber);
  std::map<string, string> metadata;
  metadata["runNumber"] = srun;
  CTPRunScalers* scl = nullptr;
  if (qc) {
    ccdbMgr.setURL("http://ali-qcdb-gpn.cern.ch:8083");
    scl = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalersQC, timeStamp, metadata);
  } else {
    scl = ccdbMgr.getSpecific<CTPRunScalers>(mCCDBPathCTPScalers, timeStamp, metadata);
  }
  if (scl == nullptr) {
    LOG(info) << "CTPRunScalers not in database, timestamp:" << timeStamp;
    return;
  }
  scl->convertRawToO2();
  std::vector<CTPScalerRecordO2> recs = scl->getScalerRecordO2();
  //
  // CTPConfiguration ctpcfg;
  ccdbMgr.setURL("http://alice-ccdb.cern.ch");
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
  int zncclsi = 255;
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
    if (cls.name.find("C1ZNC-B-NOPF-CRU") != std::string::npos) {
      int iznc = cls.getIndex();
      zncclsi = clsIndexToScaler[iznc];
      // vch = scl->getScalerIndexForClass(ivch);
      std::cout << cls.name << ":" << zncclsi << ":" << iznc << std::endl;
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
  //
  int i0 = 0;
  int ilast = 0;
  if (t0 != 0. || tlast != 0.) {
    for (int i = 0; i < n; i++) {
      double_t ttime = recs[i].epochTime - time0;
      if (!i0 && t0 < ttime) {
        i0 = i;
      }
      if (!ilast && tlast < ttime) {
        ilast = i;
      }
    }
  } else {
    ilast = n;
  }
  n = ilast - i0;
  std::cout << "i0:" << i0 << " ilast:" << ilast << std::endl;
  // Double_t x[n], znc[n], zncpp[n];
  std::vector<Double_t> xvec(n), zncvec(n), zncppvec(n), zncclassvec(n);
  Double_t* x = xvec.data();
  Double_t* znc = zncvec.data();
  Double_t* zncpp = zncppvec.data();
  Double_t* zncclass = zncclassvec.data();
  // Double_t tcetsctoznc[n], tcetoznc[n], vchtoznc[n];
  std::vector<Double_t> tcetsctozncvec(n), tcetozncvec(n), vchtozncvec(n);
  Double_t* tcetsctoznc = tcetsctozncvec.data();
  Double_t* tcetoznc = tcetozncvec.data();
  Double_t* vchtoznc = vchtozncvec.data();
  for (int i = i0; i < ilast; i++) {
    // for (int i = 30; i < 40; i++) {

    int iv = i - i0;
    x[iv] = (double_t)(recs[i + 1].intRecord.orbit + recs[i].intRecord.orbit) / 2. - orbit0;
    x[iv] *= 88e-6;
    // x[i] = (double_t)(recs[i+1].epochTime + recs[i].epochTime)/2.;
    double_t tt = (double_t)(recs[i + 1].intRecord.orbit - recs[i].intRecord.orbit);
    tt = tt * 88e-6;
    // std::cout << i << " " << iv << " " << tt << std::endl;
    //
    //  std::cout << recs[i+1].scalersInps[25] << std::endl;
    double_t znci = (double_t)(recs[i + 1].scalersInps[25] - recs[i].scalersInps[25]);
    double_t mu = -TMath::Log(1. - znci / tt / nbc / frev);
    double_t zncipp = mu * nbc * frev;
    zncpp[iv] = zncipp / 28.;
    znc[iv] = znci / 28. / tt;
    // znc class
    znci = recs[i + 1].scalers[zncclsi].l1Before - recs[i].scalers[zncclsi].l1Before;
    zncclass[iv] = znci / 28. / tt;
    // std::cout << znc[i]/zncclass[i] << std::endl;
    //
    double_t had = 0;
    if (sum) {
      had += recs[i + 1].scalers[tce].lmBefore - recs[i].scalers[tce].lmBefore;
    }
    double_t mutce = -TMath::Log(1. - had / tt / nbc / frev);
    // std::cout << recs[i+1].scalers[tce].lmBefore << std::endl;
    had += recs[i + 1].scalers[tsc].lmBefore - recs[i].scalers[tsc].lmBefore;
    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    if (zncpp[iv] > cut) {
      tcetsctoznc[iv] = (double_t)(had) / zncpp[iv] / tt;
    } else {
      tcetsctoznc[iv] = 0.;
    }
    had = recs[i + 1].scalers[tce].lmBefore - recs[i].scalers[tce].lmBefore;
    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    if (zncpp[iv] > cut) {
      tcetoznc[iv] = (double_t)(had) / zncpp[iv] / tt;
    } else {
      tcetoznc[iv] = 0.;
    }
    had = recs[i + 1].scalers[vch].lmBefore - recs[i].scalers[vch].lmBefore;
    double_t muvch = -TMath::Log(1. - had / tt / nbc / frev);

    // rat = (double_t)(had)/double_t(recs[i+1].scalersInps[25] - recs[i].scalersInps[25])*28;
    if (zncpp[iv] > cut) {
      vchtoznc[iv] = (double_t)(had) / zncpp[iv] / tt;
    } else {
      vchtoznc[iv] = 0.;
    }
    // std::cout << "muzdc:" << mu << " mu tce:" << mutce << " muvch:" << muvch << std::endl;
  }
  //
  gStyle->SetMarkerSize(0.5);
  TGraph* gr1 = new TGraph(n, x, znc);
  TGraph* gr11 = new TGraph(n, x, zncpp);    // PileuP corrected
  TGraph* gr12 = new TGraph(n, x, zncclass); // NOT PileuP corrected
  TGraph* gr2 = new TGraph(n, x, tcetsctoznc);
  TGraph* gr3 = new TGraph(n, x, tcetoznc);
  TGraph* gr4 = new TGraph(n, x, vchtoznc);
  gr1->SetMarkerStyle(20);
  gr11->SetMarkerStyle(20);
  gr12->SetMarkerStyle(20);
  gr11->SetMarkerColor(kRed);
  gr12->SetMarkerColor(kBlue);
  gr2->SetMarkerStyle(21);
  gr3->SetMarkerStyle(23);
  gr4->SetMarkerStyle(23);
  if (sum) {
    gr2->SetTitle("R=(TSC+TCE)*TVTX*B*28/ZNC; time[sec]; R");
  } else {
    gr2->SetTitle("R=(TSC)*TVTX*B*28/ZNC; time[sec]; R");
  }
  // gr2->GetHistogram()->SetMaximum(1.1);
  // gr2->GetHistogram()->SetMinimum(0.9);
  gr3->SetTitle("R=(TCE)*TVTX*B*28/ZNC; time[sec]; R");
  // gr3->GetHistogram()->SetMaximum(0.6);
  // gr3->GetHistogram()->SetMinimum(0.4);
  gr4->SetTitle("R=(VCH)*TVTX*B*28/ZNC; time[sec]; R");
  // gr4->GetHistogram()->SetMaximum(0.6);
  // gr4->GetHistogram()->SetMinimum(0.4);
  TMultiGraph* mg1 = new TMultiGraph();
  mg1->SetTitle("R=ZNC/28 rate [Hz] (red=PilUp Corrected); time[sec]; R");
  mg1->Add(gr1);
  mg1->Add(gr11);
  mg1->Add(gr12);
  TCanvas* c1 = new TCanvas("c1", srun.c_str(), 200, 10, 800, 500);
  std::string title = "RUN " + std::to_string(runNumber);
  c1->SetTitle(title.c_str());
  c1->Divide(2, 2);
  c1->cd(1);
  mg1->Draw("AP");
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
