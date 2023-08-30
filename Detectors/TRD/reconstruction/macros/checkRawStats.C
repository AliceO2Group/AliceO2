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
// ROOT header
#include <TROOT.h>
#include <TChain.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TFile.h>
#include <TTree.h>
#include <TAxis.h>
// O2 header
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/RawDataStats.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

#include <fmt/format.h>
#include <chrono>
#include <thread>
#endif

using namespace o2::trd;
using namespace o2::trd::constants;

void prepareHist(TH2F* h)
{
  h->GetXaxis()->SetTitle("Sector_Side");
  h->GetXaxis()->CenterTitle(kTRUE);
  h->GetYaxis()->SetTitle("Stack_Layer");
  h->GetYaxis()->CenterTitle(kTRUE);
  for (int sm = 0; sm < NSECTOR; ++sm) {
    for (int side = 0; side < 2; ++side) {
      std::string label = fmt::format("{0}_{1}", sm, side == 0 ? "A" : "B");
      int pos = sm * 2 + side + 1;
      h->GetXaxis()->SetBinLabel(pos, label.c_str());
    }
  }
  for (int s = 0; s < NSTACK; ++s) {
    for (int l = 0; l < NLAYER; ++l) {
      std::string label = fmt::format("{0}_{1}", s, l);
      int pos = s * NLAYER + l + 1;
      h->GetYaxis()->SetBinLabel(pos, label.c_str());
    }
  }
  h->LabelsOption("v");
}

void checkRawStats()
{
  TChain chain("stats");
  chain.AddFile("trdrawstats.root");
  std::vector<TriggerRecord> trigIn, *trigInPtr{&trigIn};
  std::vector<DataCountersPerTrigger> linkstats, *linkstatsPtr{&linkstats};
  TRDDataCountersPerTimeFrame rawstats, *rawstatsPtr{&rawstats};
  chain.SetBranchAddress("trigRec", &trigInPtr);
  chain.SetBranchAddress("linkStats", &linkstatsPtr);
  chain.SetBranchAddress("tfStats", &rawstatsPtr);

  auto h0 = new TH2F("linkError0", "link error is 0x0 (all good)", 36, 0, 36, 30, 0, 30);
  auto h1 = new TH2F("linkError1", "link error 0x1 set (data outside trigger window)", 36, 0, 36, 30, 0, 30);
  auto h2 = new TH2F("linkError2", "link error 0x2 set (timeout issue)", 36, 0, 36, 30, 0, 30);
  auto hWords = new TH2F("linkWords", "link word count (data size)", 36, 0, 36, 30, 0, 30);

  prepareHist(h0);
  prepareHist(h1);
  prepareHist(h2);
  prepareHist(hWords);

  auto c = new TCanvas("c", "c", 1900, 1200);
  c->Divide(2, 2);
  gStyle->SetOptStat("");

  for (int iEntry = 0; iEntry < chain.GetEntries(); ++iEntry) {
    chain.GetEntry(iEntry); // for each TimeFrame there is one tree entry
    int triggerCounter = 0;
    for (const auto& stats : linkstats) {
      c->Update();
      c->SetTitle(TString::Format("TF %i, Trigger %i", iEntry, triggerCounter));
      for (int hcid = 0; hcid < MAXHALFCHAMBER; ++hcid) {
        int stackLayer = HelperMethods::getStack(hcid / 2) * NLAYER + HelperMethods::getLayer(hcid / 2);
        int sectorSide = (hcid / NHCPERSEC) * 2 + (hcid % 2);
        h2->SetBinContent(sectorSide + 1, stackLayer + 1, stats.mLinkErrorFlag[hcid] & 0x2);
        h1->SetBinContent(sectorSide + 1, stackLayer + 1, stats.mLinkErrorFlag[hcid] & 0x1);
        h0->SetBinContent(sectorSide + 1, stackLayer + 1, stats.mLinkErrorFlag[hcid] == 0);
        hWords->SetBinContent(sectorSide + 1, stackLayer + 1, stats.mLinkWords[hcid]);
      }
      c->cd(1);
      h0->Draw("colz");
      c->cd(2);
      h1->Draw("colz");
      c->cd(3);
      h2->Draw("colz");
      c->cd(4);
      hWords->Draw("colz");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      ++triggerCounter;
    }
  }
}
