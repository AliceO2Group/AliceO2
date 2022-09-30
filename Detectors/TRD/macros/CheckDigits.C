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

/// \file CheckDigits.C
/// \brief Simple macro to check TRD digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TLegend.h>

#include <fairlogger/Logger.h>
#include "TRDSimulation/SimParam.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"
#endif

using namespace o2::trd;

constexpr int kMINENTRIES = 100;

void CheckDigits(std::string digifile = "trddigits.root",
                 std::string hitfile = "o2sim_HitsTRD.root",
                 std::string inputGeom = "",
                 std::string paramfile = "o2sim_par.root")
{
  TFile* fin = TFile::Open(digifile.data());
  TTree* digitTree = (TTree*)fin->Get("o2sim");
  std::vector<Digit>* digitCont = nullptr;
  digitTree->SetBranchAddress("TRDDigit", &digitCont);
  int nev = digitTree->GetEntries();

  TH1F* hDet = new TH1F("hDet", ";Detector number;Counts", 504, 0, 539);
  TH1F* hRow = new TH1F("hRow", ";Row number;Counts", 16, 0, 15);
  TH1F* hPad = new TH1F("hPad", ";Pad number;Counts", 144, 0, 143);
  TH1* hADC[540];
  for (int d = 0; d < 540; ++d) {
    hADC[d] = new TH1F(Form("hADC_%d", d), Form("ADC distribution for chamber %d;ADC value;Counts", d), 1024, 0, 1023);
  }

  TProfile* profADCperTimeBin[540];
  TH2F* hADCperTimeBinAllDetectors = new TH2F("hADCperTimeBinAllDetectors",
                                              "ADC distribution for all chambers for each time bin;Time bin;ADC",
                                              31, -0.5, 30.5, 1014, 0, 1023);
  TProfile* profADCperTimeBinAllDetectors = new TProfile("profADCperTimeBinAllDetectors",
                                                         "ADC distribution for all chambers for each time bin;Time bin;ADC",
                                                         31, -0.5, 30.5);
  for (int d = 0; d < 540; ++d) {
    profADCperTimeBin[d] = new TProfile(Form("profADCperTimeBin_%d", d),
                                        Form("ADC distribution for chamber %d for each time bin;Time bin;ADC", d),
                                        31, -0.5, 30.5);
  }
  SimParam simParam;
  LOG(info) << nev << " entries found";
  for (int iev = 0; iev < nev; ++iev) {
    digitTree->GetEvent(iev);
    for (const auto& digit : *digitCont) {
      auto adcs = digit.getADC();
      int det = digit.getDetector(); // chamber
      int row = digit.getPadRow();   // pad row
      int col = digit.getPadCol();   // pad column
      hDet->Fill(det);
      hRow->Fill(row);
      hPad->Fill(col);
      for (int tb = 0; tb < o2::trd::constants::TIMEBINS; ++tb) {
        ADC_t adc = adcs[tb];
        if (adc == (ADC_t)simParam.getADCoutRange()) {
          // LOG(info) << "Out of range ADC " << adc;
          continue;
        }
        hADC[det]->Fill(adc);
        profADCperTimeBin[det]->Fill(tb, adc);
        hADCperTimeBinAllDetectors->Fill(tb, adc);
        profADCperTimeBinAllDetectors->Fill(tb, adc);
      }
    }
  }
  TCanvas* c = new TCanvas("c", "trd digits analysis", 800, 800);
  c->DivideSquare(4);
  c->cd(1);
  hDet->Draw();
  c->cd(2);
  hRow->Draw();
  c->cd(3);
  hPad->Draw();
  c->cd(4);
  c->cd(4)->SetLogy();
  int first = 0;
  int count = 0;
  int max = 0;
  for (int d = 1; d < 540; ++d) {
    if (hADC[d]->GetEntries() < kMINENTRIES) {
      continue;
    }
    if (count > 6) {
      break;
    }
    hADC[d]->SetLineColor(count + 1);
    if (count == 0) {
      hADC[d]->Draw();
      first = d;
      if (max < hADC[d]->GetMaximum()) {
        max = hADC[d]->GetMaximum();
      }
      count++;
    } else {
      hADC[d]->Draw("SAME");
      if (max < hADC[d]->GetMaximum()) {
        max = hADC[d]->GetMaximum();
      }
      count++;
    }
  }
  hADC[first]->GetYaxis()->SetLimits(1.1, max + 100);
  c->SaveAs("testCheckDigits.pdf");

  TCanvas* c1 = new TCanvas("c1", "trd digits analysis", 600, 600);
  first = 0;
  count = 0;
  max = 0;
  std::vector<int> dets;
  for (int d = 1; d < 540; ++d) {
    if (hADC[d]->GetEntries() < kMINENTRIES) {
      continue;
    }
    if (count > 5) {
      break;
    }
    hADC[d]->SetLineColor(count + 1);
    dets.push_back(d);
    if (count == 0) {
      hADC[d]->Draw();
      first = d;
      if (max < hADC[d]->GetMaximum()) {
        max = hADC[d]->GetMaximum();
      }
      count++;
    } else {
      hADC[d]->Draw("SAME");
      if (max < hADC[d]->GetMaximum()) {
        max = hADC[d]->GetMaximum();
      }
      count++;
    }
  }
  hADC[first]->GetYaxis()->SetLimits(1.1, max + 100);
  hADC[first]->GetXaxis()->SetRangeUser(0, 512);

  TLegend* legend = new TLegend(0.7, 0.6, 0.9, 0.9);
  legend->SetBorderSize(0); // no border
  legend->SetFillStyle(0);
  legend->SetFillColor(0); // Legend background should be white
  legend->SetTextFont(42);
  legend->SetTextSize(0.03); // Increase entry font size!
  for (const auto& d : dets) {
    legend->AddEntry(hADC[d], hADC[d]->GetName(), "l");
  }
  legend->Draw();

  c1->SetLogy();
  c1->SaveAs("testCheckDigits_ADCs.pdf");

  TCanvas* c2 = new TCanvas("c2", "trd digits analysis", 600, 600);
  // profADCperTimeBinAllDetectors->Draw("HIST*");
  count = 0;
  for (const auto& det : dets) {
    profADCperTimeBin[det]->SetLineColor(count + 1);
    // profADCperTimeBin[det]->SetLineWidth(2);
    if (count == 0) {
      profADCperTimeBin[det]->Draw("HIST*");
      ++count;
    } else {
      profADCperTimeBin[det]->Draw("SAME HIST*");
      ++count;
    }
  }
  c2->SaveAs("testCheckDigits_ADCs_per_timebin.pdf");
}
