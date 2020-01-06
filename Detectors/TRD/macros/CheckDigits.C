// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <TCanvas.h>

#include "FairLogger.h"
#include "TRDBase/Digit.h"
#include "TRDBase/TRDSimParam.h"
#endif

void CheckDigits(std::string digifile = "trddigits.root",
                 std::string hitfile = "o2sim.root",
                 std::string inputGeom = "O2geometry.root",
                 std::string paramfile = "o2sim_par.root")
{
  using o2::trd::Digit;
  TFile* fin = TFile::Open(digifile.data());
  TTree* digitTree = (TTree*)fin->Get("o2sim");
  std::vector<o2::trd::Digit>* digitCont = nullptr;
  digitTree->SetBranchAddress("TRDDigit", &digitCont);
  int nev = digitTree->GetEntries();

  TH1F* hDet = new TH1F("hDet", ";Detector number;Counts", 504, 0, 539);
  TH1F* hRow = new TH1F("hRow", ";Row number;Counts", 16, 0, 15);
  TH1F* hPad = new TH1F("hPad", ";Pad number;Counts", 144, 0, 143);
  TH1* hADC = new TH1F("hADC", ";ADC value;Counts", 1024, 0, 1023);

  LOG(INFO) << nev << " entries found";
  for (int iev = 0; iev < nev; ++iev) {
    digitTree->GetEvent(iev);
    for (const auto& digit : *digitCont) {
      // loop over det, pad, row?
      auto adcs = digit.getADC();
      int det = digit.getDetector();
      int row = digit.getRow();
      int pad = digit.getPad();
      hDet->Fill(det);
      hRow->Fill(row);
      hPad->Fill(pad);
      for (int tb = 0; tb < kTimeBins; ++tb) {
        ADC_t adc = adcs[tb];
        if (adc == (ADC_t)TRDSimParam::Instance()->GetADCoutRange()) {
          LOG(INFO) << "Out of range ADC " << adc;
          continue;
        }
        hADC->Fill(adc);
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
  hADC->Draw();
  c->SaveAs("testCheckDigits.pdf");
}
