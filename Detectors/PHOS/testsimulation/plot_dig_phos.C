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

/// \file plot_dig_phos.C
/// \brief Simple macro to plot PHOS digits per event

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include "TROOT.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH2.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "PHOSBase/Geometry.h"
#endif

using namespace std;

void plot_dig_phos(int ievent = 0, TString inputfile = "o2dig.root")
{

  TFile* file1 = TFile::Open(inputfile.Data());
  TTree* digTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::phos::Digit>* mDigitsArray = nullptr;
  o2::dataformats::MCTruthContainer<o2::phos::MCLabel>* labels = nullptr;
  digTree->SetBranchAddress("PHOSDigit", &mDigitsArray);
  digTree->SetBranchAddress("PHOSDigitMCTruth", &labels);

  if (!mDigitsArray) {
    cout << "PHOS digits not in tree. Exiting..." << endl;
    return;
  }
  digTree->GetEvent(ievent);

  TH2D* vMod[5][100] = {0};
  int primLabels[5][100];
  for (int mod = 1; mod < 5; mod++)
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -999;

  o2::phos::Geometry* geom = new o2::phos::Geometry("PHOS");

  std::vector<o2::phos::Digit>::const_iterator it;
  char relId[3];

  for (it = mDigitsArray->begin(); it != mDigitsArray->end(); it++) {
    short absId = (*it).getAbsId();
    float en = (*it).getAmplitude();
    int lab = (*it).getLabel();
    geom->absToRelNumbering(absId, relId);
    // check, if this label already exist
    int j = 0;
    bool found = false;
    while (primLabels[relId[0]][j] >= -2) {
      if (primLabels[relId[0]][j] == lab) {
        found = true;
        break;
      } else {
        j++;
      }
    }
    if (!found) {
      primLabels[relId[0]][j] = lab;
    }
    if (!vMod[relId[0]][j]) {
      gROOT->cd();
      vMod[relId[0]][j] =
        new TH2D(Form("hMod%d_prim%d", relId[0], j), Form("hMod%d_prim%d", relId[0], j), 64, 0., 64., 56, 0., 56.);
    }
    vMod[relId[0]][j]->Fill(relId[1] - 0.5, relId[2] - 0.5, en);
  }

  TCanvas* c[5];
  TH2D* box = new TH2D("box", "PHOS module", 64, 0., 64., 56, 0., 56.);
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("DigitInMod%d", mod), Form("PHOS hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
    box->Draw();
    int j = 0;
    while (vMod[mod][j]) {
      vMod[mod][j]->SetLineColor(j + 1);
      if (j == 0)
        vMod[mod][j]->Draw("box");
      else
        vMod[mod][j]->Draw("boxsame");
      j++;
    }
  }
}
