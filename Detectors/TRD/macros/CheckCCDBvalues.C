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

/// \file CheckCCDBvalues.C
/// \brief Simple macro to check TRD CCDB

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>

#include <fairlogger/Logger.h>
#include "TRDBase/Calibrations.h"

#endif

void CheckCCDBvalues(const long runNumber = 297595)
{
  TTree* t = new TTree("t", "tree");
  float exb, vdrift, t0, padgainfactor, gainfactor;
  int det, row, pad;
  t->Branch("exb", &exb);
  t->Branch("vdrift", &vdrift);
  t->Branch("t0", &t0);
  t->Branch("padgainfactor", &padgainfactor);
  t->Branch("gainfactor", &gainfactor);
  t->Branch("det", &det);
  t->Branch("row", &row);
  t->Branch("pad", &pad);

  TH1F* histExB = new TH1F("histExB", ";E#timesB (VT/m);Counts", 100, -0.3, 0);
  TH1F* histVDrift = new TH1F("histVDrift", ";v_{drift} (cm/#mus);Counts", 100, 0, 2);
  TH1F* histT0 = new TH1F("histT0", ";t_{0} (#mus);Counts", 100, -3, 3);
  TH1F* histPadGainFactor = new TH1F("histPadGainFactor", ";Pad Gain Factor;Counts", 100, 0, 2);

  o2::trd::Calibrations calib;
  calib.getCCDBObjects(runNumber);
  for (det = 0; det < 540; ++det) {
    exb = calib.getExB(det);
    histExB->Fill(exb);
    for (row = 0; row < 16; ++row) {
      for (pad = 0; pad < 144; ++pad) {
        vdrift = calib.getVDrift(det, row, pad);
        t0 = calib.getT0(det, row, pad);
        padgainfactor = calib.getPadGainFactor(det, row, pad);
        // gainfactor = calib.getGainFactor(det, row, pad);
        t->Fill();
        histVDrift->Fill(vdrift);
        histT0->Fill(t0);
        histPadGainFactor->Fill(padgainfactor);
      }
    }
  }

  // Make the plots
  TCanvas c1("c1", "c1 title", 800, 600);
  c1.Divide(2, 2);
  c1.cd(1);
  histExB->Draw();
  c1.cd(2);
  histVDrift->Draw();
  c1.cd(3);
  histT0->Draw();
  c1.cd(4);
  histPadGainFactor->Draw();
  c1.SaveAs("testCheckCCDBvalues.pdf");
}
