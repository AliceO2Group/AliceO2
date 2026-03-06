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
#include "TFile.h"
#include "TH2F.h"
#include "TOFBase/CalibTOFapi.h"
#endif

void makeDRMobj_tof(const char* inputfile = "TObject_1764607157510.root", bool dummy = false)
{
  if (dummy) {
    o2::tof::Diagnostic drmDia;
    for (int j = 1; j <= 72; j++) {
      drmDia.fill(o2::tof::Diagnostic::getDRMKey(j - 1));
    }

    TFile* fo = new TFile("ccdb.root", "RECREATE");
    fo->WriteObjectAny(&drmDia, drmDia.Class_Name(), "ccdb_object");
    fo->Close();

    return;
  }

  TFile* f = new TFile(inputfile);
  TH2F* h = (TH2F*)f->Get("ccdb_object");

  float maxVal = h->GetBinContent(h->GetMaximumBin());

  if (maxVal > 1E6) { // to avoid to pass value causing overflow
    h->Scale(1E6 / maxVal);
  }

  o2::tof::Diagnostic drmDia;

  drmDia = o2::tof::CalibTOFapi::doDRMerrCalibFromQCHisto(h, "ccdb.root");
}
