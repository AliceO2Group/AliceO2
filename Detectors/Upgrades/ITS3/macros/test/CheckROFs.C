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

/// \file CheckROFs.C
/// \brief Simple macro to check ITS3 ROFs

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TROOT.h>
#include <TCanvas.h>
#include "TEfficiency.h"
#include <TClonesArray.h>
#include <TFile.h>
#include <TH2F.h>
#include <THStack.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TPad.h>
#include <TTree.h>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <span>
#endif

void CheckROFs(std::string_view tracfile = "tf1/o2trac_its.root")
{
  TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::its::TrackITS> recArr, *recArrPtr{&recArr};
  recTree->SetBranchAddress("ITSTrack", &recArrPtr);
  std::vector<o2::itsmft::ROFRecord> rofs, *rofsPtr{&rofs};
  recTree->SetBranchAddress("ITSTracksROF", &rofsPtr);

  std::vector<size_t> nTracksROF;

  for (int iEntry{0}; recTree->LoadTree(iEntry) >= 0; ++iEntry) {
    recTree->GetEntry(iEntry);

    for (const auto& rof : rofs) {
      auto trcs = std::span{recArr.cbegin() + rof.getFirstEntry(), recArr.cbegin() + rof.getFirstEntry() + rof.getNEntries()};
      nTracksROF.push_back(trcs.size());
    }
  }

  auto gNTracks = new TGraph(nTracksROF.size());
  gNTracks->GetXaxis()->SetTitle("rof");
  gNTracks->GetYaxis()->SetTitle("nTracks");
  for (int i{0}; i < nTracksROF.size(); ++i) {
    gNTracks->SetPoint(i, i, nTracksROF[i]);
  }
  gNTracks->Draw("apl");
}
