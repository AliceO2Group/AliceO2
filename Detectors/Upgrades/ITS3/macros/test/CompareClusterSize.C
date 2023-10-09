// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CompareClusterSize.C
/// \brief compare ITS2 cluster size with ITS3 one
/// \dependencies CheckClusterSize.C
/// \author felix.schlepper@cern.ch

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#endif

void CompareClusterSize(long timestamp = 0)
{
  gROOT->SetBatch();
  gStyle->SetOptStat(0);

  // Get cluster size from checkClusterSize.C output
  std::unique_ptr<TFile> its3ClusterFile(TFile::Open("checkClusterSize.root"));
  std::unique_ptr<TH1F> its3Cluster(its3ClusterFile->Get<TH1F>("outerbarrel"));
  its3Cluster->Sumw2();
  its3Cluster->SetDirectory(nullptr);
  its3Cluster->Scale(1.0 / its3Cluster->Integral());
  its3Cluster->SetLineColor(kRed);
  its3Cluster->SetLineWidth(2);
  its3Cluster->SetFillColorAlpha(kRed + 2, 0.3);

  auto hITS = new TH1F("its", "", its3Cluster->GetNbinsX(), 0,
                       its3Cluster->GetNbinsX());

  hITS->Sumw2();
  hITS->SetLineColor(kBlue);
  hITS->SetDirectory(nullptr);
  hITS->SetLineWidth(2);
  hITS->SetFillColorAlpha(kBlue + 2, 0.3);
  // get topology dict
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL("http://alice-ccdb.cern.ch");
  mgr.setTimestamp(timestamp != 0 ? timestamp
                                  : o2::ccdb::getCurrentTimestamp());
  const o2::itsmft::TopologyDictionary* dict =
    mgr.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");
  std::unique_ptr<TFile> clusFile(TFile::Open("o2clus_its.root"));
  auto clusTree = clusFile->Get<TTree>("o2sim");
  std::vector<o2::itsmft::CompClusterExt> clusArr, *clusArrP{&clusArr};
  clusTree->SetBranchAddress("ITSClusterComp", &clusArrP);
  std::vector<unsigned char> patterns;
  std::vector<unsigned char>* patternsPtr{&patterns};
  clusTree->SetBranchAddress("ITSClusterPatt", &patternsPtr);
  std::vector<o2::itsmft::ROFRecord> rofRecVec, *rofRecVecP{&rofRecVec};
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);
  clusTree->GetEntry(0);
  int nROFRec = (int)rofRecVec.size();
  auto pattIt = patternsPtr->cbegin();

  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
    // rofRec.print();

    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl;
      const auto& cluster = clusArr[clEntry];
      // cluster.print();

      auto pattId = cluster.getPatternID();
      int clusterSize = -1000;
      if (pattId == o2::itsmft::CompCluster::InvalidPatternID ||
          dict->isGroup(pattId)) {
        o2::itsmft::ClusterPattern patt(pattIt);
        clusterSize = patt.getNPixels();
      } else {
        clusterSize = dict->getNpixels(pattId);
      }
      hITS->Fill(clusterSize);
    }
  }
  hITS->Scale(1.0 / hITS->Integral());

  auto c = new TCanvas("c", "", 1200, 800);
  c->SetLogy();
  hITS->GetYaxis()->SetRangeUser(1e-4, 1.);
  hITS->GetXaxis()->SetTitle("cluster size");
  hITS->GetYaxis()->SetTitle("norm. counts");
  hITS->Draw("hist");
  its3Cluster->Draw("hist same");
  auto legend = new TLegend(0.6, 0.7, 0.9, 0.9);
  legend->AddEntry(
    hITS, Form("ITS2 data %.2f +/- %.2f", hITS->GetMean(), hITS->GetRMS()),
    "f");
  legend->AddEntry(its3Cluster.get(),
                   Form("ITS3 MC %.2f +/- %.2f", its3Cluster->GetMean(),
                        its3Cluster->GetRMS()),
                   "f");
  legend->Draw();
  c->SaveAs("comp.pdf");
}
