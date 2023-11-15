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
#include <DataFormatsITSMFT/ROFRecord.h>
#include <DataFormatsITS3/CompCluster.h>
#include <ITSBase/GeometryTGeo.h>
#include <Framework/Logger.h>
#include <DataFormatsITSMFT/TopologyDictionary.h>
#include <DetectorsCommonDataFormats/DetectorNameConf.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <CCDB/BasicCCDBManager.h>

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TColor.h>
#include <TPad.h>

#include <vector>
#include <gsl/gsl>
#endif

void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern>& pattVec, std::vector<o2::its3::CompClusterExt>* ITSclus, std::vector<unsigned char>* ITSpatt, o2::itsmft::TopologyDictionary& mdict);

void CheckSquasherITS3(const uint chipId = 0, const uint startingROF = 0, const unsigned int nRofs = 3, const string fname = "o2clus_it3.root")
{
  // TColor::InvertPalette();
  gStyle->SetOptStat(0);
  gStyle->SetPalette(kInvertedDarkBodyRadiator);
  // Geometry
  o2::base::GeometryManager::loadGeometry("");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  // Topology dictionary
  auto& cc = o2::ccdb::BasicCCDBManager::instance();
  auto mdict = cc.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");
  auto fITSclus = TFile::Open(fname.data(), "r");
  auto treeITSclus = (TTree*)fITSclus->Get("o2sim");

  std::vector<o2::its3::CompClusterExt>* ITSclus = nullptr;
  std::vector<o2::itsmft::ROFRecord>* ITSrof = nullptr;
  std::vector<unsigned char>* ITSpatt = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;

  treeITSclus->SetBranchAddress("IT3ClusterComp", &ITSclus);
  treeITSclus->SetBranchAddress("IT3ClustersROF", &ITSrof);
  treeITSclus->SetBranchAddress("IT3ClusterPatt", &ITSpatt);
  // treeITSclus->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());
  std::vector<TH2D*> hHitMapsVsFrame(nRofs);

  treeITSclus->GetEvent(0);
  LOGP(info, "there are {} rofs in this TF", ITSrof->size());

  // Get patterns
  std::vector<o2::itsmft::ClusterPattern> pattVec;
  getClusterPatterns(pattVec, ITSclus, ITSpatt, *mdict);

  for (unsigned int iR{0}; iR < nRofs; iR++) {
    LOGP(info, "Processing rof {}", iR + startingROF);
    switch(chipId/2) {
      case 0:
      {
        hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof %i; ; ; Counts", chipId, startingROF + iR), 13575, -0.5, 13574.5, 2828, -0.5, 2827.5);
        break;
      }
      case 1:
      {
        hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof %i; ; ; Counts", chipId, startingROF + iR), 13575, -0.5, 13574.5, 3770, -0.5, 3769.5);
        break;
      }
      case 2:
      {
        hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof %i; ; ; Counts", chipId, startingROF + iR), 13575, -0.5, 13574.5, 4713, -0.5, 4712.5);
        break;
      }
      default:
      {
        hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof %i; ; ; Counts", chipId, startingROF + iR), 1024, -0.5, 1023.5, 512, -0.5, 511.5);
        break;
      }
    }

    // work on data
    const auto& rof = (*ITSrof)[startingROF + iR];
    auto clustersInFrame = rof.getROFData(*ITSclus);
    auto patternsInFrame = rof.getROFData(pattVec);
    // auto pattIt = ITSpatt->cbegin();

    for (unsigned int clusInd{0}; clusInd < clustersInFrame.size(); clusInd++) {
      const auto& clus = clustersInFrame[clusInd];
      auto sID = clus.getSensorID();

      if (sID == chipId) {
        clus.print();

        // auto labels = clusLabArr->getLabels(clusInd);
        // extract pattern info
        auto col = clus.getCol();
        auto row = clus.getRow();

        std::cout << patternsInFrame[clusInd];

        std::cout << std::endl;
        int ic = 0, ir = 0;

        auto colSpan = patternsInFrame[clusInd].getColumnSpan();
        auto rowSpan = patternsInFrame[clusInd].getRowSpan();
        auto nBits = rowSpan * colSpan;

        for (int i = 2; i < patternsInFrame[clusInd].getUsedBytes() + 2; i++) {
          unsigned char tempChar = patternsInFrame[clusInd].getByte(i);
          int s = 128; // 0b10000000
          while (s > 0) {
            if ((tempChar & s) != 0) // checking active pixels
            {
              hHitMapsVsFrame[iR]->Fill(col + ic, row + ir);
            }
            ic++;
            s >>= 1;
            if ((ir + 1) * ic == nBits) {
              break;
            }
            if (ic == colSpan) {
              ic = 0;
              ir++;
            }
            if ((ir + 1) * ic == nBits) {
              break;
            }
          }
        }
      }
    }
  }
  auto canvas = new TCanvas(Form("chip%d", chipId), Form("chip%d", chipId), nRofs * 1000, 600);

  canvas->Divide(nRofs, 1);
  for (unsigned int i{0}; i < nRofs; ++i) {
    canvas->cd(i + 1);
    gPad->SetGridx();
    gPad->SetGridy();
    hHitMapsVsFrame[i]->Draw("colz");
  }
}

void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern>& pattVec, std::vector<o2::its3::CompClusterExt>* ITSclus, std::vector<unsigned char>* ITSpatt, o2::itsmft::TopologyDictionary& mdict)
{
  pattVec.reserve(ITSclus->size());
  auto pattIt = ITSpatt->cbegin();

  for (unsigned int iClus{0}; iClus < ITSclus->size(); ++iClus) {
    auto& clus = (*ITSclus)[iClus];

    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::its3::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {

      npix = mdict.getNpixels(pattID);
      patt = mdict.getPattern(pattID);
    }

    pattVec.push_back(patt);
  }
}
