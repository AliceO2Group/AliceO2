// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file drawCLusters.C
/// \brief This macro draws the clusters in the containers
/// \author Sebastian Klewin <sebastian.klewin@cern.ch>
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"

#include "TPCBase/Digit.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/Helpers.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#endif

void readClusters(std::string clusterFilename, std::string digitFilename, int sectorid)
{

  // open files and trees
  TFile* clusterFile = TFile::Open(clusterFilename.data());
  TTree* clusterTree = (TTree*)clusterFile->Get("o2sim");
  TFile* digitFile = TFile::Open(digitFilename.data());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  // register branches
  auto clusterBranch = clusterTree->GetBranch(Form("TPCClusterHW%i", sectorid));
  std::vector<o2::TPC::ClusterHardwareContainer8kb>* clusters = nullptr;
  clusterBranch->SetAddress(&clusters);

  auto digitBranch = digitTree->GetBranch(Form("TPCDigit%i", sectorid));
  std::vector<o2::TPC::Digit>* digits = nullptr;
  digitBranch->SetAddress(&digits);

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  MCLabelContainer* mcClusterTruth = nullptr;
  clusterTree->SetBranchAddress(Form("TPCClusterHWMCTruth%i", sectorid), &mcClusterTruth);
  MCLabelContainer* mcDigitTruth = nullptr;
  digitTree->SetBranchAddress(Form("TPCDigitMCTruth%i", sectorid), &mcDigitTruth);

  // create row mapping (global -> local)
  o2::TPC::Mapper& mapper = o2::TPC::Mapper::instance();
  std::vector<unsigned short> mGlobalRowToRegion;
  std::vector<unsigned short> mGlobalRowToLocalRow;
  mGlobalRowToRegion.resize(mapper.getNumberOfRows());
  mGlobalRowToLocalRow.resize(mapper.getNumberOfRows());
  unsigned short row = 0;
  for (unsigned short region = 0; region < 10; ++region) {
    for (unsigned short localRow = 0; localRow < mapper.getNumberOfRowsRegion(region); ++localRow) {
      mGlobalRowToRegion[row] = region;
      mGlobalRowToLocalRow[row] = localRow;
      ++row;
    }
  }

  // collet all digit MC labels
  // row/CRU ID                                      Label            event     digit index
  std::vector<std::unique_ptr<std::vector<std::tuple<o2::MCCompLabel, unsigned, unsigned>>>> digitLabels;
  digitLabels.resize(18 * 10);
  // and all digits
  // event                    digits
  std::vector<std::unique_ptr<std::vector<o2::TPC::Digit>>> allDigits;
  allDigits.resize(digitTree->GetEntriesFast());
  for (int i = 0; i < 18 * 10; ++i)
    digitLabels[i] = std::make_unique<std::vector<std::tuple<o2::MCCompLabel, unsigned, unsigned>>>();
  for (int iEvent = 0; iEvent < digitTree->GetEntriesFast(); ++iEvent) {
    digitTree->GetEntry(iEvent);
    allDigits[iEvent] = std::make_unique<std::vector<o2::TPC::Digit>>();
    std::cout << "Event " << iEvent << " with " << digits->size() << " Digits" << std::endl;
    for (unsigned digitCount = 0; digitCount < digits->size(); ++digitCount) {
      auto& digit = (*digits)[digitCount];
      allDigits[iEvent]->push_back(digit);
      //      for (auto& label : mcDigitTruth->getLabels(digitCount)) {
      auto label = *mcDigitTruth->getLabels(digitCount).begin(); // only most significant MC label
      digitLabels[mGlobalRowToLocalRow[digit.getRow()] * 10 + digit.getCRU()]->emplace_back(label, iEvent, digitCount);
      //      }
    }
  }

  TFile* outfile = new TFile("clusterDists.root", "RECREATE");
  TCanvas* c1 = new TCanvas("c1", "c1");
  for (int cru = 0; cru < 10; ++cru) {
    TH1F* hNumCluster = new TH1F(Form("hNumCluster%i", cru), Form("Number of Clusters per Container for CRU %i;N_{cluster};counts", cru), 407, -0.5, 406.5);
    TH1F* hClustersPerRow[18];
    TH1F* hMeanPad[18];
    TH1F* hMeanTime[18];
    for (int row = 0; row < 18; ++row) {
      hClustersPerRow[row] = new TH1F(Form("hClustersPerRow%i,%i", cru, row), Form("Number of Clusters per Row per Container for row %i in CRU %i;N_{cluster};counts", row, cru), 70, -0.5, 69.5);
      hMeanPad[row] = new TH1F(Form("hMeanPad%i,%i", cru, row), Form("Difference in pad direction for row %i in CRU %i;pad_{cluster}-pad_{digits};counts", row, cru), 160, -39.5, 40.5);
      hMeanTime[row] = new TH1F(Form("hMeanTime%i,%i", cru, row), Form("Difference in time direction for row %i in CRU %i;time_{cluster}-time_{digits};counts", row, cru), 260, -49.5, 80.5);
    }
    for (int iEvent = 0; iEvent < clusterTree->GetEntriesFast(); ++iEvent) {
      clusterTree->GetEntry(iEvent);
      std::cout << "Event " << iEvent << " with " << clusters->size() << " container" << std::endl;
      if (clusters->size() == 0)
        continue;

      int i = 0;
      unsigned mcClusterCount = 0;
      std::array<int, 18> clPerRow;
      for (auto& cont : *clusters) {
        std::cout << "container " << std::setw(4) << std::setfill(' ') << i++
                  << " with " << std::setw(3) << std::setfill(' ') << cont.getContainer()->numberOfClusters << "/" << cont.getMaxNumberOfClusters() << " clusters,"
                  << " belonging to CRU (region) " << cont.getContainer()->CRU
                  << " with time offset " << cont.getContainer()->timeBinOffset
                  << std::endl;

        auto container = cont.getContainer();
        if (container->CRU != cru) {
          mcClusterCount += container->numberOfClusters;
          continue;
        }
        hNumCluster->Fill(container->numberOfClusters);
        clPerRow.fill(0);

        for (int clusterCount = 0; clusterCount < container->numberOfClusters; ++clusterCount) {
          auto& cluster = container->clusters[clusterCount];
          ++clPerRow[cluster.getRow()];

          // with MC match
          //          for (auto& clusterLabel : mcClusterTruth->getLabels(mcClusterCount)) {
          auto clusterLabel = *mcClusterTruth->getLabels(mcClusterCount).begin(); // only most significant MC label
          for (auto& digitLabelTuple : *digitLabels[cluster.getRow() * 10 + container->CRU]) {
            if (clusterLabel != std::get<0>(digitLabelTuple))
              continue;
            if (container->CRU != (*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getCRU())
              continue;
            if (cluster.getRow() != mGlobalRowToLocalRow[(*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getRow()])
              continue;
            //              std::cout << "digit " << std::get<2>(digitLabelTuple)
            //                        << " of event " << std::get<1>(digitLabelTuple)
            //                        << " has same label " << std::get<0>(digitLabelTuple)
            //                        << " as current cluster " << mcClusterCount << std::endl;
            hMeanPad[cluster.getRow()]->Fill(cluster.getPad() -
                                             (*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getPad());
            hMeanTime[cluster.getRow()]->Fill(cluster.getTimeLocal() + container->timeBinOffset -
                                              (*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getTimeStamp());
          }
          // all digits
          //for (auto& digiEv : allDigits) {
          //  for (auto &digit : *digiEv) {
          //    if (digit.getCRU() != container->CRU) continue;
          //    if (mGlobalRowToLocalRow[digit.getRow()] != cluster.getRow()) continue;
          //    hMeanPad->Fill(cluster.getPad() - digit.getPad());
          //    hMeanTime->Fill(cluster.getTimeLocal()+container->timeBinOffset - digit.getTimeStamp());

          //}
          ++mcClusterCount;
          //          }
        }
        for (int r = 0; r < 18; ++r) {
          if (clPerRow[r] != 0)
            hClustersPerRow[r]->Fill(clPerRow[r]);
        }
      }
    }
    hNumCluster->Write();
    delete hNumCluster;
    for (int r = 0; r < 18; ++r) {
      hClustersPerRow[r]->Write();
      hClustersPerRow[r]->Draw();
      c1->SaveAs(("plots/" + std::string(hClustersPerRow[r]->GetName()) + ".pdf").c_str());
      hMeanPad[r]->Write();
      hMeanPad[r]->Draw();
      c1->SaveAs(("plots/" + std::string(hMeanPad[r]->GetName()) + ".pdf").c_str());
      hMeanTime[r]->Write();
      hMeanTime[r]->Draw();
      c1->SaveAs(("plots/" + std::string(hMeanTime[r]->GetName()) + ".pdf").c_str());
      delete hClustersPerRow[r];
      delete hMeanPad[r];
      delete hMeanTime[r];
    }
  }
  outfile->Close();
  //  delete outfile;
  //    TCanvas* c1 = new TCanvas("c1", "c1");
  //    hMeanPad->Draw();
  //    TCanvas* c2 = new TCanvas("c2", "c2");
  //    hMeanTime->Draw();
}
