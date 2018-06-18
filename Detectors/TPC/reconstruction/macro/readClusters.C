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
      for (auto& label : mcDigitTruth->getLabels(digitCount)) {
        digitLabels[mGlobalRowToLocalRow[digit.getRow()] * 10 + digit.getCRU()]->emplace_back(label, iEvent, digitCount);
      }
    }
  }

  TH1F* hMeanPad = new TH1F("hMeanPad", "Difference in pad direction;pad_{cluster}-pad_{digits};counts", 118, -29.5, 29.5);
  TH1F* hMeanTime = new TH1F("hMeanTime", "Difference in time direction;time_{cluster}-time_{digits};counts", 138, -29.5, 39.5);
  for (int iEvent = 0; iEvent < clusterTree->GetEntriesFast(); ++iEvent) {
    clusterTree->GetEntry(iEvent);
    std::cout << "Event " << iEvent << " with " << clusters->size() << " container" << std::endl;
    if (clusters->size() == 0)
      continue;

    int i = 0;
    unsigned mcClusterCount = 0;
    for (auto& cont : *clusters) {
      std::cout << "container " << std::setw(4) << std::setfill(' ') << i++
                << " with " << std::setw(3) << std::setfill(' ') << cont.getContainer()->numberOfClusters << "/" << cont.getMaxNumberOfClusters() << " clusters,"
                << " belonging to CRU (region) " << cont.getContainer()->CRU
                << " with time offset " << cont.getContainer()->timeBinOffset
                << std::endl;

      auto container = cont.getContainer();
      for (int clusterCount = 0; clusterCount < container->numberOfClusters; ++clusterCount) {
        ++mcClusterCount;
        auto& cluster = container->clusters[clusterCount];

        // with MC match
        for (auto& clusterLabel : mcClusterTruth->getLabels(mcClusterCount)) {
          for (auto& digitLabelTuple : *digitLabels[cluster.getRow() * 10 + container->CRU]) {
            if (clusterLabel == std::get<0>(digitLabelTuple)) {
              //              std::cout << "digit " << std::get<2>(digitLabelTuple)
              //                        << " of event " << std::get<1>(digitLabelTuple)
              //                        << " has same label " << std::get<0>(digitLabelTuple)
              //                        << " as current cluster " << mcClusterCount << std::endl;
              hMeanPad->Fill(cluster.getPad() -
                             (*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getPad());
              hMeanTime->Fill(cluster.getTimeLocal() + container->timeBinOffset -
                              (*allDigits[std::get<1>(digitLabelTuple)])[std::get<2>(digitLabelTuple)].getTimeStamp());
            }
          }
        }
        // all digits
        //for (auto& digiEv : allDigits) {
        //  for (auto &digit : *digiEv) {
        //    if (digit.getCRU() != container->CRU) continue;
        //    if (mGlobalRowToLocalRow[digit.getRow()] != cluster.getRow()) continue;
        //    hMeanPad->Fill(cluster.getPad() - digit.getPad());
        //    hMeanTime->Fill(cluster.getTimeLocal()+container->timeBinOffset - digit.getTimeStamp());

        //  }
        //}
      }
    }
  }
  TCanvas* c1 = new TCanvas("c1", "c1");
  hMeanPad->Draw();
  TCanvas* c2 = new TCanvas("c2", "c2");
  hMeanTime->Draw();
}
