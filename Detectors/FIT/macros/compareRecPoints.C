// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file compareRecPoints.C
/// \brief ROOT macro to compare two trees with RecPoints
///
/// \author Artur Furs artur.furs@cern.ch, Andreas Molander andreas.molander@cern.ch

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFV0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"

#include "TFile.h"
#include "TTree.h"

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#endif

void compareRecPoints(std::string filename1, std::string filename2)
{
  std::unique_ptr<TFile> file1(TFile::Open(filename1.c_str(), "READ"));
  TTree* tree1 = (TTree*)file1->Get("o2sim");

  std::unique_ptr<TFile> file2(TFile::Open(filename2.c_str(), "READ"));
  TTree* tree2 = (TTree*)file2->Get("o2sim");

  if (tree1->GetEntries() != tree2->GetEntries()) {
    std::cout << "Non equal number of entries in trees!" << std::endl;
    return;
  }

  typedef typename o2::ft0::RecPoints RecPoint;
  typedef typename o2::ft0::ChannelDataFloat ChannelDataFloat;

  std::vector<RecPoint> vecRecPoints1;
  std::vector<RecPoint>* ptrVecRecPoints1 = &vecRecPoints1;

  std::vector<ChannelDataFloat> vecChannelDataFloat1;
  std::vector<ChannelDataFloat>* ptrVecChannelDataFloat1 = &vecChannelDataFloat1;

  tree1->SetBranchAddress("FT0Cluster", &ptrVecRecPoints1);
  tree1->SetBranchAddress("FT0RecChData", &ptrVecChannelDataFloat1);

  std::vector<RecPoint> vecRecPoints2;
  std::vector<RecPoint>* ptrVecRecPoints2 = &vecRecPoints2;

  std::vector<ChannelDataFloat> vecChannelDataFloat2;
  std::vector<ChannelDataFloat>* ptrVecChannelDataFloat2 = &vecChannelDataFloat2;

  tree2->SetBranchAddress("FT0Cluster", &ptrVecRecPoints2);
  tree2->SetBranchAddress("FT0RecChData", &ptrVecChannelDataFloat2);

  for (int iEntry = 0; iEntry < tree1->GetEntries(); iEntry++) {
    tree1->GetEntry(iEntry);
    tree2->GetEntry(iEntry);

    if (vecRecPoints1 != vecRecPoints2) {
      std::cout << "Non equal RecPoints vector!" << std::endl;

      if (vecRecPoints1.size() == vecRecPoints2.size()) {
        for (int iEvent = 0; iEvent < vecRecPoints1.size(); iEvent++) {
          const auto& recPoint1 = vecRecPoints1[iEvent];
          const auto& recPoint2 = vecRecPoints2[iEvent];

          if (!(recPoint1 == recPoint2)) {
            std::cout << "First RecPoint" << std::endl;
            recPoint1.print();
            std::cout << "Second RecPoint" << std::endl;
            recPoint2.print();
          }
        }
      } else {
        std::cout << "Non equal number of RecPoints!" << std::endl;
      }
    }
    if (vecChannelDataFloat1 != vecChannelDataFloat2) {
      std::cout << "Non equal ChannelDataFloat vector!" << std::endl;

      if (vecChannelDataFloat1.size() == vecChannelDataFloat2.size()) {
        for (int iEvent = 0; iEvent < vecChannelDataFloat1.size(); iEvent++) {
          const auto& channelDataFloat1 = vecChannelDataFloat1[iEvent];
          const auto& channelDataFloat2 = vecChannelDataFloat2[iEvent];

          if (!(channelDataFloat1 == channelDataFloat2)) {
            std::cout << "First ChannelDataFloat" << std::endl;
            channelDataFloat1.print();
            std::cout << "Second ChannelDataFloat" << std::endl;
            channelDataFloat2.print();
          }
        }
      } else {
        std::cout << "Non equal number of ChannelDataFloat!" << std::endl;
      }
    }
  }

  return;
}