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

#include <filesystem>
#include <TTree.h>
#include "Framework/Logger.h"
#include "DataFormatsFDD/Digit.h"
#include <TFile.h>
#include <cstring>

using namespace o2::fdd;

int main(int argc, char* argv[])
{
  const std::string genDigDile{"fdddigits.root"};
  const std::string decDigDile{"o2_fdddigits.root"};
  const std::string branchBC{"FDDDIGITSBC"};
  const std::string branchCH{"FDDDIGITSCH"};

  if (!std::filesystem::exists(genDigDile)) {
    LOG(fatal) << "Generated digits file " << genDigDile << " is absent";
  }
  TFile flIn(genDigDile.c_str());
  std::unique_ptr<TTree> tree((TTree*)flIn.Get("o2sim"));
  if (!flIn.IsOpen() || flIn.IsZombie() || !tree) {
    LOG(fatal) << "Failed to get tree from generated digits file " << genDigDile;
  }
  std::vector<o2::fdd::Digit> digitsBC, *fddBCDataPtr = &digitsBC;
  std::vector<o2::fdd::ChannelData> digitsCh, *fddChDataPtr = &digitsCh;
  tree->SetBranchAddress("FDDDIGITSBC", &fddBCDataPtr);
  tree->SetBranchAddress("FDDDIGITSCH", &fddChDataPtr);

  if (!std::filesystem::exists(decDigDile)) {
    LOG(fatal) << "Decoded digits file " << genDigDile << " is absent";
  }

  TFile flIn2(decDigDile.c_str());
  std::unique_ptr<TTree> tree2((TTree*)flIn2.Get("o2sim"));
  if (!flIn2.IsOpen() || flIn2.IsZombie() || !tree2) {
    LOG(fatal) << "Failed to get tree from decoded digits file " << genDigDile;
  }
  std::vector<o2::fdd::Digit> digitsBC2, *fddBCDataPtr2 = &digitsBC2;
  std::vector<o2::fdd::ChannelData> digitsCh2, *fddChDataPtr2 = &digitsCh2;
  tree2->SetBranchAddress("FDDDIGITSBC", &fddBCDataPtr2);
  tree2->SetBranchAddress("FDDDIGITSCH", &fddChDataPtr2);

  int nbc = 0, nbc2 = 0, nch = 0, nch2 = 0;
  for (int ient = 0; ient < tree->GetEntries(); ient++) {
    tree->GetEntry(ient);
    int nbcEntry = digitsBC.size();
    nbc += nbcEntry;
    for (int ibc = 0; ibc < nbcEntry; ibc++) {
      auto& bcd = digitsBC[ibc];
      int bc = bcd.getBC();
      auto channels = bcd.getBunchChannelData(digitsCh);
      nch += channels.size();
    }
  }

  for (int ient = 0; ient < tree2->GetEntries(); ient++) {
    tree2->GetEntry(ient);
    int nbc2Entry = digitsBC2.size();
    nbc2 += nbc2Entry;
    for (int ibc = 0; ibc < nbc2Entry; ibc++) {
      auto& bcd2 = digitsBC2[ibc];
      int bc2 = bcd2.getBC();
      auto channels2 = bcd2.getBunchChannelData(digitsCh2);
      nch2 += channels2.size();
    }
  }
  LOG(info) << "FDD simulated: " << nbc << " triggers with " << nch << " channels";
  LOG(info) << "FDD decoded  : " << nbc2 << " triggers with " << nch2 << " channels";
  if (nbc != nbc2 || nch != nch2) {
    LOG(fatal) << "Mismatch between the number of encoded and decoded objects";
  }

  return 0;
}
