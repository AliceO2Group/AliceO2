// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file selectPVIRFrames.C
/// \brief Macro to select IRFrames for specific vertices

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>>
#include <vector>

#include <TFile.h>
#include <TTree.h>

#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonDataFormat/IRFrame.h"
#endif

void selectPVIRFrames(const char* fName = "o2_primary_vertex.root")
{
  auto fPVs = TFile::Open(fName);
  TTree* tPVs = (TTree*)fPVs->Get("o2sim");
  std::vector<o2::dataformats::PrimaryVertex> pvArr, *pvArrPtr{&pvArr};
  tPVs->SetBranchAddress("PrimaryVertex", &pvArrPtr);
  std::vector<o2::dataformats::IRFrame> irFrames;
  for (Long64_t iEntry{0}; tPVs->LoadTree(iEntry) >= 0; ++iEntry) {
    tPVs->GetEntry(iEntry);
    for (const auto& pv : pvArr) {
      // make selection of pvs
      if (pv.getNContributors() > 3000) {
        irFrames.emplace_back(pv.getIRMin(), pv.getIRMax());
      }
    }
  }
  // sort to make sure they are in the correct order
  std::sort(irFrames.begin(), irFrames.end(), [](const auto& a, const auto& b) { return a.getMin() < b.getMin(); });
  printf("Selected %zu irFrames\n", irFrames.size());
  auto fIRFrames = TFile::Open("irFrames.root", "RECREATE");
  fIRFrames->WriteObjectAny(&irFrames, "std::vector<o2::dataformats::IRFrame>", "irframes");
  fIRFrames->Close();
}
