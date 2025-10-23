// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define _USE_MATH_DEFINES

// root includes
#include "TFile.h"
#include "TH1I.h"

// o2 includes
#include "TPCQC/GPUErrorQA.h"
#include "GPUErrors.h"

ClassImp(o2::tpc::qc::GPUErrorQA);

using namespace o2::tpc::qc;

//______________________________________________________________________________
void GPUErrorQA::initializeHistograms()
{
  TH1::AddDirectory(false);

  auto const& errorNames = o2::gpu::GPUErrors::getErrorNames();

  int maxErrorCode = 1;
  for (const auto& [key, _] : errorNames) {
    if (static_cast<int>(key) > maxErrorCode) {
      maxErrorCode = key;
    }
  }

  // 1D histogram counting all reported errors
  mMapHist["ErrorCounter"] = std::make_unique<TH1I>("ErrorCounter", "ErrorCounter", maxErrorCode, -0.5, maxErrorCode - 0.5);
  mMapHist["ErrorCounter"]->GetXaxis()->SetTitle("Error Codes");
  mMapHist["ErrorCounter"]->GetYaxis()->SetTitle("Entries");
  // for convienence, label each bin with the error name
  for (size_t bin = 1; bin <= maxErrorCode; bin++) {
    auto const& it = errorNames.find(bin);
    if (it != errorNames.end()) {
      mMapHist["ErrorCounter"]->GetXaxis()->SetBinLabel(bin, it->second);
    } else {
      mMapHist["ErrorCounter"]->GetXaxis()->SetBinLabel(bin, "NO_DEF");
    }
  }
}
//______________________________________________________________________________
void GPUErrorQA::resetHistograms()
{
  for (const auto& pair : mMapHist) {
    pair.second->Reset();
  }
}
//______________________________________________________________________________
void GPUErrorQA::processErrors(std::vector<std::array<uint32_t, 4>> errors)
{
  for (const auto& error : errors) {
    uint32_t errorCode = error[0];
    mMapHist["ErrorCounter"]->AddBinContent(errorCode);
  }
}

//______________________________________________________________________________
void GPUErrorQA::dumpToFile(const std::string filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.data(), "recreate"));
  TObjArray arr;
  arr.SetName("GPUErrorQA_Hists");
  for ([[maybe_unused]] const auto& [name, hist] : mMapHist) {
    arr.Add(hist.get());
  }
  arr.Write(arr.GetName(), TObject::kSingleKey);
}
