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

/// @file   CMVContainer.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch

#include <stdexcept>
#include <fmt/format.h>

#include "TFile.h"

#include "TPCCalibration/CMVContainer.h"

namespace o2::tpc
{

void CMVPerInterval::reserve(uint32_t nTFs, uint32_t nCRUs)
{
  mCMVPerTF.resize(nTFs);
  for (auto& tfData : mCMVPerTF) {
    tfData.mDataPerTF.resize(nCRUs);
  }
}

void CMVPerInterval::clear()
{
  mCMVPerTF.clear();
  firstTF = 0;
  lastTF = 0;
}

std::string CMVPerInterval::summary() const
{
  const std::size_t nCRUs = empty() ? 0 : mCMVPerTF.front().mDataPerTF.size();
  return fmt::format("CMVPerInterval: {} TFs, {} CRU slots, firstTF={}, lastTF={}",
                     size(), nCRUs, firstTF, lastTF);
}

std::unique_ptr<TTree> CMVPerInterval::toTTree() const
{
  if (empty()) {
    throw std::runtime_error("CMVPerInterval::toTTree() called on empty container");
  }

  auto tree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
  tree->SetAutoSave(0);
  tree->SetDirectory(nullptr);

  const CMVPerInterval* ptr = this;
  tree->Branch("CMVPerInterval", &ptr);
  tree->Fill();

  tree->ResetBranchAddresses();

  return tree;
}

void CMVPerInterval::writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree) const
{
  TFile f(filename.c_str(), "RECREATE");
  if (f.IsZombie()) {
    throw std::runtime_error(fmt::format("CMVPerInterval::writeToFile: cannot open '{}'", filename));
  }
  tree->Write();
  f.Close();
}

CMVPerInterval CMVPerInterval::fromTTree(TTree* tree, int entry)
{
  if (!tree) {
    throw std::runtime_error("CMVPerInterval::fromTTree: null TTree pointer");
  }

  CMVPerInterval* ptr = nullptr;
  tree->SetBranchAddress("CMVPerInterval", &ptr);
  tree->GetEntry(entry);

  if (!ptr) {
    throw std::runtime_error("CMVPerInterval::fromTTree: failed to read object from TTree");
  }

  CMVPerInterval result = std::move(*ptr);
  delete ptr;
  return result;
}

} // namespace o2::tpc