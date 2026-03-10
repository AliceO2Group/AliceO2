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
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

void CMVContainer::reserve(uint32_t expectedTFs, uint32_t expectedCRUs)
{
  const std::size_t n = static_cast<std::size_t>(expectedTFs) * expectedCRUs * o2::tpc::cmv::NTimeBinsPerTF;
  cmvValues.reserve(n);
  cru.reserve(n);
  timebin.reserve(n);
  tf.reserve(n);
}

void CMVContainer::addEntry(float cmvVal, uint32_t cruID, uint32_t tb, uint32_t tfCounter)
{
  cmvValues.push_back(cmvVal);
  cru.push_back(cruID);
  timebin.push_back(tb);
  tf.push_back(tfCounter);
}

void CMVContainer::addPacket(const float* packet, uint32_t cruID, uint32_t tbOffset, uint32_t tfCounter)
{
  for (uint32_t tb = 0; tb < o2::tpc::cmv::NTimeBinsPerPacket; ++tb) {
    addEntry(packet[tb], cruID, tbOffset + tb, tfCounter);
  }
}

std::size_t CMVContainer::size() const { return cmvValues.size(); }

bool CMVContainer::empty() const { return cmvValues.empty(); }

void CMVContainer::clear()
{
  cmvValues.clear();
  cru.clear();
  timebin.clear();
  tf.clear();
  nTFs = 0;
  nCRUs = 0;
  firstTF = 0;
}

std::string CMVContainer::summary() const
{
  return fmt::format("CMVContainer: {} entries, {} TFs, {} CRUs, firstTF={}",
                     size(), nTFs, nCRUs, firstTF);
}

std::unique_ptr<TTree> CMVContainer::toTTree() const
{
  const std::size_t n = size();
  if (n == 0) {
    throw std::runtime_error("CMVContainer::toTTree() called on empty container");
  }

  auto tree = std::make_unique<TTree>("ccdb_object", "ccdb_object");
  tree->SetAutoSave(0);
  tree->SetDirectory(nullptr);

  // Point branches directly at the vector data
  float* pCmv = const_cast<float*>(cmvValues.data());
  uint32_t* pCru = const_cast<uint32_t*>(cru.data());
  uint32_t* pTimebin = const_cast<uint32_t*>(timebin.data());
  uint32_t* pTf = const_cast<uint32_t*>(tf.data());

  tree->Branch("cmv", pCmv, fmt::format("cmv[{}]/F", n).c_str());
  tree->Branch("cru", pCru, fmt::format("cru[{}]/i", n).c_str());
  tree->Branch("timebin", pTimebin, fmt::format("timebin[{}]/i", n).c_str());
  tree->Branch("tf", pTf, fmt::format("tf[{}]/i", n).c_str());

  tree->Fill();
  return tree;
}

void CMVContainer::writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree) const
{
  TFile f(filename.c_str(), "RECREATE");
  if (f.IsZombie()) {
    throw std::runtime_error(fmt::format("CMVContainer::writeToFile: cannot open '{}'", filename));
  }
  tree->Write();
  f.Close();
}

CMVContainer CMVContainer::fromTTree(TTree* tree, int entry)
{
  if (!tree) {
    throw std::runtime_error("CMVContainer::fromTTree: null TTree pointer");
  }

  CMVContainer c;
  const Long64_t nEntries = tree->GetEntries();
  if (nEntries <= 0) {
    return c;
  }

  // Read the array branches back into vectors in one GetEntry() call
  std::vector<float> bCmv(nEntries);
  std::vector<uint32_t> bCru(nEntries), bTimebin(nEntries), bTf(nEntries);

  tree->SetBranchAddress("cmv", bCmv.data());
  tree->SetBranchAddress("cru", bCru.data());
  tree->SetBranchAddress("timebin", bTimebin.data());
  tree->SetBranchAddress("tf", bTf.data());

  tree->GetEntry(entry);

  c.cmvValues = std::move(bCmv);
  c.cru = std::move(bCru);
  c.timebin = std::move(bTimebin);
  c.tf = std::move(bTf);

  return c;
}

} // namespace o2::tpc
