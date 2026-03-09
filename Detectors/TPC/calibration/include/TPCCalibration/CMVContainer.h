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

/// @file   CMVContainer.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Struct for storing CMVs to the CCDB

#ifndef ALICEO2_TPC_CMVCONTAINER_H_
#define ALICEO2_TPC_CMVCONTAINER_H_

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include <fmt/format.h>

#include "TTree.h"
#include "TFile.h"

#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

/// CMVContainer: accumulator for one aggregation interval
struct CMVContainer {

  uint32_t nTFs{0};  ///< number of TFs accumulated
  uint32_t nCRUs{0}; ///< number of contributing CRUs
  long firstTF{0};   ///< first TF counter in this aggregation interval

  std::vector<float> cmvValues{};  ///< CMV float values
  std::vector<uint32_t> cru{};     ///< CRU indices
  std::vector<uint32_t> timebin{}; ///< absolute timebins within the TF
  std::vector<uint32_t> tf{};      ///< TF counters

  /// Pre-allocate storage for the expected number of entries: expectedTFs × expectedCRUs × NTimeBinsPerTF
  void reserve(uint32_t expectedTFs, uint32_t expectedCRUs)
  {
    const std::size_t n = static_cast<std::size_t>(expectedTFs) * expectedCRUs * o2::tpc::cmv::NTimeBinsPerTF;
    cmvValues.reserve(n);
    cru.reserve(n);
    timebin.reserve(n);
    tf.reserve(n);
  }

  /// Append one (cmv, cru, timebin, tf) tuple
  void addEntry(float cmvVal, uint32_t cruID, uint32_t tb, uint32_t tfCounter)
  {
    cmvValues.push_back(cmvVal);
    cru.push_back(cruID);
    timebin.push_back(tb);
    tf.push_back(tfCounter);
  }

  /// Append one full CRU packet (NTimeBinsPerPacket consecutive timebins)
  /// \param packet    pointer to NTimeBinsPerPacket floats
  /// \param cruID     CRU index
  /// \param tbOffset  absolute timebin of the first sample in this packet
  /// \param tfCounter TF counter
  void addPacket(const float* packet, uint32_t cruID, uint32_t tbOffset, uint32_t tfCounter)
  {
    for (uint32_t tb = 0; tb < o2::tpc::cmv::NTimeBinsPerPacket; ++tb) {
      addEntry(packet[tb], cruID, tbOffset + tb, tfCounter);
    }
  }

  std::size_t size() const { return cmvValues.size(); }
  bool empty() const { return cmvValues.empty(); }

  /// Clear all data and reset counters
  void clear()
  {
    cmvValues.clear();
    cru.clear();
    timebin.clear();
    tf.clear();
    nTFs = 0;
    nCRUs = 0;
    firstTF = 0;
  }

  std::string summary() const
  {
    return fmt::format("CMVContainer: {} entries, {} TFs, {} CRUs, firstTF={}",
                       size(), nTFs, nCRUs, firstTF);
  }

  /// Build an in-memory TTree with one branch per field and one entry per tuple
  std::unique_ptr<TTree> toTTree() const
  {
    const std::size_t n = size();
    if (n == 0) {
      throw std::runtime_error("CMVContainer::toTTree() called on empty container");
    }

    auto tree = std::make_unique<TTree>("CMVTree", "TPC common mode values");
    tree->SetAutoSave(0);

    // Point branches directly at the vector data — single Fill() call writes all rows
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

  /// Write the container as a TTree inside a TFile on disk
  /// \param filename  path to the output ROOT file
  void writeToFile(const std::string& filename) const
  {
    TFile f(filename.c_str(), "RECREATE");
    if (f.IsZombie()) {
      throw std::runtime_error(fmt::format("CMVContainer::writeToFile: cannot open '{}'", filename));
    }
    auto tree = toTTree();
    tree->Write();
    f.Close();
  }

  /// Restore a CMVContainer from a TTree previously written by toTTree()
  static CMVContainer fromTTree(TTree* tree)
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

    tree->GetEntry(0);

    c.cmvValues = std::move(bCmv);
    c.cru = std::move(bCru);
    c.timebin = std::move(bTimebin);
    c.tf = std::move(bTf);

    return c;
  }

  ClassDefNV(CMVContainer, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_