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

#include "TTree.h"

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
  void reserve(uint32_t expectedTFs, uint32_t expectedCRUs);

  /// Append one (cmv, cru, timebin, tf) tuple
  void addEntry(float cmvVal, uint32_t cruID, uint32_t tb, uint32_t tfCounter);

  /// Append one full CRU packet (NTimeBinsPerPacket consecutive timebins)
  /// \param packet    pointer to NTimeBinsPerPacket floats
  /// \param cruID     CRU index
  /// \param tbOffset  absolute timebin of the first sample in this packet
  /// \param tfCounter TF counter
  void addPacket(const float* packet, uint32_t cruID, uint32_t tbOffset, uint32_t tfCounter);

  std::size_t size() const;
  bool empty() const;

  /// Clear all data and reset counters
  void clear();

  std::string summary() const;

  /// Build an in-memory TTree with one branch per field and one entry per tuple
  std::unique_ptr<TTree> toTTree() const;

  /// Write the container as a TTree inside a TFile on disk
  /// \param filename  path to the output ROOT file
  void writeToFile(const std::string& filename) const;

  /// Restore a CMVContainer from a TTree previously written by toTTree()
  static CMVContainer fromTTree(TTree* tree, int entry = 0);

  ClassDefNV(CMVContainer, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_