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
/// @brief  Structs for storing CMVs to the CCDB

#ifndef ALICEO2_TPC_CMVCONTAINER_H_
#define ALICEO2_TPC_CMVCONTAINER_H_

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <fmt/format.h>

#include "TTree.h"
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

/// CMV data for one TF across all CRUs
struct CMVPerTF {
  int64_t firstOrbit{0}; ///< First orbit of this TF, from heartbeatOrbit of the first CMV packet
  int64_t firstBC{0};    ///< First bunch crossing of this TF, from heartbeatBC of the first CMV packet

  /// CMV float values indexed as [CRU ID][time bin]
  std::vector<std::vector<float>> mDataPerTF;

  /// Return the CMV value for a given CRU and time bin within this TF
  float getCMV(const int cru, const int timeBin) const
  {
    if (cru < 0 || static_cast<std::size_t>(cru) >= mDataPerTF.size()) {
      throw std::out_of_range(fmt::format("CMVPerTF::getCMV: cru {} out of range [0, {})", cru, mDataPerTF.size()));
    }
    if (timeBin < 0 || static_cast<uint32_t>(timeBin) >= cmv::NTimeBinsPerTF) {
      throw std::out_of_range(fmt::format("CMVPerTF::getCMV: timeBin {} out of range [0, {})", timeBin, cmv::NTimeBinsPerTF));
    }
    return mDataPerTF[cru][timeBin];
  }

  ClassDefNV(CMVPerTF, 1)
};

/// Container holding CMVs for one aggregation interval
struct CMVPerInterval {
  int64_t firstTF{0}; ///< First TF counter seen in this interval
  int64_t lastTF{0};  ///< Last TF counter seen in this interval

  /// CMV data, one CMVPerTF entry per TF, indexed by relative TF [0, nTimeFrames)
  std::vector<CMVPerTF> mCMVPerTF;

  /// Pre-allocate nTFs TF slots; each slot gets mDataPerTF resized to nCRUs entries
  void reserve(uint32_t nTFs, uint32_t nCRUs);

  std::size_t size() const { return mCMVPerTF.size(); }
  bool empty() const { return mCMVPerTF.empty(); }

  /// Clear all data and reset counters
  void clear();

  std::string summary() const;

  /// Serialise into a TTree with a single branch holding the whole CMVPerInterval object
  std::unique_ptr<TTree> toTTree() const;

  /// Write the TTree to a ROOT file
  void writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree) const;

  /// Restore a CMVPerInterval from a TTree previously written by toTTree()
  static CMVPerInterval fromTTree(TTree* tree, int entry = 0);

  ClassDefNV(CMVPerInterval, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_