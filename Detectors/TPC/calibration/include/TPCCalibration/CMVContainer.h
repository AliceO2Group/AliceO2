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

#include <string>
#include <memory>
#include <stdexcept>
#include <fmt/format.h>

#include "TTree.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

/// CMV data for one TF across all CRUs
/// Raw 16-bit CMV values are stored in a flat C array indexed as [cru * NTimeBinsPerTF + timeBin]
/// CRU::MaxCRU and cmv::NTimeBinsPerTF are compile-time constants, so no dynamic allocation is needed
/// Each TTree entry corresponds to one CMVPerTF object (one TF)
struct CMVPerTF {
  uint32_t firstOrbit{0}; ///< First orbit of this TF, from heartbeatOrbit of the first CMV packet
  uint16_t firstBC{0};    ///< First bunch crossing of this TF, from heartbeatBC of the first CMV packet

  // Raw 16-bit CMV values, flat array indexed as [cru * NTimeBinsPerTF + timeBin]
  uint16_t mDataPerTF[CRU::MaxCRU * cmv::NTimeBinsPerTF]{};

  /// Return the raw 16-bit CMV value for a given CRU and timebin within this TF
  uint16_t getCMV(const int cru, const int timeBin) const
  {
    if (cru < 0 || cru >= static_cast<int>(CRU::MaxCRU)) {
      throw std::out_of_range(fmt::format("CMVPerTF::getCMV: cru {} out of range [0, {})", cru, static_cast<int>(CRU::MaxCRU)));
    }
    if (timeBin < 0 || static_cast<uint32_t>(timeBin) >= cmv::NTimeBinsPerTF) {
      throw std::out_of_range(fmt::format("CMVPerTF::getCMV: timeBin {} out of range [0, {})", timeBin, static_cast<int>(cmv::NTimeBinsPerTF)));
    }
    return mDataPerTF[cru * cmv::NTimeBinsPerTF + timeBin];
  }

  /// Return the float CMV value for a given CRU and timebin within this TF
  float getCMVFloat(const int cru, const int timeBin) const
  {
    auto cmv = getCMV(cru, timeBin);
    const bool positive = (cmv >> 15) & 1;          // bit 15: sign (1=positive, 0=negative)
    const float magnitude = (cmv & 0x7FFF) / 128.f; // lower 15 bits, shift right by 7 (divide by 2^7)
    return positive ? magnitude : -magnitude;
  }

  /// Serialise into a TTree; each Fill() call appends one entry (one TF)
  std::unique_ptr<TTree> toTTree() const;

  /// Write the TTree to a ROOT file
  void writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree) const;

  ClassDefNV(CMVPerTF, 8)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_