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

/// \file FEEConfig.h
/// \brief Frontend electronics configuration values
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_FEEConfig_H_
#define AliceO2_TPC_FEEConfig_H_

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "Rtypes.h"

#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"

namespace o2::tpc
{
struct CRUConfig {
  static constexpr int NConfigValues = 6; ///< number of configuration values

  bool linkOn{false};     ///< if the link is active
  bool cmcEnabled{false}; ///< if common mode correction is enabled
  bool itfEnabled{false}; ///< if ion tail filter correction is enabled
  bool zsEnabled{false};  ///< if zero suppression is enabled
  float zsOffset{0.f};    ///< zero suppression offset value
  float itCorr0{1.f};     ///< ion tail scaling parameter

  bool setValues(std::string_view cruData);

  ClassDefNV(CRUConfig, 1);
};

struct FEEConfig {
  using CalPadMapType = std::unordered_map<std::string, CalPad>;
  FEEConfig() { cruConfig.resize(CRU::MaxCRU); }
  // FEEConfig& operator=(const FEEConfig& config)
  //{
  // padMaps = config.pad
  // return *this;
  //}

  CalPadMapType padMaps;            ///< pad-wise configuration data
  std::vector<CRUConfig> cruConfig; ///< CRU configuration values

  void clear()
  {
    for ([[maybe_unused]] auto& [key, val] : padMaps) {
      val *= 0;
    }

    for (auto& val : cruConfig) {
      val = CRUConfig();
    }
  }

  ClassDefNV(FEEConfig, 1);
};

} // namespace o2::tpc
#endif
