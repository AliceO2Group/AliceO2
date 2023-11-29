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

#ifndef O2_ZDC_TOWERPARAM_H
#define O2_ZDC_TOWERPARAM_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

/// \file ZDCTowerParam.h
/// \brief ZDC Tower calibration
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct ZDCTowerParam {
  float tower_calib[NChannels] = {0};  // Tower calibration coefficients
  float tower_offset[NChannels] = {0}; // Tower offset
  std::array<bool, NChannels> modified{};
  ZDCTowerParam()
  {
    modified.fill(false);
  }
  void clearFlags();
  void setTowerCalib(uint32_t ich, float val, bool ismodified = true);
  float getTowerCalib(uint32_t ich) const;
  void setTowerOffset(uint32_t ich, float val, bool ismodified = true);
  float getTowerOffset(uint32_t ich) const;
  void print() const;
  ClassDefNV(ZDCTowerParam, 3);
};
} // namespace zdc
} // namespace o2

#endif
