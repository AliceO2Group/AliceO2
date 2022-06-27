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

#ifndef O2_ZDC_BASELINECALIBCONFIG_H
#define O2_ZDC_BASELINECALIBCONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <string>

/// \file BaselineCalibConfig.h
/// \brief Configuration of ZDC Baseline calibration procedure
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct BaselineCalibConfig {

  BaselineCalibConfig();

  int cutLow[NChannels];           /// Baseline cut low
  int cutHigh[NChannels];          /// Baseline cut high
  uint32_t min_e[NChannels] = {0}; /// Minimum entries to compute baseline
  std::string desc = "";

  void print() const;
  void resetCuts();
  void setMinEntries(uint32_t val);
  void setMinEntries(int ih, uint32_t val);
  void setCutLow(int val);
  void setCutHigh(int val);
  void setCutLow(int ih, int val);
  void setCutHigh(int ih, int val);
  void setCuts(int low, int high);
  void setCuts(int ih, int low, int high);
  void setDescription(std::string d) { desc = d; }

  ClassDefNV(BaselineCalibConfig, 1);
};
} // namespace zdc
} // namespace o2

#endif
