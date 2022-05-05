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

#ifndef O2_ZDC_WAVEFORMCALIBCONFIG_H
#define O2_ZDC_WAVEFORMCALIBCONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <string>

/// \file WaveformCalibConfig.h
/// \brief Configuration of ZDC Tower intercalibration procedure
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct WaveformCalibConfig {

  WaveformCalibConfig();

  static constexpr int NH = NTDCChannels;
  static constexpr int NBB = 3;
  static constexpr int NBA = 6;
  static constexpr int NBT = NBB + NBA + 1;

  double cutLow[NH];
  double cutHigh[NH];
  double min_e[NH] = {0.};
  std::string desc = "";
  int ibeg = - NBB;
  int iend = NBA;
  int nbun = iend - ibeg + 1;

  void print() const;
  void restrictRange(int ib, int ie);
  void resetCuts();
  void resetCutLow();
  void resetCutHigh();
  void resetCutLow(int ih);
  void resetCutHigh(int ih);
  void setMinEntries(double val);
  void setMinEntries(int ih, double val);
  void setCutLow(double val);
  void setCutHigh(double val);
  void setCutLow(int ih, double val);
  void setCutHigh(int ih, double val);
  void setCuts(double low, double high);
  void setCuts(int ih, double low, double high);
  void setDescription(std::string d) { desc = d; }

  ClassDefNV(WaveformCalibConfig, 1);
};
} // namespace zdc
} // namespace o2

#endif
