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

//#define O2_ZDC_WAVEFORMCALIB_DEBUG

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <string>

/// \file WaveformCalibConfig.h
/// \brief Configuration of ZDC Tower intercalibration procedure
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct WaveformCalibConfig {
  static constexpr int NBB = WaveformCalib_NBB;
  static constexpr int NBA = WaveformCalib_NBA;
  static constexpr int NBT = WaveformCalib_NBT;
  static constexpr int NW = WaveformCalib_NW;

  WaveformCalibConfig();

  double cutLow[NChannels]{};         /// Amplitude cut low
  double cutHigh[NChannels]{};        /// Amplitude cut high
  double min_e[NChannels]{};          /// Minimum entries to compute waveform
  double cutTimeLow[NTDCChannels]{};  /// TDC cut low
  double cutTimeHigh[NTDCChannels]{}; /// TDC cut high
  std::string desc = "";
  int ibeg = -NBB;
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
  void setTimeCuts(double low, double high);
  void setTimeCuts(int itdc, double low, double high);
  void setDescription(std::string d) { desc = d; }

  int getFirst() const
  {
    return ibeg;
  }

  int getLast() const
  {
    return iend;
  }

  ClassDefNV(WaveformCalibConfig, 1);
};
} // namespace zdc
} // namespace o2

#endif
