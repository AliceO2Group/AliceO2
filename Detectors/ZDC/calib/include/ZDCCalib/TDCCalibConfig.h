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

#ifndef O2_ZDC_TDCCALIBCONFIG_H
#define O2_ZDC_TDCCALIBCONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <string>
#include <limits>

/// \file TDCCalibConfig.h
/// \brief Configuration of ZDC TDC calibration procedure
/// \author L. Quaglia

namespace o2
{
namespace zdc
{
struct TDCCalibConfig {
  static constexpr int NTDCChannels = 10; //number of TDC channels
  double cutLow[NTDCChannels] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  double cutHigh[NTDCChannels] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  bool enabled[NTDCChannels] = {true, true, true, true, true, true, true, true, true, true}; //ZNAC, ZNAS, ZPAC, ZPAS, ZEM1, ZEM2, ZNCC, ZNCS, ZPCC, ZPCS
  int nb1[NTDCChannels] = {0};                                                               /// 1D histogram: number of bins
  double amin1[NTDCChannels] = {0};                                                          /// minimum
  double amax1[NTDCChannels] = {0};                                                          /// maximum
  int nb2[NTDCChannels] = {0};                                                               /// 2D histogram: number of bins
  double amin2[NTDCChannels] = {0};                                                          /// minimum
  double amax2[NTDCChannels] = {0};                                                          /// maximum
  double l_bnd[NTDCChannels] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double u_bnd[NTDCChannels] = {10., 10., 10., 10., 10., 10., 10., 10., 10., 10.};
  double l_bnd_o[NTDCChannels] = {-20., -20., -20., -20., -20., -20., -20., -20., -20., -20.};
  double u_bnd_o[NTDCChannels] = {20., 20., 20., 20., 20., 20., 20., 20., 20., 20.};
  double step_o[NTDCChannels] = {0};
  double min_e[NTDCChannels] = {0};
  std::string desc = "";

  void print() const;
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
  void setBinning1D(int nb, double amin, double amax);
  void setBinning2D(int nb, double amin, double amax);
  void setBinning1D(int ih, int nb, double amin, double amax);
  void setBinning2D(int ih, int nb, double amin, double amax);
  void setDescription(std::string d) { desc = d; }
  void enable(bool c0, bool c1, bool c2, bool c3, bool c4, bool c5, bool c6, bool c7, bool c8, bool c9)
  {
    enabled[0] = c0;
    enabled[1] = c1;
    enabled[2] = c2;
    enabled[3] = c3;
    enabled[4] = c4;
    enabled[5] = c5;
    enabled[6] = c6;
    enabled[7] = c7;
    enabled[8] = c8;
    enabled[9] = c9;
  }

  ClassDefNV(TDCCalibConfig, 3);
};
} // namespace zdc
} // namespace o2

#endif
