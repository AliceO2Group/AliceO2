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

#ifndef O2_ZDC_INTERCALIBCONFIG_H
#define O2_ZDC_INTERCALIBCONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>
#include <string>
#include <limits>

/// \file InterCalibConfig.h
/// \brief Configuration of ZDC Tower intercalibration procedure
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct InterCalibConfig {
  static constexpr int NH = 7; /// ZNA, ZPA, ZNC, ZPC, ZEM, ZNI, ZPI
  bool enabled[NH] = {true, true, true, true, true, true, true};
  double cutLow[NH] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  double cutHigh[NH] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  int nb1[NH] = {0};      /// 1D histogram: number of bins
  double amin1[NH] = {0}; /// minimum
  double amax1[NH] = {0}; /// maximum
  int nb2[NH] = {0};      /// 2D histogram: number of bins
  double amin2[NH] = {0}; /// minimum
  double amax2[NH] = {0}; /// maximum
  double l_bnd[NH] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double u_bnd[NH] = {10., 10., 10., 10., 10., 10., 10.};
  double l_bnd_o[NH] = {-20., -20., -20., -20., -20., -20., -20.};
  double u_bnd_o[NH] = {20., 20., 20., 20., 20., 20., 20.};
  double step_o[NH] = {0., 0., 0., 0., 0., 0., 0.};
  double min_e[NH] = {0., 0., 0., 0., 0., 0., 0.};
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
  void enable(bool c0, bool c1, bool c2, bool c3, bool c4, bool c5, bool c6)
  {
    enabled[0] = c0;
    enabled[1] = c1;
    enabled[2] = c2;
    enabled[3] = c3;
    enabled[4] = c4;
    enabled[5] = c5;
    enabled[6] = c6;
  }
  ClassDefNV(InterCalibConfig, 3);
};
} // namespace zdc
} // namespace o2

#endif
