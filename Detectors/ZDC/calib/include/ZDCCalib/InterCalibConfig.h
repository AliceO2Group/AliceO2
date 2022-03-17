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

/// \file InterCalibConfig.h
/// \brief Configuration of ZDC Tower intercalibration procedure
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct InterCalibConfig {
  static constexpr int NH = 5; /// ZNA, ZPA, ZNC, ZPC, ZEM
  double cutLow[NH] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  double cutHigh[NH] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  int nb1[NH] = {0};
  double amin1[NH] = {0};
  double amax1[NH] = {0};
  int nb2[NH] = {0};
  double amin2[NH] = {0};
  double amax2[NH] = {0};
  void print();
  void setBinning1D(int nb, double amin, double amax);
  void setBinning2D(int nb, double amin, double amax);
  {
  ClassDefNV(InterCalibConfig, 1);
};
} // namespace zdc
} // namespace o2

#endif
