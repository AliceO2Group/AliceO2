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

#ifndef ALICEO2_FT0_SLEWINGCOEF_H_
#define ALICEO2_FT0_SLEWINGCOEF_H_
////////////////////////////////////////////////
// Slewing coefficients for FT0
//////////////////////////////////////////////
#include "TGraph.h"

#include <vector>
#include <array>
#include <utility>

namespace o2
{
namespace ft0
{

struct SlewingCoef {
  constexpr static int sNCHANNELS = 208;
  constexpr static int sNAdc = 2;
  using Point_t = std::pair<double, double>;                                    // points, x and y
  using VecPoints = std::vector<Point_t>;                                       // will be processed into object(TGraph) for applying claibration
  using VecSlewingCoefs = std::array<std::array<VecPoints, sNCHANNELS>, sNAdc>; // 0 - adc0, 1 - adc1
  typedef std::array<std::array<TGraph, sNCHANNELS>, sNAdc> SlewingPlots_t;
  VecSlewingCoefs mSlewingCoefs{};
  SlewingPlots_t makeSlewingPlots() const;
  constexpr static const char* getObjectPath()
  {
    return "FT0/Calib/SlewingCoef";
  }
  ClassDefNV(SlewingCoef, 1)
};

} // namespace ft0
} // namespace o2

#endif
