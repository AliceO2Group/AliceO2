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

#include "DataFormatsFT0/SlewingCoef.h"
using namespace o2::ft0;

SlewingCoef::SlewingPlots_t SlewingCoef::makeSlewingPlots() const
{
  typename o2::ft0::SlewingCoef::SlewingPlots_t plots{};
  for (int iAdc = 0; iAdc < sNAdc; iAdc++) {
    const auto& slewingCoefs = mSlewingCoefs[iAdc];
    auto& plotsAdc = plots[iAdc];
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      const auto& points = slewingCoefs[iCh];
      const int nPoints = points.size();
      auto& plot = plotsAdc[iCh];
      plot = TGraph(nPoints);
      for (int iPoint = 0; iPoint < nPoints; iPoint++) {
        const auto& point = points[iPoint];
        plot.SetPoint(iPoint, point.first, point.second);
      }
    }
  }
  return plots;
}
