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

/// \file IndexTableUtils.h
/// \brief
///
#ifndef STRTRACKING_INCLUDE_INDEXTABLEUTILS_H_
#define STRTRACKING_INCLUDE_INDEXTABLEUTILS_H_

#include "TMath.h"

namespace o2
{
namespace strangeness_tracking
{

struct indexTableUtils {
  int getEtaBin(float eta);
  int getPhiBin(float phi);
  int getBinIndex(float eta, float phi);
  std::vector<int> getBinRect(float eta, float phi, float deltaEta, float deltaPhi);
  int mEtaBins = 64, mPhiBins = 64;
  float minEta = -1.5, maxEta = 1.5;
  float minPhi = 0., maxPhi = 2 * TMath::Pi();
};
} // namespace strangeness_tracking
} // namespace o2

#endif