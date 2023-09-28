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
/// \file StrangenessTrackingConfigParam.h
/// \brief

#ifndef ALICEO2_STRANGENESS_TRACKING_PARAM_H_
#define ALICEO2_STRANGENESS_TRACKING_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace strangeness_tracking
{

struct StrangenessTrackingParamConfig : public o2::conf::ConfigurableParamHelper<StrangenessTrackingParamConfig> {

  // parameters
  float mRadiusTolIB = .3;     // Radius tolerance for matching V0s in the IB
  float mRadiusTolOB = .1;     // Radius tolerance for matching V0s in the OB
  float mPhiBinSize = 0.02;    // Phi bin size for the matching grid
  float mEtaBinSize = 0.01;    // Eta bin size for the matching grid
  float mMinMotherClus = 3.;   // minimum number of cluster to be attached to the mother
  float mMaxChi2 = 50;         // Maximum matching chi2
  bool mVertexMatching = true; // Flag to enable/disable vertex matching

  O2ParamDef(StrangenessTrackingParamConfig, "strtracker");
};

} // namespace strangeness_tracking
} // namespace o2
#endif