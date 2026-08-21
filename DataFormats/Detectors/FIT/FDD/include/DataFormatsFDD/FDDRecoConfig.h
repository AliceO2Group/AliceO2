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

#ifndef ALICEO2_FDD_DIGIT_FILTER_PARAM
#define ALICEO2_FDD_DIGIT_FILTER_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::fdd
{
struct FDDRecoConfig : o2::conf::ConfigurableParamHelper<FDDRecoConfig> {
  double AmplitudeCutOnCollisionTimeWeights = 3;
  O2ParamDef(FDDRecoConfig, "FDDRecoConfig");
  ClassDefNV(FDDRecoConfig, 1);
};
} // namespace o2::fdd
#endif