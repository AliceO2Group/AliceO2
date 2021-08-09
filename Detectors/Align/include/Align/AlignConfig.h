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

/// @file   AlignConfig.h
/// @brief  Configuration file for global alignment

#ifndef ALICEO2_ALIGN_CONFIG_H
#define ALICEO2_ALIGN_CONFIG_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace align
{

struct AlignConfig : public o2::conf::ConfigurableParamHelper<AlignConfig> {
  enum TrackType { Collision,
                   Cosmic,
                   NTrackTypes };

  float q2PtMin[NTrackTypes] = {0.01, 0.01};
  float q2PtMax[NTrackTypes] = {10., 10.};
  float tglMax[NTrackTypes] = {3., 10.};

  int minPoints[NTrackTypes] = {4, 10};
  int minDetAcc[NTrackTypes] = {1, 1};

  int vtxMinCont = 2;     // require min number of contributors in Vtx
  int vtxMaxCont = 99999; // require max number of contributors in Vtx
  int vtxMinContVC = 20;  // min number of contributors to use as constraint

  float maxDCAforVC[2]; // DCA cut in R,Z to allow track be subjected to vertex constraint
  float maxChi2forVC;   // track-vertex chi2 cut to allow the track be subjected to vertex constraint

  O2ParamDef(AlignConfig, "align-conf");
};

} // namespace align
} // namespace o2

#endif
