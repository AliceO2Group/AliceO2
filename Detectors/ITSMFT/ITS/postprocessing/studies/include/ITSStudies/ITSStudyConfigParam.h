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

#ifndef ITS_STUDY_CONFIG_PARAM_H
#define ITS_STUDY_CONFIG_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{
namespace study
{
struct ITSCheckTracksParamConfig : public o2::conf::ConfigurableParamHelper<ITSCheckTracksParamConfig> {
  std::string outFileName = "TrackCheckStudy.root";
  size_t effHistBins = 100;
  unsigned short trackLengthMask = 0x7f;
  float effPtCutLow = 0.01;
  float effPtCutHigh = 10.;

  O2ParamDef(ITSCheckTracksParamConfig, "ITSCheckTracksParam");
};
} // namespace study
} // namespace its
} // namespace o2

#endif