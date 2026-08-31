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

/// \file ClustererParam.h
/// \brief Definition of the IOTOF clusterer settings

#ifndef ALICEO2_IOTOFCLUSTERERPARAM_H_
#define ALICEO2_IOTOFCLUSTERERPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>
#include <string>

// TO BE REMOVED BEFORE PUSH
#include "Framework/Logger.h"

namespace o2
{
namespace iotof
{
struct ClustererParam : public o2::conf::ConfigurableParamHelper<ClustererParam> {

  int maxTimeDiffNSigma = 3;        ///< maximum time difference in nsigma for clustering
  int maxFiredDigitsForCls = 16;   ///< maximum time difference in nsigma for clustering

  // boilerplate stuff + make principal key
  O2ParamDef(ClustererParam, "TF3ClustererParam");

 private:
  static constexpr float DEFNoisePerPixel()
  {
    return 1e-8; // ITS/MFT values here!!
  }
};

} // namespace iotof
} // namespace o2

#endif
