// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author ruben.shahoyan@cern.ch

/// parameters to bias precalibrated mean vertex, e.g after the alignment shift

#ifndef ALICEO2_MEANVERTEX_BIAS_PARAM_H
#define ALICEO2_MEANVERTEX_BIAS_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace dataformats
{

struct MeanVertexBiasParam : public o2::conf::ConfigurableParamHelper<MeanVertexBiasParam> {
  float xyz[3] = {};  // position bias
  float slopeX = 0.f; // x slope bias
  float slopeY = 0.f; // y slope bias

  O2ParamDef(MeanVertexBiasParam, "mvbias");
};

} // namespace dataformats
} // end namespace o2

#endif
