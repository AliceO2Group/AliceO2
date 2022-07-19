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

/// \file SACParameter.h
/// \brief Definition of the parameters for the SAC processing
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_SACPARAMETER_H_
#define ALICEO2_TPC_SACPARAMETER_H_

#include <array>
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct ParameterSAC : public o2::conf::ConfigurableParamHelper<ParameterSAC> {
  float minSAC0Median = 4;         ///< this value is used for identifying outliers (pads with high SAC0 values): "accepted SAC 0 values > median_SAC0 - stdDev * minSAC0Median"
  float maxSAC0Median = 4;         ///< this value is used for identifying outliers (pads with high SAC0 values): "accepted SAC 0 values < median_SAC0 + stdDev * maxSAC0Median"
  float maxSACDeltaValue = 100.f;  ///< maximum Delta SAC
  float minSACDeltaValue = -100.f; ///< minimum Delta SAC
  O2ParamDef(ParameterSAC, "TPCSACParam");
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterGEM_H_
