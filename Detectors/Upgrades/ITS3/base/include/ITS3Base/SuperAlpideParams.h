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

// \brief Collect all possible configurable parameters for Super ALPIDE chips

#ifndef O2_SUPER_ALPIDE_PARAMS_H
#define O2_SUPER_ALPIDE_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its3
{

/// Segmentation parameters for Super ALPIDE chips
struct SuperAlpideParams : public o2::conf::ConfigurableParamHelper<SuperAlpideParams> {
  float mPitchCol = 29.24e-4;        ///< Pixel column size (cm) //FIXME: proxy value to get same resolution as ITS2 given incorrect sensor response
  float mPitchRow = 26.88e-4;        ///< Pixel row size (cm) //FIXME: proxy value to get same resolution as ITS2 given incorrect sensor response
  float mDetectorThickness = 50.e-4; ///< Detector thickness (cm)

  // boilerplate
  O2ParamDef(SuperAlpideParams, "SuperAlpideParams");
};

} // namespace its3
} // namespace o2

#endif
