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

/// \file  FDDDigParam.h
/// \brief Configurable digitization parameters
///
/// \author Andreas Molander <andreas.molander@cern.ch>

#ifndef ALICEO2_FDD_DIG_PARAM
#define ALICEO2_FDD_DIG_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::fdd
{
struct FDDDigParam : o2::conf::ConfigurableParamHelper<FDDDigParam> {
  float hitTimeOffsetA = 0; ///< Hit time offset on the A side [ns]
  float hitTimeOffsetC = 0; ///< Hit time offset on the C side [ns]

  float pmGain = 1e6; ///< PM gain

  O2ParamDef(FDDDigParam, "FDDDigParam");
};
} // namespace o2::fdd

#endif // ALICEO2_FDD_DIG_PARAM
