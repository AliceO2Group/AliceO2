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

#ifndef TPC_IonTailCorrectionSettings_H_
#define TPC_IonTailCorrectionSettings_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::tpc
{

struct IonTailCorrectionSettings : public o2::conf::ConfigurableParamHelper<IonTailCorrectionSettings> {
  float ITMultFactor = 1; ///< fudge factor to tune IT correction
  float kTime = 0.0515;   ///< kTime constant for ion tail filter
  O2ParamDef(IonTailCorrectionSettings, "TPCITCorr");
};

} // namespace o2::tpc
#endif
