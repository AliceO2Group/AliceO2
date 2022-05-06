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

#ifndef O2_MCH_DIGITFILTERING_DIGIT_FILTER_PARAM_H_
#define O2_MCH_DIGITFILTERING_DIGIT_FILTER_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

/**
 * @class DigitFilterParam
 * @brief Configurable parameters for the digit filtering
 */
struct DigitFilterParam : public o2::conf::ConfigurableParamHelper<DigitFilterParam> {

  bool sanityCheck = false; ///< whether or not to perform some sanity checks on the input digits
  int minADCValue = 0;      ///< digits with an ADC below or equal to this value are discarded

  O2ParamDef(DigitFilterParam, "MCHDigitFilter");
};

} // namespace o2::mch

#endif
