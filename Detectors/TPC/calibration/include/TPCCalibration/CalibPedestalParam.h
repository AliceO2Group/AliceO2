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

/// \file CalibPedestalParam.h
/// \brief Implementation of the parameter class for the pedestal calibration
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_CalibPedestalParam_H_
#define ALICEO2_TPC_CalibPedestalParam_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

struct CalibPedestalParam : public o2::conf::ConfigurableParamHelper<CalibPedestalParam> {
  int FirstTimeBin{0};                                  ///< first time bin used in analysis
  int LastTimeBin{500};                                 ///< first time bin used in analysis
  int ADCMin{0};                                        ///< minimum adc value
  int ADCMax{120};                                      ///< maximum adc value
  StatisticsType StatType{StatisticsType::GausFitFast}; ///< statistics type to be used for pedestal and noise evaluation

  O2ParamDef(CalibPedestalParam, "TPCCalibPedestal");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_CalibPedestalParam_H_
