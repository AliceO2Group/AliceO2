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

/// \file DigitDumpParam.h
/// \brief Implementation of the parameter class for the hardware clusterer
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_DigitDumpParam_H_
#define ALICEO2_TPC_DigitDumpParam_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct DigitDumpParam : public o2::conf::ConfigurableParamHelper<DigitDumpParam> {
  int FirstTimeBin{0};                ///< first time bin used in analysis
  int LastTimeBin{1000};              ///< first time bin used in analysis
  int ADCMin{-100};                   ///< minimum adc value
  int ADCMax{1024};                   ///< maximum adc value
  float NoiseThreshold{-1};           ///< zero suppression threshold in noise sigma
  std::string PedestalAndNoiseFile{}; ///< file name for the pedestal and nosie file

  O2ParamDef(DigitDumpParam, "TPCDigitDump");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_HwClustererParam_H_
