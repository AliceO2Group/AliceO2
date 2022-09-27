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

#ifndef O2_ZDC_CALIBPARAMZDC_H
#define O2_ZDC_CALIBPARAMZDC_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "ZDCBase/Constants.h"
#include <string>

/// \file CalibParamZDC.h
/// \brief ZDC calibration common parameters
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct CalibParamZDC : public o2::conf::ConfigurableParamHelper<CalibParamZDC> {
  bool rootOutput = true;                // Debug output
  std::string outputDir = "./";          // ROOT files output directory
  std::string metaFileDir = "/dev/null"; // Metafile output directory
  std::string descr;                     // Calibration description
  void print();
  O2ParamDef(CalibParamZDC, "CalibParamZDC");
};
} // namespace zdc
} // namespace o2

#endif
