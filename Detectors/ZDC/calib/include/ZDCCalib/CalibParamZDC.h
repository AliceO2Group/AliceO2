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
#include "CCDB/CcdbObjectInfo.h"
#include <string>

/// \file CalibParamZDC.h
/// \brief ZDC calibration common parameters
/// \author P. Cortese

namespace o2
{
namespace zdc
{
struct CalibParamZDC : public o2::conf::ConfigurableParamHelper<CalibParamZDC> {
  bool debugOutput = true;                           // Debug output
  bool rootOutput = true;                            // Output histograms to EOS
  std::string outputDir = "./";                      // ROOT files output directory
  std::string metaFileDir = "/dev/null";             // Metafile output directory
  std::string descr;                                 // Calibration description
  int64_t eovTune = -o2::ccdb::CcdbObjectInfo::YEAR; // Tune end of validity of calibration object (eovTune>0 -> absolute, eovTune<0 increase by -eovTune)

  int updateCcdbObjectInfo(o2::ccdb::CcdbObjectInfo& info) const;
  void print() const;
  O2ParamDef(CalibParamZDC, "CalibParamZDC");
};
} // namespace zdc
} // namespace o2

#endif
