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

#include <memory>
#include <gsl/span>
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/OrbitData.h"
#include "ZDCCalib/BaselineCalibData.h"
#include "ZDCCalib/BaselineCalibConfig.h"
#ifndef ALICEO2_ZDC_BASELINECALIBEPN_H
#define ALICEO2_ZDC_BASELINECALIBEPN_H
namespace o2
{
namespace zdc
{
class BaselineCalibEPN
{
 public:
  BaselineCalibEPN() = default;
  int init();

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::OrbitData>& orbitdata);
  int endOfRun();
  int saveDebugHistos(const std::string fn = "ZDCBaselineCalibEPN.root");
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }
  void setVerbosity(int val) { mVerbosity = val; }
  BaselineCalibData mData;
  BaselineCalibData& getData() { return mData; }

 private:
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  const ModuleConfig* mModuleConfig = nullptr; /// Trigger/readout configuration object
  int32_t mVerbosity = DbgMinimal;
};
} // namespace zdc
} // namespace o2

#endif
