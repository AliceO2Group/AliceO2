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
#include "ZDCCalib/BaselineCalibData.h"
#include "ZDCCalib/BaselineCalibConfig.h"
#include "ZDCReconstruction/BaselineParam.h"
#include "CCDB/CcdbObjectInfo.h"
#ifndef ALICEO2_ZDC_BASELINECALIB_H
#define ALICEO2_ZDC_BASELINECALIB_H
namespace o2
{
namespace zdc
{
class BaselineCalib
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  BaselineCalib() = default;
  int init();

  void setConfig(const BaselineCalibConfig* param) { mConfig = param; };
  const BaselineCalibConfig* getConfig() const { return mConfig; };
  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  void setBaselineParam(const BaselineParam* param) { mParam = param; };
  const BaselineParam* getBaselineParam() const { return mParam; };

  void resetInitFlag() { mInitDone = false; };
  void clear();
  //int process(const o2::zdc::BaselineCalibSummaryData& data);
  int process(const o2::zdc::BaselineCalibSummaryData* data);
  int endOfRun();
  int saveDebugHistos(const std::string fn = "ZDCBaselineCalib.root");
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }

  CcdbObjectInfo& getCcdbObjectInfo() { return mInfo; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  BaselineCalibData& getData() { return mData; }
  BaselineParam& getParamUpd() { return mParamUpd; }

 private:
  BaselineCalibData mData;
  const BaselineParam* mParam = nullptr;
  BaselineParam mParamUpd;
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mVerbosity = DbgMinimal;
  const BaselineCalibConfig* mConfig = nullptr; /// Configuration of intercalibration
  const ModuleConfig* mModuleConfig = nullptr;  /// Trigger/readout configuration object
  CcdbObjectInfo mInfo;                         /// CCDB Info
};
} // namespace zdc
} // namespace o2

#endif
