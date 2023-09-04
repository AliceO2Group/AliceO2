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
#include "CommonDataFormat/FlatHisto1D.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCCalib/NoiseCalibData.h"
#include "ZDCReconstruction/NoiseParam.h"
#include "CCDB/CcdbObjectInfo.h"
#ifndef ALICEO2_ZDC_NOISECALIB_H
#define ALICEO2_ZDC_NOISECALIB_H
namespace o2
{
namespace zdc
{
class NoiseCalib
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  NoiseCalib() = default;
  int init();

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

  void clear();
  // int process(const o2::zdc::NoiseCalibSummaryData& data);
  int process(const o2::zdc::NoiseCalibSummaryData* data);
  void add(int ih, int iarr, o2::dataformats::FlatHisto1D<double>& h1);
  int endOfRun();
  int saveDebugHistos(const std::string fn = "ZDCNoiseCalib.root");
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }

  CcdbObjectInfo& getCcdbObjectInfo() { return mInfo; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  NoiseCalibData& getData() { return mData; }
  NoiseParam& getParam() { return mParam; }

  static constexpr int NHA = 3;
  std::array<std::array<o2::dataformats::FlatHisto1D<double>*, NChannels>, NHA> mH{};

 private:
  NoiseCalibData mData;
  NoiseParam mParam;
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  const ModuleConfig* mModuleConfig = nullptr; /// Trigger/readout configuration object
  int32_t mVerbosity = DbgMinimal;
  CcdbObjectInfo mInfo; /// CCDB Info
};
} // namespace zdc
} // namespace o2

#endif
