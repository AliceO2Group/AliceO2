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
  void clear();
  // int process(const o2::zdc::NoiseCalibSummaryData& data);
  int process(const o2::zdc::NoiseCalibSummaryData* data);
  int endOfRun();
  int saveDebugHistos(const std::string fn = "ZDCNoiseCalib.root");

  CcdbObjectInfo& getCcdbObjectInfo() { return mInfo; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }

  NoiseCalibData& getData() { return mData; }
  NoiseParam& getParam() { return mParam; }

 private:
  NoiseCalibData mData;
  NoiseParam mParam;
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mVerbosity = DbgMinimal;
  CcdbObjectInfo mInfo; /// CCDB Info
};
} // namespace zdc
} // namespace o2

#endif
