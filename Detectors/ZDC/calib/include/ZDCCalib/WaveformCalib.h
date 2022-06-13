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

#include <mutex>
#include <memory>
#include <TH1.h>
#include <TH2.h>
#include <THnBase.h>
#include <THnSparse.h>
#include <TMinuit.h>
#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibData.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include "CCDB/CcdbObjectInfo.h"
#ifndef ALICEO2_ZDC_WAVEFORMCALIB_H
#define ALICEO2_ZDC_WAVEFORMCALIB_H
namespace o2
{
namespace zdc
{
class WaveformCalib
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  WaveformCalib() = default;
  int init();
  void clear();
  int process(const WaveformCalibData& data); // Calibration of RUN3 data - aggregator node
  int endOfRun();                             // Perform minimization
  int saveDebugHistos(const std::string fn = "ZDCWaveformCalib.root");

  CcdbObjectInfo& getCcdbObjectInfo() { return mInfo; }

  void setConfig(const WaveformCalibConfig* param) { mConfig = param; };
  const WaveformCalibConfig* getConfig() const { return mConfig; };

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }

  WaveformCalibData& getData() { return mData; }

 private:
  WaveformCalibData mData;
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mVerbosity = DbgMinimal;
  const WaveformCalibConfig* mConfig = nullptr; /// Configuration of intercalibration
  CcdbObjectInfo mInfo;                         /// CCDB Info
};
} // namespace zdc
} // namespace o2

#endif
